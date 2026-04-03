"""GPU compute pipeline: frustum cull + LOD + gather on GPU.

Keeps particle data GPU-resident. Output buffers are used directly
by the wgpu render pipeline (zero-copy).
"""

import struct
import numpy as np
import wgpu
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"
WG_SIZE = 256  # workgroup size for all compute shaders


def _load_wgsl(name):
    return (SHADER_DIR / name).read_text()


def _div_ceil(n, d):
    return (n + d - 1) // d


class GPUCompute:
    """GPU-side frustum culling, LOD selection, and particle gathering."""

    def __init__(self, device):
        self.device = device
        self._n_cells = 0
        self._n_particles = 0
        self._n_levels = 0
        self._max_output = 0

        # Compile shaders
        self._cull_module = device.create_shader_module(code=_load_wgsl("frustum_cull.wgsl"))
        self._scan_module = device.create_shader_module(code=_load_wgsl("prefix_sum.wgsl"))
        self._gather_module = device.create_shader_module(code=_load_wgsl("gather.wgsl"))

        # Pipelines built lazily after upload (bind group layouts depend on data)
        self._cull_pipeline = None
        self._scan_local_pipeline = None
        self._scan_propagate_pipeline = None
        self._count_pipeline = None
        self._apply_stride_pipeline = None
        self._gather_pipeline = None

        # GPU buffers (populated by upload_snapshot)
        self._level_bufs = []  # per-level: {mass, hsml, centers, decision}
        self._particle_bufs = {}  # sorted pos/hsml/mass/qty
        self._cell_start_buf = None
        self._output_bufs = {}  # compacted output SoA (shared with renderer)
        self._cell_out_counts_buf = None
        self._counters_buf = None
        self._cull_params_buf = None
        self._gather_params_buf = None
        self._scan_params_buf = None

    def upload_snapshot(self, grid, max_output=4_000_000):
        """Upload grid structure and sorted particle data to GPU (blocking)."""
        self._prepare_snapshot(grid, max_output)
        for _ in self._upload_particle_steps(grid):
            pass

    def upload_snapshot_chunked(self, grid, max_output=4_000_000):
        """Prepare upload and return an iterator of upload steps.

        Call next() on the iterator each frame to spread the upload across frames.
        Returns None when complete.
        """
        self._prepare_snapshot(grid, max_output)
        return self._upload_particle_steps(grid)

    def _prepare_snapshot(self, grid, max_output):
        """Upload grid levels and allocate buffers (fast, <100ms)."""
        dev = self.device
        self._n_cells = grid.n_cells
        self._n_particles = len(grid.sorted_pos)
        self._n_levels = len(grid.levels)
        self._max_output = max_output

        nc3 = grid.n_cells ** 3

        # Upload per-level data (small, <10MB total)
        self._level_bufs = []
        for lv in grid.levels:
            nc = lv["nc"]
            n = nc ** 3
            hd = float(lv["half_diag"])

            centers4 = np.zeros((n, 4), dtype=np.float32)
            centers4[:, :3] = lv["centers"]
            centers4[:, 3] = hd

            buf_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            level_data = {
                "nc": nc,
                "mass": dev.create_buffer_with_data(
                    data=lv["mass"].astype(np.float32), usage=buf_usage),
                "hsml": dev.create_buffer_with_data(
                    data=lv["hsml"].astype(np.float32), usage=buf_usage),
                "centers": dev.create_buffer_with_data(
                    data=centers4, usage=buf_usage),
                "decision": dev.create_buffer(
                    size=n * 4, usage=buf_usage),
                "com": lv["com"],
                "hsml_np": lv["hsml"].copy(),
                "qty": lv["qty"],
                "cov": lv["cov"],
                "mh2": lv["mh2"],
                "mass_np": lv["mass"],
                "cs": lv["cs"],
            }
            self._level_bufs.append(level_data)

        # Allocate particle buffers (empty, fast)
        n = self._n_particles
        particle_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        self._particle_bufs = {
            "pos": dev.create_buffer(size=n * 16, usage=particle_usage),
            "hsml": dev.create_buffer(size=n * 4, usage=particle_usage),
            "mass": dev.create_buffer(size=n * 4, usage=particle_usage),
            "qty": dev.create_buffer(size=n * 4, usage=particle_usage),
        }

        # Cell start (convert int64 → u32)
        cell_start_u32 = grid.cell_start.astype(np.uint32)
        self._cell_start_buf = dev.create_buffer_with_data(
            data=cell_start_u32, usage=wgpu.BufferUsage.STORAGE)

        # Pre-allocate output buffers
        self._output_bufs = {
            "pos": dev.create_buffer(
                size=max_output * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "hsml": dev.create_buffer(
                size=max_output * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "mass": dev.create_buffer(
                size=max_output * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "qty": dev.create_buffer(
                size=max_output * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
        }

        # Work buffers
        self._cell_out_counts_buf = dev.create_buffer(
            size=nc3 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self._counters_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

        # Prefix sum block sums
        n_blocks_1 = _div_ceil(nc3, WG_SIZE)
        n_blocks_2 = _div_ceil(n_blocks_1, WG_SIZE)
        self._scan_block_sums_1 = dev.create_buffer(
            size=max(n_blocks_1 * 4, 4), usage=wgpu.BufferUsage.STORAGE)
        self._scan_block_sums_2 = dev.create_buffer(
            size=max(n_blocks_2 * 4, 4), usage=wgpu.BufferUsage.STORAGE)

        self._dummy_decision_buf = dev.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE)

        # Uniform buffers
        self._cull_params_buf = dev.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._gather_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._scan_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._upload_ready = False

    def _upload_particle_steps(self, grid):
        """Generator that yields after each ~64MB chunk of buffer writes."""
        dev = self.device
        CHUNK = 4_000_000  # particles per chunk (~64MB for pos, ~16MB for scalars)

        n = self._n_particles
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = grid.sorted_pos

        # Upload positions in chunks
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            dev.queue.write_buffer(
                self._particle_bufs["pos"], start * 16, pos4[start:end].tobytes())
            yield f"pos {start//CHUNK}"

        # Upload scalar arrays in chunks (yield per chunk)
        for name, arr in [("hsml", grid.sorted_hsml),
                          ("mass", grid.sorted_mass),
                          ("qty", grid.sorted_qty)]:
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                dev.queue.write_buffer(
                    self._particle_bufs[name], start * 4, arr[start:end].tobytes())
                yield f"{name} {start//CHUNK}"

        # Build pipelines
        self._build_pipelines()
        self._upload_ready = True
        yield "done"

    def upload_weights(self, grid):
        """Re-upload mass/qty arrays and per-level moments after field switch.

        Keeps particle pos/hsml and cell_start unchanged.
        """
        dev = self.device

        # Re-upload sorted mass/qty
        dev.queue.write_buffer(self._particle_bufs["mass"], 0, grid.sorted_mass.tobytes())
        dev.queue.write_buffer(self._particle_bufs["qty"], 0, grid.sorted_qty.tobytes())

        # Re-upload per-level data that depends on mass/qty
        for i, lv in enumerate(grid.levels):
            lb = self._level_bufs[i]
            dev.queue.write_buffer(lb["mass"], 0, lv["mass"].astype(np.float32).tobytes())
            dev.queue.write_buffer(lb["hsml"], 0, lv["hsml"].astype(np.float32).tobytes())
            # Update numpy arrays for summary readback
            lb["com"] = lv["com"]
            lb["hsml_np"] = lv["hsml"].copy()
            lb["qty"] = lv["qty"]
            lb["cov"] = lv["cov"]
            lb["mh2"] = lv["mh2"]
            lb["mass_np"] = lv["mass"]

    def _build_pipelines(self):
        """Build all compute pipelines."""
        dev = self.device

        # --- Frustum cull pipeline ---
        self._cull_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        cull_layout = dev.create_pipeline_layout(bind_group_layouts=[self._cull_bgl])
        self._cull_pipeline = dev.create_compute_pipeline(
            layout=cull_layout,
            compute={"module": self._cull_module, "entry_point": "main"},
        )

        # --- Prefix sum pipelines ---
        self._scan_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        ])
        scan_layout = dev.create_pipeline_layout(bind_group_layouts=[self._scan_bgl])
        self._scan_local_pipeline = dev.create_compute_pipeline(
            layout=scan_layout,
            compute={"module": self._scan_module, "entry_point": "scan_local"},
        )
        self._scan_propagate_pipeline = dev.create_compute_pipeline(
            layout=scan_layout,
            compute={"module": self._scan_module, "entry_point": "propagate"},
        )

        # --- Gather pipelines ---
        self._gather_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        self._gather_bgl1 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        ])
        self._gather_bgl2 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        gather_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._gather_bgl0, self._gather_bgl1, self._gather_bgl2])

        self._count_pipeline = dev.create_compute_pipeline(
            layout=gather_layout,
            compute={"module": self._gather_module, "entry_point": "count_particles"},
        )
        self._apply_stride_pipeline = dev.create_compute_pipeline(
            layout=gather_layout,
            compute={"module": self._gather_module, "entry_point": "apply_stride"},
        )
        self._gather_pipeline = dev.create_compute_pipeline(
            layout=gather_layout,
            compute={"module": self._gather_module, "entry_point": "gather_particles"},
        )

        # Build static bind groups
        self._gather_bg1 = dev.create_bind_group(
            layout=self._gather_bgl1,
            entries=[
                {"binding": 0, "resource": {"buffer": self._particle_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._particle_bufs["hsml"]}},
                {"binding": 2, "resource": {"buffer": self._particle_bufs["mass"]}},
                {"binding": 3, "resource": {"buffer": self._particle_bufs["qty"]}},
            ],
        )
        self._gather_bg2 = dev.create_bind_group(
            layout=self._gather_bgl2,
            entries=[
                {"binding": 0, "resource": {"buffer": self._output_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._output_bufs["hsml"]}},
                {"binding": 2, "resource": {"buffer": self._output_bufs["mass"]}},
                {"binding": 3, "resource": {"buffer": self._output_bufs["qty"]}},
            ],
        )

    def dispatch_cull(self, camera, max_particles, lod_pixels=4,
                      viewport_width=2048, summary_overlap=0.0):
        """Run GPU frustum cull + LOD + gather. Returns (n_particles, summary_data).

        n_particles: number of gathered particles in output buffers.
        summary_data: (pos, hsml, mass, qty, cov) numpy arrays for aniso summaries,
                      read back from CPU since summaries are few and need covariance assembly.
        """
        dev = self.device
        nc = self._n_cells
        nc3 = nc ** 3
        fov_rad = float(np.radians(camera.fov))
        pix_per_rad = viewport_width / (2.0 * np.tan(fov_rad / 2))

        # === Pass 1: Multi-level frustum cull + LOD ===
        # Each level must be a separate submit so the uniform buffer write
        # takes effect before the dispatch.
        for li in range(self._n_levels - 1, -1, -1):
            lb = self._level_bufs[li]
            level_nc = lb["nc"]
            level_nc3 = level_nc ** 3

            is_coarsest = 1 if li == self._n_levels - 1 else 0
            is_finest = 1 if li == 0 else 0
            parent_nc = self._level_bufs[li + 1]["nc"] if li < self._n_levels - 1 else 0

            params_data = struct.pack(
                "fff f fff f fff f fff f ffff IIII",
                *camera.position, 0.0,
                *camera.forward, 0.0,
                *camera.right, 0.0,
                *camera.up, 0.0,
                fov_rad, camera.aspect, pix_per_rad, float(lod_pixels),
                is_coarsest, is_finest, parent_nc, level_nc,
            )
            dev.queue.write_buffer(self._cull_params_buf, 0, params_data)

            parent_decision_buf = (
                self._level_bufs[li + 1]["decision"] if li < self._n_levels - 1
                else self._dummy_decision_buf
            )

            cull_bg = dev.create_bind_group(
                layout=self._cull_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._cull_params_buf}},
                    {"binding": 1, "resource": {"buffer": lb["mass"]}},
                    {"binding": 2, "resource": {"buffer": lb["hsml"]}},
                    {"binding": 3, "resource": {"buffer": lb["centers"]}},
                    {"binding": 4, "resource": {"buffer": parent_decision_buf}},
                    {"binding": 5, "resource": {"buffer": lb["decision"]}},
                ],
            )

            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._cull_pipeline)
            cpass.set_bind_group(0, cull_bg)
            cpass.dispatch_workgroups(_div_ceil(level_nc3, WG_SIZE))
            cpass.end()
            dev.queue.submit([encoder.finish()])

        # === Read back decisions to collect summaries (CPU) ===
        # Summaries are typically few (<10K) so CPU assembly of covariance is fine.
        # Collect summaries from all non-finest levels where decision == SUMMARY
        summary_parts = []
        for li in range(self._n_levels - 1, 0, -1):  # skip finest (li=0)
            lb = self._level_bufs[li]
            level_nc3 = lb["nc"] ** 3
            decision_data = dev.queue.read_buffer(lb["decision"], size=level_nc3 * 4)
            decisions = np.frombuffer(decision_data, dtype=np.uint32)
            s_idx = np.where(decisions == 1)[0]
            if len(s_idx) > 0:
                summary_parts.append((
                    lb["com"][s_idx],
                    lb["hsml_np"][s_idx],
                    lb["mass_np"][s_idx],
                    lb["qty"][s_idx],
                    lb["cov"][s_idx],
                    lb["mh2"][s_idx] / np.maximum(lb["mass_np"][s_idx], 1e-30),
                    lb["cs"],
                ))

        # === Pass 2: Count particles in EMIT cells ===
        # Clear counters
        dev.queue.write_buffer(self._counters_buf, 0, struct.pack("IIII", 0, 0, 0, 0))

        # Write gather params (stride=1 initially)
        dev.queue.write_buffer(self._gather_params_buf, 0,
                               struct.pack("IIII", nc3, max_particles, 0, 1))

        finest_decision_buf = self._level_bufs[0]["decision"]
        gather_bg0 = dev.create_bind_group(
            layout=self._gather_bgl0,
            entries=[
                {"binding": 0, "resource": {"buffer": self._gather_params_buf}},
                {"binding": 1, "resource": {"buffer": finest_decision_buf}},
                {"binding": 2, "resource": {"buffer": self._cell_start_buf}},
                {"binding": 3, "resource": {"buffer": self._cell_out_counts_buf}},
                {"binding": 4, "resource": {"buffer": self._counters_buf}},
            ],
        )

        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._count_pipeline)
        cpass.set_bind_group(0, gather_bg0)
        cpass.set_bind_group(1, self._gather_bg1)
        cpass.set_bind_group(2, self._gather_bg2)
        cpass.dispatch_workgroups(_div_ceil(nc3, WG_SIZE))
        cpass.end()
        dev.queue.submit([encoder.finish()])

        # Read total visible count
        counter_data = dev.queue.read_buffer(self._counters_buf, size=16)
        total_visible = struct.unpack("I", counter_data[:4])[0]

        # Compute stride — grow output buffer if needed.
        # Use ~25% of max buffer size for output (rest is source data + overhead).
        # 28 bytes/particle (pos=16 + hsml=4 + mass=4 + qty=4), pos is largest at 16.
        max_buf_bytes = self.device.adapter.limits.get("max-buffer-size", 2**30)
        MAX_OUTPUT_CAP = max_buf_bytes // (16 * 4)  # conservative: 25% of limit for pos buffer
        n_summaries = sum(len(p[0]) for p in summary_parts) if summary_parts else 0
        budget = min(max(max_particles - n_summaries, max_particles // 2), MAX_OUTPUT_CAP)
        if budget > self._max_output:
            self._grow_output_buffers(budget)
        stride = max(1, total_visible // budget) if total_visible > budget else 1

        # === Pass 3: Apply stride to get per-cell output counts ===
        dev.queue.write_buffer(self._gather_params_buf, 0,
                               struct.pack("IIII", nc3, budget, total_visible, stride))

        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._apply_stride_pipeline)
        cpass.set_bind_group(0, gather_bg0)
        cpass.set_bind_group(1, self._gather_bg1)
        cpass.set_bind_group(2, self._gather_bg2)
        cpass.dispatch_workgroups(_div_ceil(nc3, WG_SIZE))
        cpass.end()
        dev.queue.submit([encoder.finish()])

        # === Pass 4: Prefix sum on cell_out_counts ===
        self._prefix_sum(self._cell_out_counts_buf, nc3)

        # Read exact output count from counters[1] (written by apply_stride)
        counter_data2 = dev.queue.read_buffer(self._counters_buf, size=16)
        n_output = struct.unpack("I", counter_data2[4:8])[0]
        n_output = min(n_output, self._max_output)

        # === Pass 5: Gather particles ===
        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._gather_pipeline)
        cpass.set_bind_group(0, gather_bg0)
        cpass.set_bind_group(1, self._gather_bg1)
        cpass.set_bind_group(2, self._gather_bg2)
        cpass.dispatch_workgroups(_div_ceil(nc3, WG_SIZE))
        cpass.end()
        dev.queue.submit([encoder.finish()])

        # === Assemble summary output ===
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros(0, dtype=np.float32)
        z6 = np.zeros((0, 6), dtype=np.float32)

        if summary_parts:
            s_pos = np.concatenate([p[0] for p in summary_parts]).astype(np.float32)
            s_hsml = np.concatenate([p[1] for p in summary_parts]).astype(np.float32)
            s_mass = np.concatenate([p[2] for p in summary_parts]).astype(np.float32)
            s_qty = np.concatenate([p[3] for p in summary_parts]).astype(np.float32)
            s_cov = np.concatenate([p[4] for p in summary_parts]).astype(np.float32)
            s_mean_h2 = np.concatenate([p[5] for p in summary_parts]).astype(np.float32)
            s_cs2 = np.concatenate([
                np.broadcast_to((p[6] ** 2)[None, :], (len(p[0]), 3))
                for p in summary_parts
            ]).astype(np.float32)
            # Add kernel smoothing and overlap padding to covariance diagonal
            s_cov[:, 0] += 0.225 * s_mean_h2
            s_cov[:, 3] += 0.225 * s_mean_h2
            s_cov[:, 5] += 0.225 * s_mean_h2
            alpha = summary_overlap
            s_cov[:, 0] += alpha * s_cs2[:, 0]
            s_cov[:, 3] += alpha * s_cs2[:, 1]
            s_cov[:, 5] += alpha * s_cs2[:, 2]
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return n_output, total_visible, (s_pos, s_hsml, s_mass, s_qty, s_cov)

    def _prefix_sum(self, buf, n):
        """Run multi-level prefix sum on a u32 storage buffer. Returns total sum."""
        dev = self.device
        n_blocks_1 = _div_ceil(n, WG_SIZE)

        # Level 1: scan local blocks
        dev.queue.write_buffer(self._scan_params_buf, 0, struct.pack("IIII", n, 0, 0, 0))
        scan_bg = dev.create_bind_group(
            layout=self._scan_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf}},
                {"binding": 1, "resource": {"buffer": self._scan_block_sums_1}},
                {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
            ],
        )
        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._scan_local_pipeline)
        cpass.set_bind_group(0, scan_bg)
        cpass.dispatch_workgroups(n_blocks_1)
        cpass.end()
        dev.queue.submit([encoder.finish()])

        if n_blocks_1 > 1:
            # Level 2: scan block sums
            n_blocks_2 = _div_ceil(n_blocks_1, WG_SIZE)
            dev.queue.write_buffer(self._scan_params_buf, 0,
                                   struct.pack("IIII", n_blocks_1, 0, 0, 0))
            scan_bg2 = dev.create_bind_group(
                layout=self._scan_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._scan_block_sums_1}},
                    {"binding": 1, "resource": {"buffer": self._scan_block_sums_2}},
                    {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                ],
            )
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._scan_local_pipeline)
            cpass.set_bind_group(0, scan_bg2)
            cpass.dispatch_workgroups(n_blocks_2)
            cpass.end()
            dev.queue.submit([encoder.finish()])

            if n_blocks_2 > 1:
                # Level 3: scan level-2 block sums (for nc3=262144, n_blocks_2=4, fits in one WG)
                n_blocks_3 = _div_ceil(n_blocks_2, WG_SIZE)
                dev.queue.write_buffer(self._scan_params_buf, 0,
                                       struct.pack("IIII", n_blocks_2, 0, 0, 0))
                scan_bg3 = dev.create_bind_group(
                    layout=self._scan_bgl,
                    entries=[
                        {"binding": 0, "resource": {"buffer": self._scan_block_sums_2}},
                        {"binding": 1, "resource": {"buffer": self._scan_block_sums_2}},  # dummy
                        {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                    ],
                )
                encoder = dev.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                cpass.set_pipeline(self._scan_local_pipeline)
                cpass.set_bind_group(0, scan_bg3)
                cpass.dispatch_workgroups(1)
                cpass.end()
                dev.queue.submit([encoder.finish()])

                # Propagate level-2 block sums into level-1 block sums
                dev.queue.write_buffer(self._scan_params_buf, 0,
                                       struct.pack("IIII", n_blocks_1, 0, 0, 0))
                prop_bg2 = dev.create_bind_group(
                    layout=self._scan_bgl,
                    entries=[
                        {"binding": 0, "resource": {"buffer": self._scan_block_sums_1}},
                        {"binding": 1, "resource": {"buffer": self._scan_block_sums_2}},
                        {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                    ],
                )
                encoder = dev.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                cpass.set_pipeline(self._scan_propagate_pipeline)
                cpass.set_bind_group(0, prop_bg2)
                cpass.dispatch_workgroups(n_blocks_1)
                cpass.end()
                dev.queue.submit([encoder.finish()])

            # Propagate level-1 block sums into data
            dev.queue.write_buffer(self._scan_params_buf, 0, struct.pack("IIII", n, 0, 0, 0))
            prop_bg = dev.create_bind_group(
                layout=self._scan_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf}},
                    {"binding": 1, "resource": {"buffer": self._scan_block_sums_1}},
                    {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                ],
            )
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._scan_propagate_pipeline)
            cpass.set_bind_group(0, prop_bg)
            cpass.dispatch_workgroups(_div_ceil(n, WG_SIZE))
            cpass.end()
            dev.queue.submit([encoder.finish()])

        # Total output count not needed here — caller computes from counters
        return 0

    # ---- GPU-side LOS projection ----

    def upload_vector_field(self, grid, field_name, data_manager):
        """Upload a sorted 3-component vector field to GPU for LOS projection.

        Args:
            grid: SpatialGrid (for sort_order)
            field_name: e.g. "Velocities"
            data_manager: SnapshotData instance
        """
        dev = self.device
        vec = data_manager.get_vector_field(field_name)  # (N, 3) float32
        sorted_vec = vec[grid.sort_order].astype(np.float32)

        # Pack as vec4 for alignment
        n = len(sorted_vec)
        vec4 = np.zeros((n, 4), dtype=np.float32)
        vec4[:, :3] = sorted_vec

        buf_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        self._los_vec_buf = dev.create_buffer_with_data(data=vec4, usage=buf_usage)
        self._los_field_name = field_name
        self._los_n = n

        # Compile LOS shader if not yet done
        if not hasattr(self, '_los_pipeline'):
            los_module = dev.create_shader_module(code=_load_wgsl("los_project.wgsl"))
            self._los_bgl = dev.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            ])
            los_layout = dev.create_pipeline_layout(bind_group_layouts=[self._los_bgl])
            self._los_pipeline = dev.create_compute_pipeline(
                layout=los_layout,
                compute={"module": los_module, "entry_point": "main"},
            )
            self._los_params_buf = dev.create_buffer(
                size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def dispatch_los_project(self, camera):
        """Project the uploaded vector field along camera.forward into the particle qty buffer.

        Updates _particle_bufs["qty"] in-place on GPU. No CPU readback needed.
        Call this when the camera rotates while a LOS vector field is active.
        """
        if not hasattr(self, '_los_vec_buf') or self._los_vec_buf is None:
            return

        dev = self.device
        n = self._los_n

        # 2D dispatch to stay within 65535 workgroups per dimension
        n_wg = _div_ceil(n, WG_SIZE)
        dispatch_x = min(n_wg, 65535)
        dispatch_y = _div_ceil(n_wg, dispatch_x)

        params = struct.pack("fffI IIII", *camera.forward, n, dispatch_x, 0, 0, 0)
        dev.queue.write_buffer(self._los_params_buf, 0, params)

        los_bg = dev.create_bind_group(
            layout=self._los_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._los_params_buf}},
                {"binding": 1, "resource": {"buffer": self._los_vec_buf}},
                {"binding": 2, "resource": {"buffer": self._particle_bufs["qty"]}},
            ],
        )

        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._los_pipeline)
        cpass.set_bind_group(0, los_bg)
        cpass.dispatch_workgroups(dispatch_x, dispatch_y)
        cpass.end()
        dev.queue.submit([encoder.finish()])

    def has_los_field(self):
        return hasattr(self, '_los_vec_buf') and self._los_vec_buf is not None

    def _grow_output_buffers(self, new_max):
        """Re-allocate output buffers for a larger budget."""
        dev = self.device
        self._max_output = new_max
        self._output_bufs = {
            "pos": dev.create_buffer(
                size=new_max * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "hsml": dev.create_buffer(
                size=new_max * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "mass": dev.create_buffer(
                size=new_max * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
            "qty": dev.create_buffer(
                size=new_max * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC),
        }
        # Rebuild gather bind group for new output buffers
        self._gather_bg2 = dev.create_bind_group(
            layout=self._gather_bgl2,
            entries=[
                {"binding": 0, "resource": {"buffer": self._output_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._output_bufs["hsml"]}},
                {"binding": 2, "resource": {"buffer": self._output_bufs["mass"]}},
                {"binding": 3, "resource": {"buffer": self._output_bufs["qty"]}},
            ],
        )

    def get_output_buffers(self):
        """Return output SoA buffers for direct use by the render pipeline."""
        return self._output_bufs

    def get_sort_index_buffer(self):
        """Return sort index buffer (valid after dispatch_sort)."""
        return self._sort_index_a

    # ---- Depth sorting ----

    def _init_sort(self, n):
        """Initialize sort buffers for n particles."""
        dev = self.device
        buf_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

        self._sort_keys_a = dev.create_buffer(size=n * 4, usage=buf_usage)
        self._sort_keys_b = dev.create_buffer(size=n * 4, usage=buf_usage)
        self._sort_index_a = dev.create_buffer(size=n * 4, usage=buf_usage)
        self._sort_index_b = dev.create_buffer(size=n * 4, usage=buf_usage)
        self._sort_n = n

        # Histogram: 16 digits × n_workgroups
        n_wg = _div_ceil(n, WG_SIZE)
        hist_size = 16 * n_wg
        self._sort_histogram = dev.create_buffer(size=hist_size * 4, usage=buf_usage)
        self._sort_n_wg = n_wg
        self._sort_hist_size = hist_size

        # Prefix sum block sums for histogram
        n_hist_blocks = _div_ceil(hist_size, WG_SIZE)
        n_hist_blocks2 = _div_ceil(n_hist_blocks, WG_SIZE)
        self._sort_hist_block_sums_1 = dev.create_buffer(
            size=max(n_hist_blocks * 4, 4), usage=buf_usage)
        self._sort_hist_block_sums_2 = dev.create_buffer(
            size=max(n_hist_blocks2 * 4, 4), usage=buf_usage)

        # Compile sort shaders
        self._depth_module = dev.create_shader_module(code=_load_wgsl("depth_keys.wgsl"))
        self._sort_module = dev.create_shader_module(code=_load_wgsl("radix_sort.wgsl"))

        # Depth keys pipeline
        self._depth_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        depth_layout = dev.create_pipeline_layout(bind_group_layouts=[self._depth_bgl])
        self._depth_pipeline = dev.create_compute_pipeline(
            layout=depth_layout,
            compute={"module": self._depth_module, "entry_point": "main"},
        )

        # Radix sort pipelines
        self._sort_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        sort_layout = dev.create_pipeline_layout(bind_group_layouts=[self._sort_bgl])

        self._histogram_pipeline = dev.create_compute_pipeline(
            layout=sort_layout,
            compute={"module": self._sort_module, "entry_point": "build_histogram"},
        )
        self._scatter_pipeline = dev.create_compute_pipeline(
            layout=sort_layout,
            compute={"module": self._sort_module, "entry_point": "scatter"},
        )

        # Depth params uniform
        self._depth_params_buf = dev.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._sort_params_buf_sort = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def dispatch_sort(self, camera, n_particles):
        """Sort gathered particles by depth (back-to-front).

        Must be called after dispatch_cull(). Uses the output pos buffer
        for depth computation. After sort, sort_index buffer contains
        the permutation for back-to-front rendering.

        Returns the sort index buffer.
        """
        dev = self.device

        if not hasattr(self, '_sort_n') or self._sort_n < n_particles:
            self._init_sort(n_particles)

        n = n_particles
        n_wg = _div_ceil(n, WG_SIZE)

        # Stage 0: Compute depth keys and initialize sort indices
        depth_data = struct.pack("fff f fff I",
                                 *camera.position, 0.0,
                                 *camera.forward, n)
        dev.queue.write_buffer(self._depth_params_buf, 0, depth_data)

        depth_bg = dev.create_bind_group(
            layout=self._depth_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._depth_params_buf}},
                {"binding": 1, "resource": {"buffer": self._output_bufs["pos"]}},
                {"binding": 2, "resource": {"buffer": self._sort_keys_a}},
                {"binding": 3, "resource": {"buffer": self._sort_index_a}},
            ],
        )

        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._depth_pipeline)
        cpass.set_bind_group(0, depth_bg)
        cpass.dispatch_workgroups(n_wg)
        cpass.end()
        dev.queue.submit([encoder.finish()])

        # Stage 1-8: Radix sort passes (4-bit radix, 8 passes for 32-bit keys)
        keys_a, keys_b = self._sort_keys_a, self._sort_keys_b
        idx_a, idx_b = self._sort_index_a, self._sort_index_b

        for pass_idx in range(8):
            bit_offset = pass_idx * 4

            sort_data = struct.pack("IIII", n, bit_offset, n_wg, 0)
            dev.queue.write_buffer(self._sort_params_buf_sort, 0, sort_data)

            # Clear histogram
            dev.queue.write_buffer(self._sort_histogram, 0,
                                   bytes(16 * n_wg * 4))

            sort_bg = dev.create_bind_group(
                layout=self._sort_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._sort_params_buf_sort}},
                    {"binding": 1, "resource": {"buffer": keys_a}},
                    {"binding": 2, "resource": {"buffer": idx_a}},
                    {"binding": 3, "resource": {"buffer": keys_b}},
                    {"binding": 4, "resource": {"buffer": idx_b}},
                    {"binding": 5, "resource": {"buffer": self._sort_histogram}},
                ],
            )

            # Build histogram
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._histogram_pipeline)
            cpass.set_bind_group(0, sort_bg)
            cpass.dispatch_workgroups(n_wg)
            cpass.end()
            dev.queue.submit([encoder.finish()])

            # Prefix sum on histogram
            hist_n = 16 * n_wg
            self._prefix_sum_sort(self._sort_histogram, hist_n)

            # Scatter
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._scatter_pipeline)
            cpass.set_bind_group(0, sort_bg)
            cpass.dispatch_workgroups(n_wg)
            cpass.end()
            dev.queue.submit([encoder.finish()])

            # Swap buffers for next pass
            keys_a, keys_b = keys_b, keys_a
            idx_a, idx_b = idx_b, idx_a

        # After 8 passes (even number), result is in keys_a/idx_a = the original buffers
        # _sort_index_a now contains the sorted permutation
        self._sort_index_a = idx_a
        return self._sort_index_a

    def _prefix_sum_sort(self, buf, n):
        """Prefix sum for sort histogram (may be larger than cell grid)."""
        # Reuse the existing prefix sum infrastructure but with sort-specific block sums
        dev = self.device
        n_blocks_1 = _div_ceil(n, WG_SIZE)

        dev.queue.write_buffer(self._scan_params_buf, 0, struct.pack("IIII", n, 0, 0, 0))
        scan_bg = dev.create_bind_group(
            layout=self._scan_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf}},
                {"binding": 1, "resource": {"buffer": self._sort_hist_block_sums_1}},
                {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
            ],
        )
        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._scan_local_pipeline)
        cpass.set_bind_group(0, scan_bg)
        cpass.dispatch_workgroups(n_blocks_1)
        cpass.end()
        dev.queue.submit([encoder.finish()])

        if n_blocks_1 > 1:
            n_blocks_2 = _div_ceil(n_blocks_1, WG_SIZE)
            dev.queue.write_buffer(self._scan_params_buf, 0,
                                   struct.pack("IIII", n_blocks_1, 0, 0, 0))
            scan_bg2 = dev.create_bind_group(
                layout=self._scan_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._sort_hist_block_sums_1}},
                    {"binding": 1, "resource": {"buffer": self._sort_hist_block_sums_2}},
                    {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                ],
            )
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._scan_local_pipeline)
            cpass.set_bind_group(0, scan_bg2)
            cpass.dispatch_workgroups(n_blocks_2)
            cpass.end()
            dev.queue.submit([encoder.finish()])

            if n_blocks_2 > 1:
                dev.queue.write_buffer(self._scan_params_buf, 0,
                                       struct.pack("IIII", n_blocks_2, 0, 0, 0))
                scan_bg3 = dev.create_bind_group(
                    layout=self._scan_bgl,
                    entries=[
                        {"binding": 0, "resource": {"buffer": self._sort_hist_block_sums_2}},
                        {"binding": 1, "resource": {"buffer": self._sort_hist_block_sums_2}},
                        {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                    ],
                )
                encoder = dev.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                cpass.set_pipeline(self._scan_local_pipeline)
                cpass.set_bind_group(0, scan_bg3)
                cpass.dispatch_workgroups(1)
                cpass.end()
                dev.queue.submit([encoder.finish()])

                dev.queue.write_buffer(self._scan_params_buf, 0,
                                       struct.pack("IIII", n_blocks_1, 0, 0, 0))
                prop_bg2 = dev.create_bind_group(
                    layout=self._scan_bgl,
                    entries=[
                        {"binding": 0, "resource": {"buffer": self._sort_hist_block_sums_1}},
                        {"binding": 1, "resource": {"buffer": self._sort_hist_block_sums_2}},
                        {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                    ],
                )
                encoder = dev.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                cpass.set_pipeline(self._scan_propagate_pipeline)
                cpass.set_bind_group(0, prop_bg2)
                cpass.dispatch_workgroups(n_blocks_1)
                cpass.end()
                dev.queue.submit([encoder.finish()])

            dev.queue.write_buffer(self._scan_params_buf, 0, struct.pack("IIII", n, 0, 0, 0))
            prop_bg = dev.create_bind_group(
                layout=self._scan_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf}},
                    {"binding": 1, "resource": {"buffer": self._sort_hist_block_sums_1}},
                    {"binding": 2, "resource": {"buffer": self._scan_params_buf}},
                ],
            )
            encoder = dev.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._scan_propagate_pipeline)
            cpass.set_bind_group(0, prop_bg)
            cpass.dispatch_workgroups(_div_ceil(n, WG_SIZE))
            cpass.end()
            dev.queue.submit([encoder.finish()])

    def release(self):
        self._level_bufs = []
        self._particle_bufs = {}
        self._output_bufs = {}
