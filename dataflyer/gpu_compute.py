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


def _make_bind_group(dev, layout, buffers):
    """Create bind group from layout + ordered list of buffers."""
    return dev.create_bind_group(
        layout=layout,
        entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)],
    )


def _make_compute_bgl(dev, buffer_types):
    """Create compute bind group layout from list of buffer type strings.

    Each entry is 'uniform', 'read-only-storage', or 'storage'.
    """
    return dev.create_bind_group_layout(entries=[
        {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
        for i, t in enumerate(buffer_types)
    ])


def _make_compute_pipeline(dev, bgl_list, module, entry_point):
    """Create compute pipeline from bind group layouts, shader module, and entry point."""
    layout = dev.create_pipeline_layout(bind_group_layouts=bgl_list)
    return dev.create_compute_pipeline(
        layout=layout, compute={"module": module, "entry_point": entry_point})


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
        self._subsample_module = device.create_shader_module(
            code=_load_wgsl("subsample_cull.wgsl"))

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

    def _dispatch(self, pipeline, bind_groups, workgroups, wy=1):
        """Submit a single compute dispatch."""
        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        if isinstance(bind_groups, list):
            for i, bg in enumerate(bind_groups):
                cpass.set_bind_group(i, bg)
        else:
            cpass.set_bind_group(0, bind_groups)
        cpass.dispatch_workgroups(workgroups, wy)
        cpass.end()
        self.device.queue.submit([encoder.finish()])

    def upload_subsample_only(self, grid, max_output=4_000_000):
        """Minimal upload for the brute-force subsample cull pipeline.

        Uses the grid's raw particle arrays (no Morton sort, no tree).
        Works with deferred-build AdaptiveOctree where sorted_* are None.
        """
        dev = self.device

        # Prefer raw (unsorted) arrays if available, else sorted.
        pos_src = getattr(grid, "_raw_pos", None)
        if pos_src is None:
            pos_src = grid.sorted_pos
            hsml_src = grid.sorted_hsml
            mass_src = grid.sorted_mass
            qty_src = grid.sorted_qty
        else:
            hsml_src = grid._raw_hsml
            mass_src = grid._raw_mass
            qty_src = grid._raw_qty

        n = len(pos_src)
        self._n_particles = n
        self._max_output = max_output

        particle_usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                          | wgpu.BufferUsage.COPY_SRC)
        self._particle_bufs = {
            "pos": dev.create_buffer(size=n * 16, usage=particle_usage),
            "hsml": dev.create_buffer(size=n * 4, usage=particle_usage),
            "mass": dev.create_buffer(size=n * 4, usage=particle_usage),
            "qty": dev.create_buffer(size=n * 4, usage=particle_usage),
        }

        out_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        self._output_bufs = {
            "pos": dev.create_buffer(size=max_output * 16, usage=out_usage),
            "hsml": dev.create_buffer(size=max_output * 4, usage=out_usage),
            "mass": dev.create_buffer(size=max_output * 4, usage=out_usage),
            "qty": dev.create_buffer(size=max_output * 4, usage=out_usage),
        }

        # Upload positions (chunked to keep submits small)
        CHUNK = 4_000_000
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = pos_src
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            dev.queue.write_buffer(
                self._particle_bufs["pos"], start * 16, pos4[start:end].tobytes())
        for name, arr in [("hsml", hsml_src),
                          ("mass", mass_src),
                          ("qty", qty_src)]:
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                dev.queue.write_buffer(
                    self._particle_bufs[name], start * 4,
                    arr[start:end].tobytes())

        # Build only the subsample pipeline (not the tree-LOD ones)
        self._build_subsample_pipeline()
        self._upload_ready = True

    def _build_subsample_pipeline(self):
        """Build just the brute-force subsample pipeline."""
        dev = self.device
        _bgl = _make_compute_bgl
        _bg = _make_bind_group
        _pipe = _make_compute_pipeline

        self._subsample_bgl0 = _bgl(dev, [
            "uniform", "read-only-storage", "read-only-storage",
            "read-only-storage", "read-only-storage", "storage"])
        self._subsample_bgl1 = _bgl(dev, ["storage"] * 4)
        self._subsample_pipeline = _pipe(
            dev, [self._subsample_bgl0, self._subsample_bgl1],
            self._subsample_module, "main")
        self._subsample_params_buf = dev.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._subsample_counter_buf = dev.create_buffer(
            size=16,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST))
        pb = self._particle_bufs
        ob = self._output_bufs
        self._subsample_bg0 = _bg(dev, self._subsample_bgl0, [
            self._subsample_params_buf,
            pb["pos"], pb["hsml"], pb["mass"], pb["qty"],
            self._subsample_counter_buf])
        self._subsample_bg1 = _bg(dev, self._subsample_bgl1, [
            ob["pos"], ob["hsml"], ob["mass"], ob["qty"]])

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

        # Upload per-level data (small, <10MB total)
        self._level_bufs = []
        for lv in grid.levels:
            n = len(lv["mass"])  # number of nodes at this level
            hd = float(lv["half_diag"])

            centers4 = np.zeros((n, 4), dtype=np.float32)
            centers4[:, :3] = lv["centers"]
            centers4[:, 3] = hd

            buf_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            # Pack summary data for GPU gather
            com4 = np.zeros((n, 4), dtype=np.float32)
            com4[:, :3] = lv["com"]
            cov_packed = np.zeros((n * 2, 4), dtype=np.float32)
            cov_packed[0::2, :3] = lv["cov"][:, :3]  # xx, xy, xz
            cov_packed[1::2, :3] = lv["cov"][:, 3:]  # yy, yz, zz

            # Upload parent_idx buffer (maps each node to its parent in the parent level).
            # AdaptiveOctree provides this directly; for SpatialGrid we compute it
            # from the 3D grid structure: parent_idx = (ix/2)*pnc^2 + (iy/2)*pnc + (iz/2)
            if "parent_idx" in lv:
                parent_idx = lv["parent_idx"]
            else:
                nc = lv["nc"]  # cells per side (SpatialGrid convention)
                pnc = nc // 2 if nc > 2 else 1
                ix = np.arange(n, dtype=np.uint32) // (nc * nc)
                iy = (np.arange(n, dtype=np.uint32) // nc) % nc
                iz = np.arange(n, dtype=np.uint32) % nc
                parent_idx = (ix // 2) * pnc * pnc + (iy // 2) * pnc + (iz // 2)
            level_data = {
                "n_nodes": n,
                "mass": dev.create_buffer_with_data(
                    data=lv["mass"].astype(np.float32), usage=buf_usage),
                "hsml": dev.create_buffer_with_data(
                    data=lv["hsml"].astype(np.float32), usage=buf_usage),
                "centers": dev.create_buffer_with_data(
                    data=centers4, usage=buf_usage),
                "decision": dev.create_buffer(
                    size=n * 4, usage=buf_usage),
                "com_gpu": dev.create_buffer_with_data(data=com4, usage=buf_usage),
                "qty_gpu": dev.create_buffer_with_data(
                    data=lv["qty"].astype(np.float32), usage=buf_usage),
                "cov_gpu": dev.create_buffer_with_data(data=cov_packed, usage=buf_usage),
                "mh2_gpu": dev.create_buffer_with_data(
                    data=lv["mh2"].astype(np.float32), usage=buf_usage),
                "parent_idx": dev.create_buffer_with_data(
                    data=parent_idx.astype(np.uint32), usage=buf_usage),
                "cs": lv["cs"].copy(),
                "half_diag": hd,
                # Numpy copies for upload_weights refresh
                "mass_np": lv["mass"],
                "hsml_np": lv["hsml"].copy(),
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

        # Per-slot persistent buffers for composite mode (avoid re-uploading every frame)
        self._slot_bufs = [
            {"mass": dev.create_buffer(size=n * 4, usage=particle_usage),
             "qty": dev.create_buffer(size=n * 4, usage=particle_usage),
             "id": None},  # slot config id for cache invalidation
            {"mass": dev.create_buffer(size=n * 4, usage=particle_usage),
             "qty": dev.create_buffer(size=n * 4, usage=particle_usage),
             "id": None},
        ]

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

        # Work buffers (sized for leaf cells)
        n_leaves = len(grid.levels[0]["mass"])
        self._n_leaves = n_leaves
        self._cell_out_counts_buf = dev.create_buffer(
            size=n_leaves * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self._counters_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

        # Prefix sum block sums
        n_blocks_1 = _div_ceil(n_leaves, WG_SIZE)
        n_blocks_2 = _div_ceil(n_blocks_1, WG_SIZE)
        self._scan_block_sums_1 = dev.create_buffer(
            size=max(n_blocks_1 * 4, 4), usage=wgpu.BufferUsage.STORAGE)
        self._scan_block_sums_2 = dev.create_buffer(
            size=max(n_blocks_2 * 4, 4), usage=wgpu.BufferUsage.STORAGE)

        # Summary output buffers (max ~50K summaries is generous)
        MAX_SUMMARIES = 50_000
        self._max_summaries = MAX_SUMMARIES
        out_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        self._summary_bufs = {
            "pos": dev.create_buffer(size=MAX_SUMMARIES * 16, usage=out_usage),
            "mass": dev.create_buffer(size=MAX_SUMMARIES * 4, usage=out_usage),
            "qty": dev.create_buffer(size=MAX_SUMMARIES * 4, usage=out_usage),
            "cov": dev.create_buffer(size=MAX_SUMMARIES * 2 * 16, usage=out_usage),  # 2 vec4 per summary
        }
        # Per-level summary count buffer (reused for each level's prefix sum)
        max_level_nodes = max(len(lv["mass"]) for lv in grid.levels)
        self._summary_counts_buf = dev.create_buffer(
            size=max_level_nodes * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self._summary_counters_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self._summary_params_buf = dev.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._dummy_decision_buf = dev.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE)

        # Uniform buffers
        self._cull_params_buf = dev.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._gather_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._scan_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # Per-level cull uniform buffers (one per level, written once per frame)
        n_levels = len(grid.levels)
        self._per_level_cull_params = [
            dev.create_buffer(size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            for _ in range(n_levels)
        ]
        # Per-level summary params buffers
        self._per_level_summary_params = [
            dev.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            for _ in range(n_levels)
        ]

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

        # Re-upload mass/qty. In subsample-only mode the particle buffers
        # come from the grid's raw arrays (no Morton sort), so use those
        # if sorted_* are not populated. Cast to float32 to match the
        # buffer's element size.
        mass_src = grid.sorted_mass if grid.sorted_mass is not None else grid._raw_mass
        qty_src = grid.sorted_qty if grid.sorted_qty is not None else grid._raw_qty
        mass_f32 = np.ascontiguousarray(mass_src, dtype=np.float32)
        qty_f32 = np.ascontiguousarray(qty_src, dtype=np.float32)

        # Subsample-only mode splits particles across multiple per-chunk
        # buffers. Write each chunk's slice into its own buffer; otherwise
        # we'd overrun the (single-chunk) legacy alias and hang Metal.
        chunks = getattr(self, "_chunk_bufs", None)
        if chunks:
            # In SurfaceDensity mode update_weights sets _raw_qty to the
            # same array as _raw_mass. The qty buffer is then unused by
            # the resolve shader, so we can skip uploading it entirely
            # and halve the staging traffic.
            qty_is_mass = qty_src is mass_src
            for cb in chunks:
                start, cn = cb["start"], cb["n"]
                dev.queue.write_buffer(
                    cb["mass"], 0, mass_f32[start:start + cn].tobytes())
                if not qty_is_mass:
                    dev.queue.write_buffer(
                        cb["qty"], 0, qty_f32[start:start + cn].tobytes())
            # One blocking sync at the end forces all the queued writes to
            # complete before we return to the render loop. Without this,
            # the next dispatch_subsample_cull's read_buffer deadlocks
            # behind ~1 GB of pending uploads. read_buffer here works
            # because we are still inside the overlay click callback, not
            # inside the render frame callback.
            dev.queue.read_buffer(chunks[0]["mass"], size=4)
        else:
            dev.queue.write_buffer(self._particle_bufs["mass"], 0, mass_f32.tobytes())
            dev.queue.write_buffer(self._particle_bufs["qty"], 0, qty_f32.tobytes())

        # Subsample-only mode has no level buffers — skip per-level upload.
        if not self._level_bufs:
            return

        # Re-upload per-level data that depends on mass/qty
        for i, lv in enumerate(grid.levels):
            lb = self._level_bufs[i]
            n = lb["n_nodes"]
            dev.queue.write_buffer(lb["mass"], 0, lv["mass"].astype(np.float32).tobytes())
            dev.queue.write_buffer(lb["hsml"], 0, lv["hsml"].astype(np.float32).tobytes())
            # Summary GPU buffers
            com4 = np.zeros((n, 4), dtype=np.float32)
            com4[:, :3] = lv["com"]
            dev.queue.write_buffer(lb["com_gpu"], 0, com4.tobytes())
            dev.queue.write_buffer(lb["qty_gpu"], 0, lv["qty"].astype(np.float32).tobytes())
            cov_packed = np.zeros((n * 2, 4), dtype=np.float32)
            cov_packed[0::2, :3] = lv["cov"][:, :3]
            cov_packed[1::2, :3] = lv["cov"][:, 3:]
            dev.queue.write_buffer(lb["cov_gpu"], 0, cov_packed.tobytes())
            dev.queue.write_buffer(lb["mh2_gpu"], 0, lv["mh2"].astype(np.float32).tobytes())
            lb["mass_np"] = lv["mass"]
            lb["hsml_np"] = lv["hsml"].copy()

    def _build_pipelines(self):
        """Build all compute pipelines."""
        dev = self.device
        _bgl = _make_compute_bgl
        _bg = _make_bind_group
        _pipe = _make_compute_pipeline

        # --- Frustum cull pipeline (7 bindings: params, mass, hsml, centers, parent_decision, decision, parent_idx) ---
        self._cull_bgl = _bgl(dev, [
            "uniform", "read-only-storage", "read-only-storage",
            "read-only-storage", "read-only-storage", "storage",
            "read-only-storage"])
        self._cull_pipeline = _pipe(dev, [self._cull_bgl], self._cull_module, "main")

        # --- Compute stride pipeline ---
        stride_module = dev.create_shader_module(code=_load_wgsl("compute_stride.wgsl"))
        self._stride_bgl = _bgl(dev, ["uniform", "storage"])
        self._stride_pipeline = _pipe(dev, [self._stride_bgl], stride_module, "main")

        # --- Prefix sum pipelines ---
        self._scan_bgl = _bgl(dev, ["storage", "storage", "uniform"])
        self._scan_local_pipeline = _pipe(dev, [self._scan_bgl], self._scan_module, "scan_local")
        self._scan_propagate_pipeline = _pipe(dev, [self._scan_bgl], self._scan_module, "propagate")

        # --- Gather pipelines ---
        self._gather_bgl0 = _bgl(dev, [
            "uniform", "read-only-storage", "read-only-storage", "storage", "storage"])
        self._gather_bgl1 = _bgl(dev, ["read-only-storage"] * 4)
        self._gather_bgl2 = _bgl(dev, ["storage"] * 4)
        gather_bgls = [self._gather_bgl0, self._gather_bgl1, self._gather_bgl2]
        self._count_pipeline = _pipe(dev, gather_bgls, self._gather_module, "count_particles")
        self._apply_stride_pipeline = _pipe(dev, gather_bgls, self._gather_module, "apply_stride")
        self._gather_pipeline = _pipe(dev, gather_bgls, self._gather_module, "gather_particles")

        # --- Summary gather pipeline ---
        summary_module = dev.create_shader_module(code=_load_wgsl("gather_summaries.wgsl"))
        self._summary_bgl0 = _bgl(dev, ["uniform", "read-only-storage", "storage"])
        self._summary_bgl1 = _bgl(dev, ["read-only-storage"] * 6)
        self._summary_bgl2 = _bgl(dev, ["storage"] * 4)
        summary_bgls = [self._summary_bgl0, self._summary_bgl1, self._summary_bgl2]
        self._summary_gather_pipeline = _pipe(dev, summary_bgls, summary_module, "gather_summaries")

        # Summary output bind group (static)
        sb = self._summary_bufs
        self._summary_out_bg = _bg(dev, self._summary_bgl2,
                                   [sb["pos"], sb["mass"], sb["qty"], sb["cov"]])

        # Build static bind groups
        pb = self._particle_bufs
        self._gather_bg1 = _bg(dev, self._gather_bgl1,
                               [pb["pos"], pb["hsml"], pb["mass"], pb["qty"]])
        ob = self._output_bufs
        self._gather_bg2 = _bg(dev, self._gather_bgl2,
                               [ob["pos"], ob["hsml"], ob["mass"], ob["qty"]])

        # Pre-build per-level cull bind groups (7 bindings including parent_idx)
        self._per_level_cull_bgs = []
        for li in range(self._n_levels):
            lb = self._level_bufs[li]
            parent_decision_buf = (
                self._level_bufs[li + 1]["decision"] if li < self._n_levels - 1
                else self._dummy_decision_buf
            )
            self._per_level_cull_bgs.append(_bg(dev, self._cull_bgl, [
                self._per_level_cull_params[li], lb["mass"], lb["hsml"],
                lb["centers"], parent_decision_buf, lb["decision"],
                lb["parent_idx"]]))

        # Pre-build per-level summary bind groups
        self._per_level_summary_bg0s = []
        self._per_level_summary_bg1s = []
        for li in range(self._n_levels):
            lb = self._level_bufs[li]
            self._per_level_summary_bg0s.append(_bg(dev, self._summary_bgl0, [
                self._per_level_summary_params[li], lb["decision"],
                self._summary_counters_buf]))
            self._per_level_summary_bg1s.append(_bg(dev, self._summary_bgl1, [
                lb["com_gpu"], lb["hsml"], lb["mass"],
                lb["qty_gpu"], lb["cov_gpu"], lb["mh2_gpu"]]))

        # Brute-force subsample cull pipeline (shares particle/output buffers)
        self._build_subsample_pipeline()

    def dispatch_cull(self, camera, max_particles, lod_pixels=4,
                      viewport_width=2048, summary_overlap=0.0):
        """Run GPU frustum cull + LOD + gather. Returns (n_particles, summary_data).

        n_particles: number of gathered particles in output buffers.
        summary_data: (pos, hsml, mass, qty, cov) numpy arrays for aniso summaries,
                      read back from CPU since summaries are few and need covariance assembly.
        """
        dev = self.device
        n_leaves = self._n_leaves
        fov_rad = float(np.radians(camera.fov))
        pix_per_rad = viewport_width / (2.0 * np.tan(fov_rad / 2))

        # === Pass 1: Multi-level frustum cull + LOD (batched) ===
        for li in range(self._n_levels - 1, -1, -1):
            lb = self._level_bufs[li]
            n_nodes = lb["n_nodes"]
            is_coarsest = 1 if li == self._n_levels - 1 else 0
            is_finest = 1 if li == 0 else 0
            params_data = struct.pack(
                "fff f fff f fff f fff f ffff IIII",
                *camera.position, 0.0,
                *camera.forward, 0.0,
                *camera.right, 0.0,
                *camera.up, 0.0,
                fov_rad, camera.aspect, pix_per_rad, float(lod_pixels),
                is_coarsest, is_finest, n_nodes, 0,
            )
            dev.queue.write_buffer(self._per_level_cull_params[li], 0, params_data)

        encoder = dev.create_command_encoder()
        for li in range(self._n_levels - 1, -1, -1):
            n_nodes_li = self._level_bufs[li]["n_nodes"]
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._cull_pipeline)
            cpass.set_bind_group(0, self._per_level_cull_bgs[li])
            cpass.dispatch_workgroups(_div_ceil(n_nodes_li, WG_SIZE))
            cpass.end()
        dev.queue.submit([encoder.finish()])

        # === GPU Summary Gather (batched, atomic offset) ===
        dev.queue.write_buffer(self._summary_counters_buf, 0, struct.pack("IIII", 0, 0, 0, 0))
        for li in range(self._n_levels - 1, 0, -1):
            lb = self._level_bufs[li]
            cs = lb["cs"]
            dev.queue.write_buffer(self._per_level_summary_params[li], 0,
                                   struct.pack("IffffIII", lb["n_nodes"], summary_overlap,
                                               float(cs[0]**2), float(cs[1]**2),
                                               float(cs[2]**2), self._max_summaries, 0, 0))

        encoder = dev.create_command_encoder()
        for li in range(self._n_levels - 1, 0, -1):
            n_nodes_li = self._level_bufs[li]["n_nodes"]
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._summary_gather_pipeline)
            cpass.set_bind_group(0, self._per_level_summary_bg0s[li])
            cpass.set_bind_group(1, self._per_level_summary_bg1s[li])
            cpass.set_bind_group(2, self._summary_out_bg)
            cpass.dispatch_workgroups(_div_ceil(n_nodes_li, WG_SIZE))
            cpass.end()
        dev.queue.submit([encoder.finish()])

        # === Pass 2: Count + stride + apply stride + prefix sum ===
        dev.queue.write_buffer(self._counters_buf, 0, struct.pack("IIII", 0, 0, 0, 0))

        max_buf_bytes = self.device.adapter.limits.get("max-buffer-size", 2**30)
        MAX_OUTPUT_CAP = max_buf_bytes // (16 * 4)
        budget = min(max(max_particles // 2, max_particles - self._max_summaries), MAX_OUTPUT_CAP)
        if budget > self._max_output:
            self._grow_output_buffers(budget)

        dev.queue.write_buffer(self._gather_params_buf, 0,
                               struct.pack("IIII", n_leaves, budget, 0, 1))

        finest_decision_buf = self._level_bufs[0]["decision"]
        gather_bg0 = _make_bind_group(dev, self._gather_bgl0, [
            self._gather_params_buf, finest_decision_buf, self._cell_start_buf,
            self._cell_out_counts_buf, self._counters_buf])
        gather_bgs = [gather_bg0, self._gather_bg1, self._gather_bg2]

        # Count particles
        self._dispatch(self._count_pipeline, gather_bgs, _div_ceil(n_leaves, WG_SIZE))

        # Read total visible, compute stride on CPU
        counter_data = dev.queue.read_buffer(self._counters_buf, size=16)
        total_visible = struct.unpack("I", counter_data[:4])[0]
        stride = max(1, _div_ceil(total_visible, budget)) if total_visible > budget else 1

        # Apply stride
        dev.queue.write_buffer(self._gather_params_buf, 0,
                               struct.pack("IIII", n_leaves, budget, total_visible, stride))
        self._dispatch(self._apply_stride_pipeline, gather_bgs, _div_ceil(n_leaves, WG_SIZE))

        # Prefix sum
        self._prefix_sum(self._cell_out_counts_buf, n_leaves)

        # Read n_output from apply_stride
        counter_data2 = dev.queue.read_buffer(self._counters_buf, size=16)
        n_output = struct.unpack("I", counter_data2[4:8])[0]
        n_output = min(n_output, self._max_output)

        # === Pass 5: Gather particles ===
        self._dispatch(self._gather_pipeline, gather_bgs, _div_ceil(n_leaves, WG_SIZE))

        # Read n_summaries
        sc_data = dev.queue.read_buffer(self._summary_counters_buf, size=4)
        n_summaries = struct.unpack("I", sc_data)[0]
        n_summaries = min(n_summaries, self._max_summaries)

        # === Read back summary data from GPU ===
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros(0, dtype=np.float32)
        z6 = np.zeros((0, 6), dtype=np.float32)

        if n_summaries > 0:
            s_pos4 = np.frombuffer(
                dev.queue.read_buffer(self._summary_bufs["pos"], size=n_summaries * 16),
                dtype=np.float32).reshape(-1, 4)
            s_pos = s_pos4[:, :3].copy()
            s_mass = np.frombuffer(
                dev.queue.read_buffer(self._summary_bufs["mass"], size=n_summaries * 4),
                dtype=np.float32).copy()
            s_qty = np.frombuffer(
                dev.queue.read_buffer(self._summary_bufs["qty"], size=n_summaries * 4),
                dtype=np.float32).copy()
            s_cov_packed = np.frombuffer(
                dev.queue.read_buffer(self._summary_bufs["cov"], size=n_summaries * 2 * 16),
                dtype=np.float32).reshape(-1, 4)
            # Unpack: [xx,xy,xz,_] [yy,yz,zz,_] → (n, 6)
            s_cov = np.zeros((n_summaries, 6), dtype=np.float32)
            s_cov[:, :3] = s_cov_packed[0::2, :3]
            s_cov[:, 3:] = s_cov_packed[1::2, :3]
            # hsml not stored in summaries — use sqrt(trace(cov)) as proxy
            s_hsml = np.sqrt(np.maximum(s_cov[:, 0] + s_cov[:, 3] + s_cov[:, 5], 1e-30))
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return n_output, total_visible, (s_pos, s_hsml, s_mass, s_qty, s_cov)

    def grow_subsample_output(self, max_output):
        """Reallocate output buffers and the subsample bind group if the
        requested max_output exceeds the current capacity. Used when the
        renderer's max_render_particles is bumped at runtime.

        Caps max_output to fit within the device's max storage buffer
        size. The largest per-element entry is pos (16 bytes), so the
        cap is max-storage-buffer-binding-size / 16.
        """
        try:
            max_buf = int(self.device.limits["max-storage-buffer-binding-size"])
        except Exception:
            max_buf = 2**31
        max_output = min(max_output, max_buf // 16)
        if max_output <= self._max_output:
            return
        dev = self.device
        out_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        self._output_bufs = {
            "pos": dev.create_buffer(size=max_output * 16, usage=out_usage),
            "hsml": dev.create_buffer(size=max_output * 4, usage=out_usage),
            "mass": dev.create_buffer(size=max_output * 4, usage=out_usage),
            "qty": dev.create_buffer(size=max_output * 4, usage=out_usage),
        }
        self._max_output = max_output
        ob = self._output_bufs
        self._subsample_bg1 = _make_bind_group(dev, self._subsample_bgl1, [
            ob["pos"], ob["hsml"], ob["mass"], ob["qty"]])

    def dispatch_subsample_cull(self, camera, stride, max_output=None):
        """Brute-force per-particle frustum cull + stride subsample.

        Tests all particles. Particles passing the hash stride filter AND
        the frustum test atomically write to the output buffers, but the
        atomic write stops once `max_output` in-frustum particles have
        been emitted. So the budget caps the in-frustum count sent to
        the renderer.
        """
        dev = self.device
        n = self._n_particles
        stride = max(int(stride), 1)
        if max_output is None:
            max_output = self._max_output
        else:
            max_output = min(int(max_output), self._max_output)

        ratio = float(stride)
        h_scale = ratio ** (1.0 / 3.0)
        fov_rad = float(np.radians(camera.fov))

        # Zero the counter
        dev.queue.write_buffer(self._subsample_counter_buf, 0,
                               struct.pack("IIII", 0, 0, 0, 0))

        # Pack params (96 bytes — vec3+pad ×4 = 64, then 8 floats/ints = 32)
        # Layout: cam_pos+pad, cam_fwd+pad, cam_right+pad, cam_up+pad,
        #         fov_rad, aspect, stride, n_particles, h_scale, mass_scale,
        #         max_output, _pad
        params_data = struct.pack(
            "fff f fff f fff f fff f ffII ffII",
            float(camera.position[0]), float(camera.position[1]),
            float(camera.position[2]), 0.0,
            float(camera.forward[0]), float(camera.forward[1]),
            float(camera.forward[2]), 0.0,
            float(camera.right[0]), float(camera.right[1]),
            float(camera.right[2]), 0.0,
            float(camera.up[0]), float(camera.up[1]),
            float(camera.up[2]), 0.0,
            fov_rad, float(camera.aspect), stride, n,
            h_scale, ratio, max_output, 0,
        )
        dev.queue.write_buffer(self._subsample_params_buf, 0, params_data)

        # Wrap workgroup count into a 2D grid (max 65535 per dim).
        total_wg = _div_ceil(n, WG_SIZE)
        wgx = min(total_wg, 65535)
        wgy = _div_ceil(total_wg, wgx)

        encoder = dev.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._subsample_pipeline)
        cpass.set_bind_group(0, self._subsample_bg0)
        cpass.set_bind_group(1, self._subsample_bg1)
        cpass.dispatch_workgroups(wgx, wgy)
        cpass.end()
        dev.queue.submit([encoder.finish()])

        # Read back the counter to get the actual output count
        counter_data = dev.queue.read_buffer(self._subsample_counter_buf, size=4)
        n_out = struct.unpack("I", counter_data)[0]
        n_out = min(n_out, self._max_output)
        return n_out

    def _prefix_sum(self, buf, n, block_sums_1=None, block_sums_2=None):
        """Run multi-level prefix sum on a u32 storage buffer.

        Args:
            buf: GPU buffer to scan in-place.
            n: Number of elements.
            block_sums_1/2: Scratch buffers for hierarchical scan. Defaults to
                the cell-grid block sum buffers allocated in _prepare_snapshot.
        """
        if block_sums_1 is None:
            block_sums_1 = self._scan_block_sums_1
        if block_sums_2 is None:
            block_sums_2 = self._scan_block_sums_2

        dev = self.device
        _bg = _make_bind_group
        params = self._scan_params_buf
        n_blocks_1 = _div_ceil(n, WG_SIZE)

        def _scan_level(data_buf, out_buf, count, workgroups):
            dev.queue.write_buffer(params, 0, struct.pack("IIII", count, 0, 0, 0))
            self._dispatch(self._scan_local_pipeline,
                           _bg(dev, self._scan_bgl, [data_buf, out_buf, params]),
                           workgroups)

        def _propagate(data_buf, sums_buf, count):
            dev.queue.write_buffer(params, 0, struct.pack("IIII", count, 0, 0, 0))
            self._dispatch(self._scan_propagate_pipeline,
                           _bg(dev, self._scan_bgl, [data_buf, sums_buf, params]),
                           _div_ceil(count, WG_SIZE))

        # Level 1: scan local blocks
        _scan_level(buf, block_sums_1, n, n_blocks_1)

        if n_blocks_1 > 1:
            n_blocks_2 = _div_ceil(n_blocks_1, WG_SIZE)
            _scan_level(block_sums_1, block_sums_2, n_blocks_1, n_blocks_2)

            if n_blocks_2 > 1:
                # Level 3: scan level-2 block sums (fits in one WG)
                _scan_level(block_sums_2, block_sums_2, n_blocks_2, 1)
                _propagate(block_sums_1, block_sums_2, n_blocks_1)

            _propagate(buf, block_sums_1, n)

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
            self._los_bgl = _make_compute_bgl(dev, ["uniform", "read-only-storage", "storage"])
            self._los_pipeline = _make_compute_pipeline(dev, [self._los_bgl], los_module, "main")
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

        los_bg = _make_bind_group(dev, self._los_bgl,
                                   [self._los_params_buf, self._los_vec_buf,
                                    self._particle_bufs["qty"]])
        self._dispatch(self._los_pipeline, los_bg, dispatch_x, wy=dispatch_y)

    def upload_slot_data(self, slot_idx, slot_id, sorted_mass, sorted_qty):
        """Write sorted mass/qty to persistent per-slot buffers. Only writes on config change."""
        sb = self._slot_bufs[slot_idx]
        if sb["id"] == slot_id:
            return  # already up to date
        dev = self.device
        dev.queue.write_buffer(sb["mass"], 0, sorted_mass.tobytes())
        dev.queue.write_buffer(sb["qty"], 0, sorted_qty.tobytes())
        sb["id"] = slot_id

    def activate_slot(self, slot_idx):
        """Copy per-slot mass/qty into the active particle buffers for cull/gather."""
        dev = self.device
        sb = self._slot_bufs[slot_idx]
        n = self._n_particles
        encoder = dev.create_command_encoder()
        encoder.copy_buffer_to_buffer(sb["mass"], 0, self._particle_bufs["mass"], 0, n * 4)
        encoder.copy_buffer_to_buffer(sb["qty"], 0, self._particle_bufs["qty"], 0, n * 4)
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
        ob = self._output_bufs
        self._gather_bg2 = _make_bind_group(dev, self._gather_bgl2,
                                            [ob["pos"], ob["hsml"], ob["mass"], ob["qty"]])

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
        self._depth_bgl = _make_compute_bgl(dev, [
            "uniform", "read-only-storage", "storage", "storage"])
        self._depth_pipeline = _make_compute_pipeline(dev, [self._depth_bgl],
                                                      self._depth_module, "main")

        # Radix sort pipelines
        self._sort_bgl = _make_compute_bgl(dev, [
            "uniform", "read-only-storage", "read-only-storage",
            "storage", "storage", "storage"])
        self._histogram_pipeline = _make_compute_pipeline(
            dev, [self._sort_bgl], self._sort_module, "build_histogram")
        self._scatter_pipeline = _make_compute_pipeline(
            dev, [self._sort_bgl], self._sort_module, "scatter")

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

        depth_bg = _make_bind_group(dev, self._depth_bgl, [
            self._depth_params_buf, self._output_bufs["pos"],
            self._sort_keys_a, self._sort_index_a])
        self._dispatch(self._depth_pipeline, depth_bg, n_wg)

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

            sort_bg = _make_bind_group(dev, self._sort_bgl, [
                self._sort_params_buf_sort, keys_a, idx_a,
                keys_b, idx_b, self._sort_histogram])

            # Build histogram
            self._dispatch(self._histogram_pipeline, sort_bg, n_wg)

            # Prefix sum on histogram
            hist_n = 16 * n_wg
            self._prefix_sum(self._sort_histogram, hist_n,
                             self._sort_hist_block_sums_1, self._sort_hist_block_sums_2)

            # Scatter
            self._dispatch(self._scatter_pipeline, sort_bg, n_wg)

            # Swap buffers for next pass
            keys_a, keys_b = keys_b, keys_a
            idx_a, idx_b = idx_b, idx_a

        # After 8 passes (even number), result is in keys_a/idx_a = the original buffers
        # _sort_index_a now contains the sorted permutation
        self._sort_index_a = idx_a
        return self._sort_index_a

    def release(self):
        self._level_bufs = []
        self._particle_bufs = {}
        self._output_bufs = {}
