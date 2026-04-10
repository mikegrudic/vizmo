"""GPU compute helper for the subsample splat path.

The renderer's splat_subsample.wgsl pipeline reads particle pos/hsml/mass/qty
directly from per-chunk storage buffers and does the cull + sampling inline.
This module just allocates the chunk buffers, uploads the source arrays
(pre-shuffled so the first K particles of each chunk are a uniform random
K-subset), and provides per-slot mass/qty buffers for composite mode.
"""

import numpy as np
import wgpu


BRICK_SIZE = 4096  # Must match BRICK_SIZE in multigrid_bin.wgsl


def _assign_spatial_bricks(pos_f32, n_bricks):
    """Assign each particle to a spatial brick via 3D grid binning.

    Each particle maps to a grid cell; each cell IS a brick. O(N) with
    no sort. The grid resolution is chosen so the number of cells is
    close to n_bricks (cubed root, rounded up).
    """
    if n_bricks <= 1:
        pmin = pos_f32.min(axis=0)
        pmax = pos_f32.max(axis=0)
        extent = np.maximum(pmax - pmin, np.float32(1e-30))
        return np.zeros(len(pos_f32), dtype=np.uint32), 1, 1, pmin, extent

    pmin = pos_f32.min(axis=0)
    pmax = pos_f32.max(axis=0)
    extent = np.maximum(pmax - pmin, np.float32(1e-30))

    # Grid resolution: cube root of n_bricks, at least 2.
    res = max(2, int(np.ceil(n_bricks ** (1.0 / 3.0))))
    # Scale factors to map positions directly to grid cell indices.
    scale = (np.float64(res) / extent).astype(np.float32)
    # Compute flattened brick id per particle in one fused pass,
    # avoiding a full (N,3) int32 intermediate.
    r2 = np.int32(res * res)
    r1 = np.int32(res)
    rmax = np.int32(res - 1)
    x = np.clip(((pos_f32[:, 0] - pmin[0]) * scale[0]).astype(np.int32), 0, rmax)
    y = np.clip(((pos_f32[:, 1] - pmin[1]) * scale[1]).astype(np.int32), 0, rmax)
    z = np.clip(((pos_f32[:, 2] - pmin[2]) * scale[2]).astype(np.int32), 0, rmax)
    brick_ids = (x * r2 + y * r1 + z).astype(np.uint32)
    return brick_ids, res * res * res, res, pmin, extent


class GPUCompute:
    """Allocates and manages the per-chunk source particle buffers and
    the per-slot mass/qty buffers used by composite mode.
    """

    def __init__(self, device):
        self.device = device
        self._n_particles = 0
        self._chunk_bufs = []
        self._slot_chunks = [None, None]
        self._slot_ids = [None, None]
        self._shuffle_perm = None
        # Kept in float64 so the world-origin shift is exact: the
        # subsequent hi/lo split (pos_hi/pos_lo) only recovers extra
        # precision if the offset itself isn't already truncated.
        self._pos_offset = np.zeros(3, dtype=np.float64)
        self._upload_ready = False
        # Legacy alias retained for code paths that touch _particle_bufs.
        self._particle_bufs = {}

    # ---- Initial upload ----

    def upload_subsample_only(self, pos, hsml, mass, qty):
        """Allocate per-chunk source particle buffers for the splat
        pipeline. Particles are pre-shuffled and the bounding-box center
        is subtracted from positions before upload (world-origin shift,
        for float32 precision on cosmological-scale snapshots).
        """
        dev = self.device
        pos_src, hsml_src, mass_src, qty_src = pos, hsml, mass, qty
        n = len(pos_src)
        self._n_particles = n

        # World-origin shift: cosmological coords are ~1e4-1e5 absolute,
        # which loses several decimal digits of precision in the
        # vertex-shader matrix multiply when stored as float32. Subtract
        # the bbox center on upload; the renderer subtracts it from the
        # camera position to match.
        #
        # The offset itself stays in float64; uploads are stored as a
        # DSFUN90-style hi/lo pair (pos_hi, pos_lo) that the shader
        # cancels against the camera's matching hi/lo pair, so neither
        # the offset nor the per-particle position loses precision in
        # the cast to f32.
        pmin = np.asarray(pos_src.min(axis=0), dtype=np.float64)
        pmax = np.asarray(pos_src.max(axis=0), dtype=np.float64)
        self._pos_offset = (pmin + pmax) * 0.5

        # 256 MB cap per binding (Apple Metal practical limit) → 16 M
        # particles per chunk for pos at 16 bytes each.
        SAFE_CHUNK_BYTES = 256 * 1024 * 1024
        chunk_n = SAFE_CHUNK_BYTES // 16

        # Assign each particle to a spatially compact brick BEFORE
        # shuffling. The brick_id array will be shuffled along with
        # everything else so the GPU can look up each particle's brick.
        n_bricks_approx = (n + BRICK_SIZE - 1) // BRICK_SIZE
        # Compute offset-shifted f32 positions (needed for brick AABBs
        # and spatial assignment).
        shifted_all = (np.asarray(pos_src, dtype=np.float64) - self._pos_offset)
        pos_hi_all = shifted_all.astype(np.float32)
        brick_ids, n_bricks_total, grid_res, pmin, extent = \
            _assign_spatial_bricks(pos_hi_all, n_bricks_approx)

        # Per-brick AABBs from grid geometry (no particle iteration).
        # Each brick = one grid cell, so the AABB is the cell boundary.
        cell_size = extent / grid_res
        # Build grid-cell coordinates for all bricks.
        bids = np.arange(n_bricks_total, dtype=np.int32)
        ix = bids // (grid_res * grid_res)
        iy = (bids // grid_res) % grid_res
        iz = bids % grid_res
        brick_aabb_data = np.zeros((n_bricks_total, 8), dtype=np.float32)
        for ax, gi in enumerate((ix, iy, iz)):
            brick_aabb_data[:, ax] = pmin[ax] + gi * cell_size[ax]
            brick_aabb_data[:, 4 + ax] = pmin[ax] + (gi + 1) * cell_size[ax]
        # h_max per brick still needs a scatter-reduce over particles.
        np.maximum.at(brick_aabb_data[:, 7], brick_ids, hsml_src)

        # Pre-shuffle the source arrays so a per-chunk dispatch of K
        # instances draws a uniformly random K-subset of the chunk via
        # plain ii-indexing in the vertex shader (no hash, no
        # collisions, no source-order imprinting).
        perm = np.random.default_rng(0).permutation(n)
        self._shuffle_perm = perm
        pos_src = np.ascontiguousarray(pos_src[perm])
        hsml_src = np.ascontiguousarray(hsml_src[perm])
        mass_src = np.ascontiguousarray(mass_src[perm])
        qty_src = np.ascontiguousarray(qty_src[perm])
        brick_ids = np.ascontiguousarray(brick_ids[perm])

        # mapped_at_creation lets us write directly to device memory
        # without going through Metal's staging-buffer queue.
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)
        self._chunk_bufs = []
        for start in range(0, n, chunk_n):
            cn = min(chunk_n, n - start)
            # DSFUN90 hi/lo split, computed in f64:
            #   shifted = pos - offset                       (f64)
            #   pos_hi  = (f32)shifted                       (f32)
            #   pos_lo  = (f32)(shifted - (f64)pos_hi)       (f32)
            # Reconstruction: (f64)pos_hi + (f64)pos_lo == shifted
            # to ~14 decimal digits.
            shifted = (np.asarray(pos_src[start:start + cn], dtype=np.float64)
                       - self._pos_offset)
            pos_hi_xyz = shifted.astype(np.float32)
            pos_lo_xyz = (shifted - pos_hi_xyz.astype(np.float64)).astype(np.float32)

            pos_buf = dev.create_buffer(
                size=cn * 16, usage=usage, mapped_at_creation=True)
            pos4 = np.zeros((cn, 4), dtype=np.float32)
            pos4[:, :3] = pos_hi_xyz
            pos_buf.write_mapped(pos4.tobytes())
            pos_buf.unmap()

            pos_lo_buf = dev.create_buffer(
                size=cn * 16, usage=usage, mapped_at_creation=True)
            pos4_lo = np.zeros((cn, 4), dtype=np.float32)
            pos4_lo[:, :3] = pos_lo_xyz
            pos_lo_buf.write_mapped(pos4_lo.tobytes())
            pos_lo_buf.unmap()

            hsml_buf = dev.create_buffer(
                size=cn * 4, usage=usage, mapped_at_creation=True)
            hsml_buf.write_mapped(hsml_src[start:start + cn].tobytes())
            hsml_buf.unmap()

            mass_buf = dev.create_buffer(
                size=cn * 4, usage=usage, mapped_at_creation=True)
            mass_buf.write_mapped(mass_src[start:start + cn].tobytes())
            mass_buf.unmap()

            qty_buf = dev.create_buffer(
                size=cn * 4, usage=usage, mapped_at_creation=True)
            qty_buf.write_mapped(qty_src[start:start + cn].tobytes())
            qty_buf.unmap()

            # Identity index buffer. The splat vertex shader reads
            # local_idx = s_index[ii], so the default identity behaves
            # exactly like the old direct ii lookup. The multigrid bin
            # compute pass overwrites this buffer per frame to scatter
            # particles by their (camera-dependent) level.
            index_buf = dev.create_buffer(
                size=cn * 4, usage=usage, mapped_at_creation=True)
            index_buf.write_mapped(np.arange(cn, dtype=np.uint32).tobytes())
            index_buf.unmap()

            # Per-particle brick_id buffer: maps each (shuffled) particle
            # to its spatial brick. The cull shader writes per-brick
            # visibility; the splat/multigrid shaders read brick_id to
            # check it.
            brick_id_buf = dev.create_buffer(
                size=cn * 4, usage=usage, mapped_at_creation=True)
            brick_id_buf.write_mapped(brick_ids[start:start + cn].tobytes())
            brick_id_buf.unmap()

            # Global brick AABB + visibility buffers (shared across all
            # chunks since brick_ids are globally assigned).  Each chunk
            # gets its own copy of the full AABB/vis arrays — they're
            # small (~n_bricks * 32 B / 4 B) and this avoids cross-chunk
            # bind group complications.
            brick_aabb_buf = dev.create_buffer(
                size=n_bricks_total * 32, usage=usage, mapped_at_creation=True)
            brick_aabb_buf.write_mapped(brick_aabb_data.tobytes())
            brick_aabb_buf.unmap()

            brick_vis_buf = dev.create_buffer(
                size=n_bricks_total * 4, usage=usage, mapped_at_creation=True)
            brick_vis_buf.write_mapped(np.ones(n_bricks_total, dtype=np.uint32).tobytes())
            brick_vis_buf.unmap()

            self._chunk_bufs.append({
                "pos": pos_buf, "pos_lo": pos_lo_buf, "hsml": hsml_buf,
                "mass": mass_buf, "qty": qty_buf, "index": index_buf,
                "brick_id": brick_id_buf,
                "brick_aabb": brick_aabb_buf, "brick_vis": brick_vis_buf,
                "n": cn, "n_bricks": n_bricks_total, "start": start,
            })

        self._particle_bufs = self._chunk_bufs[0]
        self._upload_ready = True

    def get_chunk_bufs(self):
        return self._chunk_bufs

    def get_pos_offset(self):
        return self._pos_offset

    # ---- Per-slot mass/qty buffers (composite mode) ----

    def get_or_create_slot_chunks(self, slot_idx):
        """Lazily allocate per-chunk mass+qty buffers for a composite
        slot. pos+hsml stay shared with self._chunk_bufs.
        """
        if self._slot_chunks[slot_idx] is not None:
            return self._slot_chunks[slot_idx]
        dev = self.device
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)
        out = []
        for cb in self._chunk_bufs:
            cn = cb["n"]
            out.append({
                "mass": dev.create_buffer(size=cn * 4, usage=usage),
                "qty": dev.create_buffer(size=cn * 4, usage=usage),
                "n": cn, "start": cb["start"],
            })
        self._slot_chunks[slot_idx] = out
        return out

    def upload_subsample_slot(self, slot_idx, slot_id, mass, qty):
        """Upload mass/qty into a composite slot's per-chunk buffers.
        Cached by slot_id so re-uploading the same slot is a no-op.
        """
        slot_chunks = self.get_or_create_slot_chunks(slot_idx)
        if self._slot_ids[slot_idx] == slot_id:
            return slot_chunks
        dev = self.device
        perm = self._shuffle_perm
        mass_f32 = np.ascontiguousarray(mass, dtype=np.float32)
        if perm is not None:
            mass_f32 = mass_f32[perm]
        qty_is_mass = qty is mass or qty is None
        if not qty_is_mass:
            qty_f32 = np.ascontiguousarray(qty, dtype=np.float32)
            if perm is not None:
                qty_f32 = qty_f32[perm]
        for sc in slot_chunks:
            start, cn = sc["start"], sc["n"]
            dev.queue.write_buffer(
                sc["mass"], 0, mass_f32[start:start + cn].tobytes())
            if not qty_is_mass:
                dev.queue.write_buffer(
                    sc["qty"], 0, qty_f32[start:start + cn].tobytes())
        # Final sync so the next render doesn't wedge behind a Metal
        # staging-buffer backlog.
        dev.queue.read_buffer(slot_chunks[-1]["mass"], size=4)
        self._slot_ids[slot_idx] = slot_id
        return slot_chunks

    # ---- Re-upload of mass/qty after a render-mode change (non-composite) ----

    def upload_weights(self, mass, qty):
        """Re-upload mass/qty for every chunk after a field swap.
        Both arrays are in raw particle order; the shuffle perm built
        on initial upload is reapplied so they line up with the
        already-shuffled pos/hsml chunks.
        """
        dev = self.device
        perm = self._shuffle_perm
        mass_f32 = np.ascontiguousarray(mass, dtype=np.float32)
        if perm is not None:
            mass_f32 = mass_f32[perm]
        qty_is_mass = qty is mass or qty is None
        if not qty_is_mass:
            qty_f32 = np.ascontiguousarray(qty, dtype=np.float32)
            if perm is not None:
                qty_f32 = qty_f32[perm]
        # Queue all writes, then a single blocking read on the LAST
        # chunk to drain the staging queue. Per-chunk reads serialize
        # the upload and turn it into a multi-second beachball; one
        # final sync is enough because Metal's queue is FIFO.
        for cb in self._chunk_bufs:
            start, cn = cb["start"], cb["n"]
            dev.queue.write_buffer(
                cb["mass"], 0, mass_f32[start:start + cn].tobytes())
            if not qty_is_mass:
                dev.queue.write_buffer(
                    cb["qty"], 0, qty_f32[start:start + cn].tobytes())
        dev.queue.read_buffer(self._chunk_bufs[-1]["mass"], size=4)
        # The next dispatch in either composite slot will see different
        # data, so invalidate the slot caches.
        self._slot_ids = [None, None]

    # ---- Cleanup ----

    def release(self):
        # GPUBuffer release happens automatically on garbage collection;
        # nothing to explicitly free here.
        pass
