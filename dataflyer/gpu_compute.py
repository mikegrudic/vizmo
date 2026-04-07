"""GPU compute helper for the subsample splat path.

The renderer's splat_subsample.wgsl pipeline reads particle pos/hsml/mass/qty
directly from per-chunk storage buffers and does the cull + sampling inline.
This module just allocates the chunk buffers, uploads the source arrays
(pre-shuffled so the first K particles of each chunk are a uniform random
K-subset), and provides per-slot mass/qty buffers for composite mode.
"""

import numpy as np
import wgpu


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
        self._pos_offset = np.zeros(3, dtype=np.float32)
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
        pmin = np.asarray(pos_src.min(axis=0), dtype=np.float64)
        pmax = np.asarray(pos_src.max(axis=0), dtype=np.float64)
        self._pos_offset = ((pmin + pmax) * 0.5).astype(np.float32)

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

        # 256 MB cap per binding (Apple Metal practical limit) → 16 M
        # particles per chunk for pos at 16 bytes each.
        SAFE_CHUNK_BYTES = 256 * 1024 * 1024
        chunk_n = SAFE_CHUNK_BYTES // 16

        # mapped_at_creation lets us write directly to device memory
        # without going through Metal's staging-buffer queue.
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)
        self._chunk_bufs = []
        for start in range(0, n, chunk_n):
            cn = min(chunk_n, n - start)
            pos_buf = dev.create_buffer(
                size=cn * 16, usage=usage, mapped_at_creation=True)
            pos4 = np.zeros((cn, 4), dtype=np.float32)
            pos4[:, :3] = pos_src[start:start + cn]
            pos4[:, :3] -= self._pos_offset
            pos_buf.write_mapped(pos4.tobytes())
            pos_buf.unmap()

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

            self._chunk_bufs.append({
                "pos": pos_buf, "hsml": hsml_buf,
                "mass": mass_buf, "qty": qty_buf,
                "n": cn, "start": start,
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
