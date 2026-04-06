"""True adaptive octree for frustum culling and LOD rendering.

Morton-sorted octree where each branch subdivides independently until
the cell particle count is <= leaf_size or max_depth is reached.
No cell ever has more than leaf_size particles (unless max_depth caps it).

Drop-in replacement for SpatialGrid with the same public interface.
"""

import math
import numpy as np
from numba import njit, prange
from .spatial_grid import CellMoments
from .spatial_grid import _gather_subsampled_direct, _gather_importance_sampled


# ---- Morton code utilities ----

@njit(cache=True)
def _split_by_3(x):
    """Spread 21-bit integer into every 3rd bit position for Morton encoding."""
    x = np.uint64(x) & np.uint64(0x1fffff)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return x


@njit(cache=True)
def _morton_encode(ix, iy, iz):
    return _split_by_3(ix) | (_split_by_3(iy) << np.uint64(1)) | (_split_by_3(iz) << np.uint64(2))


@njit(parallel=True, cache=True)
def _compute_morton_codes(positions, pmin, inv_box, max_coord):
    """Compute 63-bit Morton codes for all particles."""
    n = len(positions)
    codes = np.empty(n, dtype=np.uint64)
    mc = np.int64(max_coord)
    for i in prange(n):
        fx = (positions[i, 0] - pmin[0]) * inv_box[0]
        fy = (positions[i, 1] - pmin[1]) * inv_box[1]
        fz = (positions[i, 2] - pmin[2]) * inv_box[2]
        ix = min(max(int(fx * mc), 0), mc - 1)
        iy = min(max(int(fy * mc), 0), mc - 1)
        iz = min(max(int(fz * mc), 0), mc - 1)
        codes[i] = _morton_encode(ix, iy, iz)
    return codes


@njit(cache=True)
def _compact_by_3(x):
    """Inverse of _split_by_3: extract every 3rd bit into contiguous integer."""
    x = x & np.uint64(0x1249249249249249)
    x = (x | (x >> np.uint64(2)))  & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x >> np.uint64(4)))  & np.uint64(0x100f00f00f00f00f)
    x = (x | (x >> np.uint64(8)))  & np.uint64(0x1f0000ff0000ff)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x1f00000000ffff)
    x = (x | (x >> np.uint64(32))) & np.uint64(0x1fffff)
    return x


@njit(cache=True)
def _morton_decode(code):
    """Decode Morton code to (ix, iy, iz)."""
    ix = np.int64(_compact_by_3(code))
    iy = np.int64(_compact_by_3(code >> np.uint64(1)))
    iz = np.int64(_compact_by_3(code >> np.uint64(2)))
    return ix, iy, iz


@njit(parallel=True, cache=True)
def _compute_centers_vec(codes, depth, pmin, cell_size):
    """Compute cell center coordinates from Morton codes."""
    n = len(codes)
    centers = np.empty((n, 3), dtype=np.float32)
    for i in prange(n):
        ix, iy, iz = _morton_decode(codes[i])
        centers[i, 0] = pmin[0] + (ix + np.float32(0.5)) * cell_size[0]
        centers[i, 1] = pmin[1] + (iy + np.float32(0.5)) * cell_size[1]
        centers[i, 2] = pmin[2] + (iz + np.float32(0.5)) * cell_size[2]
    return centers


def _compute_centers_from_codes(codes, depth, pmin, cell_size):
    """Compute centers, handling depth 0 (root) specially."""
    if depth == 0:
        return np.array([[pmin[0] + cell_size[0] * 0.5,
                          pmin[1] + cell_size[1] * 0.5,
                          pmin[2] + cell_size[2] * 0.5]], dtype=np.float32)
    return _compute_centers_vec(codes, depth, pmin.astype(np.float32),
                                cell_size.astype(np.float32))


# ---- Adaptive subdivision ----

def _adaptive_subdivide(sorted_codes, n, leaf_size, max_depth):
    """Find all nodes via adaptive subdivision.

    Returns:
        leaves: list of (start, end, depth, code) for each leaf cell
        internals: set of (depth, code) for each internal node
    """
    leaves = []
    internals = set()
    # Stack: (start_idx, end_idx, depth, code_at_depth)
    stack = [(0, n, 0, np.uint64(0))]

    while stack:
        start, end, depth, code = stack.pop()
        count = end - start
        if count == 0:
            continue
        if count <= leaf_size or depth >= max_depth:
            leaves.append((start, end, depth, int(code)))
            continue

        internals.add((depth, int(code)))

        # Subdivide into 8 children using Morton code structure.
        # Particles in [start, end) have codes whose depth-d prefix == code.
        # Children at depth d+1 have prefix code*8 + c for c in 0..7.
        remaining_bits = np.uint64(3 * (max_depth - depth - 1))
        sub = sorted_codes[start:end]
        for c in range(7, -1, -1):  # reverse order for DFS
            child_code = np.uint64(int(code) * 8 + c)
            lo_full = child_code << remaining_bits
            hi_full = (child_code + np.uint64(1)) << remaining_bits
            lo = start + int(np.searchsorted(sub, lo_full))
            hi = start + int(np.searchsorted(sub, hi_full))
            if hi > lo:
                stack.append((lo, hi, depth + 1, child_code))

    return leaves, internals


def _build_adaptive_tree(sorted_codes, n, leaf_size, max_depth, pmin, box):
    """Build a true adaptive octree from sorted Morton codes.

    Returns (levels, cell_start) where levels[0] is finest, levels[-1] is root.
    Leaves exist at variable depths; no leaf has more than leaf_size particles
    (unless capped by max_depth).
    """
    leaves, internals = _adaptive_subdivide(sorted_codes, n, leaf_size, max_depth)

    # Sort leaves by start index for a proper CSR
    leaves.sort(key=lambda x: x[0])
    n_leaves = len(leaves)
    cell_start = np.empty(n_leaves + 1, dtype=np.int64)
    for i, (s, e, d, c) in enumerate(leaves):
        cell_start[i] = s
    cell_start[n_leaves] = n

    # Map (depth, code) → leaf index for fast lookup
    leaf_lookup = {}
    for i, (s, e, d, c) in enumerate(leaves):
        leaf_lookup[(d, c)] = i

    # Collect all nodes by depth. Each node is (code, is_leaf, leaf_idx).
    nodes_by_depth = {}
    for i, (s, e, d, c) in enumerate(leaves):
        nodes_by_depth.setdefault(d, []).append((c, True, i))
    for d, c in internals:
        nodes_by_depth.setdefault(d, []).append((c, False, -1))

    # Sort each depth by Morton code
    for d in nodes_by_depth:
        nodes_by_depth[d].sort(key=lambda x: x[0])

    # Build levels from finest depth to coarsest (root)
    populated_depths = sorted(nodes_by_depth.keys(), reverse=True)
    levels = []
    depth_to_level_idx = {}

    for d in populated_depths:
        nodes = nodes_by_depth[d]
        nc = len(nodes)
        codes = np.array([nd[0] for nd in nodes], dtype=np.uint64)
        is_leaf = np.array([nd[1] for nd in nodes], dtype=np.bool_)
        leaf_indices = np.array([nd[2] for nd in nodes], dtype=np.int64)

        cell_size = box / (2 ** d) if d > 0 else box.copy()
        half_diag = float(np.linalg.norm(cell_size) * 0.5)
        centers = _compute_centers_from_codes(codes, d, pmin, cell_size)

        level = {
            "nc": nc,
            "depth": d,
            "cs": cell_size.astype(np.float32),
            "half_diag": half_diag,
            "centers": centers,
            "cell_codes": codes,
            "is_leaf": is_leaf,
            "leaf_indices": leaf_indices,
        }
        levels.append(level)
        depth_to_level_idx[d] = len(levels) - 1

    # Build parent_idx: each node's parent is in the next coarser level
    # Build code→index maps for each level for fast lookup
    code_to_idx_by_level = {}
    for li, lv in enumerate(levels):
        code_to_idx_by_level[li] = {int(c): i for i, c in enumerate(lv["cell_codes"])}

    for li in range(len(levels) - 1):
        child_lv = levels[li]
        child_depth = child_lv["depth"]
        parent_depth = child_depth - 1
        # Find the level containing the parent depth
        parent_li = depth_to_level_idx.get(parent_depth)
        if parent_li is None:
            # Parent depth has no nodes — shouldn't happen with a connected tree
            child_lv["parent_idx"] = np.zeros(child_lv["nc"], dtype=np.uint32)
            continue
        parent_code_map = code_to_idx_by_level[parent_li]
        parent_codes = child_lv["cell_codes"] >> np.uint64(3)
        parent_idx = np.array([parent_code_map[int(pc)] for pc in parent_codes],
                              dtype=np.uint32)
        child_lv["parent_idx"] = parent_idx

    # Root level has no parent
    levels[-1]["parent_idx"] = np.zeros(levels[-1]["nc"], dtype=np.uint32)

    # Build child_offset: for each non-leaf node, index of first child in child level.
    # Process from coarsest to finest: levels[-1] down to levels[1].
    for li in range(len(levels) - 1, 0, -1):
        parent_lv = levels[li]
        child_li = li - 1
        child_lv = levels[child_li]

        # Only works if child_depth == parent_depth + 1
        if child_lv["depth"] != parent_lv["depth"] + 1:
            # Non-adjacent depths: no direct children
            parent_lv["child_offset"] = np.zeros(parent_lv["nc"], dtype=np.uint32)
            continue

        n_parents = parent_lv["nc"]
        n_children = child_lv["nc"]
        pi_arr = child_lv["parent_idx"]

        child_offset = np.empty(n_parents, dtype=np.uint32)
        ci = 0
        for pi in range(n_parents):
            child_offset[pi] = ci
            while ci < n_children and pi_arr[ci] == pi:
                ci += 1
        parent_lv["child_offset"] = child_offset

    # Finest level has no children
    if levels:
        levels[0].setdefault("child_offset", np.zeros(levels[0]["nc"], dtype=np.uint32))

    return levels, cell_start


# ---- Moment computation ----

@njit(parallel=True, cache=True)
def _coarsen_from_children(child_offset, n_parents,
                           child_mass, child_mp, child_mq, child_mh2, child_mxx,
                           n_children_total):
    """Sum children's extensive moments into parent nodes.

    Only sums over children within each parent's [child_offset[p], child_offset[p+1]) range.
    """
    parent_mass = np.empty(n_parents, dtype=np.float32)
    parent_mp = np.empty((n_parents, 3), dtype=np.float32)
    parent_mq = np.empty(n_parents, dtype=np.float32)
    parent_mh2 = np.empty(n_parents, dtype=np.float32)
    parent_mxx = np.empty((n_parents, 6), dtype=np.float32)

    for p in prange(n_parents):
        cs = child_offset[p]
        ce = child_offset[p + 1] if p < n_parents - 1 else n_children_total

        m = np.float64(0.0)
        mpx = np.float64(0.0)
        mpy = np.float64(0.0)
        mpz = np.float64(0.0)
        mq = np.float64(0.0)
        mh2 = np.float64(0.0)
        v0 = np.float64(0.0)
        v1 = np.float64(0.0)
        v2 = np.float64(0.0)
        v3 = np.float64(0.0)
        v4 = np.float64(0.0)
        v5 = np.float64(0.0)

        for c in range(cs, ce):
            m += np.float64(child_mass[c])
            mpx += np.float64(child_mp[c, 0])
            mpy += np.float64(child_mp[c, 1])
            mpz += np.float64(child_mp[c, 2])
            mq += np.float64(child_mq[c])
            mh2 += np.float64(child_mh2[c])
            v0 += np.float64(child_mxx[c, 0])
            v1 += np.float64(child_mxx[c, 1])
            v2 += np.float64(child_mxx[c, 2])
            v3 += np.float64(child_mxx[c, 3])
            v4 += np.float64(child_mxx[c, 4])
            v5 += np.float64(child_mxx[c, 5])

        parent_mass[p] = np.float32(m)
        parent_mp[p, 0] = np.float32(mpx)
        parent_mp[p, 1] = np.float32(mpy)
        parent_mp[p, 2] = np.float32(mpz)
        parent_mq[p] = np.float32(mq)
        parent_mh2[p] = np.float32(mh2)
        parent_mxx[p, 0] = np.float32(v0)
        parent_mxx[p, 1] = np.float32(v1)
        parent_mxx[p, 2] = np.float32(v2)
        parent_mxx[p, 3] = np.float32(v3)
        parent_mxx[p, 4] = np.float32(v4)
        parent_mxx[p, 5] = np.float32(v5)

    return parent_mass, parent_mp, parent_mq, parent_mh2, parent_mxx


# ---- Main class ----

class AdaptiveOctree:
    """True adaptive octree for frustum culling and LOD rendering.

    Each branch subdivides independently until cell count <= leaf_size
    or max_depth is reached. Drop-in replacement for SpatialGrid.
    """

    def __init__(self, positions, masses, hsml, quantity, leaf_size=1024, max_depth=15):
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        pmin = positions.min(axis=0).astype(np.float64)
        pmax = positions.max(axis=0).astype(np.float64)
        box = pmax - pmin
        box[box == 0] = 1.0
        # Pad to cubic so all octree cells are cubes (avoids axis-aligned splats)
        max_extent = box.max()
        center = 0.5 * (pmin + pmax)
        self.pmin = (center - 0.5 * max_extent).astype(np.float32)
        self.pmax = (center + 0.5 * max_extent).astype(np.float32)
        box = np.full(3, max_extent, dtype=np.float64)
        self._box = box

        # Morton sort
        inv_box = 1.0 / box
        max_coord = 2 ** max_depth
        codes = _compute_morton_codes(
            positions.astype(np.float32), self.pmin.astype(np.float64),
            inv_box, max_coord)
        self.sort_order = np.argsort(codes)
        sorted_codes = codes[self.sort_order]

        # Store sorted particle data
        so = self.sort_order
        self.sorted_pos = positions[so].astype(np.float32)
        self.sorted_hsml = hsml[so].astype(np.float32)
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Build tree structure
        self.levels, self.cell_start = _build_adaptive_tree(
            sorted_codes, len(positions), leaf_size, max_depth, self.pmin, box)
        self.n_cells = sum(1 for lv in self.levels for f in lv["is_leaf"] if f)
        self.is_adaptive = True  # GPU compute not yet supported for adaptive trees

        # Compute moments
        self._compute_all_moments()

    def _compute_all_moments(self):
        """Compute extensive and intensive moments for all levels."""
        # Phase 1: compute leaf moments from particles
        n_leaves = len(self.cell_start) - 1
        leaf_counts = np.diff(self.cell_start).astype(np.int64)
        leaf_moments = CellMoments(n_leaves)
        leaf_moments.accumulate_from_particles(
            self.cell_start, self.sorted_pos, self.sorted_mass,
            self.sorted_hsml, self.sorted_qty)
        leaf_com, leaf_qty, leaf_cov, leaf_hsml = leaf_moments.derive()

        # Phase 2: fill each level's moments bottom-up
        for li in range(len(self.levels)):
            lv = self.levels[li]
            nc = lv["nc"]
            is_leaf = lv["is_leaf"]
            leaf_idx = lv["leaf_indices"]

            moments = CellMoments(nc)
            npart = np.zeros(nc, dtype=np.int64)

            # Copy leaf node moments from leaf_moments
            leaf_mask = np.where(is_leaf)[0]
            if len(leaf_mask) > 0:
                li_idx = leaf_idx[leaf_mask]
                moments.mass[leaf_mask] = leaf_moments.mass[li_idx]
                moments.mp[leaf_mask] = leaf_moments.mp[li_idx]
                moments.mq[leaf_mask] = leaf_moments.mq[li_idx]
                moments.mh2[leaf_mask] = leaf_moments.mh2[li_idx]
                moments.mxx[leaf_mask] = leaf_moments.mxx[li_idx]
                npart[leaf_mask] = leaf_counts[li_idx]

            # Coarsen internal node moments from children
            internal_mask = np.where(~is_leaf)[0]
            if len(internal_mask) > 0 and li > 0:
                child_lv = self.levels[li - 1]
                child_moments = child_lv["moments"]
                child_npart = child_lv["npart"]
                co = lv["child_offset"]

                p_mass, p_mp, p_mq, p_mh2, p_mxx = _coarsen_from_children(
                    co, nc,
                    child_moments.mass, child_moments.mp, child_moments.mq,
                    child_moments.mh2, child_moments.mxx,
                    child_lv["nc"])

                # Sum child particle counts for internal nodes
                for pi in internal_mask:
                    cs = co[pi]
                    ce = co[pi + 1] if pi < nc - 1 else child_lv["nc"]
                    npart[pi] = child_npart[cs:ce].sum()

                # Only overwrite internal nodes (leaves already filled)
                moments.mass[internal_mask] = p_mass[internal_mask]
                moments.mp[internal_mask] = p_mp[internal_mask]
                moments.mq[internal_mask] = p_mq[internal_mask]
                moments.mh2[internal_mask] = p_mh2[internal_mask]
                moments.mxx[internal_mask] = p_mxx[internal_mask]

            com, qty, cov, cell_hsml = moments.derive()

            # Compute hsml_max from largest eigenvalue of (cov + kernel padding).
            # Used as the opening criterion when anisotropic summaries are active.
            safe_mass = np.maximum(moments.mass, 1e-30)
            mean_h2 = moments.mh2 / safe_mass
            padded_cov = cov.copy()
            padded_cov[:, 0] += 0.225 * mean_h2
            padded_cov[:, 3] += 0.225 * mean_h2
            padded_cov[:, 5] += 0.225 * mean_h2
            mats = np.zeros((nc, 3, 3), dtype=np.float64)
            mats[:, 0, 0] = padded_cov[:, 0]
            mats[:, 0, 1] = mats[:, 1, 0] = padded_cov[:, 1]
            mats[:, 0, 2] = mats[:, 2, 0] = padded_cov[:, 2]
            mats[:, 1, 1] = padded_cov[:, 3]
            mats[:, 1, 2] = mats[:, 2, 1] = padded_cov[:, 4]
            mats[:, 2, 2] = padded_cov[:, 5]
            evals_max = np.linalg.eigvalsh(mats)[:, -1]  # largest eigenvalue
            hsml_max = np.sqrt(np.maximum(evals_max / 0.225, 1e-30)).astype(np.float32)

            lv.update({
                "moments": moments, "mass": moments.mass, "com": com,
                "hsml": cell_hsml, "hsml_max": hsml_max, "qty": qty, "cov": cov,
                "mxx": moments.mxx, "mh2": moments.mh2, "npart": npart,
            })

    def update_weights(self, masses, quantity=None):
        """Re-weight the tree without rebuilding structure."""
        if quantity is None:
            quantity = masses
        so = self.sort_order
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)
        self._compute_all_moments()

    # ---- CPU query ----

    @staticmethod
    def _classify_nodes(h_pix, is_leaf, is_root_level, lod_pixels):
        """Decide which nodes to summarize vs refine.

        Standalone subroutine for the node opening criterion. Easy to swap
        for experimentation.

        Args:
            h_pix: per-node angular size in pixels (geometric, e.g. half_diag/dist).
            is_leaf: bool array, True for leaf nodes.
            is_root_level: True if this is the coarsest (root) level — forces
                refinement of small root levels regardless of size.
            lod_pixels: angular threshold; nodes smaller than this are summarized.

        Returns:
            (summary_sel, refine_sel) — bool arrays of equal length.
        """
        if is_root_level:
            # Always refine root-level internal nodes; leaves get summarized
            refine_sel = ~is_leaf
            summary_sel = is_leaf
        else:
            small = h_pix <= lod_pixels
            # Leaves always summarize; internal nodes refine if large
            summary_sel = small | is_leaf
            refine_sel = ~small & ~is_leaf
        return summary_sel, refine_sel

    def query_frustum_lod(self, camera, max_particles, lod_pixels=4,
                          importance_sampling=False, viewport_width=2048,
                          summary_overlap=0.0, anisotropic=False, **kwargs):
        """Top-down multi-level LOD query. Returns same format as SpatialGrid."""
        if lod_pixels <= 2 or len(self.levels) <= 1:
            return self._frustum_cull_leaves(camera, max_particles, importance_sampling)

        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        pix_per_rad = viewport_width / (2.0 * np.tan(fov_rad / 2))
        half_tan = np.tan(fov_rad / 2)
        summary_parts = []
        emit_leaf_indices = []

        # Start from coarsest level (root)
        lv = self.levels[-1]
        centers = lv["centers"]
        hd = lv["half_diag"]

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        cell_extent = hd + np.minimum(lv["hsml"], 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        has_mass = lv["mass"] > 0
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        dist = np.sqrt(depths**2 + rights**2 + ups**2)
        safe_dist = np.maximum(dist, 0.01)
        # Opening criterion uses cell geometry (half_diag) so all cells
        # at the same depth and distance make the same refine/summary decision.
        h_pix = hd / safe_dist * pix_per_rad

        # Classify coarsest level via the standalone opening criterion
        is_leaf = lv["is_leaf"]
        summary_sel, refine_sel = self._classify_nodes(
            h_pix, is_leaf, is_root_level=(lv["nc"] <= 8), lod_pixels=lod_pixels)
        summary_mask = visible & has_mass & summary_sel
        refine_mask = visible & has_mass & refine_sel

        s_idx = np.where(summary_mask)[0]
        if len(s_idx) > 0:
            summary_parts.append((
                lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                lv["qty"][s_idx], lv["cov"][s_idx],
                lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                lv["npart"][s_idx],
            ))

        refine_cells = np.where(refine_mask)[0]

        # Traverse finer levels
        for li in range(len(self.levels) - 2, -1, -1):
            if len(refine_cells) == 0:
                break

            lv = self.levels[li]
            parent_lv = self.levels[li + 1]
            hd = lv["half_diag"]
            is_leaf = lv["is_leaf"]

            # Expand parents to children
            child_indices = []
            for pi in refine_cells:
                cs = parent_lv["child_offset"][pi]
                ce = (parent_lv["child_offset"][pi + 1]
                      if pi < parent_lv["nc"] - 1 else lv["nc"])
                for ci in range(cs, ce):
                    child_indices.append(ci)

            if not child_indices:
                break
            child_flat = np.array(child_indices, dtype=np.int64)

            valid = lv["mass"][child_flat] > 0
            child_flat = child_flat[valid]
            if len(child_flat) == 0:
                break

            centers = lv["centers"][child_flat]
            diff = centers - cam_pos
            dist = np.sqrt((diff * diff).sum(axis=1))
            safe_dist = np.maximum(dist, 0.01)
            h_pix = hd / safe_dist * pix_per_rad

            child_is_leaf = is_leaf[child_flat]
            summary_sel, refine_sel = self._classify_nodes(
                h_pix, child_is_leaf, is_root_level=False, lod_pixels=lod_pixels)

            s_idx = child_flat[summary_sel]
            if len(s_idx) > 0:
                summary_parts.append((
                    lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                    lv["qty"][s_idx], lv["cov"][s_idx],
                    lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                    lv["npart"][s_idx],
                ))

            refine_cells = child_flat[refine_sel]

        # Gather particles from emitted leaves
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros(0, dtype=np.float32)
        z6 = np.zeros((0, 6), dtype=np.float32)

        n_visible_real = 0
        r_pos = r_hsml = r_mass = r_qty = z3, z1, z1, z1

        if emit_leaf_indices:
            vis_cells = np.array(emit_leaf_indices, dtype=np.int64)
            n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
            budget = max(max_particles - n_summaries, max_particles // 2)

            if importance_sampling:
                # Compute distances for importance sampling
                leaf_centers = np.array([
                    self.levels[0]["centers"][0]  # placeholder
                ] * len(vis_cells), dtype=np.float32)
                # Use cell_start midpoints as approximate centers
                vis_depths = np.ones(len(vis_cells), dtype=np.float64)
                vis_cell_h = np.ones(len(vis_cells), dtype=np.float64)
                for i, li_idx in enumerate(vis_cells):
                    ps = self.cell_start[li_idx]
                    pe = self.cell_start[li_idx + 1]
                    mid = (ps + pe) // 2
                    c = self.sorted_pos[mid]
                    d = np.sqrt(np.sum((c - cam_pos)**2))
                    vis_depths[i] = max(d, 0.01)
                    vis_cell_h[i] = self.sorted_hsml[ps:pe].mean() if pe > ps else 1.0
                r_pos, r_hsml, r_mass, r_qty, n_visible_real = _gather_importance_sampled(
                    self.cell_start, vis_cells, budget,
                    self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                    vis_depths, vis_cell_h)
            else:
                r_pos, r_hsml, r_mass, r_qty, n_visible_real = _gather_subsampled_direct(
                    self.cell_start, vis_cells, budget,
                    self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty)

        n_visible_real = int(n_visible_real) if not isinstance(n_visible_real, int) else n_visible_real
        if isinstance(r_pos, tuple):
            r_pos, r_hsml, r_mass, r_qty = z3, z1, z1, z1

        n_sampled = len(r_pos)
        if not importance_sampling and n_sampled > 0 and n_visible_real > n_sampled:
            ratio = n_visible_real / n_sampled
            r_mass = r_mass * ratio
            r_hsml = r_hsml * (ratio ** (1.0 / 3.0))

        # Assemble summary output
        N_ISOTROPIC = 64  # cells with fewer particles revert to isotropic splats
        if summary_parts:
            s_pos = np.concatenate([p[0] for p in summary_parts]).astype(np.float32)
            s_hsml = np.concatenate([p[1] for p in summary_parts]).astype(np.float32)
            s_mass = np.concatenate([p[2] for p in summary_parts]).astype(np.float32)
            s_qty = np.concatenate([p[3] for p in summary_parts]).astype(np.float32)
            s_cov = np.concatenate([p[4] for p in summary_parts]).astype(np.float32)
            s_mean_h2 = np.concatenate([p[5] for p in summary_parts]).astype(np.float32)
            # Cell width (not half_diag) for each summary
            s_cs = np.concatenate([
                np.broadcast_to(p[6][None, :], (len(p[0]), 3))
                for p in summary_parts
            ]).astype(np.float32)
            s_npart = np.concatenate([p[7] for p in summary_parts])

            if anisotropic:
                # Kernel smoothing padding
                s_cov[:, 0] += 0.225 * s_mean_h2
                s_cov[:, 3] += 0.225 * s_mean_h2
                s_cov[:, 5] += 0.225 * s_mean_h2

                # Force isotropic for cells with too few particles
                iso = s_npart < N_ISOTROPIC
                if iso.any():
                    trace_iso = s_cov[iso, 0] + s_cov[iso, 3] + s_cov[iso, 5]
                    iso_var = trace_iso / 3.0
                    s_cov[iso, 0] = iso_var
                    s_cov[iso, 1] = 0
                    s_cov[iso, 2] = 0
                    s_cov[iso, 3] = iso_var
                    s_cov[iso, 4] = 0
                    s_cov[iso, 5] = iso_var

                # Recompute hsml from covariance
                new_trace = s_cov[:, 0] + s_cov[:, 3] + s_cov[:, 5]
                s_hsml = np.sqrt(np.maximum(
                    (1.0 / 0.225) * new_trace, 1e-30)).astype(np.float32)
            else:
                # Isotropic mode: hsml = 2 * cell_width
                s_hsml = (2.0 * s_cs[:, 0]).astype(np.float32)
                # Zero out covariance (unused in isotropic path)
                s_cov[:] = 0
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return (
            r_pos.astype(np.float32), r_hsml.astype(np.float32),
            r_mass.astype(np.float32), r_qty.astype(np.float32),
            s_pos, s_hsml, s_mass, s_qty, s_cov,
        )

    def _frustum_cull_leaves(self, camera, max_particles, importance_sampling=False):
        """Fast path: frustum cull all leaf cells directly."""
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        half_tan = np.tan(fov_rad / 2)

        # Collect all leaf centers and half_diags
        all_centers = []
        all_hsml = []
        all_leaf_idx = []
        all_half_diag = []
        for lv in self.levels:
            leaf_mask = np.where(lv["is_leaf"])[0]
            if len(leaf_mask) > 0:
                all_centers.append(lv["centers"][leaf_mask])
                all_hsml.append(lv["hsml"][leaf_mask])
                all_leaf_idx.append(lv["leaf_indices"][leaf_mask])
                all_half_diag.append(
                    np.full(len(leaf_mask), lv["half_diag"], dtype=np.float32))

        if not all_centers:
            z3 = np.zeros((0, 3), dtype=np.float32)
            z1 = np.zeros(0, dtype=np.float32)
            return z3, z1, z1, z1

        centers = np.concatenate(all_centers)
        hsml = np.concatenate(all_hsml)
        leaf_idx = np.concatenate(all_leaf_idx)
        hd = np.concatenate(all_half_diag)

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        cell_extent = hd + np.minimum(hsml, 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        visible = in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        vis_idx = leaf_idx[visible]
        if len(vis_idx) == 0:
            z3 = np.zeros((0, 3), dtype=np.float32)
            z1 = np.zeros(0, dtype=np.float32)
            return z3, z1, z1, z1

        vis_cells = vis_idx.astype(np.int64)

        if importance_sampling:
            dist = np.sqrt(depths[visible]**2 + rights[visible]**2 + ups[visible]**2)
            safe_dist = np.maximum(dist, 0.01).astype(np.float64)
            vis_cell_h = hsml[visible].astype(np.float64)
            pos, h, mass, qty, n_vis = _gather_importance_sampled(
                self.cell_start, vis_cells, max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                safe_dist, vis_cell_h)
        else:
            pos, h, mass, qty, n_vis = _gather_subsampled_direct(
                self.cell_start, vis_cells, max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty)

        n_vis = int(n_vis)
        n_sampled = len(pos)
        if not importance_sampling and n_sampled > 0 and n_vis > n_sampled:
            ratio = n_vis / n_sampled
            mass = mass * ratio
            h = h * (ratio ** (1.0 / 3.0))

        return (
            pos.astype(np.float32), h.astype(np.float32),
            mass.astype(np.float32), qty.astype(np.float32),
        )
