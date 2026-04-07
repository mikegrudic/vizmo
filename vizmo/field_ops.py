"""Shared helpers for vector field projection, field combination, and app state."""

import numpy as np

# Constants shared between both app backends
SD_OPS = ["*", "+", "-", "/", "min", "max"]
RENDER_MODES = ["SurfaceDensity", "WeightedAverage", "WeightedVariance", "Composite"]
VECTOR_PROJECTIONS = ["LOS", "|v|", "|v|^2"]


def max_entropy_limits(vals, weights, n_bins=128, n_search=32, log_scale=False):
    """Find (lo, hi) that maximize mass-weighted Shannon entropy of the
    color histogram.

    The search runs on the *displayed* axis: when `log_scale=True`, the
    objective is the entropy of the log10(value) histogram, and the
    returned limits are still in linear units (the caller log-transforms
    them itself).

    Args:
        vals: (N,) value array.
        weights: (N,) per-sample weight (e.g. accumulated mass).
        n_bins: histogram bin count.
        n_search: grid resolution per axis (lo, hi).
        log_scale: when True, transform vals to log10 before searching
            and return exp10 of the optimal limits.
    """
    vals = np.asarray(vals)
    weights = np.asarray(weights)

    if log_scale:
        pos = vals > 0
        if not pos.any():
            return float(vals.min()) if len(vals) else 0.0, \
                   float(vals.max()) if len(vals) else 1.0
        vals = np.log10(vals[pos])
        weights = weights[pos]

    # Random subsample to cap sort cost. Strided subsampling is biased on
    # spatially-coherent inputs (the accum buffer is row-major), so use a
    # seeded RNG for reproducibility without ordering artifacts.
    if len(vals) > 100_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(vals), size=100_000, replace=False)
        vals = vals[idx]
        weights = weights[idx]

    order = np.argsort(vals)
    sv = vals[order]
    sw = weights[order]
    cw = np.cumsum(sw)
    total = cw[-1]
    if total <= 0:
        lo_lin = float(sv[0])
        hi_lin = float(sv[-1])
        if log_scale:
            return float(10 ** lo_lin), float(10 ** hi_lin)
        return lo_lin, hi_lin
    cw /= total

    # Candidate lo/hi values from mass-weighted CDF fractions.
    # Wider window than the original [0,.25]/[.75,1] so the search can
    # reach the optimum on heavy-tailed / skewed fields, where the true
    # best lo/hi can sit well past the 25/75 percentiles.
    lo_fracs = np.linspace(0.0, 0.45, n_search)
    hi_fracs = np.linspace(0.55, 1.0, n_search)
    lo_vals = np.interp(lo_fracs, cw, sv)
    hi_vals = np.interp(hi_fracs, cw, sv)

    # Bin-edge fractions [0, 1/n_bins, ..., 1]
    t = np.arange(n_bins + 1, dtype=np.float64) / n_bins

    # All edge positions: shape (n_search, n_search, n_bins+1)
    span = hi_vals[None, :, None] - lo_vals[:, None, None]
    edges = lo_vals[:, None, None] + t * span

    # Single vectorized CDF lookup
    cdf_at_edges = np.interp(edges.ravel(), sv, cw).reshape(edges.shape)

    # Bin probabilities
    p = np.diff(cdf_at_edges, axis=2)
    p[:, :, 0] += cdf_at_edges[:, :, 0]
    p[:, :, -1] += 1.0 - cdf_at_edges[:, :, -1]

    # Shannon entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    H = -np.sum(p * log_p, axis=2)

    # Mask invalid pairs (hi <= lo)
    H[hi_vals[None, :] <= lo_vals[:, None]] = -np.inf

    i, j = np.unravel_index(H.argmax(), H.shape)
    lo_opt = float(lo_vals[i])
    hi_opt = float(hi_vals[j])
    if log_scale:
        return float(10 ** lo_opt), float(10 ** hi_opt)
    return lo_opt, hi_opt


def make_default_app_state(data):
    """Create the default shared state dict for both app backends.

    Args:
        data: SnapshotData instance.

    Returns:
        dict with all field/mode/slot defaults.
    """
    sd_fields = data.available_fields()
    vector_fields = data.available_vector_fields()
    has_vel = "Velocities" in vector_fields
    return {
        "sd_fields": sd_fields,
        "vector_fields": vector_fields,
        "sd_field": "Masses",
        "sd_field2": "None",
        "sd_op": "*",
        "render_mode_name": "SurfaceDensity",
        "wa_data_field": "Masses",
        "vector_projection": "LOS",
        "composite": False,
        "slot": [
            {"mode": "SurfaceDensity", "weight": "Masses", "data": "Masses",
             "weight2": "None", "op": "*", "proj": "LOS",
             "min": -1.0, "max": 3.0, "log": 1, "resolve": 0},
            {"mode": "WeightedVariance" if has_vel else "SurfaceDensity",
             "weight": "Masses",
             "data": "Velocities" if has_vel else "Masses",
             "weight2": "None", "op": "*", "proj": "LOS",
             "min": -1.0, "max": 3.0, "log": 1,
             "resolve": 2 if has_vel else 0},
        ],
    }


def uses_vector_field(render_mode_name, wa_data_field, sd_field, sd_field2, vector_fields):
    """Check if any active field is a vector field."""
    if render_mode_name in ("WeightedAverage", "WeightedVariance"):
        if wa_data_field in vector_fields:
            return True
    if sd_field in vector_fields:
        return True
    if sd_field2 != "None" and sd_field2 in vector_fields:
        return True
    return False


def is_los_stale(render_mode_name, wa_data_field, sd_field, sd_field2,
                 vector_fields, vector_projection,
                 los_camera_pos, camera_position,
                 pos_threshold=None):
    """Check if the LOS projection needs recomputing.

    The LOS vector field projects each particle's vector onto the unit
    vector from the *camera position* to that particle. That direction
    is invariant under pure camera rotation — only translation
    invalidates it.

    `pos_threshold` is the maximum tolerated translation in world units
    before the cache is considered stale; if None it falls back to 1.0.
    """
    if not uses_vector_field(render_mode_name, wa_data_field, sd_field, sd_field2,
                             vector_fields):
        return False
    if vector_projection != "LOS":
        return False
    if los_camera_pos is None:
        return True
    thr = pos_threshold if pos_threshold is not None else 1.0
    return float(np.linalg.norm(np.asarray(camera_position)
                                - np.asarray(los_camera_pos))) > thr


def project_vector(vec, projection, camera_forward,
                   camera_position=None, positions=None):
    """Project an (N, 3) vector field to a scalar array.

    Args:
        vec: (N, 3) float array.
        projection: "LOS", "|v|", or "|v|^2".
        camera_forward: (3,) unit vector. Used as the LOS direction
            when `camera_position` and `positions` are not both supplied.
        camera_position: optional (3,) float array. When provided
            together with `positions`, the LOS direction is computed
            per-particle as the unit vector from the camera to each
            particle (instead of using the global `camera_forward`).
        positions: optional (N, 3) float array of particle positions.

    Returns:
        (N,) float32 array.
    """
    if projection == "LOS":
        if camera_position is not None and positions is not None:
            # Per-particle line-of-sight: dot the vector field with the
            # unit vector from the camera to each particle.
            d = positions.astype(np.float32, copy=False) - np.asarray(
                camera_position, dtype=np.float32)
            inv_len = 1.0 / np.maximum(
                np.linalg.norm(d, axis=1), np.float32(1e-30))
            d *= inv_len[:, None]
            out = (vec * d).sum(axis=1)
        else:
            out = vec @ camera_forward
    elif projection == "|v|":
        out = np.linalg.norm(vec, axis=1)
    else:  # |v|^2
        out = (vec * vec).sum(axis=1)
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def combine_fields(w, w2, op):
    """Combine two scalar field arrays with the given operator.

    Args:
        w, w2: (N,) arrays.
        op: one of "*", "+", "-", "/", "min", "max".

    Returns:
        (N,) combined array.
    """
    if op == "*":
        return w * w2
    elif op == "+":
        return w + w2
    elif op == "-":
        return w - w2
    elif op == "/":
        return w / np.maximum(np.abs(w2), 1e-30) * np.sign(w2)
    elif op == "min":
        return np.minimum(w, w2)
    elif op == "max":
        return np.maximum(w, w2)
    return w * w2


def resolve_field(field_name, vector_fields, data_manager, projection,
                  camera_forward, camera_position=None):
    """Load a field and project to scalar if it's a vector field.

    Args:
        field_name: HDF5 field name.
        vector_fields: set of field names that are vectors.
        data_manager: SnapshotData instance.
        projection: "LOS", "|v|", or "|v|^2".
        camera_forward: (3,) unit vector.
        camera_position: optional (3,) float. When provided and the
            projection is "LOS", the line-of-sight is computed
            per-particle (vector dotted with the unit vector from the
            camera to the particle) instead of using a single global
            forward direction.

    Returns:
        (N,) float32 array.
    """
    if field_name not in vector_fields:
        return data_manager.get_field(field_name)

    # Cache the projected scalar so repeat reweights at the same camera
    # pose (e.g. switching Field 2 between two vector fields) don't
    # recompute a 134M-element dot product every time.
    cache = getattr(data_manager, "_projected_cache", None)
    if cache is None:
        cache = {}
        data_manager._projected_cache = cache
    if projection == "LOS":
        if camera_position is not None:
            # Per-particle LOS depends only on the camera *position*
            # (the direction from camera to each particle is invariant
            # under camera rotation). Quantize so trivial float jitter
            # doesn't invalidate the entry.
            qp = tuple(int(round(float(c) * 1000)) for c in camera_position)
            key = (field_name, "LOS_pp", qp)
        else:
            qf = tuple(int(round(c * 10000)) for c in camera_forward)
            key = (field_name, "LOS", qf)
    else:
        key = (field_name, projection)
    hit = cache.get(key)
    if hit is not None:
        return hit
    vec = data_manager.get_vector_field(field_name)
    if projection == "LOS" and camera_position is not None:
        positions = data_manager.get_vector_field("Coordinates")
        out = project_vector(vec, projection, camera_forward,
                             camera_position=camera_position,
                             positions=positions)
    else:
        out = project_vector(vec, projection, camera_forward)
    cache[key] = out
    # Bound the cache to avoid unbounded growth from camera poses.
    if len(cache) > 32:
        # Drop an arbitrary old entry.
        cache.pop(next(iter(cache)))
    return out


def compute_weights(sd_field, sd_field2, sd_op, vector_fields, data_manager,
                    projection, camera_forward, camera_position=None):
    """Compute the final weight array from primary + optional secondary field.

    Returns:
        (N,) float32 array.
    """
    w = resolve_field(sd_field, vector_fields, data_manager, projection,
                      camera_forward, camera_position=camera_position)
    if sd_field2 != "None":
        w2 = resolve_field(sd_field2, vector_fields, data_manager, projection,
                           camera_forward, camera_position=camera_position)
        w = combine_fields(w, w2, sd_op)
    return w


def compute_slot_fields(slot, vector_fields, data_manager, camera_forward,
                        camera_position=None):
    """Compute weights and qty for a composite slot dict.

    Args:
        slot: dict with keys "weight", "weight2", "op", "mode", "data", "proj".
        vector_fields: set of field names that are vectors.
        data_manager: SnapshotData instance.
        camera_forward: (3,) unit vector.
        camera_position: optional (3,) float; when given, LOS becomes
            per-particle (camera-to-particle direction).

    Returns:
        (weights, qty) where qty is None for SurfaceDensity mode.
        Also sets slot["resolve"] to the resolve mode int.
    """
    proj = slot.get("proj", "LOS")

    # Weight field
    weights = resolve_field(slot["weight"], vector_fields, data_manager, proj,
                            camera_forward, camera_position=camera_position)

    # Optional second weight field
    w2_name = slot.get("weight2", "None")
    if w2_name != "None":
        w2 = resolve_field(w2_name, vector_fields, data_manager, proj,
                           camera_forward, camera_position=camera_position)
        weights = combine_fields(weights, w2, slot.get("op", "*"))

    # Data (qty) field for weighted average / variance
    if slot["mode"] in ("WeightedAverage", "WeightedVariance"):
        qty = resolve_field(slot["data"], vector_fields, data_manager, proj,
                            camera_forward, camera_position=camera_position)
    else:
        qty = None

    slot["resolve"] = {"SurfaceDensity": 0, "WeightedAverage": 1, "WeightedVariance": 2}[slot["mode"]]
    return weights, qty
