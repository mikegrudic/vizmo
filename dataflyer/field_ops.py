"""Shared helpers for vector field projection, field combination, and app state."""

import numpy as np

# Constants shared between both app backends
SD_OPS = ["*", "+", "-", "/", "min", "max"]
RENDER_MODES = ["SurfaceDensity", "WeightedAverage", "WeightedVariance", "Composite"]
VECTOR_PROJECTIONS = ["LOS", "|v|", "|v|^2"]


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
        "los_camera_fwd": None,
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
                 vector_fields, vector_projection, los_camera_fwd, camera_forward):
    """Check if the LOS projection needs recomputing due to camera rotation."""
    if not uses_vector_field(render_mode_name, wa_data_field, sd_field, sd_field2,
                             vector_fields):
        return False
    if vector_projection != "LOS":
        return False
    if los_camera_fwd is None:
        return True
    dot = float(np.dot(los_camera_fwd, camera_forward))
    return dot < 0.9998


def project_vector(vec, projection, camera_forward):
    """Project an (N, 3) vector field to a scalar array.

    Args:
        vec: (N, 3) float array.
        projection: "LOS", "|v|", or "|v|^2".
        camera_forward: (3,) unit vector for LOS projection.

    Returns:
        (N,) float32 array.
    """
    if projection == "LOS":
        return (vec @ camera_forward).astype(np.float32)
    elif projection == "|v|":
        return np.linalg.norm(vec, axis=1).astype(np.float32)
    else:  # |v|^2
        return (vec * vec).sum(axis=1).astype(np.float32)


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


def resolve_field(field_name, vector_fields, data_manager, projection, camera_forward):
    """Load a field and project to scalar if it's a vector field.

    Args:
        field_name: HDF5 field name.
        vector_fields: set of field names that are vectors.
        data_manager: SnapshotData instance.
        projection: "LOS", "|v|", or "|v|^2".
        camera_forward: (3,) unit vector.

    Returns:
        (N,) float32 array.
    """
    if field_name in vector_fields:
        vec = data_manager.get_vector_field(field_name)
        return project_vector(vec, projection, camera_forward)
    return data_manager.get_field(field_name)


def compute_weights(sd_field, sd_field2, sd_op, vector_fields, data_manager,
                    projection, camera_forward):
    """Compute the final weight array from primary + optional secondary field.

    Returns:
        (N,) float32 array.
    """
    w = resolve_field(sd_field, vector_fields, data_manager, projection, camera_forward)
    if sd_field2 != "None":
        w2 = resolve_field(sd_field2, vector_fields, data_manager, projection, camera_forward)
        w = combine_fields(w, w2, sd_op)
    return w


def compute_slot_fields(slot, vector_fields, data_manager, camera_forward):
    """Compute weights and qty for a composite slot dict.

    Args:
        slot: dict with keys "weight", "weight2", "op", "mode", "data", "proj".
        vector_fields: set of field names that are vectors.
        data_manager: SnapshotData instance.
        camera_forward: (3,) unit vector.

    Returns:
        (weights, qty) where qty is None for SurfaceDensity mode.
        Also sets slot["resolve"] to the resolve mode int.
    """
    proj = slot.get("proj", "LOS")

    # Weight field
    weights = resolve_field(slot["weight"], vector_fields, data_manager, proj, camera_forward)

    # Optional second weight field
    w2_name = slot.get("weight2", "None")
    if w2_name != "None":
        w2 = resolve_field(w2_name, vector_fields, data_manager, proj, camera_forward)
        weights = combine_fields(weights, w2, slot.get("op", "*"))

    # Data (qty) field for weighted average / variance
    if slot["mode"] in ("WeightedAverage", "WeightedVariance"):
        qty = resolve_field(slot["data"], vector_fields, data_manager, proj, camera_forward)
    else:
        qty = None

    slot["resolve"] = {"SurfaceDensity": 0, "WeightedAverage": 1, "WeightedVariance": 2}[slot["mode"]]
    return weights, qty
