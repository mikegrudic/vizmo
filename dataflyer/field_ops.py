"""Shared helpers for vector field projection and field combination."""

import numpy as np


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
