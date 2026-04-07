"""Snapshot zoom benchmark — single-level vs multigrid LOD.

Loads an HDF5 snapshot (default: snapshot_600.hdf5 in repo root), starts
the camera at the default auto-scale position (above the data, looking
down -z), and zooms exponentially toward the mass-weighted median of
the gas distribution with a 3-second e-folding time, stopping at 10 kpc.

Renders the same camera path under each requested multigrid level
count and prints a summary table.

    # default: snapshot_600.hdf5 in repo root, single-level vs 6-level
    python tests/bench_snapshot_zoom.py

    # custom snapshot, just multigrid
    python tests/bench_snapshot_zoom.py --snap path/to/snap.hdf5 --multigrid 6

    # cheaper run for iteration
    python tests/bench_snapshot_zoom.py --res 1024 --fps 15
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import wgpu

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = REPO_ROOT / "snapshot_600.hdf5"

from vizmo.camera import Camera
from vizmo.colormaps import colormap_to_texture_data
from vizmo.gpu_compute import GPUCompute
from vizmo.wgpu_renderer import WGPURenderer


def load_snapshot(path):
    """Load gas positions/masses/hsml. Caches the meshoid smoothing
    lengths to <snap>.hsml.npy because building them on a 100M-particle
    snapshot takes a while.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        pos = np.array(f["PartType0/Coordinates"][:], dtype=np.float32)
        mass = np.array(f["PartType0/Masses"][:], dtype=np.float32)
    print(f"  loaded {len(pos):,} gas particles from {path.name}")

    cache = path.with_suffix(path.suffix + ".hsml.npy")
    if cache.exists():
        hsml = np.load(cache)
        if len(hsml) == len(pos):
            print(f"  loaded cached smoothing lengths from {cache.name}")
            return pos, mass, hsml.astype(np.float32)

    print("  computing smoothing lengths (one-time)...")
    from meshoid import Meshoid

    hsml = Meshoid(pos.astype(np.float64)).SmoothingLength().astype(np.float32)
    try:
        np.save(cache, hsml)
        print(f"  cached to {cache.name}")
    except OSError:
        pass
    return pos, mass, hsml


def mass_weighted_median(pos, mass):
    n = len(pos)
    step = max(1, n // 100)
    sp = pos[::step]
    sw = np.asarray(mass[::step], dtype=np.float64)
    out = np.empty(3, dtype=np.float64)
    for axis in range(3):
        order = np.argsort(sp[:, axis])
        cw = np.cumsum(sw[order])
        idx = min(int(np.searchsorted(cw, cw[-1] * 0.5)), len(order) - 1)
        out[axis] = float(sp[order[idx], axis])
    return out


def setup(pos, mass, hsml, res, multigrid_levels):
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    feats = set()
    if "float32-blendable" in adapter.features:
        feats.add("float32-blendable")
    device = adapter.request_device_sync(required_features=feats)

    r = WGPURenderer(device, canvas_context=None, present_format="bgra8unorm")
    r.set_colormap(colormap_to_texture_data("magma"))
    r.kernel = "cubic_spline"
    r.resolve_mode = 0
    r.log_scale = 1
    r.multigrid_levels = multigrid_levels
    r.set_particles(pos, hsml, mass)

    gc = GPUCompute(device)
    gc.upload_subsample_only(pos, hsml, mass, mass)
    r.set_subsample_chunks(gc.get_chunk_bufs(), world_offset=gc.get_pos_offset())
    r._ensure_fbo(res, res, which=1)

    cam = Camera(fov=60, aspect=1.0)
    cam.auto_scale(pos, masses=mass)
    return device, r, cam


def run(label, device, r, cam, target, res, n_frames, dt, tau, stop_dist, cap):
    """Step the camera exponentially toward target. Returns list of
    (distance, ms_per_frame) records.
    """
    r.set_subsample_max_per_frame(cap)
    pos0 = cam.position.copy()
    target = np.asarray(target, dtype=np.float32)
    init_dist = float(np.linalg.norm(pos0 - target))
    direction = (target - pos0) / max(init_dist, 1e-30)

    # Warmup
    for _ in range(3):
        r._render_accum(cam, res, res, r._accum_textures)
    device.queue.on_submitted_work_done_sync()

    records = []
    t = 0.0
    for i in range(n_frames):
        # Exponential decay of distance with e-folding time tau, clipped
        # at stop_dist.
        dist = init_dist * np.exp(-t / tau)
        if dist < stop_dist:
            dist = stop_dist
        cam.position = (target - direction * dist).astype(np.float32)
        cam._dirty = True

        device.queue.on_submitted_work_done_sync()
        t0 = time.perf_counter()
        r._render_accum(cam, res, res, r._accum_textures)
        device.queue.on_submitted_work_done_sync()
        ms = (time.perf_counter() - t0) * 1000.0
        records.append((dist, ms))
        t += dt
        if dist <= stop_dist and t > 4 * tau:
            break
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--snap",
        default=str(DEFAULT_SNAPSHOT),
        help=f"HDF5 snapshot path (default: {DEFAULT_SNAPSHOT.name} in repo root)",
    )
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--multigrid", type=int, nargs="+", default=[1, 6])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--stop", type=float, default=10.0, help="stop distance in same units as snapshot (kpc)")
    p.add_argument("--cap", type=int, default=4_000_000)
    args = p.parse_args()

    if not Path(args.snap).exists():
        sys.exit(f"snapshot not found: {args.snap}\n" f"pass --snap PATH or place snapshot_600.hdf5 in {REPO_ROOT}")

    pos, mass, hsml = load_snapshot(args.snap)
    target = mass_weighted_median(pos, mass)
    print(f"  mass-weighted median target: {target}")
    print(f"  data extent: {pos.max(axis=0) - pos.min(axis=0)}")

    dt = 1.0 / args.fps
    n_frames = int(args.fps * 6 * args.tau)  # zoom for ~6 e-folds

    summary = []
    for ml in args.multigrid:
        print(f"\n=== multigrid_levels={ml} ===")
        device, r, cam = setup(pos, mass, hsml, args.res, ml)
        recs = run(f"ml={ml}", device, r, cam, target, args.res, n_frames, dt, args.tau, args.stop, args.cap)
        ds = np.array([d for d, _ in recs])
        ms = np.array([m for _, m in recs])
        print(f"  {'frame':>6} {'dist':>12} {'ms':>10}")
        step = max(1, len(recs) // 15)
        for i in range(0, len(recs), step):
            print(f"  {i:>6d} {ds[i]:>12.2f} {ms[i]:>10.3f}")
        print(
            f"  total {ms.sum():>8.1f} ms / {len(recs)} frames " f"| mean {ms.mean():.2f} | median {np.median(ms):.2f}"
        )
        summary.append((ml, ms.mean(), np.median(ms), ms.sum(), len(recs)))
        r.release()

    print("\n=== summary ===")
    print(f"{'levels':>8} {'mean ms':>10} {'median ms':>12} {'total ms':>12} {'frames':>8}")
    for ml, mean, med, tot, n in summary:
        print(f"{ml:>8d} {mean:>10.2f} {med:>12.2f} {tot:>12.1f} {n:>8d}")
    if len(summary) >= 2:
        base = summary[0][1]
        for ml, mean, *_ in summary[1:]:
            print(f"  speedup levels={ml} vs levels={summary[0][0]}: " f"{base / mean:.2f}x")


if __name__ == "__main__":
    main()
