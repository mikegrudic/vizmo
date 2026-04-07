"""Benchmark splat throughput vs subsample cap (LOD stride).

Renders the same particle set at increasing strides and reports per-frame
ms. Used to characterise the diminishing-returns plateau of the random
subsampling LOD and to measure the effect of the multigrid path.

Usage:
    python tests/bench_multigrid.py [--n N] [--res R] [--multigrid LEVELS]
"""

import argparse
import time

import numpy as np
import wgpu
from meshoid import Meshoid

from vizmo.camera import Camera
from vizmo.colormaps import colormap_to_texture_data
from vizmo.gpu_compute import GPUCompute
from vizmo.wgpu_renderer import WGPURenderer


def make_particles(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, (n, 3)).astype(np.float64)
    masses = (np.ones(n) / n).astype(np.float64)
    hsml = Meshoid(pos, boxsize=1.0).SmoothingLength()
    return pos.astype(np.float32), masses.astype(np.float32), hsml.astype(np.float32)


def setup_renderer(pos, hsml, mass, res, multigrid_levels=1, hsml_scale=1.0):
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    feats = set()
    if "float32-blendable" in adapter.features:
        feats.add("float32-blendable")
    device = adapter.request_device_sync(required_features=feats)

    r = WGPURenderer(device, canvas_context=None, present_format="bgra8unorm")
    r.set_colormap(colormap_to_texture_data("magma"))
    r.kernel = "cubic_spline"
    r.resolve_mode = 0
    r.log_scale = 0
    r.multigrid_levels = multigrid_levels
    r.hsml_scale = hsml_scale
    r.set_particles(pos, hsml, mass)

    gc = GPUCompute(device)
    gc.upload_subsample_only(pos, hsml, mass, mass)
    r.set_subsample_chunks(gc.get_chunk_bufs(), world_offset=gc.get_pos_offset())
    r._ensure_fbo(res, res, which=1)

    cam = Camera(fov=90, aspect=1.0)
    cam.position = np.array([0.5, 0.5, 1.5], dtype=np.float32)
    cam._forward = np.array([0, 0, -1], dtype=np.float32)
    cam._up = np.array([0, 1, 0], dtype=np.float32)
    cam._dirty = True
    cam.near = 1e-4
    cam.far = 10.0
    return device, r, cam


def time_frames(device, r, cam, res, n_frames):
    # Warmup
    for _ in range(3):
        r._render_accum(cam, res, res, r._accum_textures)
    device.queue.on_submitted_work_done_sync()
    t0 = time.perf_counter()
    for _ in range(n_frames):
        r._render_accum(cam, res, res, r._accum_textures)
    device.queue.on_submitted_work_done_sync()
    return (time.perf_counter() - t0) * 1000.0 / n_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--multigrid", type=int, default=1, help="number of multigrid levels (1 = disabled)")
    p.add_argument(
        "--hsml-scale", type=float, default=1.0, help="multiplier on smoothing length (drives fragment cost)"
    )
    args = p.parse_args()

    print(f"benchmark: N={args.n} res={args.res} multigrid_levels={args.multigrid}")
    pos, mass, hsml = make_particles(args.n)
    device, r, cam = setup_renderer(
        pos, mass, hsml, args.res, multigrid_levels=args.multigrid, hsml_scale=args.hsml_scale
    )

    print(f"{'cap':>10} {'stride':>8} {'ms/frame':>12}")
    for cap_frac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0156, 0.0078):
        cap = max(1000, int(args.n * cap_frac))
        r.set_subsample_max_per_frame(cap)
        ms = time_frames(device, r, cam, args.res, args.frames)
        stride = args.n / min(args.n, cap)
        print(f"{cap:>10d} {stride:>8.1f} {ms:>12.3f}")

    r.release()


if __name__ == "__main__":
    main()
