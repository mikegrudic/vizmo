"""Compare wgpu vs moderngl backend performance on the same snapshot."""

import time
import sys
import types
import numpy as np

SNAPSHOT = "/Users/mgrudic/code/bubblebuddies_gizmo/popeye/bubblebuddies_gizmo/SN_512/snapshot_060.hdf5"
RES = 512
FOV = 90
N_REPEATS = 5


def load_snapshot(path):
    import h5py
    with h5py.File(path, "r") as f:
        pos = f["PartType0/Coordinates"][:].astype(np.float32)
        masses = f["PartType0/Masses"][:].astype(np.float32)
        boxsize = float(f["Header"].attrs["BoxSize"])
        for field in ("KernelMaxRadius", "SmoothingLength"):
            if field in f["PartType0"]:
                hsml = f["PartType0"][field][:].astype(np.float32)
                break
        else:
            raise KeyError("No smoothing length field found")
    return pos, masses, hsml, boxsize


def make_camera(boxsize, fov):
    from dataflyer.camera import Camera
    camera = Camera(fov=fov, aspect=1.0)
    center = boxsize / 2
    camera.position = np.array([center, center, center * 1.5], dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera.near = boxsize * 1e-6
    camera.far = boxsize * 10
    return camera


def fmt(times_s):
    arr = np.array(times_s) * 1000
    return f"{np.median(arr):7.1f} ms  (range {arr.min():.1f}-{arr.max():.1f})"


def bench_moderngl(pos, masses, hsml, boxsize):
    """Benchmark the moderngl backend."""
    import moderngl
    from dataflyer.renderer import SplatRenderer
    from dataflyer.colormaps import create_colormap_texture_safe

    ctx = moderngl.create_standalone_context()
    renderer = SplatRenderer(ctx)
    renderer.colormap_tex = create_colormap_texture_safe(ctx, "magma")
    renderer.resolve_mode = 0
    renderer.log_scale = 1
    renderer._viewport_width = RES

    camera = make_camera(boxsize, FOV)

    # Build grid
    t0 = time.perf_counter()
    renderer.set_particles(pos, hsml, masses)
    t_build = time.perf_counter() - t0

    # Warmup
    for _ in range(3):
        renderer.update_visible(camera)

    # Benchmark cull (= query_frustum_lod + upload)
    cull_times = []
    upload_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer.update_visible(camera)
        cull_times.append(time.perf_counter() - t0)
        upload_times.append(renderer._last_upload_ms / 1000)

    # Benchmark render (headless)
    original_render = renderer.render.__func__
    original_screen = type(ctx).screen

    def _headless_render(self, cam, w, h):
        try:
            type(ctx).screen = property(lambda s: (_ for _ in ()).throw(StopIteration))
            original_render(self, cam, w, h)
        except (StopIteration, AttributeError):
            pass
        finally:
            type(ctx).screen = original_screen

    renderer.render = types.MethodType(_headless_render, renderer)

    renderer.render(camera, RES, RES)
    ctx.finish()

    render_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer.render(camera, RES, RES)
        ctx.finish()
        render_times.append(time.perf_counter() - t0)

    n_vis = renderer.n_particles + getattr(renderer, 'n_big', 0) + getattr(renderer, 'n_aniso', 0)

    renderer.release()
    ctx.release()

    return {
        "build": t_build,
        "cull_total": cull_times,
        "upload": upload_times,
        "render": render_times,
        "n_vis": n_vis,
    }


def bench_wgpu(pos, masses, hsml, boxsize):
    """Benchmark the wgpu backend."""
    import wgpu
    from dataflyer.wgpu_renderer import WGPURenderer
    from dataflyer.colormaps import colormap_to_texture_data

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    req_features = set()
    if "float32-blendable" in adapter.features:
        req_features.add("float32-blendable")
    device = adapter.request_device_sync(
        required_features=req_features,
        required_limits={"max_storage_buffer_binding_size": 2**30},
    )

    renderer = WGPURenderer(device)
    rgba = colormap_to_texture_data("magma")
    renderer.set_colormap(rgba)
    renderer.resolve_mode = 0
    renderer.log_scale = 1
    renderer._viewport_width = RES

    camera = make_camera(boxsize, FOV)

    # Build grid
    t0 = time.perf_counter()
    renderer.set_particles(pos, hsml, masses)
    t_build = time.perf_counter() - t0

    # Warmup
    for _ in range(3):
        renderer.update_visible(camera)

    # Benchmark cull (= query_frustum_lod + buffer upload to wgpu)
    cull_times = []
    upload_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer.update_visible(camera)
        cull_times.append(time.perf_counter() - t0)
        upload_times.append(renderer._last_upload_ms / 1000)

    # Benchmark render (headless — accumulation only, no resolve since no canvas)
    renderer._ensure_fbo(RES, RES, which=1)
    renderer._render_accum(camera, RES, RES, renderer._accum_textures)

    render_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer._render_accum(camera, RES, RES, renderer._accum_textures)
        render_times.append(time.perf_counter() - t0)

    n_vis = renderer.n_particles + renderer.n_aniso

    renderer.release()

    return {
        "build": t_build,
        "cull_total": cull_times,
        "upload": upload_times,
        "render": render_times,
        "n_vis": n_vis,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else SNAPSHOT
    print(f"Loading {path}...")
    pos, masses, hsml, boxsize = load_snapshot(path)
    print(f"  {len(pos):,} particles, boxsize={boxsize:.1f}\n")

    print("=" * 70)
    print("MODERNGL BACKEND")
    print("=" * 70)
    mgl = bench_moderngl(pos, masses, hsml, boxsize)
    print(f"  Build:          {mgl['build']:.2f} s")
    print(f"  Cull (total):   {fmt(mgl['cull_total'])}")
    print(f"  Upload:         {fmt(mgl['upload'])}")
    print(f"  Render:         {fmt(mgl['render'])}")
    print(f"  Visible:        {mgl['n_vis']:,}")

    cull_med_mgl = np.median(mgl['cull_total']) * 1000
    upload_med_mgl = np.median(mgl['upload']) * 1000
    render_med_mgl = np.median(mgl['render']) * 1000
    total_mgl = cull_med_mgl + render_med_mgl
    print(f"  TOTAL/frame:    {total_mgl:.1f} ms ({1000/total_mgl:.0f} fps)")

    print(f"\n{'=' * 70}")
    print("WGPU BACKEND")
    print("=" * 70)
    wgp = bench_wgpu(pos, masses, hsml, boxsize)
    print(f"  Build:          {wgp['build']:.2f} s")
    print(f"  Cull (total):   {fmt(wgp['cull_total'])}")
    print(f"  Upload:         {fmt(wgp['upload'])}")
    print(f"  Render:         {fmt(wgp['render'])}")
    print(f"  Visible:        {wgp['n_vis']:,}")

    cull_med_wgpu = np.median(wgp['cull_total']) * 1000
    upload_med_wgpu = np.median(wgp['upload']) * 1000
    render_med_wgpu = np.median(wgp['render']) * 1000
    total_wgpu = cull_med_wgpu + render_med_wgpu
    print(f"  TOTAL/frame:    {total_wgpu:.1f} ms ({1000/total_wgpu:.0f} fps)")

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    print(f"  {'Stage':<20s} {'moderngl':>12s} {'wgpu':>12s} {'speedup':>10s}")
    print(f"  {'-'*54}")

    for label, mgl_t, wgpu_t in [
        ("Cull+upload", cull_med_mgl, cull_med_wgpu),
        ("  (upload only)", upload_med_mgl, upload_med_wgpu),
        ("Render", render_med_mgl, render_med_wgpu),
        ("TOTAL/frame", total_mgl, total_wgpu),
    ]:
        speedup = mgl_t / max(wgpu_t, 0.01)
        print(f"  {label:<20s} {mgl_t:>9.1f} ms {wgpu_t:>9.1f} ms {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
