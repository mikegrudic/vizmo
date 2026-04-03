"""Detailed timing breakdown of the DataFlyer2 pipeline.

Measures each stage independently: grid build, moment accumulation,
coarsening, frustum cull/LOD, particle gathering, GPU upload, render.
Also measures update_weights (field switch) cost.

Usage:
    python tests/bench_detailed.py [snapshot_path]
"""

import time
import sys
import types
import numpy as np

SNAPSHOT_DEFAULT = "/Users/mgrudic/code/bubblebuddies_gizmo/popeye/bubblebuddies_gizmo/SN_512/snapshot_060.hdf5"
RES = 512
FOV = 90
N_REPEATS = 5  # repeat each measurement for stable medians


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


def bench_grid_build(pos, masses, hsml):
    """Measure full SpatialGrid construction (sort + moments + coarsen)."""
    from dataflyer.spatial_grid import SpatialGrid
    times = []
    grid = None
    for _ in range(max(1, N_REPEATS // 2)):  # fewer repeats for expensive op
        t0 = time.perf_counter()
        grid = SpatialGrid(pos, masses, hsml, masses)
        times.append(time.perf_counter() - t0)
    return grid, times


def bench_grid_build_breakdown(pos, masses, hsml):
    """Break down grid construction into sub-stages."""
    from dataflyer.spatial_grid import SpatialGrid, _accumulate_cell_moments, CellMoments

    n_cells = 64
    pmin = pos.min(axis=0).astype(np.float32)
    pmax = pos.max(axis=0).astype(np.float32)
    box = pmax - pmin
    box[box == 0] = 1.0
    cs = box / n_cells

    # Stage 1: cell assignment + argsort
    t_sort_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        cell_idx = np.clip(((pos - pmin) / cs).astype(np.int32), 0, n_cells - 1)
        cell_id = cell_idx[:, 0] * n_cells * n_cells + cell_idx[:, 1] * n_cells + cell_idx[:, 2]
        sort_order = np.argsort(cell_id)
        t_sort_times.append(time.perf_counter() - t0)

    sorted_cell_id = cell_id[sort_order]

    # Stage 2: cell_start (CSR) construction
    t_csr_times = []
    nc3 = n_cells ** 3
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        cell_start = np.zeros(nc3 + 1, dtype=np.int64)
        unique_cells, counts = np.unique(sorted_cell_id, return_counts=True)
        cell_start[unique_cells + 1] = counts
        np.cumsum(cell_start, out=cell_start)
        t_csr_times.append(time.perf_counter() - t0)

    # Stage 3: permute particle arrays
    t_perm_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        sorted_pos = pos[sort_order].astype(np.float32)
        sorted_hsml = hsml[sort_order].astype(np.float32)
        sorted_mass = masses[sort_order].astype(np.float32)
        sorted_qty = masses[sort_order].astype(np.float32)
        t_perm_times.append(time.perf_counter() - t0)

    # Stage 4: moment accumulation (numba kernel)
    # Warmup numba JIT
    moments = CellMoments(nc3)
    moments.accumulate_from_particles(cell_start, sorted_pos, sorted_mass, sorted_hsml, sorted_qty)

    t_moments_times = []
    for _ in range(N_REPEATS):
        moments = CellMoments(nc3)
        t0 = time.perf_counter()
        moments.accumulate_from_particles(cell_start, sorted_pos, sorted_mass, sorted_hsml, sorted_qty)
        t_moments_times.append(time.perf_counter() - t0)

    # Stage 5: derive intensive quantities
    t_derive_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        com, qty, cov, cell_hsml = moments.derive()
        t_derive_times.append(time.perf_counter() - t0)

    # Stage 6: coarsen hierarchy (5 levels)
    # Build finest level dict first
    centers = np.zeros((nc3, 3), dtype=np.float32)  # placeholder
    finest = {
        "nc": n_cells, "cs": cs, "moments": moments,
        "mass": moments.mass, "com": com, "hsml": cell_hsml,
        "qty": qty, "cov": cov, "mxx": moments.mxx, "mh2": moments.mh2,
    }

    t_coarsen_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        prev = finest
        nc = n_cells
        while nc > 2:
            child_moments = prev["moments"]
            parent_nc = nc // 2
            parent_moments = CellMoments(parent_nc ** 3)
            parent_moments.coarsen_2x2x2(child_moments, prev["com"], prev["qty"], nc)
            p_com, p_qty, p_cov, p_hsml = parent_moments.derive()
            prev = {
                "nc": parent_nc, "moments": parent_moments,
                "mass": parent_moments.mass, "com": p_com, "hsml": p_hsml,
                "qty": p_qty, "cov": p_cov, "mxx": parent_moments.mxx, "mh2": parent_moments.mh2,
            }
            nc = parent_nc
        t_coarsen_times.append(time.perf_counter() - t0)

    return {
        "argsort": t_sort_times,
        "csr_build": t_csr_times,
        "permute_arrays": t_perm_times,
        "moment_accumulation": t_moments_times,
        "derive_intensive": t_derive_times,
        "coarsen_hierarchy": t_coarsen_times,
    }


def bench_update_weights(grid, masses):
    """Measure update_weights (field switch) cost."""
    # Use a different qty field to simulate switching
    qty = np.random.rand(len(masses)).astype(np.float32)

    # Warmup
    grid.update_weights(masses, qty)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        grid.update_weights(masses, qty)
        times.append(time.perf_counter() - t0)
    return times


def bench_query_frustum_lod(grid, camera, max_particles=4_000_000, lod_pixels=4):
    """Measure query_frustum_lod cost, broken into cull + gather."""
    # Warmup
    grid.query_frustum_lod(camera, max_particles, lod_pixels=lod_pixels, viewport_width=RES)

    times = []
    n_real = []
    n_summary = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = grid.query_frustum_lod(
            camera, max_particles, lod_pixels=lod_pixels, viewport_width=RES
        )
        times.append(time.perf_counter() - t0)
        if len(result) == 9:
            n_real.append(len(result[0]))
            n_summary.append(len(result[4]))
        else:
            n_real.append(len(result[0]))
            n_summary.append(0)
    return times, int(np.median(n_real)), int(np.median(n_summary))


def bench_query_finest_only(grid, camera, max_particles=4_000_000):
    """Measure _frustum_cull_finest (no LOD, no summaries)."""
    # Warmup
    grid._frustum_cull_finest(camera, max_particles)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        grid._frustum_cull_finest(camera, max_particles)
        times.append(time.perf_counter() - t0)
    return times


def bench_gpu_upload(ctx, renderer, grid, camera):
    """Measure GPU buffer upload cost (the CPU→GPU bottleneck)."""
    result = grid.query_frustum_lod(camera, 4_000_000, lod_pixels=4, viewport_width=RES)

    if len(result) == 9:
        r_pos, r_hsml, r_mass, r_qty = result[:4]
        s_pos, s_hsml, s_mass, s_qty, s_cov = result[4:]
    else:
        r_pos, r_hsml, r_mass, r_qty = result
        s_pos = np.zeros((0, 3), np.float32)
        s_cov = np.zeros((0, 6), np.float32)
        s_mass = np.zeros(0, np.float32)
        s_qty = np.zeros(0, np.float32)

    # Warmup
    renderer._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)

    upload_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)
        upload_times.append(time.perf_counter() - t0)

    aniso_times = []
    if len(s_pos) > 0:
        renderer._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            renderer._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
            aniso_times.append(time.perf_counter() - t0)

    data_size_mb = (r_pos.nbytes + r_hsml.nbytes + r_mass.nbytes + r_qty.nbytes) / 1e6
    return upload_times, aniso_times, data_size_mb


def bench_render(ctx, renderer, camera):
    """Measure GPU render pass cost."""
    from dataflyer.renderer import SplatRenderer

    # Patch to skip screen framebuffer
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

    # Warmup
    renderer.render(camera, RES, RES)
    ctx.finish()

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer.render(camera, RES, RES)
        ctx.finish()
        times.append(time.perf_counter() - t0)

    return times


def bench_numpy_argsort_standalone(n):
    """Measure hypothetical CPU depth-sort cost for n particles."""
    keys = np.random.rand(n).astype(np.float32)
    # Warmup
    np.argsort(keys)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        np.argsort(keys)
        times.append(time.perf_counter() - t0)
    return times


def fmt(times_s, unit="ms"):
    """Format timing list as median with range."""
    arr = np.array(times_s)
    if unit == "ms":
        arr *= 1000
    elif unit == "s":
        pass
    med = np.median(arr)
    lo, hi = arr.min(), arr.max()
    return f"{med:8.1f} {unit}  (range {lo:.1f}-{hi:.1f})"


def main():
    import moderngl
    from dataflyer.renderer import SplatRenderer
    from dataflyer.colormaps import create_colormap_texture_safe

    path = sys.argv[1] if len(sys.argv) > 1 else SNAPSHOT_DEFAULT
    print(f"Loading {path}...")
    pos, masses, hsml, boxsize = load_snapshot(path)
    n = len(pos)
    print(f"  {n:,} particles, boxsize={boxsize:.1f}")
    data_mb = (pos.nbytes + masses.nbytes + hsml.nbytes) / 1e6
    print(f"  Raw data: {data_mb:.0f} MB\n")

    camera = make_camera(boxsize, FOV)

    # === Grid Build Breakdown ===
    print("=" * 70)
    print("GRID BUILD BREAKDOWN")
    print("=" * 70)
    breakdown = bench_grid_build_breakdown(pos, masses, hsml)
    total_build = 0
    for stage, times in breakdown.items():
        med = np.median(times)
        total_build += med
        print(f"  {stage:<25s} {fmt(times)}")
    print(f"  {'TOTAL (sum of medians)':<25s} {total_build*1000:8.1f} ms")

    # === Full grid build (end-to-end) ===
    print(f"\n  Full SpatialGrid():")
    grid, build_times = bench_grid_build(pos, masses, hsml)
    print(f"  {'end-to-end':<25s} {fmt(build_times)}")

    # === Update Weights (field switch) ===
    print(f"\n{'=' * 70}")
    print("UPDATE WEIGHTS (field switch simulation)")
    print("=" * 70)
    uw_times = bench_update_weights(grid, masses)
    print(f"  {'update_weights()':<25s} {fmt(uw_times)}")

    # === Frustum Cull + LOD ===
    print(f"\n{'=' * 70}")
    print("FRUSTUM CULL + LOD QUERY")
    print("=" * 70)

    for lod_px in [4, 2, 1]:
        label = f"query_frustum_lod(lod={lod_px})"
        if lod_px <= 2:
            # lod_pixels <= 2 triggers fast path (_frustum_cull_finest)
            times = bench_query_finest_only(grid, camera)
            n_r, n_s = "N/A", "N/A"
            print(f"  {label:<35s} {fmt(times)}  (fastest path, no summaries)")
        else:
            times, n_r, n_s = bench_query_frustum_lod(grid, camera, lod_pixels=lod_px)
            print(f"  {label:<35s} {fmt(times)}  ({n_r:,} real + {n_s:,} summary)")

    # Different camera positions
    print(f"\n  Varying camera distance:")
    for dist_frac, label in [(0.1, "close"), (0.5, "medium"), (1.5, "far"), (5.0, "very far")]:
        cam = make_camera(boxsize, FOV)
        cam.position = np.array([boxsize/2, boxsize/2, boxsize/2 + boxsize*dist_frac], dtype=np.float32)
        times, n_r, n_s = bench_query_frustum_lod(grid, cam, lod_pixels=4)
        print(f"    {label:<15s} (d={dist_frac:.1f}L) {fmt(times)}  ({n_r:,} + {n_s:,})")

    # === GPU Upload ===
    print(f"\n{'=' * 70}")
    print("GPU UPLOAD (CPU -> GPU transfer)")
    print("=" * 70)
    ctx = moderngl.create_standalone_context()
    renderer = SplatRenderer(ctx)
    renderer.colormap_tex = create_colormap_texture_safe(ctx, "magma")
    renderer.resolve_mode = 0
    renderer.log_scale = 1
    renderer._viewport_width = RES

    renderer.set_particles(pos, hsml, masses)

    upload_times, aniso_times, data_mb = bench_gpu_upload(ctx, renderer, grid, camera)
    print(f"  {'_upload_arrays()':<25s} {fmt(upload_times)}  ({data_mb:.1f} MB)")
    if aniso_times:
        print(f"  {'_upload_aniso_summaries':<25s} {fmt(aniso_times)}")

    # === Render ===
    print(f"\n{'=' * 70}")
    print("GPU RENDER (draw calls + resolve)")
    print("=" * 70)
    renderer.update_visible(camera)
    render_times = bench_render(ctx, renderer, camera)
    print(f"  {'render()':<25s} {fmt(render_times)}")

    # === Hypothetical CPU depth sort ===
    print(f"\n{'=' * 70}")
    print("HYPOTHETICAL CPU DEPTH SORT (for opacity rendering)")
    print("=" * 70)
    for n_sort in [1_000_000, 2_000_000, 4_000_000]:
        sort_times = bench_numpy_argsort_standalone(n_sort)
        print(f"  np.argsort({n_sort/1e6:.0f}M f32)     {fmt(sort_times)}")

    # === End-to-end update_visible ===
    print(f"\n{'=' * 70}")
    print("END-TO-END update_visible() (cull + upload)")
    print("=" * 70)
    # Warmup
    renderer.update_visible(camera)

    e2e_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        renderer.update_visible(camera)
        e2e_times.append(time.perf_counter() - t0)
    cull_ms = renderer._last_cull_ms
    upload_ms = renderer._last_upload_ms
    print(f"  {'end-to-end':<25s} {fmt(e2e_times)}")
    print(f"  Last breakdown: cull={cull_ms:.1f}ms  upload={upload_ms:.1f}ms")

    # === Summary ===
    print(f"\n{'=' * 70}")
    print("SUMMARY — where time goes per frame")
    print("=" * 70)

    cull_med = np.median([t for t, _, _ in [bench_query_frustum_lod(grid, camera)]]) * 1000
    upload_med = np.median(upload_times) * 1000
    render_med = np.median(render_times) * 1000

    total = cull_med + upload_med + render_med
    print(f"  {'Cull + LOD + gather':<25s} {cull_med:7.1f} ms  ({cull_med/total*100:4.1f}%)")
    print(f"  {'GPU upload':<25s} {upload_med:7.1f} ms  ({upload_med/total*100:4.1f}%)")
    print(f"  {'GPU render':<25s} {render_med:7.1f} ms  ({render_med/total*100:4.1f}%)")
    print(f"  {'TOTAL':<25s} {total:7.1f} ms  (= {1000/total:.0f} fps theoretical)")
    print()

    renderer.release()
    ctx.release()


if __name__ == "__main__":
    main()
