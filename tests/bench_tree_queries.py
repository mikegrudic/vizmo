"""Benchmark cKDTree segment-style queries on synthetic data sized like the
problem snapshot (~86M gas, 40 stars). Goal: figure out whether the slow
star-column update is intrinsic to scipy at this scale or specific to the
M2e2 gas distribution / smoothing-length spread.

Usage:
    python tests/bench_tree_queries.py
"""

import time
import numpy as np
from scipy.spatial import cKDTree


def fmt(t):
    if t < 1e-3:
        return f"{t*1e6:7.1f} us"
    if t < 1.0:
        return f"{t*1e3:7.1f} ms"
    return f"{t:7.2f}  s"


def bench(label, fn, repeats=1):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    dt = (time.perf_counter() - t0) / repeats
    print(f"  {label:50s} {fmt(dt)}")
    return out


def main():
    rng = np.random.default_rng(0)
    n_gas = 86_000_000
    n_stars = 40
    box = 100.0
    print(f"Synthetic uniform cube: n_gas={n_gas:,}, n_stars={n_stars}, box={box}")

    print("\n[alloc] random positions...")
    t0 = time.perf_counter()
    xgas = rng.uniform(-box / 2, box / 2, size=(n_gas, 3)).astype(np.float64)
    print(f"  positions  {fmt(time.perf_counter() - t0)}  ({xgas.nbytes/1e9:.2f} GB)")

    # Smoothing lengths: tight log-normal centered around what gives ~32
    # neighbors in a uniform field of this density.
    rho = n_gas / box**3
    h_typ = (32.0 / (4 / 3 * np.pi * rho)) ** (1 / 3)
    print(f"  h_typical = {h_typ:.4g}")
    h = rng.lognormal(mean=np.log(h_typ), sigma=0.3, size=n_gas).astype(np.float64)
    h_max = float(h.max())
    print(f"  h range:  {h.min():.4g} .. {h_max:.4g}  (max/typ = {h_max/h_typ:.2f})")

    print("\n[build] cKDTree(xgas) ...")
    t0 = time.perf_counter()
    tree = cKDTree(xgas)
    print(f"  build      {fmt(time.perf_counter() - t0)}")

    # Stars: random positions inside box
    xstar = rng.uniform(-box / 2, box / 2, size=(n_stars, 3)).astype(np.float64)
    cam_pos = np.array([0.0, 0.0, 1.5 * box], dtype=np.float64)

    # ----- Single point query benchmarks -----
    print("\n[single ball queries]")
    for r_factor in [1.0, 2.0, 4.0, 8.0]:
        r = r_factor * h_typ

        def fn():
            return tree.query_ball_point(xstar[0], r=r)

        out = bench(f"query_ball_point(r={r_factor}*h_typ={r:.3g})", fn, repeats=3)
        print(f"     -> {len(out):,} candidates")

    # ----- Multi-point segment-style query (one star, subdivided segment) -----
    print("\n[segment subdivision query, 1 star]")
    for n_chunks in [16, 64, 256, 1024, 4096]:
        ray = cam_pos - xstar[0]
        d_obs = float(np.linalg.norm(ray))
        ray_dir = ray / d_obs
        step = d_obs / n_chunks
        ts = (np.arange(n_chunks) + 0.5) * step
        centers = xstar[0][None, :] + ts[:, None] * ray_dir[None, :]
        radius = 0.5 * step + h_max

        def fn():
            return tree.query_ball_point(centers, r=radius)

        out = bench(f"n_chunks={n_chunks:5d}, r={radius:.3g}", fn, repeats=2)
        n_total = sum(len(x) for x in out)
        print(f"     -> {n_total:,} total candidates ({n_total/n_chunks:.0f}/chunk)")

    # ----- 40-star realistic h-binned scheme -----
    print("\n[full h-binned scheme, all 40 stars]")

    # Build octave bins like the real code
    pos_h = h[h > 0]
    h_min = float(pos_h.min())
    n_oct = max(1, int(np.ceil(np.log2(h_max / h_min))) + 1)
    edges = h_min * (2.0 ** np.arange(n_oct + 1))
    edges[-1] = max(edges[-1], h_max * 1.0001)
    print(f"  building {n_oct} bins, edges ~ {edges[0]:.3g} .. {edges[-1]:.3g}")
    bins = []
    t0 = time.perf_counter()
    for i in range(n_oct):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((h >= lo) & (h < hi))[0]
        if idx.size == 0:
            continue
        sub_tree = cKDTree(xgas[idx])
        bins.append((idx, sub_tree, float(h[idx].max())))
        print(f"    bin h<{hi:.3g}: {idx.size:>12,} pts, build cum {fmt(time.perf_counter()-t0)}")

    print(f"  total bin build: {fmt(time.perf_counter() - t0)}")

    def run_one_frame():
        total_cands = 0
        for s in range(n_stars):
            ray = cam_pos - xstar[s]
            d_obs = float(np.linalg.norm(ray))
            ray_dir = ray / d_obs
            for bin_idx, sub_tree, h_bin in bins:
                seg_len = max(2.0 * h_bin, 1e-30)
                n_chunks = max(1, int(np.ceil(d_obs / seg_len)))
                if n_chunks > 4096:
                    n_chunks = 4096
                step = d_obs / n_chunks
                ts = (np.arange(n_chunks) + 0.5) * step
                centers = xstar[s][None, :] + ts[:, None] * ray_dir[None, :]
                radius = 0.5 * step + h_bin
                lists = sub_tree.query_ball_point(centers, r=radius)
                for l in lists:
                    total_cands += len(l)
        return total_cands

    t0 = time.perf_counter()
    total = run_one_frame()
    dt = time.perf_counter() - t0
    print(f"  one full frame (workers=1): {fmt(dt)}  ({total:,} candidate points)")

    # Same loop, but with workers=-1
    def run_one_frame_mt():
        total_cands = 0
        for s in range(n_stars):
            ray = cam_pos - xstar[s]
            d_obs = float(np.linalg.norm(ray))
            ray_dir = ray / d_obs
            for bin_idx, sub_tree, h_bin in bins:
                seg_len = max(2.0 * h_bin, 1e-30)
                n_chunks = max(1, int(np.ceil(d_obs / seg_len)))
                if n_chunks > 4096:
                    n_chunks = 4096
                step = d_obs / n_chunks
                ts = (np.arange(n_chunks) + 0.5) * step
                centers = xstar[s][None, :] + ts[:, None] * ray_dir[None, :]
                radius = 0.5 * step + h_bin
                lists = sub_tree.query_ball_point(centers, r=radius, workers=-1)
                for l in lists:
                    total_cands += len(l)
        return total_cands

    t0 = time.perf_counter()
    total = run_one_frame_mt()
    dt = time.perf_counter() - t0
    print(f"  one full frame (workers=-1): {fmt(dt)}  ({total:,} candidate points)")

    # Variant: batch ALL stars × all centers per bin into one query call.
    def run_one_frame_batched_mt():
        total_cands = 0
        for bin_idx, sub_tree, h_bin in bins:
            seg_len = max(2.0 * h_bin, 1e-30)
            all_centers = []
            all_radii = []
            for s in range(n_stars):
                ray = cam_pos - xstar[s]
                d_obs = float(np.linalg.norm(ray))
                ray_dir = ray / d_obs
                n_chunks = max(1, int(np.ceil(d_obs / seg_len)))
                if n_chunks > 4096:
                    n_chunks = 4096
                step = d_obs / n_chunks
                ts = (np.arange(n_chunks) + 0.5) * step
                centers = xstar[s][None, :] + ts[:, None] * ray_dir[None, :]
                all_centers.append(centers)
                all_radii.append(0.5 * step + h_bin)
            centers_cat = np.concatenate(all_centers, axis=0)
            # query_ball_point allows scalar r; use the largest (close enough across stars).
            r = max(all_radii)
            lists = sub_tree.query_ball_point(centers_cat, r=r, workers=-1)
            for l in lists:
                total_cands += len(l)
        return total_cands

    t0 = time.perf_counter()
    total = run_one_frame_batched_mt()
    dt = time.perf_counter() - t0
    print(f"  one full frame (batched all stars/bin, workers=-1): {fmt(dt)}  ({total:,} candidates)")


if __name__ == "__main__":
    main()
