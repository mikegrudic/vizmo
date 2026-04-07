"""Orbit-around-center PID-loop benchmark.

Loads a snapshot, places the camera on a circular orbit around the
center of the simulation, looking inward, and runs the live app's
auto-LOD cap controller frame-by-frame for a fixed wall-clock duration.

Reports per-frame ms and the per-frame instance cap so PID/bang-bang
oscillation is visible directly. Use to reproduce LOD oscillation
under camera motion and to A/B controller tunings.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import wgpu

from vizmo.camera import Camera
from vizmo.colormaps import colormap_to_texture_data
from vizmo.gpu_compute import GPUCompute
from vizmo.wgpu_renderer import WGPURenderer

DEFAULT_SNAP = Path.home() / "code/bubblebuddies_gizmo/popeye/bubblebuddies_gizmo/SN_512/snapshot_060.hdf5"


def load_snapshot(path):
    path = Path(path)
    with h5py.File(path, "r") as f:
        pos = np.array(f["PartType0/Coordinates"][:], dtype=np.float32)
        mass = np.array(f["PartType0/Masses"][:], dtype=np.float32)
    print(f"  loaded {len(pos):,} gas particles from {path.name}")
    cache = path.with_suffix(path.suffix + ".hsml.npy")
    if cache.exists():
        hsml = np.load(cache)
        if len(hsml) == len(pos):
            print(f"  cached hsml from {cache.name}")
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


# --- controller variants ---------------------------------------------------


class BangBang:
    """The current live-app controller (wgpu_app.py:712-741)."""

    name = "bangbang(0.9/1.2,*1.10/*0.85)"

    def __init__(self, target_ms):
        self.target_ms = target_ms
        self.smooth_ms = 0.0

    def step(self, last_render_ms, cap, ceiling, dt_s=None):
        if last_render_ms > 0:
            if self.smooth_ms > 0:
                self.smooth_ms = 0.85 * self.smooth_ms + 0.15 * last_render_ms
            else:
                self.smooth_ms = last_render_ms
        if self.smooth_ms <= 0:
            return cap
        if self.smooth_ms < self.target_ms * 0.9:
            return min(ceiling, int(cap * 1.10) + 1)
        if self.smooth_ms > self.target_ms * 1.2:
            return max(1000, int(cap * 0.85))
        return cap


class WideDeadband:
    """Bang-bang with a much wider deadband and smaller multipliers.

    Multigrid's cost-vs-cap curve is nearly flat, so the controller
    rarely needs to act; the wide deadband keeps the cap pinned at the
    current value most of the time.
    """

    def __init__(self, target_ms, lo=0.6, hi=1.5, up=1.05, dn=0.92, alpha=0.20):
        self.target_ms = target_ms
        self.smooth_ms = 0.0
        self.lo, self.hi, self.up, self.dn, self.alpha = lo, hi, up, dn, alpha
        self.name = f"wide({lo:.2f}/{hi:.2f},*{up:.2f}/*{dn:.2f},a={alpha:.2f})"

    def step(self, last_render_ms, cap, ceiling, dt_s=None):
        if last_render_ms > 0:
            self.smooth_ms = (
                (1 - self.alpha) * self.smooth_ms + self.alpha * last_render_ms
                if self.smooth_ms > 0
                else last_render_ms
            )
        if self.smooth_ms <= 0:
            return cap
        if self.smooth_ms < self.target_ms * self.lo:
            return min(ceiling, int(cap * self.up) + 1)
        if self.smooth_ms > self.target_ms * self.hi:
            return max(1000, int(cap * self.dn))
        return cap


class TimeBased:
    """Framerate-independent proportional controller.

    Time-domain parameters:
      tau_smooth_s   EMA time constant for render_ms (seconds)
      tau_resp_s     controller response time constant (seconds)
      max_rate_log2  max log2 cap-change rate per *second* (clip)

    Per-frame update uses real wall-clock dt:
      alpha   = 1 - exp(-dt / tau_smooth_s)
      smooth += alpha * (render_ms - smooth)
      err     = (smooth - target_ms) / target_ms
      step    = -err * dt / tau_resp_s          # log2 units
      step    = clamp(step, ±max_rate_log2 * dt)
      cap    *= 2**step

    This makes the bandwidth, settling time, and rate limit identical
    at 15, 30, or 60 fps targets — only the granularity of corrections
    changes with frame rate.
    """

    def __init__(self, target_ms, tau_smooth_s=0.15, tau_resp_s=0.4, max_rate_log2=2.0):
        self.target_ms = target_ms
        self.tau_smooth_s = tau_smooth_s
        self.tau_resp_s = tau_resp_s
        self.max_rate_log2 = max_rate_log2
        self.smooth_ms = 0.0
        self.name = f"time(tau_s={tau_smooth_s:.2f},tau_r={tau_resp_s:.2f}," f"rate={max_rate_log2:.1f}/s)"

    def step(self, last_render_ms, cap, ceiling, dt_s):
        if last_render_ms <= 0 or dt_s <= 0:
            return cap
        alpha = 1.0 - np.exp(-dt_s / self.tau_smooth_s)
        if self.smooth_ms <= 0:
            self.smooth_ms = last_render_ms
        else:
            self.smooth_ms += alpha * (last_render_ms - self.smooth_ms)
        err = (self.smooth_ms - self.target_ms) / self.target_ms
        log_step = -err * dt_s / self.tau_resp_s
        max_step = self.max_rate_log2 * dt_s
        if log_step > max_step:
            log_step = max_step
        elif log_step < -max_step:
            log_step = -max_step
        new_cap = int(cap * (2.0**log_step))
        return max(1000, min(ceiling, new_cap))


class Proportional:
    """True proportional controller with rate-limited output.

    err = (smooth_ms - target_ms) / target_ms
    log_step = -K * err   (negative err → grow, positive → shrink)
    log_step is clamped to ±max_step so a single bad frame can't move
    the cap by more than ~10% (default).
    """

    def __init__(self, target_ms, K=0.25, max_step_log=0.05, alpha=0.25):
        self.target_ms = target_ms
        self.smooth_ms = 0.0
        self.K = K
        self.max_step_log = max_step_log
        self.alpha = alpha
        self.name = f"P(K={K:.2f},clip={max_step_log:.2f},a={alpha:.2f})"

    def step(self, last_render_ms, cap, ceiling, dt_s=None):
        if last_render_ms > 0:
            self.smooth_ms = (
                (1 - self.alpha) * self.smooth_ms + self.alpha * last_render_ms
                if self.smooth_ms > 0
                else last_render_ms
            )
        if self.smooth_ms <= 0:
            return cap
        err = (self.smooth_ms - self.target_ms) / self.target_ms
        log_step = -self.K * err
        if log_step > self.max_step_log:
            log_step = self.max_step_log
        elif log_step < -self.max_step_log:
            log_step = -self.max_step_log
        new_cap = int(cap * (2.0**log_step))
        return max(1000, min(ceiling, new_cap))


class BangBangGentle:
    """Same shape, smaller multipliers, wider deadband, slower EMA."""

    def __init__(self, target_ms, lo=0.80, hi=1.30, up=1.04, dn=0.94, alpha=0.07):
        self.target_ms = target_ms
        self.smooth_ms = 0.0
        self.lo, self.hi, self.up, self.dn, self.alpha = lo, hi, up, dn, alpha
        self.name = f"bangbang({lo:.2f}/{hi:.2f},*{up:.2f}/*{dn:.2f},a={alpha:.2f})"

    def step(self, last_render_ms, cap, ceiling, dt_s=None):
        if last_render_ms > 0:
            if self.smooth_ms > 0:
                self.smooth_ms = (1 - self.alpha) * self.smooth_ms + self.alpha * last_render_ms
            else:
                self.smooth_ms = last_render_ms
        if self.smooth_ms <= 0:
            return cap
        if self.smooth_ms < self.target_ms * self.lo:
            return min(ceiling, int(cap * self.up) + 1)
        if self.smooth_ms > self.target_ms * self.hi:
            return max(1000, int(cap * self.dn))
        return cap


# --- orbit driver ----------------------------------------------------------


def run_orbit(
    label,
    device,
    r,
    cam,
    target_center,
    ctrl,
    *,
    res,
    duration_s,
    target_fps,
    ceiling,
    orbit_radius,
    orbit_period_s,
    edge_radius,
    n_passes,
):
    """Camera path: angular orbit + smooth radial in/out passes.

    Radius follows a raised-cosine schedule between `orbit_radius`
    (inner) and `edge_radius` (outer), making `n_passes` complete
    in-out cycles over `duration_s`. Angular position keeps sweeping
    at `omega` so the camera traces a smooth pinwheel-style path
    through both light- and heavy-load regimes.
    """
    target_center = np.asarray(target_center, dtype=np.float32)
    omega = 2 * np.pi / orbit_period_s
    target_ms = 1000.0 / target_fps
    radial_omega = 2 * np.pi * n_passes / max(duration_s, 1e-6)
    r_mid = 0.5 * (orbit_radius + edge_radius)
    r_amp = 0.5 * (edge_radius - orbit_radius)

    # Warmup so the first measured frame doesn't include shader compile.
    for _ in range(3):
        r._render_accum(cam, res, res, r._accum_textures)
    device.queue.on_submitted_work_done_sync()

    cap = r._subsample_max_per_frame
    records = []  # (t_wall, render_ms, cap)
    t0 = time.perf_counter()
    last_frame_wall = t0
    while True:
        now = time.perf_counter()
        t = now - t0
        if t > duration_s:
            break
        dt_s = now - last_frame_wall
        last_frame_wall = now
        # Position the camera on the orbit (looking inward) with a
        # raised-cosine radial sweep between orbit_radius and edge_radius.
        ang = omega * t
        radius = r_mid - r_amp * np.cos(radial_omega * t)
        offset = np.array([radius * np.cos(ang), 0.0, radius * np.sin(ang)], dtype=np.float32)
        cam.position = (target_center + offset).astype(np.float32)
        cam._forward = -offset / max(np.linalg.norm(offset), 1e-30)
        cam._up = np.array([0, 1, 0], dtype=np.float32)
        cam._dirty = True

        # Render and time it.
        device.queue.on_submitted_work_done_sync()
        rt0 = time.perf_counter()
        r._render_accum(cam, res, res, r._accum_textures)
        device.queue.on_submitted_work_done_sync()
        render_ms = (time.perf_counter() - rt0) * 1000.0

        records.append((t, render_ms, cap))

        # Drive the controller and apply the new cap.
        cap = ctrl.step(render_ms, cap, ceiling, dt_s)
        r.set_subsample_max_per_frame(cap)
    return records


def summarize(label, recs):
    ts = np.array([r[0] for r in recs])
    ms = np.array([r[1] for r in recs])
    cap = np.array([r[2] for r in recs])
    # Cap oscillation: log10 std of consecutive ratios.
    ratios = cap[1:] / np.maximum(cap[:-1], 1)
    log_swings = np.log2(np.maximum(ratios, 1e-9))
    print(f"\n=== {label} ===")
    print(f"  frames {len(recs)}   wall {ts[-1]:.2f}s")
    print(
        f"  ms:   mean {ms.mean():6.2f}  median {np.median(ms):6.2f}  "
        f"std {ms.std():6.2f}  p5 {np.percentile(ms,5):6.2f}  p95 {np.percentile(ms,95):6.2f}"
    )
    print(
        f"  cap:  mean {cap.mean()/1e6:6.2f}M  std {cap.std()/1e6:6.2f}M  "
        f"min {cap.min()/1e6:6.2f}M  max {cap.max()/1e6:6.2f}M"
    )
    print(f"  cap log2-step std: {log_swings.std():.3f}  (lower = smoother)")
    return dict(ms=ms, cap=cap, ts=ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snap", default=str(DEFAULT_SNAP))
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--multigrid", type=int, default=6)
    p.add_argument(
        "--target-fps", type=float, nargs="+", default=[15.0, 30.0, 60.0], help="one or more target FPS values to test"
    )
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--orbit-period", type=float, default=8.0)
    p.add_argument(
        "--orbit-radius-frac", type=float, default=0.05, help="inner orbit radius as a fraction of the data extent"
    )
    p.add_argument(
        "--edge-radius-frac", type=float, default=0.45, help="outer (edge) radius as a fraction of the data extent"
    )
    p.add_argument("--radial-passes", type=int, default=3, help="number of full in-out cycles over the run")
    p.add_argument("--ceiling", type=int, default=0, help="cap ceiling (0 = full N)")
    args = p.parse_args()

    if not Path(args.snap).exists():
        sys.exit(f"snapshot not found: {args.snap}")
    pos, mass, hsml = load_snapshot(args.snap)

    # Center of the simulation = box midpoint of the data.
    center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
    extent = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    orbit_r = extent * args.orbit_radius_frac
    edge_r = extent * args.edge_radius_frac
    print(f"  center {center}")
    print(f"  extent {extent:.2f}  inner {orbit_r:.2f}  outer {edge_r:.2f}")

    device, r, cam = setup(pos, mass, hsml, args.res, args.multigrid)
    ceiling = args.ceiling if args.ceiling > 0 else int(len(pos))
    print(f"  cap ceiling: {ceiling/1e6:.1f}M")

    all_results = []  # (target_fps, ctrl_name, summary)
    for fps in args.target_fps:
        target_ms = 1000.0 / fps
        controllers = [
            ("current", BangBang(target_ms)),
            ("time(0.10/0.25)", TimeBased(target_ms, tau_smooth_s=0.10, tau_resp_s=0.25, max_rate_log2=3.0)),
            ("time(0.07/0.15)", TimeBased(target_ms, tau_smooth_s=0.07, tau_resp_s=0.15, max_rate_log2=4.0)),
            ("time(0.05/0.10)", TimeBased(target_ms, tau_smooth_s=0.05, tau_resp_s=0.10, max_rate_log2=5.0)),
        ]
        print(f"\n--- target {fps:.0f} fps (target_ms={target_ms:.2f}) ---")
        for label, ctrl in controllers:
            r.set_subsample_max_per_frame(ceiling)
            recs = run_orbit(
                label,
                device,
                r,
                cam,
                center,
                ctrl,
                res=args.res,
                duration_s=args.duration,
                target_fps=fps,
                ceiling=ceiling,
                orbit_radius=orbit_r,
                orbit_period_s=args.orbit_period,
                edge_radius=edge_r,
                n_passes=args.radial_passes,
            )
            s = summarize(f"{fps:.0f}fps {ctrl.name}", recs)
            all_results.append((fps, ctrl.name, s))

    # Final compact comparison table per fps target.
    print("\n\n=== summary table (ms std lower = smoother) ===")
    print(f"{'fps':>5} {'controller':<48} {'mean ms':>10} {'ms std':>10} " f"{'p95 ms':>10} {'cap step':>10}")
    for fps, name, s in all_results:
        log_step = np.log2(s["cap"][1:] / np.maximum(s["cap"][:-1], 1)).std()
        print(
            f"{fps:>5.0f} {name:<48} {s['ms'].mean():>10.2f} "
            f"{s['ms'].std():>10.2f} {np.percentile(s['ms'],95):>10.2f} "
            f"{log_step:>10.3f}"
        )

    r.release()


if __name__ == "__main__":
    main()
