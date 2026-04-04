"""Shared benchmark utilities for both app backends."""

import time
import numpy as np


def rodrigues(v, axis, angle):
    """Rotate vector v around unit axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def slerp_vec(a, b, t):
    """Interpolate unit vectors via slerp-like blend."""
    blend = a * (1 - t) + b * t
    n = np.linalg.norm(blend)
    return blend / n if n > 1e-8 else a


def build_benchmark_keyframes(camera, center, extent, n_frames):
    """Build benchmark keyframes: look around, fly across, orbit.

    Args:
        camera: Camera instance (uses current position/forward/up).
        center: (3,) data center.
        extent: float, data extent (e.g. boxsize).
        n_frames: total number of keyframes to generate.

    Returns:
        list of (position, forward, up, label) tuples.
    """
    start_pos = camera.position.copy()
    fwd0 = camera.forward.copy()
    up0 = camera.up.copy()
    right0 = camera.right.copy()

    keyframes = []
    def kf(pos, fwd, up, label):
        keyframes.append((pos.astype(np.float32), fwd.astype(np.float32),
                          up.astype(np.float32), label))

    kf(start_pos, fwd0, up0, "start")
    kf(start_pos, rodrigues(fwd0, right0, np.radians(30)), up0, "look up")
    kf(start_pos, rodrigues(fwd0, right0, np.radians(-30)), up0, "look down")
    kf(start_pos, rodrigues(fwd0, up0, np.radians(45)), up0, "look left")
    kf(start_pos, rodrigues(fwd0, up0, np.radians(-45)), up0, "look right")
    kf(start_pos, fwd0, rodrigues(up0, fwd0, np.radians(30)), "roll left")
    kf(start_pos, fwd0, rodrigues(up0, fwd0, np.radians(-30)), "roll right")
    kf(start_pos, fwd0, up0, "reset")

    # Fly to opposite side
    far_pos = start_pos + fwd0 * extent * 1.5
    n_fly = max(n_frames // 4, 10)
    for i in range(n_fly):
        t = (i + 1) / n_fly
        pos = start_pos * (1 - t) + far_pos * t
        kf(pos, fwd0, up0, f"fly out {int(t*100)}%")

    # Turn around
    fwd_back = -fwd0
    kf(far_pos, fwd_back, up0, "turn around")

    # Fly back
    for i in range(n_fly):
        t = (i + 1) / n_fly
        pos = far_pos * (1 - t) + start_pos * t
        kf(pos, fwd_back, up0, f"fly back {int(t*100)}%")

    # Orbit around data center for remaining frames
    n_orbit = max(n_frames - len(keyframes), 0)
    if n_orbit > 0:
        radius = extent * 0.6
        angles = np.linspace(0, 2 * np.pi, n_orbit, endpoint=False)
        for theta in angles:
            pos = np.array([center[0] + radius * np.sin(theta), center[1],
                            center[2] + radius * np.cos(theta)], dtype=np.float32)
            fwd = (center - pos).astype(np.float32)
            fwd = fwd / np.linalg.norm(fwd)
            kf(pos, fwd, up0, "orbit")

    return keyframes[:n_frames]


def print_benchmark_results(frame_times, cull_times, phase_list, n_total, backend=""):
    """Print benchmark results summary.

    Args:
        frame_times: array of per-frame times in ms.
        cull_times: array of per-frame cull times in ms.
        phase_list: array of phase strings ("transition" or "hold").
        n_total: total particle count.
        backend: optional backend name for header.
    """
    frame_times = np.asarray(frame_times)
    cull_times = np.asarray(cull_times)
    phases = np.asarray(phase_list)

    prefix = f"{backend} " if backend else ""
    print(f"\n--- {prefix}Benchmark Results ({len(frame_times)} frames) ---")
    for phase_name in ("transition", "hold"):
        mask = phases == phase_name
        if not mask.any():
            continue
        ft = frame_times[mask]
        ct = cull_times[mask]
        fps = 1000.0 / ft
        print(f"  {phase_name.capitalize()} ({mask.sum()} frames):")
        print(f"    Frame time:  median={np.median(ft):.1f}ms  "
              f"p5={np.percentile(ft, 5):.1f}ms  "
              f"p95={np.percentile(ft, 95):.1f}ms")
        print(f"    FPS:         median={np.median(fps):.0f}  "
              f"p5={np.percentile(fps, 5):.0f}  "
              f"p95={np.percentile(fps, 95):.0f}")
        print(f"    Cull time:   median={np.median(ct):.1f}ms  "
              f"p95={np.percentile(ct, 95):.1f}ms")
    print(f"  Particles:   {n_total:,} total")
