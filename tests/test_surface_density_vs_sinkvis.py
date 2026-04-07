"""Compare dataflyer wgpu surface density against CrunchSnaps/SinkVis.

Generates random particles in a unit cube, computes smoothing lengths with
Meshoid, renders surface density with both pipelines, and checks
quantitative agreement (mass ratio, log-pixel correlation, median |Δlog|).
"""

import os

import numpy as np
import pytest
from meshoid import Meshoid, GridSurfaceDensity


# ---------------------------------------------------------------------------
# Reference: SinkVis-style perspective surface density
# ---------------------------------------------------------------------------

def sinkvis_surface_density(positions, masses, hsml, center, camera_distance,
                            res=128, fov=90):
    """Reproduces SinkVis.SetupCoordsAndWeights + SinkVisSigmaGas.GenerateMaps
    without needing an HDF5 file. Camera looks down -z from
    center + (0, 0, camera_distance), with a perspective projection onto a
    unit-distance plane spanning [-rmax, rmax] in tan-angle, where
    rmax = fov/90.
    """
    pos = positions - center
    pos[:, 2] -= camera_distance
    r = np.abs(pos[:, 2])
    m = masses.copy()
    h = hsml.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pos[:, :2] = pos[:, :2] / (-pos[:, 2][:, None])
        h[:] = h / r
        m[:] = m / r**2
    behind = pos[:, 2] >= 0
    h[behind] = 0
    m[behind] = 0

    rmax = fov / 90.0
    h = np.clip(h, 2 * rmax / res, np.inf)

    sigma = GridSurfaceDensity(
        m, pos, h, np.zeros(3), 2 * rmax, res=res, parallel=True,
    ).T
    return sigma


# ---------------------------------------------------------------------------
# Subject under test: dataflyer wgpu accumulation
# ---------------------------------------------------------------------------

def dataflyer_surface_density(positions, masses, hsml, center, camera_distance,
                              boxsize=1.0, res=128, fov=90):
    """Render the surface density (denominator accumulation texture)
    using the wgpu splat path, with the camera matched to SinkVis.
    """
    import wgpu
    from dataflyer.wgpu_renderer import WGPURenderer
    from dataflyer.gpu_compute import GPUCompute
    from dataflyer.colormaps import colormap_to_texture_data
    from dataflyer.camera import Camera

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    req_features = set()
    if "float32-blendable" in adapter.features:
        req_features.add("float32-blendable")
    device = adapter.request_device_sync(required_features=req_features)

    renderer = WGPURenderer(device, canvas_context=None, present_format="bgra8unorm")
    renderer.set_colormap(colormap_to_texture_data("magma"))
    renderer.kernel = "cubic_spline"
    renderer.resolve_mode = 0
    renderer.log_scale = 0

    pos32 = positions.astype(np.float32)
    hsml32 = hsml.astype(np.float32)
    mass32 = masses.astype(np.float32)
    renderer.set_particles(pos32, hsml32, mass32)

    gpu_compute = GPUCompute(device)
    gpu_compute.upload_subsample_only(pos32, hsml32, mass32, mass32)
    renderer.set_subsample_chunks(gpu_compute.get_chunk_bufs(),
                                  world_offset=gpu_compute.get_pos_offset())
    # Render every particle once: cap = N, so eff_stride = 1 and h_scale = 1.
    renderer.set_subsample_max_per_frame(len(pos32) + 1)

    camera = Camera(fov=fov, aspect=1.0)
    camera.position = np.array(
        [center[0], center[1], center[2] + camera_distance], dtype=np.float32
    )
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera._dirty = True
    extent = boxsize
    camera.near = extent * 1e-4
    camera.far = extent * 10

    renderer._ensure_fbo(res, res, which=1)
    renderer._render_accum(camera, res, res, renderer._accum_textures)

    # Denominator texture = Σ mass·W(r)/h² = surface density.
    # wgpu textures use a top-left origin (row 0 is the top of the
    # framebuffer); SinkVis's GridSurfaceDensity returns a map with
    # row 0 at the bottom. Flip vertically to match.
    den_flat = renderer._read_accum_texture_r(
        renderer._accum_textures["textures"][1], size=(res, res))
    sigma = np.flipud(den_flat.reshape(res, res))

    renderer.release()
    return sigma


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def particle_data():
    """Random particles in a unit cube with Meshoid smoothing lengths."""
    rng = np.random.default_rng(42)
    N = 10_000
    boxsize = 1.0
    positions = rng.uniform(0, boxsize, (N, 3)).astype(np.float64)
    masses = np.ones(N, dtype=np.float64) / N  # uniform mass, total = 1
    hsml = Meshoid(positions, boxsize=boxsize).SmoothingLength()
    return positions, masses, hsml, boxsize


def test_surface_density_perspective(particle_data):
    """The dataflyer wgpu surface density should agree with SinkVis's
    perspective projection at fov=90, camera_distance=1, to within a
    few percent on integrated mass and ~0.1 dex per pixel."""
    positions, masses, hsml, boxsize = particle_data
    center = np.array([0.5, 0.5, 0.5])
    camera_distance = 1.0
    res = 128
    fov = 90

    sigma_sinkvis = sinkvis_surface_density(
        positions.copy(), masses.copy(), hsml.copy(),
        center, camera_distance, res=res, fov=fov,
    )
    sigma_dataflyer = dataflyer_surface_density(
        positions.copy(), masses.copy(), hsml.copy(),
        center, camera_distance, boxsize=boxsize, res=res, fov=fov,
    )

    # Same units (mass per unit world area on the unit-distance plane), so
    # both maps should integrate to the same total mass.
    pixel_area = (2 * fov / 90.0 / res) ** 2
    total_sinkvis = sigma_sinkvis.sum() * pixel_area
    total_dataflyer = sigma_dataflyer.sum() * pixel_area

    assert total_dataflyer > 0, "dataflyer produced an empty map"
    assert total_sinkvis > 0, "SinkVis produced an empty map"
    mass_ratio = total_dataflyer / total_sinkvis
    assert 0.9 < mass_ratio < 1.1, (
        f"Total mass mismatch: dataflyer={total_dataflyer:.4g}, "
        f"sinkvis={total_sinkvis:.4g}, ratio={mass_ratio:.3f}"
    )

    mask = (sigma_sinkvis > 0) & (sigma_dataflyer > 0)
    assert mask.sum() > res * res * 0.5, "Too few overlapping pixels with signal"

    log_sv = np.log10(sigma_sinkvis[mask])
    log_df = np.log10(sigma_dataflyer[mask])
    correlation = np.corrcoef(log_sv, log_df)[0, 1]
    log_ratio = np.abs(log_sv - log_df)
    median_log_ratio = np.median(log_ratio)

    # Always save the comparison image — it's the most useful diagnostic
    # when the assertions below trip.
    _save_comparison(sigma_sinkvis, sigma_dataflyer, correlation,
                     median_log_ratio, mass_ratio)

    assert correlation > 0.99, (
        f"Log surface density correlation too low: {correlation:.3f}"
    )
    assert median_log_ratio < 0.1, (
        f"Median |log10(sinkvis/dataflyer)| = {median_log_ratio:.3f}, "
        "expected < 0.1 dex"
    )


def _save_comparison(sigma_sv, sigma_df, correlation, median_log_ratio, mass_ratio):
    """3-panel diagnostic image: SinkVis | dataflyer | log ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    pos_sv = sigma_sv[sigma_sv > 0]
    pos_df = sigma_df[sigma_df > 0]
    if len(pos_sv) == 0 or len(pos_df) == 0:
        return
    vmin = min(pos_sv.min(), pos_df.min())
    vmax = max(pos_sv.max(), pos_df.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(sigma_sv, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap="magma", origin="lower")
    axes[0].set_title("SinkVis (GridSurfaceDensity)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sigma_df, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap="magma", origin="lower")
    axes[1].set_title("dataflyer (wgpu splats)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (sigma_sv > 0) & (sigma_df > 0),
            np.log10(sigma_df / sigma_sv),
            0.0,
        )
    vlim = max(abs(ratio.min()), abs(ratio.max()), 0.05)
    im2 = axes[2].imshow(ratio, vmin=-vlim, vmax=vlim,
                         cmap="coolwarm", origin="lower")
    axes[2].set_title(r"$\log_{10}$(dataflyer / SinkVis)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"corr={correlation:.4f}   "
        r"median $|\Delta\log_{10}|$" + f"={median_log_ratio:.4f} dex"
        f"   mass ratio={mass_ratio:.4f}",
        fontsize=11,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "surface_density_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nComparison image saved to {out}")
