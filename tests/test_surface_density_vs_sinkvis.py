"""Compare dataflyer GPU surface density against CrunchSnaps/SinkVis.

Generates random particles in a unit cube, computes smoothing lengths with
Meshoid, writes a minimal HDF5 snapshot, renders surface density with both
pipelines, and checks quantitative agreement.
"""

import tempfile
import os

import numpy as np
import h5py
import pytest
from meshoid import Meshoid, GridSurfaceDensity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_snapshot(path, positions, masses, hsml, boxsize=1.0):
    """Write a minimal GIZMO-style HDF5 snapshot."""
    N = len(masses)
    with h5py.File(path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["Time"] = 0.0
        hdr.attrs["BoxSize"] = boxsize
        hdr.attrs["NumFilesPerSnapshot"] = 1
        hdr.attrs["ComovingIntegrationOn"] = 0
        hdr.attrs["HubbleParam"] = 1.0
        hdr.attrs["UnitLength_In_CGS"] = 1.0
        hdr.attrs["UnitVelocity_In_CGS"] = 1.0
        hdr.attrs["UnitMass_In_CGS"] = 1.0
        npart = np.zeros(6, dtype=np.int32)
        npart[0] = N
        hdr.attrs["NumPart_ThisFile"] = npart
        hdr.attrs["NumPart_Total"] = npart.astype(np.uint64)

        pt0 = f.create_group("PartType0")
        pt0.create_dataset("Coordinates", data=positions.astype(np.float64))
        pt0.create_dataset("Masses", data=masses.astype(np.float64))
        pt0.create_dataset("SmoothingLength", data=hsml.astype(np.float64))
        pt0.create_dataset("Velocities", data=np.zeros_like(positions))
        pt0.create_dataset("InternalEnergy", data=np.ones(N, dtype=np.float64))
        pt0.create_dataset("Density", data=np.ones(N, dtype=np.float64))
        pt0.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint64))


def sinkvis_surface_density(positions, masses, hsml, center, camera_distance,
                            boxsize=1.0, res=128, fov=90):
    """Compute surface density the way SinkVis does with perspective projection.

    Reproduces the coordinate transform from SinkVis.SetupCoordsAndWeights +
    SinkVisSigmaGas.GenerateMaps, without needing an HDF5 file.
    """
    pos = positions - center
    # Camera looks along -z from z = +camera_distance (relative to center)
    pos[:, 2] -= camera_distance
    r = np.abs(pos[:, 2])
    m = masses.copy()
    h = hsml.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pos[:, :2] = pos[:, :2] / (-pos[:, 2][:, None])
        h[:] = h / r
        m[:] = m / r**2
    behind = pos[:, 2] >= 0  # particles at or behind camera
    h[behind] = 0
    m[behind] = 0

    # rmax in angular units for perspective
    rmax = fov / 90.0
    # Clip smoothing length to pixel scale (matching SinkVis line 346)
    h = np.clip(h, 2 * rmax / res, np.inf)

    sigma = GridSurfaceDensity(
        m, pos, h, np.zeros(3), 2 * rmax, res=res, parallel=True,
    ).T
    return sigma


def dataflyer_surface_density(positions, masses, hsml, center, camera_distance,
                              boxsize=1.0, res=128, fov=90):
    """Render surface density using dataflyer's GPU splat renderer.

    Uses a headless moderngl standalone context. Returns the denominator
    accumulation texture (= surface density for mode 0).
    """
    import moderngl
    from dataflyer.renderer import SplatRenderer
    from dataflyer.colormaps import create_colormap_texture_safe
    from dataflyer.camera import Camera

    ctx = moderngl.create_standalone_context()
    renderer = SplatRenderer(ctx)
    renderer.colormap_tex = create_colormap_texture_safe(ctx, "magma")
    renderer.mode = 0  # surface density
    renderer.log_scale = 0
    renderer.use_tree = False  # render all particles directly, no LOD

    # Upload all particles (surface density: qty = mass, but mode=0 uses
    # denominator which accumulates kernel-weighted mass = surface density)
    renderer.set_particles(positions.astype(np.float32),
                           hsml.astype(np.float32),
                           masses.astype(np.float32))

    # Set up camera to match SinkVis: at center + camera_distance along +z,
    # looking along -z, with +y up
    camera = Camera(fov=fov, aspect=1.0)
    camera.position = np.array(
        [center[0], center[1], center[2] + camera_distance], dtype=np.float32
    )
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)

    # Set clip planes
    extent = boxsize
    camera.near = extent * 1e-4
    camera.far = extent * 10

    # Cull and upload visible particles
    renderer._viewport_width = res
    renderer.max_render_particles = len(masses) + 1
    renderer.update_visible(camera)

    # Run only the accumulation pass (Pass 1) — standalone context has no
    # screen framebuffer, so we skip the resolve pass entirely.
    view = np.ascontiguousarray(camera.view_matrix().T)
    proj = np.ascontiguousarray(camera.projection_matrix().T)
    renderer._ensure_accum_fbo(res, res)
    renderer._accum_fbo.use()
    renderer._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

    kernel_id = renderer.KERNELS.index(renderer.kernel)
    renderer.prog_additive["u_view"].write(view.tobytes())
    renderer.prog_additive["u_proj"].write(proj.tobytes())
    renderer.prog_additive["u_viewport_size"].value = (float(res), float(res))
    renderer.prog_additive["u_kernel"].value = kernel_id

    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.ONE, moderngl.ONE)
    ctx.disable(moderngl.DEPTH_TEST)

    if renderer.vao_additive is not None and renderer.n_particles > 0:
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        renderer.vao_additive.render(moderngl.POINTS)

    if renderer.vao_quad is not None and renderer.n_big > 0:
        renderer.prog_quad["u_view"].write(view.tobytes())
        renderer.prog_quad["u_proj"].write(proj.tobytes())
        renderer.prog_quad["u_kernel"].value = kernel_id
        renderer.vao_quad.render(moderngl.TRIANGLES, instances=renderer.n_big)

    # Read back the denominator (surface density)
    den_data = np.frombuffer(
        renderer._accum_tex_den.read(), dtype=np.float32
    ).reshape(res, res)

    renderer.release()
    ctx.release()
    return den_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def particle_data():
    """Generate random particles in a unit cube with Meshoid smoothing lengths."""
    rng = np.random.default_rng(42)
    N = 10_000
    boxsize = 1.0
    positions = rng.uniform(0, boxsize, (N, 3)).astype(np.float64)
    masses = np.ones(N, dtype=np.float64) / N  # uniform mass, total = 1
    hsml = Meshoid(positions, boxsize=boxsize).SmoothingLength()
    return positions, masses, hsml, boxsize


def test_surface_density_perspective(particle_data):
    """Surface density from dataflyer GPU renderer should match SinkVis
    perspective projection (camera_distance=1) to within a few percent."""
    positions, masses, hsml, boxsize = particle_data
    center = np.array([0.5, 0.5, 0.5])
    camera_distance = 1.0
    res = 128
    fov = 90

    sigma_sinkvis = sinkvis_surface_density(
        positions.copy(), masses.copy(), hsml.copy(),
        center, camera_distance, boxsize=boxsize, res=res, fov=fov,
    )

    sigma_dataflyer = dataflyer_surface_density(
        positions.copy(), masses.copy(), hsml.copy(),
        center, camera_distance, boxsize=boxsize, res=res, fov=fov,
    )

    # Both maps should have the same total mass (integral of surface density)
    pixel_area_sinkvis = (2 * fov / 90.0 / res) ** 2  # angular pixel area
    total_sinkvis = sigma_sinkvis.sum() * pixel_area_sinkvis
    total_dataflyer = sigma_dataflyer.sum() * pixel_area_sinkvis

    # Total mass should be close (within 10% — kernel differences are expected)
    assert total_dataflyer > 0, "dataflyer produced an empty map"
    assert total_sinkvis > 0, "SinkVis produced an empty map"
    mass_ratio = total_dataflyer / total_sinkvis
    assert 0.9 < mass_ratio < 1.1, (
        f"Total mass mismatch: dataflyer={total_dataflyer:.4g}, "
        f"sinkvis={total_sinkvis:.4g}, ratio={mass_ratio:.3f}"
    )

    # Per-pixel correlation: the maps should be highly correlated
    mask = (sigma_sinkvis > 0) & (sigma_dataflyer > 0)
    assert mask.sum() > res * res * 0.5, "Too few overlapping pixels with signal"

    log_sv = np.log10(sigma_sinkvis[mask])
    log_df = np.log10(sigma_dataflyer[mask])
    correlation = np.corrcoef(log_sv, log_df)[0, 1]
    assert correlation > 0.99, (
        f"Log surface density correlation too low: {correlation:.3f}"
    )

    # Median absolute log ratio (per-pixel accuracy)
    log_ratio = np.abs(log_sv - log_df)
    median_log_ratio = np.median(log_ratio)
    assert median_log_ratio < 0.1, (
        f"Median |log10(sinkvis/dataflyer)| = {median_log_ratio:.3f}, "
        "expected < 0.1 dex"
    )

    # Save side-by-side comparison image
    _save_comparison(sigma_sinkvis, sigma_dataflyer, correlation,
                     median_log_ratio, mass_ratio)


def _save_comparison(sigma_sv, sigma_df, correlation, median_log_ratio, mass_ratio):
    """Generate a 3-panel comparison: SinkVis | dataflyer | log ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Common log-scale limits from the union of both maps
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
    axes[1].set_title("dataflyer (GPU splats)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Log ratio panel
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


def test_snapshot_roundtrip(particle_data):
    """Verify the test snapshot is well-formed and can be loaded."""
    positions, masses, hsml, boxsize = particle_data
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = os.path.join(tmpdir, "snapshot_000.hdf5")
        make_test_snapshot(snap_path, positions, masses, hsml, boxsize=boxsize)

        with h5py.File(snap_path, "r") as f:
            assert "PartType0" in f
            assert f["PartType0/Coordinates"].shape == (len(masses), 3)
            assert f["Header"].attrs["BoxSize"] == boxsize
            coords = f["PartType0/Coordinates"][:]
            m = f["PartType0/Masses"][:]
            h = f["PartType0/SmoothingLength"][:]

        assert np.allclose(coords, positions)
        assert np.allclose(m, masses)
        assert np.allclose(h, hsml)

        # Verify GridSurfaceDensity can render from the loaded data
        center = np.array([0.5, 0.5, 0.5])
        sigma = GridSurfaceDensity(
            m.astype(np.float64), (coords - center).astype(np.float64),
            h.astype(np.float64), np.zeros(3), boxsize, res=64, parallel=True,
        )
        assert sigma.shape == (64, 64)
        assert sigma.sum() > 0
