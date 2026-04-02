"""Test that composite mode produces consistent results across mode switches."""

import numpy as np
import pytest
import moderngl
import types


def make_renderer_and_camera(pos, masses, hsml):
    from dataflyer.renderer import SplatRenderer
    from dataflyer.colormaps import create_colormap_texture_safe
    from dataflyer.camera import Camera

    ctx = moderngl.create_standalone_context()
    renderer = SplatRenderer(ctx)
    renderer.colormap_tex = create_colormap_texture_safe(ctx, "magma")
    renderer.log_scale = 1
    renderer.use_tree = True

    camera = Camera(fov=90, aspect=1.0)
    camera.position = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._dirty = True
    camera.near = 0.01
    camera.far = 10.0
    renderer._viewport_width = 64

    renderer.set_particles(pos, hsml, masses)
    renderer.update_visible(camera)

    # Patch for headless rendering
    original = renderer.render.__func__
    orig_screen = type(ctx).screen

    def _hr(self, cam, w, h):
        try:
            type(ctx).screen = property(lambda s: (_ for _ in ()).throw(StopIteration))
            original(self, cam, w, h)
        except (StopIteration, AttributeError):
            pass
        finally:
            type(ctx).screen = orig_screen

    renderer.render = types.MethodType(_hr, renderer)

    return ctx, renderer, camera


def render_composite(renderer, camera, masses, qty, res=64):
    """Render a composite frame and return both FBO contents."""
    renderer._ensure_fbo(res, res, which=1)
    renderer._ensure_fbo(res, res, which=2)

    # Slot 0: SurfaceDensity(Masses)
    renderer.resolve_mode = 0
    renderer.update_weights(masses)
    renderer.update_visible(camera)
    renderer._render_accum(camera, res, res, renderer._accum_fbo)

    # Slot 1: WeightedAverage(qty)
    renderer.resolve_mode = 1
    renderer.update_weights(masses, qty)
    renderer.update_visible(camera)
    renderer._render_accum(camera, res, res, renderer._accum_fbo2)

    den1 = np.frombuffer(renderer._accum_tex_den.read(), dtype=np.float32).copy()
    den2 = np.frombuffer(renderer._accum_tex_den2.read(), dtype=np.float32).copy()
    num2 = np.frombuffer(renderer._accum_tex_num2.read(), dtype=np.float32).copy()
    return den1, den2, num2


def render_surface_density(renderer, camera, masses, res=64):
    """Render a plain SurfaceDensity frame."""
    renderer.resolve_mode = 0
    renderer.update_weights(masses)
    renderer.update_visible(camera)
    renderer._ensure_fbo(res, res, which=1)
    renderer._render_accum(camera, res, res, renderer._accum_fbo)
    return np.frombuffer(renderer._accum_tex_den.read(), dtype=np.float32).copy()


@pytest.fixture
def particle_data():
    rng = np.random.default_rng(42)
    N = 5000
    pos = rng.uniform(0, 1, (N, 3)).astype(np.float32)
    masses = np.ones(N, dtype=np.float32) / N
    hsml = np.full(N, 0.05, dtype=np.float32)
    qty = rng.uniform(100, 10000, N).astype(np.float32)
    return pos, masses, hsml, qty


def test_composite_consistent_after_mode_switch(particle_data):
    """Composite results should be identical before and after switching to
    SurfaceDensity and back."""
    pos, masses, hsml, qty = particle_data
    ctx, renderer, camera = make_renderer_and_camera(pos, masses, hsml)

    # First composite render
    den1_a, den2_a, num2_a = render_composite(renderer, camera, masses, qty)

    # Switch to SurfaceDensity, render, move camera, render again
    sd1 = render_surface_density(renderer, camera, masses)

    camera.position = np.array([0.5, 0.5, 1.5], dtype=np.float32)
    camera._dirty = True
    sd2 = render_surface_density(renderer, camera, masses)

    # Move camera back
    camera.position = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    camera._dirty = True

    # Second composite render — should match the first
    den1_b, den2_b, num2_b = render_composite(renderer, camera, masses, qty)

    assert np.allclose(den1_a, den1_b, atol=1e-6), \
        f"Slot 0 denominator changed: max diff = {np.abs(den1_a - den1_b).max()}"
    assert np.allclose(den2_a, den2_b, atol=1e-6), \
        f"Slot 1 denominator changed: max diff = {np.abs(den2_a - den2_b).max()}"
    assert np.allclose(num2_a, num2_b, atol=1e-6), \
        f"Slot 1 numerator changed: max diff = {np.abs(num2_a - num2_b).max()}"

    renderer.release()
    ctx.release()


def test_composite_consistent_via_apply_render_mode(particle_data):
    """Test using the app-level _apply_render_mode + _render_composite_frame flow."""
    pos, masses, hsml, qty = particle_data
    ctx, renderer, camera = make_renderer_and_camera(pos, masses, hsml)

    # Simulate app state
    class FakeApp:
        pass

    app = FakeApp()
    app.renderer = renderer
    app.camera = camera
    app.data = type('D', (), {
        'get_field': lambda self, n: masses if n == "Masses" else qty,
        'get_vector_field': lambda self, n: None,
    })()
    app._vector_fields = []
    app._sd_field = "Masses"
    app._sd_field2 = "None"
    app._sd_op = "*"
    app._SD_OPS = ["*"]
    app._render_mode_name = "SurfaceDensity"
    app._render_mode = None
    app._composite = False
    app._wa_data_field = "Masses"
    app._vector_projection = "LOS"
    app._los_camera_fwd = None
    app._needs_auto_range = False
    app.width = 64
    app.height = 64
    app._slot = [
        {"mode": "SurfaceDensity", "weight": "Masses", "data": "Masses",
         "weight2": "None", "op": "*", "proj": "LOS",
         "min": -1.0, "max": 3.0, "log": 1, "resolve": 0},
        {"mode": "SurfaceDensity", "weight": "Masses", "data": "Masses",
         "weight2": "None", "op": "*", "proj": "LOS",
         "min": -1.0, "max": 3.0, "log": 1, "resolve": 0},
    ]

    # Import the methods we need
    from dataflyer.app import DataFlyerApp
    app._compute_slot = DataFlyerApp._compute_slot.__get__(app)
    app._compute_weights = DataFlyerApp._compute_weights.__get__(app)
    app._project_field = DataFlyerApp._project_field.__get__(app)
    app._render_composite_frame = DataFlyerApp._render_composite_frame.__get__(app)
    app._apply_render_mode = DataFlyerApp._apply_render_mode.__get__(app)
    app._msg = lambda text: None
    app._uses_vector_field = DataFlyerApp._uses_vector_field.__get__(app)
    app._is_los_stale = DataFlyerApp._is_los_stale.__get__(app)

    def render_both_slots(app):
        """Render both composite slots into FBOs, return denominator contents."""
        r = app.renderer
        r._ensure_fbo(64, 64, which=1)
        r._ensure_fbo(64, 64, which=2)
        fbos = [r._accum_fbo, r._accum_fbo2]
        for i in range(2):
            w, q = app._compute_slot(app._slot[i])
            r.update_weights(w, q)
            r.update_visible(app.camera)
            r._render_accum(app.camera, 64, 64, fbos[i])
        den1 = np.frombuffer(r._accum_tex_den.read(), dtype=np.float32).copy()
        den2 = np.frombuffer(r._accum_tex_den2.read(), dtype=np.float32).copy()
        return den1, den2

    # 1. Enter Composite, render both slots
    app._render_mode_name = "Composite"
    app._apply_render_mode(auto_range=False)
    den1_a, den2_a = render_both_slots(app)

    # 2. Switch to SurfaceDensity
    app._render_mode_name = "SurfaceDensity"
    app._apply_render_mode(auto_range=False)
    renderer.update_visible(camera)

    # 3. Move camera, render SD
    camera.position = np.array([0.5, 0.5, 1.5], dtype=np.float32)
    camera._dirty = True
    renderer.update_visible(camera)

    # 4. Move back, switch to Composite, render both slots
    camera.position = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    camera._dirty = True
    app._render_mode_name = "Composite"
    app._apply_render_mode(auto_range=False)
    den1_b, den2_b = render_both_slots(app)

    assert np.allclose(den1_a, den1_b, atol=1e-6), \
        f"Slot 0 denom changed after round-trip: max diff = {np.abs(den1_a - den1_b).max()}"
    assert np.allclose(den2_a, den2_b, atol=1e-6), \
        f"Slot 1 denom changed after round-trip: max diff = {np.abs(den2_a - den2_b).max()}"

    renderer.release()
    ctx.release()


def test_composite_after_stray_update_visible(particle_data):
    """Simulate the main loop calling update_visible between composite renders.

    The main loop's progressive refinement calls update_visible with whatever
    state is in the grid. This test checks that a stray update_visible between
    mode switches doesn't corrupt composite results.
    """
    pos, masses, hsml, qty = particle_data
    ctx, renderer, camera = make_renderer_and_camera(pos, masses, hsml)

    # First composite render
    den1_a, den2_a, num2_a = render_composite(renderer, camera, masses, qty)

    # Simulate what happens in the main loop: the progressive refinement
    # calls update_visible with stale renderer state (from last slot 1 render)
    renderer.update_visible(camera)

    # Simulate switching to SurfaceDensity mode
    renderer.resolve_mode = 0
    renderer.update_weights(masses)
    renderer.update_visible(camera)

    # Simulate more stray update_visible calls (progressive refinement)
    renderer.update_visible(camera)
    renderer.update_visible(camera)

    # Switch back to composite
    den1_b, den2_b, num2_b = render_composite(renderer, camera, masses, qty)

    assert np.allclose(den1_a, den1_b, atol=1e-6), \
        f"Slot 0 denom changed: max diff = {np.abs(den1_a - den1_b).max()}"
    assert np.allclose(num2_a, num2_b, atol=1e-6), \
        f"Slot 1 numerator changed: max diff = {np.abs(num2_a - num2_b).max()}"

    renderer.release()
    ctx.release()


def test_composite_consistent_after_camera_move(particle_data):
    """Composite results at same camera position should be identical
    regardless of intermediate camera moves."""
    pos, masses, hsml, qty = particle_data
    ctx, renderer, camera = make_renderer_and_camera(pos, masses, hsml)

    # Render at position A
    den1_a, den2_a, num2_a = render_composite(renderer, camera, masses, qty)

    # Move to B, render composite there
    camera.position = np.array([0.3, 0.7, 1.0], dtype=np.float32)
    camera._forward = np.array([0.2, -0.3, -0.9], dtype=np.float32)
    camera._dirty = True
    render_composite(renderer, camera, masses, qty)

    # Move back to A
    camera.position = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._dirty = True
    den1_b, den2_b, num2_b = render_composite(renderer, camera, masses, qty)

    assert np.allclose(den1_a, den1_b, atol=1e-6), \
        f"Slot 0 changed after camera roundtrip: max diff = {np.abs(den1_a - den1_b).max()}"
    assert np.allclose(num2_a, num2_b, atol=1e-6), \
        f"Slot 1 changed after camera roundtrip: max diff = {np.abs(num2_a - num2_b).max()}"

    renderer.release()
    ctx.release()
