# vizmo

Real-time 3D fly-through explorer for unstructured simulation data. Loads HDF5 snapshots from GIZMO/Gadget simulations and renders interactive surface density maps, mass-weighted averages, velocity dispersions, and composite CoolMap visualizations on the GPU via WebGPU.

This is a **vibecoded** app. I have no idea how its front- or backends work, and am not sure if I could really properly support or get it running for you 10 years from now! But it would not exist otherwise. Software is weird now. It is what it is. 

But I hope you enjoy it and please feel free to report bugs or request features.

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.9
- WebGPU-capable GPU (Metal on macOS, Vulkan on Linux, D3D12 on Windows)

## Quickstart

Point `vizmo` at an HDF5 snapshot file:

```bash
vizmo path/to/snapshot.hdf5
```

The snapshot must contain gas particle data (`PartType0`) with at minimum `Coordinates`, `Masses`, and `KernelMaxRadius` (or `SmoothingLength`) fields. Star particles (`PartType5`) are also supported if present.

### CLI options

```
vizmo snapshot.hdf5 [--width 1920] [--height 1080] [--fov 90]
                        [--fullscreen] [--screenshot OUT.png]
                        [--profile OUT.pstats]
```

## Rendering backend

vizmo uses a WebGPU backend ([wgpu-py](https://github.com/pygfx/wgpu-py)) with GPU-resident particle data. Compute shaders perform frustum culling, LOD selection, and per-cell summary gathering with zero CPU↔GPU per-frame transfer. On unified-memory systems (Apple Silicon) field switches are also near-zero copy.

The renderer uses progressive refinement and an auto-LOD subsample cap that adapts within a user-controlled ceiling to keep interaction smooth during motion and sharpen on idle.

## Controls

**Camera:**
- `W/A/S/D` — Move forward/left/back/right
- `Z/X` — Move up/down
- `Q/E` — Roll left/right
- Mouse (click + drag) — Look around
- Scroll wheel — Adjust speed

**Visualization:**
- `Tab` — Hide/show all UI
- `C` — Cycle colormap
- `L` — Toggle log/linear scale
- `R` — Auto-range color scale (composite: Color slot)
- `T` — Auto-range composite Lightness slot
- `+/-` — Contract/expand color range
- `[/]` — Coarser/finer LOD pixel size
- `,/.` — Lower/raise the auto-LOD subsample-cap ceiling
- `P` — Save screenshot
- `F1` or `\` — Toggle dev overlay
- `Esc` — Quit

## Render Modes

Select from the **Mode** dropdown in the user menu:

- **SurfaceDensity** — Projected surface density of a weight field. Supports combining two fields with arithmetic operators (Op / Field 2).
- **WeightedAverage** — Mass-weighted line-of-sight average of a data field.
- **WeightedVariance** — Mass-weighted line-of-sight standard deviation (e.g. velocity dispersion).
- **Composite** — CoolMap-style dual-field visualization. Encodes one field in lightness and another in colormap hue. Each channel has independent render mode, field selection, limits, and scaling.

### Vector Fields

3D vector fields (e.g. Velocities) are automatically detected. When selected as a weight or data field, a **Proj** dropdown appears:

- **LOS** — Per-particle line-of-sight component (dot product with the unit vector from the camera to the particle). Invariant under camera rotation; recomputed when the camera translates past a small threshold.
- **|v|** — Euclidean norm.
- **|v|^2** — Squared norm.

## Architecture

```
vizmo/
  app.py            - CLI entry point
  wgpu_app.py       - Main loop, key actions, progressive refinement, auto-LOD
  wgpu_renderer.py  - WGPURenderer: RenderMode, accumulate + resolve + composite passes
  gpu_compute.py    - GPUCompute: GPU-resident data, compute cull/LOD/gather
  wgpu_overlay.py   - WGPUDevOverlay, WGPUUserMenu (wgpu panel rendering)
  overlay.py        - Panel/PanelStyle base, DevOverlay, UserMenu
  camera.py         - 6DOF camera with cached basis vectors
  data_manager.py   - HDF5 I/O with lazy loading and cosmological corrections
  field_ops.py      - Field arithmetic and vector projections
  colormaps.py      - Matplotlib colormap to GPU texture
  shaders/          - WGSL shaders (common, splat_subsample, resolve, composite,
                      star, text)
```
