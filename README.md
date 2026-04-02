# DataFlyer

Real-time 3D fly-through explorer for SPH simulation data. Loads HDF5 snapshots from GIZMO/Gadget simulations and renders interactive surface density maps, mass-weighted averages, velocity dispersions, and composite CoolMap visualizations at 60+ FPS.

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.9
- OpenGL 3.3+ capable GPU

## Quickstart

Point `dataflyer` at an HDF5 snapshot file:

```bash
dataflyer path/to/snapshot.hdf5
```

The snapshot must contain gas particle data (`PartType0`) with at minimum `Coordinates`, `Masses`, and `KernelMaxRadius` (or `SmoothingLength`) fields. Star particles (`PartType5`) are also supported if present.

If a directory contains multiple snapshots, they are automatically discovered and you can step through them with the arrow keys.

### CLI options

```
dataflyer snapshot.hdf5 [--width 1920] [--height 1080] [--fov 90]
                        [--screenshot output.png] [--benchmark N]
```

## Controls

**Camera:**
- `W/A/S/D` - Move forward/left/back/right
- `Z/X` - Move up/down
- `Q/E` - Roll left/right
- Mouse (click + drag) - Look around
- Scroll wheel - Adjust speed

**Visualization:**
- `Tab` - Hide/show all UI
- `C` - Cycle colormap
- `L` - Toggle log/linear scale
- `R` - Auto-range color scale
- `+/-` - Contract/expand color range
- `[/]` - Decrease/increase LOD detail
- `,/.` - Fewer/more particles
- `Left/Right` - Previous/next snapshot
- `P` - Save screenshot

**Advanced:**
- `\` - Toggle dev overlay
- `I` - Toggle importance sampling
- `K` - Cycle SPH kernel
- `T` - Toggle spatial tree
- `H` - Print help
- `Esc` - Quit

## Render Modes

Select from the **Mode** dropdown in the user menu:

- **SurfaceDensity** - Projected surface density of a weight field. Supports combining two fields with arithmetic operators (Op / Field 2).
- **WeightedAverage** - Mass-weighted line-of-sight average of a data field.
- **WeightedVariance** - Mass-weighted line-of-sight standard deviation (e.g. velocity dispersion).
- **Composite** - CoolMap-style dual-field visualization. Encodes one field in brightness and another in colormap hue via HSV blending. Each channel has independent render mode, field selection, limits, and scaling.

### Vector Fields

3D vector fields (e.g. Velocities) are automatically detected. When selected as a weight or data field, a **Proj** dropdown appears:

- **LOS** - Line-of-sight component (dot product with camera forward). Recomputed automatically when the camera rotates.
- **|v|** - Euclidean norm.
- **|v|^2** - Squared norm.

## Architecture

```
dataflyer/
  app.py           - Main loop, key actions, render mode orchestration
  renderer.py      - RenderMode, ParticleLayer, BufferSet, SplatRenderer
  spatial_grid.py  - CellMoments, SpatialGrid, numba gather kernels
  camera.py        - 6DOF camera with cached basis vectors
  data_manager.py  - HDF5 I/O with lazy loading and cosmological corrections
  overlay.py       - Panel/PanelStyle base, DevOverlay, UserMenu
  colormaps.py     - Matplotlib colormap to GPU texture
  shaders/         - GLSL vertex/fragment shaders
```

## Tests

```bash
pytest tests/
```

- `test_surface_density_vs_sinkvis.py` - Validates GPU surface density against CrunchSnaps/Meshoid
- `test_composite_state.py` - Verifies composite mode state consistency across mode switches
- `bench_perf_regression.py` - Performance benchmark (`python tests/bench_perf_regression.py`)
