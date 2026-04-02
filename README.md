# DataFlyer

Real-time 3D fly-through explorer for SPH simulation data.

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

The snapshot must contain gas particle data (`PartType0`) with at minimum `Coordinates`, `Masses`, and `KernelMaxRadius` fields. Star particles (`PartType5`) are also supported if present.

If a directory contains multiple snapshots, they are automatically discovered and you can step through them with the arrow keys.

### Controls

**Camera:**
- `W/A/S/D` - Move forward/left/back/right
- `Z/X` - Move up/down
- `Q/E` - Roll left/right
- Mouse (click + drag) - Look around
- Scroll wheel - Adjust speed

**Visualization:**
- `C` - Cycle colormap
- `L` - Toggle log/linear scale
- `R` - Auto-range color scale
- `+/-` - Contract/expand color range
- `[/]` - Decrease/increase LOD detail
- `,/.` - Fewer/more particles
- `I` - Toggle importance sampling
- `K` - Cycle kernel
- `T` - Toggle tree
- `Left/Right` - Previous/next snapshot
- `P` - Save screenshot
- `\` - Toggle dev overlay
- `H` - Print help
- `Esc` - Quit

### CLI options

```
dataflyer snapshot.hdf5 [--width 1920] [--height 1080] [--fov 90]
                        [--screenshot output.png] [--benchmark N]
```
