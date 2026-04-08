#!/usr/bin/env bash
# Convert a screen recording (.mov) to a README-friendly .gif.
# Usage: scripts/mov2gif.sh INPUT.mov OUTPUT.gif [WIDTH] [FPS] [COLORS]
# Defaults: WIDTH=600, FPS=8, COLORS=48 (~8MB for ~50s of UI footage).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT.mov OUTPUT.gif [WIDTH] [FPS] [COLORS]" >&2
    exit 1
fi

input=$1
output=$2
width=${3:-600}
fps=${4:-8}
colors=${5:-48}

ffmpeg -y -i "$input" \
    -vf "fps=${fps},scale=${width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=${colors}[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
    "$output"

ls -lh "$output"
