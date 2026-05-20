# Bump the version in pyproject.toml first — PyPI rejects re-uploads.
set -e
rm -rf dist/ build/ *.egg-info
python -m build
unzip -l dist/vizmo-*.whl | grep -E 'wgsl|assets'
twine upload dist/*
