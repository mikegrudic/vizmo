# bump version 0.1.2 -> 0.1.3 in pyproject.toml
rm -rf dist/ build/ *.egg-info
python -m build
unzip -l dist/vizmo-0.1.3-*.whl | grep -E 'wgsl|assets'
twine upload dist/*
