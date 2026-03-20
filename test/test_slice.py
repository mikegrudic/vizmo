from vizmo import coordinate_grid
from meshoid import Meshoid, particle_glass
import numpy as np
import pytest


def test_slice():
    """Tests that a slice reconstructs the coordinate functions exactly,
    and consistency between the coordinate grid function and the meshoid slice grid"""
    x = particle_glass(16**3)
    sliceargs = {"res": 128, "size": 1.0, "center": 0.5 * np.ones(3)}
    X, Y = coordinate_grid(**sliceargs)
    xslice = Meshoid(x).Slice(x[:, 0], **sliceargs, order=2)  # 2nd order
    assert X == pytest.approx(xslice, abs=1e-15)
    xslice = Meshoid(x).Slice(x[:, 1], **sliceargs, order=2)  # 2nd order
    assert Y == pytest.approx(xslice, abs=1e-15)
    xslice = Meshoid(x).Slice(x[:, 0], **sliceargs, order=1)  # 1st order
    assert X == pytest.approx(xslice, abs=1e-15)
    xslice = Meshoid(x).Slice(x[:, 1], **sliceargs, order=1)  # 1st order
    assert Y == pytest.approx(xslice, abs=1e-15)
    xslice = Meshoid(x).Slice(x[:, 0], **sliceargs, order=0)  # 0'th order (error should be on the order of cell size)
    assert X == pytest.approx(xslice, abs=0.1)
    xslice = Meshoid(x).Slice(x[:, 1], **sliceargs, order=0)  # 0'th order (error should be on the order of cell size)
    assert Y == pytest.approx(xslice, abs=0.1)
