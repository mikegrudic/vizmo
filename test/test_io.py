from vizmo.io import (
    get_snapshot_data,
    snapdata_to_meshoid,
    snapshot_to_meshoid,
    get_snapshot_timeline,
    get_snapdata_at_time,
)
from vizmo.test import download_test_data


def test_io():
    """Simple routine that calls the various IO routines just for the sake of coverage"""
    download_test_data()
    rundir = "output"
    snappath = "output/snapshot_640.hdf5"
    data = get_snapshot_data(snappath)
    snapdata_to_meshoid(data)
    snapshot_to_meshoid(snappath)
    get_snapshot_timeline(rundir)
    get_snapdata_at_time(rundir, 1.0)
