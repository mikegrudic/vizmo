from vizmo.io import (
    get_snapshot_data,
    snapdata_to_meshoid,
    snapshot_to_meshoid,
    get_snapshot_timeline,
    get_snapdata_at_time,
)
from os import mkdir, system


def download_example_data():
    path = "https://users.flatironinstitute.org/~mgrudic/starforge_data/STARFORGE_RT/STARFORGE_v1.2/M2e2_R1/M2e2_R1_Z1_S0_A2_B0.1_I1_Res58_n2_sol0.5_42/output/"

    try:
        mkdir("output")
    except:
        pass

    for snapnum in 640, 650, 980:
        system(f"wget --directory-prefix='output/' {path}snapshot_{snapnum}.hdf5")


def test_io():
    """Simple routine that calls the various IO routines just for the sake of coverage"""
    download_example_data()
    rundir = "output"
    snappath = "output/snapshot_640.hdf5"
    data = get_snapshot_data(snappath)
    snapdata_to_meshoid(data)
    snapshot_to_meshoid(snappath)
    get_snapshot_timeline(rundir)
    get_snapdata_at_time(rundir, 1.0)
