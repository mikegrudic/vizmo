from os import mkdir
from os.path import isdir
from urllib.request import urlretrieve


def download_test_data():
    path = "https://users.flatironinstitute.org/~mgrudic/starforge_data/STARFORGE_RT/STARFORGE_v1.2/M2e2_R1/M2e2_R1_Z1_S0_A2_B0.1_I1_Res58_n2_sol0.5_42/output/"

    if not isdir("./output"):
        mkdir("output")

    for snapnum in 640, 650, 980:
        urlretrieve(path + f"/snapshot_{snapnum}.hdf5", f"output/snapshot_{snapnum}.hdf5")
