import numpy as np
import pandas as pd
from loader import numpy_loading
from os import listdir
from os.path import isfile, join
from graphic import plot_interactive

folder_path = "./dataset"
all_files = [folder_path + "/" + f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".dat")]
all_files = ["./dataset/data_000000.dat"]

for filename in all_files:
    mat = numpy_loading(filename, output=False, analyze=False)
    # print("Caricato il file", filename)
    # print(sum(mat[:, 2] == 1744617))
df = pd.DataFrame(data=mat, columns=["TDC", "BX", "ORBIT", "CHAN", "FPGA", "HEAD"])

occ = df["ORBIT"].value_counts()
print(occ[occ >= 12])


def display(df):
    pass


time_offset_by_chamber = np.array(
    [95.0 - 1.1,  # Ch 0
     95.0 + 6.4,  # Ch 1
     95.0 + 0.5,  # Ch 2
     95.0 - 2.6]  # Ch 3
)

groups = []
for (orbit, df_orbit) in df.groupby("ORBIT"):
    if len(df_orbit) < 12:
        continue
    # df_orbit["TIME"] = 25 * df.BX + df.TDC * 25 / 30 - time_offset_by_chamber[df_orbit["CHAMBER"]]
    # df_orbit["TIME"] = 25 * df.BX + df.TDC * 25 / 30

    t0 = df_orbit[(df_orbit.CHAN == 128) & (df_orbit.FPGA == 1)]
    if not len(t0):
        # print("Non ho trovato un t0")
        continue

    if len(t0) > 1:
        print("Ho trovato più di un t0")
        continue

    print(df_orbit.loc[t0.index])
    df_orbit = df_orbit.drop_duplicates(["CHAN", "FPGA"])

    # df_orbit = df_orbit.drop(t0.index)
    df_orbit = df_orbit[df_orbit.CHAN < 128]
    df_orbit["CHAMBER"] = np.round(df.FPGA * 2 + df.CHAN // 64)
    df_orbit = df_orbit[df_orbit.CHAMBER != 1]
    df_orbit.CHAN = df_orbit.CHAN - (df_orbit.CHAMBER % 2) * 64
    # print(df_orbit.sort_values("TIME"))
    groups.append(df_orbit)

    if len(groups) == 20:
        break

plot_interactive(groups, landscape=False)
