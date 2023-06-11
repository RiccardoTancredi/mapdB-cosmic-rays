import numpy as np
import pandas as pd
from loader import numpy_loading
from os import listdir
from os.path import isfile, join
from graphic import plot_interactive

# folder_path = "../dataset"
# all_files = [folder_path + "/" + f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".dat")]
all_files = ["./dataset/data_000000.dat"]

time_offset_by_chamber = np.array(
    [95.0 - 1.1,  # Ch 0
     95.0 + 6.4,  # Ch 1
     95.0 + 0.5,  # Ch 2
     95.0 - 2.6]  # Ch 3
)


def load_dataframe(filename):
    # for filename in all_files:
    mat = numpy_loading(filename, output=False, analyze=False)

    df = pd.DataFrame(data=mat, columns=["TDC", "BX", "ORBIT", "CHANNEL", "FPGA", "HEAD"])
    return df


MIN_DIFFERENT_LAYERS_HIT = 2


def manipulate_dataframe(df):

    print(f"Eventi Iniziali: {len(df.ORBIT.value_counts())}")

    # we eliminate all hits on chamber 1
    df = df[~((df.FPGA == 0) & (df.CHANNEL >= 64) & (df.CHANNEL <= 127))]

    # we keep only the first hit in the same cell
    df = df.sort_values(by=["ORBIT", "BX", "TDC"])
    df = df.drop_duplicates(["ORBIT", "CHANNEL", "FPGA"])

    # we create the column for the time (we don't include orbit)
    df["TIME"] = 25 * df.BX + df.TDC * 25 / 30

    # We keep only the orbits with a t0 (todo: check if there is only one t0)
    mask_t0 = (df.FPGA == 1) & (df.CHANNEL == 128)
    orbits_with_t0 = df.loc[mask_t0, ('ORBIT', "TIME")]
    df = df[df.ORBIT.isin(orbits_with_t0["ORBIT"].unique())]

    # we rename the TIME columns for the t0s in T0 and then merge with the original df on ORBIT
    orbits_with_t0.rename(columns={'TIME': 'T0'}, inplace=True)
    df = pd.merge(df, orbits_with_t0, on='ORBIT', how='inner')

    # We remove the column with t0 and the infamous channel 138
    df = df[(df.CHANNEL != 128) & (df.CHANNEL != 138)]

    df["CHAMBER"] = np.round(df.FPGA * 2 + df.CHANNEL // 64)
    df["CELL"] = (df.CHANNEL - (df.CHAMBER % 2) * 64)
    df["LAYER"] = df.CELL // 16

    # We remove the one who have less than MIN_DIFFERENT_LAYERS_HIT * 3
    # equivalent to SQL 'HAVING COUNT(X) >= ... '
    df = df.groupby(["ORBIT"]).filter(lambda x: len(x) >= MIN_DIFFERENT_LAYERS_HIT * 2)

    # Here we keep only the chambers with at least 'MIN_DIFFERENT_LAYERS_HIT' different layers hit
    df = df.groupby(["ORBIT", "CHAMBER"]).filter(lambda x: len(x.LAYER.value_counts()) >= MIN_DIFFERENT_LAYERS_HIT)

    # Here we keep only the orbit which have at least two different chambers (with the right amount of hits)
    df = df.groupby(["ORBIT"]).filter(lambda x: len(x.CHAMBER.value_counts()) >= 2)

    # todo time correction

    print(f"Eventi Finali: {len(df.ORBIT.value_counts())}")

    return df


def manipulate_df_loop(df):
    groups = []

    groupby = df.groupby("ORBIT")
    print("total", len(groupby))

    for i, (orbit, df_orbit) in enumerate(groupby):
        df_orbit = df_orbit[~((df_orbit.FPGA == 0) & (df_orbit.CHANNEL >= 64) & (df_orbit.CHANNEL <= 127))]

        df_orbit = df_orbit[(df_orbit.CHANNEL != 138)]
        if len(df_orbit) < MIN_DIFFERENT_LAYERS_HIT * 3 + 1:
            continue

        # we keep only the first hit in the same cell
        df_orbit = df_orbit.sort_values(by=["ORBIT", "BX", "TDC"])
        df_orbit = df_orbit.drop_duplicates(["CHANNEL", "FPGA"])

        # we create the column for the time (we don't include orbit)
        df_orbit["TIME"] = 25 * df_orbit.BX + df_orbit.TDC * 25 / 30

        t0 = df_orbit[(df_orbit.CHANNEL == 128) & (df_orbit.FPGA == 1)]
        if not len(t0):
            # print("Non ho trovato un t0")
            continue

        if len(t0) > 1:
            print("Ho trovato più di un t0")
            continue

        df_orbit["T0"] = t0["TIME"].values[0]

        # We remove the column with t0 and the infamous channel 138
        df_orbit = df_orbit[(df_orbit.CHANNEL != 128)]

        df_orbit["CHAMBER"] = np.round(df_orbit.FPGA * 2 + df_orbit.CHANNEL // 64)
        df_orbit["LAYER"] = (df_orbit.CHANNEL - (df_orbit.CHAMBER % 2) * 64) // 16

        for cham in [0, 2, 3]:
            diff_layers = len(df_orbit[df_orbit.CHAMBER == cham].LAYER.value_counts())

            if diff_layers < MIN_DIFFERENT_LAYERS_HIT:
                continue

        groups.append(df_orbit)

        if len(groups) % 25 == 0:
            print(i, len(groups))
    # plot_interactive(groups, landscape=False)


def main():
    df = load_dataframe("../dataset/data_000001.dat")
    df_filtered = manipulate_dataframe(df)
    groups = [dff for _, dff in df_filtered.groupby("ORBIT")]
    plot_interactive(groups, landscape=True)

    # manipulate_df_loop(df)


if __name__ == "__main__":
    main()
