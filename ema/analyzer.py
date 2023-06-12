import pickle

import numpy as np
import pandas as pd

from ema.graphic import plot_interactive, plot_event
from loader import numpy_loading
from scipy import stats
import matplotlib.pyplot as plt

# folder_path = "../dataset"
# all_files = [folder_path + "/" + f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".dat")]
all_files = ["./dataset/data_000000.dat"]

time_offset_by_chamber = np.array(
    [95.0 - 1.1,  # Ch 0
     95.0 + 6.4,  # Ch 1
     95.0 + 0.5,  # Ch 2
     95.0 - 2.6]  # Ch 3
)

SPACE_OFFSETS = np.array([219.8, 977.3, 1035.6, 1819.8])
CELL_WIDTH = 42
CELL_HEIGHT = 13


def load_dataframe(filename):
    # for filename in all_files:
    mat = numpy_loading(filename, output=False, analyze=False)

    df = pd.DataFrame(data=mat, columns=["TDC", "BX", "ORBIT", "CHANNEL", "FPGA", "HEAD"])
    return df


MIN_UNIQUE_LAYERS_HIT = 2
MAX_HITS_PER_CHAMBER = 12
MIN_HITS_PER_CHAMBER = MIN_UNIQUE_LAYERS_HIT
MAX_HITS_PER_LAYER = 3
MIN_GOOD_CHAMBERS = 2


def manipulate_dataframe(df):
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
    # Here we assign the correct LAYER only to the cell belonging to layer 0 and 3
    df["LAYER"] = df.CELL % 4
    # Here we swap layer 1 with layer 2 (the incorrect ones)
    df["LAYER"] = np.where((df.LAYER == 1) | (df.LAYER == 2), df.LAYER % 2 + 1, df.LAYER)

    df["CELL_X"] = (df.CELL // 4) * CELL_WIDTH + CELL_WIDTH * 0.5 + CELL_WIDTH * 0.5 * (df.LAYER % 2)
    df["CELL_Y"] = SPACE_OFFSETS[df.CHAMBER] + (4 - df.LAYER) * CELL_HEIGHT - CELL_HEIGHT * 0.5

    # time correction
    df["T0"] = df.T0 - time_offset_by_chamber[df.CHAMBER]
    # We calculate the distance, and then convert it from um to mm
    df["DISTANCE"] = (df.TIME - df.T0) * 53.8 / 1000

    hits_before = len(df)
    df = df[(df.DISTANCE >= 0) & (df.DISTANCE <= 21)]
    print(f"Sono stati rimossi {hits_before - len(df)} eventi perchè avevano distanze sbagliate")

    # In this function we check if there are at least 'MIN_GOOD_CHAMBERS' chambers
    # What is a good chamber? A good chamber is when
    # the chamber has at least 'MIN_HITS_PER_CHAMBER' hits in total
    # the chamber has at least 'MIN_UNIQUE_LAYERS_HIT' layers hit
    #
    # But if we find a chamber with more than 'MAX_HITS_PER_CHAMBER' hits, then we discard the whole event
    def filter_chambers(x):
        good_ch = []
        for ch, df_ch in x.groupby("CHAMBER"):
            if len(df_ch) > MAX_HITS_PER_CHAMBER:
                return False

            if len(df_ch) >= MIN_HITS_PER_CHAMBER:
                if len(df_ch.LAYER.value_counts()) >= MIN_UNIQUE_LAYERS_HIT:
                    if np.all(df_ch.LAYER.value_counts() <= MAX_HITS_PER_LAYER):
                        good_ch.append(ch)

        return len(good_ch) >= MIN_GOOD_CHAMBERS

    df = df.groupby(["ORBIT"]).filter(filter_chambers)

    df = df.sort_values(by=["ORBIT"])
    return df


# def manipulate_df_loop(df):
#     groups = []
#
#     groupby = df.groupby("ORBIT")
#     print("total", len(groupby))
#
#     for i, (orbit, df_orbit) in enumerate(groupby):
#         df_orbit = df_orbit[~((df_orbit.FPGA == 0) & (df_orbit.CHANNEL >= 64) & (df_orbit.CHANNEL <= 127))]
#
#         df_orbit = df_orbit[(df_orbit.CHANNEL != 138)]
#         if len(df_orbit) < MIN_UNIQUE_LAYERS_HIT * 3 + 1:
#             continue
#
#         # we keep only the first hit in the same cell
#         df_orbit = df_orbit.sort_values(by=["ORBIT", "BX", "TDC"])
#         df_orbit = df_orbit.drop_duplicates(["CHANNEL", "FPGA"])
#
#         # we create the column for the time (we don't include orbit)
#         df_orbit["TIME"] = 25 * df_orbit.BX + df_orbit.TDC * 25 / 30
#
#         t0 = df_orbit[(df_orbit.CHANNEL == 128) & (df_orbit.FPGA == 1)]
#         if not len(t0):
#             # print("Non ho trovato un t0")
#             continue
#
#         if len(t0) > 1:
#             print("Ho trovato più di un t0")
#             continue
#
#         df_orbit["T0"] = t0["TIME"].values[0]
#
#         # We remove the column with t0 and the infamous channel 138
#         df_orbit = df_orbit[(df_orbit.CHANNEL != 128)]
#
#         df_orbit["CHAMBER"] = np.round(df_orbit.FPGA * 2 + df_orbit.CHANNEL // 64)
#         df_orbit["LAYER"] = (df_orbit.CHANNEL - (df_orbit.CHAMBER % 2) * 64) // 16
#
#         for cham in [0, 2, 3]:
#             diff_layers = len(df_orbit[df_orbit.CHAMBER == cham].LAYER.value_counts())
#
#             if diff_layers < MIN_UNIQUE_LAYERS_HIT:
#                 continue
#
#         groups.append(df_orbit)
#
#         if len(groups) % 25 == 0:
#             print(i, len(groups))
#     # plot_interactive(groups, landscape=False)


def alg1(x1, x2, x, y):
    res = []
    for i in range(4):
        p1 = np.array([x1[i], y[i]])
        p2 = np.array([x2[i], y[i]])
        a = p1 - np.column_stack([x, y])
        b = p2 - np.column_stack([x, y])
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        sa = np.sum(la)
        sb = np.sum(lb)
        if sa < sb:
            res.append(x1[i])
        elif sb < sa:
            res.append(x2[i])
        else:
            print("UGUALLEEEEEEEE")
            res.append(x1[i])
    return np.array(res)


def alg2(x1, x2, x, y):
    res = []
    for i in range(4):
        p1 = x1[i]
        p2 = x2[i]
        a = p1 - x
        b = p2 - x
        la = np.abs(a)
        lb = np.abs(b)
        sa = np.sum(la)
        sb = np.sum(lb)
        if sa < sb:
            res.append(x1[i])
        elif sb < sa:
            res.append(x2[i])
        else:
            print("UGUALLEEEEEEEE")
            res.append(x1[i])
    return np.array(res)


def isolate_local_tracks(df_ch):
    plot_event()


# hits for a chamber
def calculate_local_track(df):
    x1 = df.CELL_X - df.DISTANCE
    x2 = df.CELL_X + df.DISTANCE
    y = df.CELL_Y
    xgood1 = alg1(x1.values, x2.values, df.CELL_X.values, df.CELL_Y.values)
    xgood2 = alg2(x1.values, x2.values, df.CELL_X.values, df.CELL_Y.values)
    res1 = stats.linregress(xgood1, y)
    res2 = stats.linregress(xgood2, y)
    # slope, intercept = res.slope, res.intercept
    print(res1.slope, res1.intercept)
    print(res2.slope, res2.intercept)
    return res1, res2


# def generate_data_provider(dfs, local_tracks=True):
#     def data_provider(index: int):
#         res = []
#         res.append(dfs[index])
#
#         if local_tracks:
#             a = calculate_local_track(df)
#             res.append()
#
#
#     return data_provider


def main():
    df = load_dataframe("../dataset/data_000001.dat")

    print(f"Eventi Iniziali: {len(df.ORBIT.value_counts())}")
    df_filtered = manipulate_dataframe(df)
    print(f"Eventi Finali: {len(df_filtered.ORBIT.value_counts())}")
    print(np.unique(df.ORBIT.values))

    groups = [dff for _, dff in df_filtered.groupby("ORBIT")]

    # with open("./pickled/data.bin", "wb") as f:
    #     pickle.dump([df, df_filtered, groups], f)
    #
    # with open("./pickled/data.bin", "rb") as f:
    #     df, df_filtered, groups = pickle.load(f)
    #
    # grouped_ch = [a for _, a in groups[0].groupby("CHAMBER")]
    # regr_data = [[], [], [], []]
    # for i in range(len(grouped_ch)):
    #     res1, res2 = calculate_local_track(grouped_ch[i])
    #     chami = grouped_ch[i].CHAMBER.values[0]
    #     regr_data[chami].append([res1.slope, res1.intercept])
    #     regr_data[chami].append([res2.slope, res2.intercept])
    #     # graphic._axes.scatter(x1, y, s=10)

    plot_interactive(groups, landscape=True, regr_data=None, show=False)
    plt.show(block=True)


    # # good_orbit = np.in1d(np.unique(df.ORBIT.values), (np.unique(df_filtered.ORBIT.values)))
    # plot_interactive(groups, landscape=False, add_info=None)

    # manipulate_df_loop(df)


if __name__ == "__main__":
    main()
