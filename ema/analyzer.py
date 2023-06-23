import math
import pickle
import time

import numpy as np
import pandas as pd

from ema import graphic, fitter
from ema.fitter import get_residuals_eucl_squared
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
    df["REL_TIME"] = df.TIME - df.T0
    df["DISTANCE"] = df.REL_TIME * 53.8 / 1000

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

    df = df.sort_values(by=["ORBIT", "CHAMBER", "LAYER", "CELL"])
    return df


def isolate_local_tracks(df):
    for orbit, df_orbit in df.groupby("ORBIT"):
        for ch, df_ch in df_orbit.groupby("CHAMBER"):
            plot_event(df_ch.CHAMBER, df_ch.CELL, df_ch.DISTANCE, focus_chamber=ch)
            graphic._axes.set_title(f"Orbit: {orbit}, ch: {ch}", y=1.0, pad=-14)
            plt.waitforbuttonpress()


# hits for a chamber
def calculate_local_track(df):
    count = 0
    chs = 0
    PLOT = False

    orbit_groupby = df.groupby("ORBIT")

    tracks = np.zeros(shape=(len(orbit_groupby) * 3, 6))
    for orbit, df_orbit in orbit_groupby:
        for ch, df_ch in df_orbit.groupby("CHAMBER"):
            if len(df_ch) != 4:
                continue

            x1 = (df_ch.CELL_X - df_ch.DISTANCE).values
            x2 = (df_ch.CELL_X + df_ch.DISTANCE).values
            x_cell = df_ch.CELL_X.values
            y_cell = df_ch.CELL_Y.values

            res_bf, comb_bf, debug_bf = fitter.fit_by_bruteforce(x1, x2, x_cell, y_cell, debug=True)

            bf_lr_sorted = sorted(debug_bf[0], key=lambda x: x[2])
            slope_silver, inter_silver, comb_silver = None, None, None

            slope_silver = bf_lr_sorted[1][0].slope
            inter_silver = bf_lr_sorted[1][0].intercept
            comb_silver = bf_lr_sorted[1][1]

            tracks[chs, 0:4] = [orbit, ch, res_bf.slope, res_bf.intercept]
            df.loc[df_ch.index, "HIT_X"] = np.where(comb_bf == 0, x1, x2)

            if slope_silver is not None:
                tracks[chs, 4:] = [slope_silver, inter_silver]
                df.loc[df_ch.index, "HIT_X2"] = np.where(comb_silver == 0, x1, x2)

            chs += 1
            # if np.all(comb1 == comb_bf):
            #     count += 1

            if PLOT:
                regr_data = [[], [], [], []]
                regr_data[ch].append([res_bf.slope, res_bf.intercept])
                if slope_silver:
                    regr_data[ch].append([slope_silver, inter_silver])

                with np.printoptions(precision=3, suppress=True):
                    print(f"{comb_bf}, {res_bf.slope:.3f}, {res_bf.intercept:.3f}, {debug_bf[2]}")

                plot_event(df_ch.CHAMBER, df_ch.CELL, df_ch.DISTANCE, regr_data=regr_data, focus_chamber=ch)
                graphic._axes.set_title(f"Orbit: {orbit}, ch: {ch}", y=1.0, pad=-14)
                plt.legend()
                plt.waitforbuttonpress()
    print(count, chs, count / chs)
    tracks = pd.DataFrame(data=tracks[tracks[:, 0] != 0],
                          columns=["ORBIT", "CHAMBER", "SLOPE", "INTERCEPT", "SLOPE2", "INTERCEPT2"])
    return df, tracks


def calculate_global_track(df, tracks):
    diff_angles = [[], [], [], []]
    for orbit, df_orbit in df.groupby("ORBIT"):

        df_track = tracks[tracks.ORBIT == orbit]
        if len(df_track) < 2:
            continue

        chambers = df_track.CHAMBER.unique().astype(int)

        hits_x_list = []
        hits_y_list = []
        for ch in [0, 2, 3]:
            ch_hits = df_orbit[df_orbit.CHAMBER == ch]
            if len(ch_hits) > 0 and not np.any(np.isnan(ch_hits.HIT_X)):
                hits_x_list.append((ch_hits.HIT_X, ch_hits.HIT_X2))
                hits_y_list.append(ch_hits.CELL_Y)

        result_list = fitter.fit_chambers_by_bruteforce(*hits_x_list, ys=hits_y_list)

        slopes = np.zeros(4)
        intercepts = np.zeros(4)

        slopes[chambers] = df_track.SLOPE.values
        intercepts[chambers] = df_track.INTERCEPT.values

        # slopes, intercepts = df_track.SLOPE.values, df_track.INTERCEPT.values
        angles = np.arctan(slopes) * 180 / np.pi
        slope, intercept = result_list[0][0].slope, result_list[0][0].intercept
        angle = np.arctan(slope) * 180 / np.pi
        for i, ch in enumerate(chambers):
            diff_angles[ch].append(angle - angles[ch])

        slope_diff = np.abs(slopes[0] - slopes[1])
        same_dir = (slopes[0] * slopes[1]) > 0
        delta_x = (df_orbit.CELL_Y - intercepts[0]) / slopes[0] - df_orbit.HIT_X

        if same_dir:
            continue

        # plot_event(df_orbit.CHAMBER, df_orbit.CELL, df_orbit.DISTANCE)
        # x_range = np.linspace(-10, 700, 50)
        # for i, (res_lr, comb, sres, res) in enumerate(result_list):
        #     original = np.all(comb == 0)
        #     if i >= 1 and not original:
        #         continue
        #     label = "Original" if original else str(comb)
        #     graphic._axes.plot(x_range, x_range * res_lr.slope + res_lr.intercept, label=f"{i} " + label)
        # for slope_i, intercept_i in zip(slopes, intercepts):
        #     graphic._axes.plot(x_range, x_range * slope_i + intercept_i, label=f"{i} ", ls="dashed")
        # graphic._axes.set_title(f"Orbit: {orbit}")
        # plt.legend()
        #
        # while not plt.waitforbuttonpress():
        #     pass

    MAX_ANGLE = 15
    plt.figure(figsize=(12, 10))
    for i, ch in enumerate([0, 2, 3]):
        plt.subplot(2, 3, i + 1 + 3)
        nbins = int(np.sqrt(len(diff_angles[ch])))
        plt.hist(diff_angles[ch], bins=nbins)

        values = np.array(diff_angles[ch])
        values = values[(values > -MAX_ANGLE) & (values < MAX_ANGLE)]

        plt.subplot(2, 3, i + 1)
        nbins = int(np.sqrt(len(values))) or 1
        plt.hist(values, bins=nbins)
    plt.show()


def get_pickled(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_pickled(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    filenames = ["../dataset/data_000000.dat", "../dataset/data_000001.dat", "../dataset/data_000002.dat",
                 "../dataset/data_000003.dat", "../dataset/data_000004.dat", "../dataset/data_000005.dat"]

    # big_df_filtered = None
    # for filename in filenames:
    #     print(f"Carico il file: {filename}")
    #     df = load_dataframe(filename)
    #     print(f"Eventi Iniziali: {len(df.ORBIT.value_counts())}")
    #     df_filtered = manipulate_dataframe(df)
    #     print(f"Eventi Finali: {len(df_filtered.ORBIT.value_counts())}")
    #
    #     if big_df_filtered is None:
    #         big_df_filtered = df_filtered
    #     else:
    #         big_df_filtered = pd.concat([big_df_filtered, df_filtered], ignore_index=True)
    #
    #     print("--------------------------------------")
    #
    # big_df_filtered, tracks = calculate_local_track(big_df_filtered)
    # set_pickled("./pickled/big_things.bin", [big_df_filtered, tracks])

    big_df_filtered, tracks = get_pickled("./pickled/big_things.bin")
    calculate_global_track(big_df_filtered, tracks)

    # df = load_dataframe("../dataset/data_000001.dat")
    # print(f"Eventi Iniziali: {len(df.ORBIT.value_counts())}")
    # df_filtered = manipulate_dataframe(df)
    # print(f"Eventi Finali: {len(df_filtered.ORBIT.value_counts())}")
    # print(np.unique(df.ORBIT.values))

    # with open("./pickled/data.bin", "wb") as f:
    #     pickle.dump([df, df_filtered, groups], f)

    # with open("./pickled/data.bin", "rb") as f:
    #     df, df_filtered, groups = pickle.load(f)

    # df_filtered, tracks = calculate_local_track(df_filtered, df_raw=df)
    # with open("./pickled/tracks.bin", "wb") as f:
    #     pickle.dump([df_filtered, tracks], f)

    # with open("./pickled/tracks.bin", "rb") as f:
    #     df_filtered, tracks = pickle.load(f)
    #
    # calculate_global_track(df_filtered, tracks)

    # regr_data = [[], [], [], []]
    # for i in range(len(grouped_ch)):
    #     res1, res2 = calculate_local_track(grouped_ch[i])
    #     chami = grouped_ch[i].CHAMBER.values[0]
    #     regr_data[chami].append([res1.slope, res1.intercept])
    #     regr_data[chami].append([res2.slope, res2.intercept])
    #     # graphic._axes.scatter(x1, y, s=10)

    # plot_interactive(groups, landscape=True, regr_data=None, show=False)
    # plt.show(block=True)

    # # good_orbit = np.in1d(np.unique(df.ORBIT.values), (np.unique(df_filtered.ORBIT.values)))
    # plot_interactive(groups, landscape=False, add_info=None)

    # manipulate_df_loop(df)


if __name__ == "__main__":
    main()
