import pickle

import numpy as np
import pandas as pd
from scipy import stats

import graphic, fitter
from fitter import simple_fit, get_residuals_eucl_squared
from graphic import plot_interactive, plot_event, plot_pattern_recognition_steps, plot_grouping_result, write_text, \
    plot_interactive2
from loader import numpy_loading
import matplotlib.pyplot as plt

TIME_OFFSETS = np.array(
    [95.0 - 1.1,  # Ch 0
     95.0 + 6.4,  # Ch 1
     95.0 + 0.5,  # Ch 2
     95.0 - 2.6]  # Ch 3
)

SPACE_OFFSETS = np.array([219.8, 977.3, 1035.6, 1819.8])
CELL_WIDTH = 42
CELL_HEIGHT = 13


def load_dataframe(filename):
    mat = numpy_loading(filename, output=False, analyze=False)
    df = pd.DataFrame(data=mat, columns=["TDC", "BX", "ORBIT", "CHANNEL", "FPGA", "HEAD"])
    return df


# Numero minimo di layer diversi colpiti per camera
MIN_UNIQUE_LAYERS_HIT = 2
# Numero massimo di hits in una camera
MAX_HITS_PER_CHAMBER = 12
# Numero minimo di hits per camera (in questo caso uguale al numero minimo di layer colpiti)
MIN_HITS_PER_CHAMBER = MIN_UNIQUE_LAYERS_HIT
# Numero massimo di hits per layer (di una camera)
MAX_HITS_PER_LAYER = 3
# Numero minimo di 'buone camere' per considerare l'intero evento, la definizione di
# buona camera è specificata più sotto
MIN_GOOD_CHAMBERS = 2

MIN_PR_HIT = 3


def get_layers(cells):
    # Here we assign the correct LAYER only to the cell belonging to layer 0 and 3
    # because the modulo operator is good only for the layer 0 and 3, not the 1 and 2
    # because the 4*k cell is in the first row, the 4*k+3 cell is in the fourth row, but
    # the 4*k+1 is in the third and the 4*k+2 in the second, so these last two are flipped
    layers = cells % 4
    # Here we swap layer 1 with layer 2 (the incorrect ones)
    return np.where((layers == 1) | (layers == 2), layers % 2 + 1, layers)


# Questa funzione ha lo scopo di prendere il dataframe grezzo e produrne uno con soltanto gli eventi
# che ci interessano e aggiungengo tutta una serie di colonne utili
def manipulate_dataframe(df):
    # we eliminate all hits on chamber 1
    df = df[~((df.FPGA == 0) & (df.CHANNEL >= 64) & (df.CHANNEL <= 127))]

    # we keep only the first hit in the same cell (to do that we sort by 'time' and keep the first)
    df = df.sort_values(by=["ORBIT", "BX", "TDC"])
    df = df.drop_duplicates(["ORBIT", "CHANNEL", "FPGA"])  # by default it keeps the first

    # we create the column for the time (we don't include orbit because we don't want to compare
    # hits in different event, but only hits from the same events so adding the orbit it's just
    # like adding an offset equal for every hit
    df["TIME"] = 25 * df.BX + df.TDC * 25 / 30

    # We keep only the orbits with a t0 (todo: check if there is only one t0)
    mask_t0 = (df.FPGA == 1) & (df.CHANNEL == 128)
    orbits_with_t0 = df.loc[mask_t0, ('ORBIT', "TIME")]  # list of orbits which contain a t0 row
    df = df[df.ORBIT.isin(orbits_with_t0.ORBIT.unique())]  # we keep only the orbit in that list

    # we rename the TIME columns for the t0s to T0 and then merge with the original df on ORBIT
    # in this way we obtain a new column called T0 for every hit which contains the t0 of the event
    # associated with that orbit
    orbits_with_t0.rename(columns={'TIME': 'T0'}, inplace=True)
    df = pd.merge(df, orbits_with_t0, on='ORBIT', how='inner')

    # We remove the column with t0 and the infamous channel 138 (we don't know what is that)
    df = df[(df.CHANNEL != 128) & (df.CHANNEL != 138)]

    # Here we create some column
    df["CHAMBER"] = np.round(df.FPGA * 2 + df.CHANNEL // 64)
    # Cell is a number from 0-63 for every chamber (like shown in the pdf)
    df["CELL"] = (df.CHANNEL - (df.CHAMBER % 2) * 64)
    df["LAYER"] = get_layers(df.CELL)

    # Here we assign to every hit the position of the central cable of that cell (basically the center of the cell)
    df["CELL_X"] = (df.CELL // 4) * CELL_WIDTH + CELL_WIDTH * 0.5 + CELL_WIDTH * 0.5 * (df.LAYER % 2)
    df["CELL_Y"] = SPACE_OFFSETS[df.CHAMBER] + (4 - df.LAYER) * CELL_HEIGHT - CELL_HEIGHT * 0.5

    # time correction (there is a different t0 for each chamber)
    df["T0"] = df.T0 - TIME_OFFSETS[df.CHAMBER]
    # We calculate the distance using the drift time, and then convert it from um to mm
    df["REL_TIME"] = df.TIME - df.T0
    df["DISTANCE"] = df.REL_TIME * 53.8 / 1000

    hits_before = len(df)
    # We keep only the distances in the range [0, 21] because are the only distances possible (half cell width)
    df = df[(df.DISTANCE >= 0) & (df.DISTANCE <= 21)]
    print(f"Sono stati rimossi {hits_before - len(df)} eventi perchè avevano distanze sbagliate")

    # In this function we check if there are at least 'MIN_GOOD_CHAMBERS' chambers
    # What is a good chamber? A good chamber is when
    # the chamber has at least 'MIN_HITS_PER_CHAMBER' hits in total
    # the chamber has at least 'MIN_UNIQUE_LAYERS_HIT' layers hit
    #
    # But if we find a chamber with more than 'MAX_HITS_PER_CHAMBER' hits, then we discard the whole event
    def filter_chambers(x):
        # x is dataframe only containing the hits from the same orbit
        good_ch = 0
        # we now subset that dataframe grouping by chamber
        for ch, df_ch in x.groupby("CHAMBER"):
            # if the numbers of hits in this chamber is greater than the maximum allowed we discard
            # the whole event (if we return False, pandas know to not consider the orbit in input to
            # this function)
            if len(df_ch) > MAX_HITS_PER_CHAMBER:
                return False

            # If in this chamber there is the minimum number of hits
            if len(df_ch) >= MIN_HITS_PER_CHAMBER:
                # and if there is the minimum number of layer hit
                if len(df_ch.LAYER.value_counts()) >= MIN_UNIQUE_LAYERS_HIT:
                    # and the number if hits in the same layer is allowed
                    # then this chamber is a good chamber
                    if np.all(df_ch.LAYER.value_counts() <= MAX_HITS_PER_LAYER):
                        # we increment the good chamber count
                        good_ch += 1

        # if this event (same orbit) has at least the minimum number of good chamber then we keep them
        # otherwise we return false and filter it out
        return good_ch >= MIN_GOOD_CHAMBERS

    # We groupby orbit and then each subgroup is sent to the function to check if it's good (return True)
    # or we have to filter it out (return False)
    df = df.groupby(["ORBIT"]).filter(filter_chambers)

    # We finally sort the hits
    df = df.sort_values(by=["ORBIT", "CHAMBER", "LAYER", "CELL"])
    return df


def group_hits(df):
    n = len(df)
    matx = df.CELL_X.values.reshape(-1, 1) - df.CELL_X.values.reshape(1, -1)
    maty = df.CELL_Y.values.reshape(-1, 1) - df.CELL_Y.values.reshape(1, -1)
    # 45 approx sqrt(21^2 + (3*13)^2) = 44.29
    mat = np.sqrt(matx ** 2 + maty ** 2) < 45  # todo do not sqrt

    to_visit_global = set(range(n))
    groups = []
    while to_visit_global:
        i = to_visit_global.pop()
        group = set(np.where(mat[i, :])[0])
        to_visit_local = group - {i}
        while to_visit_local:
            j = to_visit_local.pop()
            j_nodes = np.where(mat[j, :])[0]
            group.update(j_nodes)

            if len(j_nodes) == n:
                to_visit_global.clear()
                break

            to_visit_global.remove(j)
            to_visit_local.update(to_visit_global.intersection(j_nodes))

        groups.append(df.iloc[list(group)])
    return groups


def count_hits_in_cone(cell, layer, other_cells):
    layer_map = [[0, np.array([-2, 2, -3, 1, 5, -5, -1, 3, 7])],
                 [2, np.array([0, 4, 1, 5, -1, 3, 7])],
                 [1, np.array([-2, 2, -4, 0, 4, 3, -1])],
                 [3, np.array([1, 5, -2, 2, 6, -4, 0, 4, 8])]]
    offset, data = layer_map[layer]
    data = data + (cell - offset)
    data = data[(data >= 0) & (data <= 63)]
    return np.sum([1 for c in other_cells if c in data])


def discard_obvious_points(df):
    nlayers = len(df.LAYER.value_counts())
    np.empty(shape=(len(df), 2))
    x_values = np.concatenate([df.CELL_X - df.DISTANCE, df.CELL_X + df.DISTANCE])
    y_values = np.tile(df.CELL_Y.values, 2)

    layers = np.tile(df.LAYER.values, 2)
    offsets = np.array([0, 45, 70, 95])
    matx = x_values.reshape(-1, 1) - x_values.reshape(1, -1)
    maty = y_values.reshape(-1, 1) - y_values.reshape(1, -1)
    mat = np.sqrt(matx ** 2 + maty ** 2)  # < offsets[layers]

    a = 1


def solve_layer_ambiguity(df, debug=False):
    cell_per_layers = df.LAYER.value_counts()
    if np.max(cell_per_layers) > 1:
        for layer, occ in cell_per_layers.items():
            if occ == 1:
                continue

            df_layer = df[df.LAYER == layer]
            df_olayer = df[df.LAYER != layer]
            # x_values = np.empty(len(df_layer) * 2, dtype=float)
            # x_values[0::2] = df_layer.CELL_X - df_layer.DISTANCE
            # x_values[1::2] = df_layer.CELL_X + df_layer.DISTANCE
            # y_values = np.repeat(df_layer.CELL_Y.values, repeats=2)
            x_values = np.concatenate([df_layer.CELL_X - df_layer.DISTANCE, df_layer.CELL_X + df_layer.DISTANCE])
            y_values = np.tile(df_layer.CELL_Y.values, 2)
            # x_ovalues = np.empty(len(df_olayer) * 2, dtype=float)
            # x_ovalues[0::2] = df_olayer.CELL_X - df_olayer.DISTANCE
            # x_ovalues[1::2] = df_olayer.CELL_X + df_olayer.DISTANCE
            # y_ovalues = np.repeat(df_olayer.CELL_Y.values, repeats=2)
            x_ovalues = np.concatenate([df_olayer.CELL_X - df_olayer.DISTANCE, df_olayer.CELL_X + df_olayer.DISTANCE])
            y_ovalues = np.tile(df_olayer.CELL_Y.values, 2)
            layers = np.abs(layer - np.tile(df_olayer.LAYER.values, 2).astype(int))
            offsets = np.array([0, 45, 70, 95])
            matx = (x_values.reshape(-1, 1) - x_ovalues.reshape(1, -1))
            maty = y_values.reshape(-1, 1) - y_ovalues.reshape(1, -1)
            mat = np.sqrt(matx ** 2 + maty ** 2) < offsets[layers]

            a = 1
            # best_res = 0
            # best_cells = []
            # for cell in df[df.LAYER == layer].CELL:
            #     res = count_hits_in_cone(cell, layer, df[df.LAYER != layer].CELL)
            #     if res == best_res:
            #         best_cells.append(cell)
            #     elif res > best_res:
            #         best_res = res
            #         best_cells = [cell]
            # if debug:
            #     print(f"Con il metodo del cono ho scelto la celle {best_cells}")
            #
            # best_res = 0
            # best_cells = []
            # for cell in df[df.LAYER == layer].CELL:
            #     res = count_continuous_piece(cell, df[df.LAYER != layer].CELL)
            #     if res == best_res:
            #         best_cells.append(cell)
            #     elif res > best_res:
            #         best_res = res
            #         best_cells = [cell]
            # if debug:
            #     print(f"Con il metodo del blocco ho scelto le celle {best_cells}")


def pattern_recognition(df, debug_info):
    info = []
    while len(df) >= MIN_PR_HIT:
        lr, res = simple_fit(df.CELL_X, df.CELL_Y, minimize_vertical=False, res_method="y")
        res2 = get_residuals_eucl_squared(df.CELL_Y, df.CELL_X, lr.slope, lr.intercept)
        if np.max(res) > (CELL_WIDTH * 0.75) ** 2:
            rm_index = res.idxmax()
            cell = df.loc[rm_index].CELL
            if debug_info:
                info.append((lr, [res, res2], df, cell))
                if debug_info:
                    print(f"Ho rimosso la cella {cell}")
            df = df.drop(rm_index)
        else:
            if debug_info:
                info.append((lr, [res, res2], df, None))
                return df, info
            return df
    if debug_info:
        return None, info
    else:
        return None


def count_continuous_piece(cell, other_cells):
    layer_map = [[0, np.array([4, 2, -2, -4])],
                 [2, np.array([0, 4, 6, 5, 1, -2])],
                 [1, np.array([2, 5, 3, -2, -3, -1])],
                 [3, np.array([1, 5, 7, -1])]]
    layer = get_layers(cell)
    offset, data = layer_map[layer]
    data = set(data + (cell - offset))

    scanned = []
    while True:
        to_scan = [c for c in other_cells if c in data and c not in scanned]
        if not to_scan:
            break
        for c in to_scan:
            l = get_layers(c)
            offset_, data_ = layer_map[l]
            data_ = set(data_ + (c - offset_))
            data.update(data_)
            scanned.append(c)

    return np.sum([1 for c in other_cells if c in data])


# Sceglie il gruppo 'migliore', codice brutto
def get_index_good_group(groups):
    lenghts = np.array([np.arange(len(groups)),
                        [len(a) for a in groups]]).T
    if 4 in lenghts:
        return lenghts[lenghts[:, 1] == 4][0][0]

    lenghts = lenghts[(lenghts[:, 1] <= 5) & (lenghts[:, 1] >= 3)]
    if len(lenghts) > 0:
        lenghts = lenghts[lenghts[:, 1].argsort()[::-1]]
        return lenghts[0, 0]


# Here we want to construct the track for a chamber only (the local track)
def calculate_local_track(df):
    PLOT = False

    if PLOT:
        plot_interactive2()
    orbit_groupby = df.groupby("ORBIT")

    # The tracks parameters are saved in a matrix with 6 columns (orbit, chamber, slope1, intercept1,
    # slope2, intercept2) what those are, is specified below
    tracks = np.zeros(shape=(len(orbit_groupby) * 3, 6))
    counter = 0
    for orbit, df_orbit in orbit_groupby:
        # for every orbit we group by chamber
        for ch, df_ch in df_orbit.groupby("CHAMBER"):
            # if orbit != 666316 or ch != 0:
            #     continue

            if len(df_ch) >= 8:
                continue

            groups = group_hits(df_ch)
            df_ch_index = get_index_good_group(groups)

            # discard_obvious_points(df_ch)
            # solve_layer_ambiguity(df_ch)

            if PLOT:
                plot_grouping_result(ch, groups, df_ch_index)
                graphic.wait_for_event()

            if df_ch_index is None:
                continue

            df_ch = groups[df_ch_index]

            # df_ch, pr_info = pattern_recognition(df_ch, debug_info=True)
            # if PLOT:
            #     plot_pattern_recognition_steps(pr_steps=pr_info)

            # x1 are the distances on the left side, x2 on the right side
            x1 = (df_ch.CELL_X - df_ch.DISTANCE).values
            x2 = (df_ch.CELL_X + df_ch.DISTANCE).values
            # x_cell are the x-positions of the center of the cell
            x_cell = df_ch.CELL_X.values
            # y_cell are the y-positions of the center of the cell
            y_cell = df_ch.CELL_Y.values

            # We calculate the linear regression of every possibile combination of right and left
            # res_bf is the best regression result
            res_bf, comb_bf, debug_bf = fitter.fit_by_bruteforce(x1, x2, x_cell, y_cell, debug=True)

            # we sort the result by the lowest residual square, where the residual is the euclidean
            # distance from the points to the line
            bf_lr_sorted = sorted(debug_bf[0], key=lambda x: x[2])

            # here we get the parameters of the second element of that array, so the parameters
            # of the second-best line (silver medal)
            slope_silver = bf_lr_sorted[1][0].slope
            inter_silver = bf_lr_sorted[1][0].intercept
            # comb is an array of 0 or 1, that tells us if we used the left (0) or the right (1) distance
            comb_silver = bf_lr_sorted[1][1]

            # we save the result in the tracks-matrix of the best line
            tracks[counter, 0:4] = [orbit, ch, res_bf.slope, res_bf.intercept]
            # and we add in the hits-dataframe a column with the real distance of the hits we found
            # that means that if we found right-right-left-left we get the center distance of the cell
            # and we add-add-sub-sub the distance found from the drift time
            # hit_y is not needed because is equal to the y-center of the cell
            df.loc[df_ch.index, "HIT_X"] = np.where(comb_bf == 0, x1, x2)

            # if we have the second best line (it should be always the case) we add its info
            if slope_silver is not None:
                tracks[counter, 4:] = [slope_silver, inter_silver]
                df.loc[df_ch.index, "HIT_X2"] = np.where(comb_silver == 0, x1, x2)

            counter += 1

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
                graphic.wait_for_event()

    # we convert the numpy tracks-matrix to a tracks-dataframe
    tracks = tracks[~np.all(tracks == 0, axis=1)]
    tracks = pd.DataFrame(data=tracks[tracks[:, 0] != 0],
                          columns=["ORBIT", "CHAMBER", "SLOPE", "INTERCEPT", "SLOPE2", "INTERCEPT2"])
    return df, tracks


# Here we try to combine the local track to form a global track
def calculate_global_track(df, tracks):
    PLOT = False
    if PLOT:
        plot_interactive2(landscape=False)

    # array that will contain the difference in degrees of the global track vs the local track
    # of each chamber
    columns_names = ['G_SLOPE', 'G_INTERCEPT', 'CHAMBER0', 'CHAMBER1', 'CHAMBER2', 'CHAMBER3']
    diff_angles = np.full(shape=(tracks.shape[0], len(columns_names)), fill_value=np.nan)
    # So, for each orbit
    for index, (orbit, df_orbit) in enumerate(df.groupby("ORBIT")):

        # we get the tracks associated with this orbit from the tracks-dataframe
        df_track = tracks[tracks.ORBIT == orbit]
        # we require that there are at least two local track, can be lowered to 1
        if len(df_track) < 2:
            continue

        # chambers is just a list of the chambers with hits in this event
        chambers = df_track.CHAMBER.unique().astype(int)

        # forse toglierli ancora prima, all'inizio o nel local track
        df_orbit = df_orbit[~np.isnan(df_orbit.HIT_X)]
        ch_counts = df_orbit.CHAMBER.value_counts()
        xmat = np.full(shape=(len(ch_counts), 2, max(ch_counts)), fill_value=np.nan)
        ymat = np.full(shape=(len(ch_counts), max(ch_counts)), fill_value=np.nan)
        for i, (ch, n) in enumerate(ch_counts.sort_index().items()):
            df_hits = df_orbit[df_orbit.CHAMBER == ch]
            xmat[i, 0, :n] = df_hits.HIT_X
            xmat[i, 1, :n] = df_hits.HIT_X2
            ymat[i, :n] = df_hits.CELL_Y

        # We have two lines for each chamber (the best and the 2-nd best), we try to find the best possible line
        # using a set of points for each chamber (so we try all the combinations of best and 2nd-best to
        # get the global track)
        result_list = fitter.fit_chambers_by_bruteforce(xmat, ymat)

        slopes = np.zeros(4)
        intercepts = np.zeros(4)

        # here we get put into a list the paramters of the local tracks
        slopes[chambers] = df_track.SLOPE.values
        intercepts[chambers] = df_track.INTERCEPT.values

        # we convert the local slopes into angles (degree)
        angles = np.arctan(slopes) * 180 / np.pi
        # we get the result of the best global line
        slope, intercept = result_list[0][0].slope, result_list[0][0].intercept
        # we convert the global slope in angle
        angle = np.arctan(slope) * 180 / np.pi
        # and for each chamber we save the difference
        diff_angles[index, 0:2] = slope, intercept

        for ch in chambers:
            diff_angles[index, ch+2] = angle - angles[ch]

        if PLOT:
            regr_data = [[], [], [], []]
            slopes2, intercepts2 = np.zeros(4), np.zeros(4)
            slopes2[chambers] = df_track.SLOPE2.values
            intercepts2[chambers] = df_track.INTERCEPT2.values
            for i in [0, 2, 3]:
                regr_data[i].append([slopes[i], intercepts[i]])
                regr_data[i].append([slopes2[i], intercepts2[i]])
            # regr_data[ch].append([res_bf.slope, res_bf.intercept])

            plot_event(df_orbit.CHAMBER, df_orbit.CELL, df_orbit.DISTANCE, regr_data=regr_data, landscape=False)
            x_range = np.linspace(-50, 2500, 50)
            graphic._axes.plot(x_range, x_range / slope - intercept/slope, ls="dotted", lw=1)
            # graphic._axes.plot(x_range, x_range * slope + intercept)
            graphic._axes.set_title(f"Orbit: {orbit}, ch: {ch}", y=1.0, pad=-14)
            plt.legend()
            graphic.wait_for_event()

    # now outside the cycle, we plot the histograms of the differences
    MAX_ANGLE = 15
    plt.figure(figsize=(12, 10))
    for i, ch in enumerate([0, 2, 3]):
        plt.subplot(2, 3, i + 1 + 3)
        nbins = int(np.sqrt(len(diff_angles[ch])))
        plt.hist(diff_angles[ch], bins=nbins)
        plt.title(f"differences of ch {ch}")

        values = np.array(diff_angles[ch])
        values = values[(values > -MAX_ANGLE) & (values < MAX_ANGLE)]

        plt.subplot(2, 3, i + 1)
        nbins = int(np.sqrt(len(values))) or 1
        plt.hist(values, bins=nbins)
        plt.title(f"differences of ch {ch}, zoom from -{MAX_ANGLE}° to {MAX_ANGLE}°")
    plt.show()

    return pd.DataFrame(data=diff_angles, columns=columns_names).dropna(how="all").drop('CHAMBER1', axis=1)

def get_pickled(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_pickled(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    # filenames = ["../dataset/data_000000.dat", "../dataset/data_000001.dat", "../dataset/data_000002.dat",
    #              "../dataset/data_000003.dat", "../dataset/data_000004.dat", "../dataset/data_000005.dat"]
    filenames = ["./dataset/data_000001.dat", "./dataset/data_000002.dat"]
    # filenames = ["../dataset/data_000000.dat"]
    #
    # We join the dataset (after filtering one by one (we don't want a gigantc raw df))
    big_df_filtered = None
    for filename in filenames:
        print(f"Carico il file: {filename}")
        df = load_dataframe(filename)
        print(f"Eventi Iniziali: {len(df.ORBIT.value_counts())}")
        df_filtered = manipulate_dataframe(df)
        print(f"Eventi Finali: {len(df_filtered.ORBIT.value_counts())}")

        if big_df_filtered is None:
            big_df_filtered = df_filtered
        else:
            big_df_filtered = pd.concat([big_df_filtered, df_filtered], ignore_index=True)

        print("--------------------------------------")

    print("Dataframe loaded")
    # set_pickled("./pickled/big_things.bin", big_df_filtered)
    # big_df_filtered = get_pickled("./pickled/big_things.bin")
    big_df_filtered, tracks = calculate_local_track(big_df_filtered)
    # We save the result so we can use get_pickled to avoid doing all the calculation at every startup
    # for developing-purpouse only
    # set_pickled("./pickled/big_things.bin", [big_df_filtered, tracks])
    print(big_df_filtered)
    print("\n")
    print(tracks)
    print("\n")
    # if you are working only on the global_tracks and want to avoid calculating all the time the local
    # tracks, you can comment all the code above and just run these two lines
    # big_df_filtered, tracks = get_pickled("./pickled/big_things.bin")
    res = calculate_global_track(big_df_filtered, tracks)
    print("Global tracks calculated")
    print(res)

    ### OLD STUFFS ###

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
