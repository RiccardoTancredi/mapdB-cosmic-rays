import itertools
import numpy as np
from scipy import stats


def get_yresiduals_squared(x, y, slope, intercept):
    return (x * slope + intercept - y) ** 2


def get_xresiduals_squared(x, y, slope, intercept):
    return ((y - intercept) / slope - x) ** 2


def get_residuals_eucl_squared(x, y, slope, intercept):
    # distance from the line y_ = ax_ + b from point (x, y)
    return ((slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)) ** 2


# def fit_by_pair_dist(x1, x2, x, y, debug=False, only_x=False):
#     stack = np.column_stack([x, y])
#     stack1 = np.column_stack([x1, y])
#     stack2 = np.column_stack([x2, y])
#     stack_pair = np.row_stack([stack1, stack2])
#
#     svalue1 = np.linalg.norm(stack1.reshape(-1, 1, 2) - stack_pair, axis=2) ** 2
#     svalue2 = np.linalg.norm(stack2.reshape(-1, 1, 2) - stack_pair, axis=2) ** 2
#
#     value1 = np.round(np.sum(svalue1, axis=1), 2)
#     value2 = np.round(np.sum(svalue2, axis=1), 2)
#
#     hres = np.where(value1 < value2, 0, np.where(value1 > value2, 1, 2))
#     x1_chosen = np.where(value1 < value2, x1, np.where(value1 > value2, x2, x1))
#
#     if 2 in hres:
#         hint = np.where(hres == 2, np.nan, hres)
#         print(f"Devo farmi aiutare dal brute force, hint: {hint}")
#         fit_by_bruteforce(x1, x2, x, y, hint=hint)
#
#     res_lr = stats.linregress(x1_chosen, y)
#
#     debug_data = None
#     if debug:
#         too_close = np.any(np.abs(x1 - x) < 2)
#         residuals = get_residuals_eucl_squared(x1_chosen, y, res_lr.slope, res_lr.intercept)
#         debug_data = [value1, value2, residuals, too_close]
#
#     return res_lr, hres, debug_data


def fit_by_dist(x1, x2, x, y, debug=False, only_x=False):
    if only_x:
        stack = np.column_stack([x, y])
        stack1 = np.column_stack([x1, y])
        stack2 = np.column_stack([x2, y])

        svalue1 = np.linalg.norm(stack1.reshape(-1, 1, 2) - stack, axis=2) ** 2
        svalue2 = np.linalg.norm(stack2.reshape(-1, 1, 2) - stack, axis=2) ** 2
    else:
        svalue1 = (x1.reshape(-1, 1) - x) ** 2
        svalue2 = (x2.reshape(-1, 1) - x) ** 2

    value1 = np.round(np.sum(svalue1, axis=1), 2)
    value2 = np.round(np.sum(svalue2, axis=1), 2)

    hres = np.where(value1 < value2, 0, np.where(value1 > value2, 1, 2))
    x1_chosen = np.where(value1 < value2, x1, np.where(value1 > value2, x2, x1))

    if 2 in hres:
        hint = np.where(hres == 2, np.nan, hres)
        if debug:
            print(f"Devo farmi aiutare dal brute force, hint: {hint}")
        return fit_by_bruteforce(x1, x2, x, y, hint=hint)

    res_lr = stats.linregress(x1_chosen, y)

    debug_data = None
    if debug:
        too_close = np.any(np.abs(x1 - x) < 2)
        residuals = get_residuals_eucl_squared(x1_chosen, y, res_lr.slope, res_lr.intercept)
        debug_data = [value1, value2, residuals, too_close]

    return res_lr, hres, debug_data


# res_method can be "x", "y", "xy"
# hint: array of 1, 0 or np.nan: force the alg to do only the cases that match this array:
# np.nan stands for any, so [np.nan, 1, 0, np.nan] checks only the combination with
# a 1 in the second place and a 0 in the third place
def fit_by_bruteforce(x1, x2, x, y, hint=None, res_method="xy", debug=False):
    combs = np.array(list(itertools.product([0, 1], repeat=len(x1))))
    stack = np.column_stack((x1, x2))

    if hint is not None:
        # help = np.array(help, dtype=np.int)
        good_combs_i = np.all(np.isnan(hint) | (np.array(combs) == hint), axis=1)
        combs = combs[good_combs_i, :]

    res_list = []
    for i, comb in enumerate(combs):
        x_data = stack[np.arange(len(x1)), comb]
        y_data = y
        res = stats.linregress(x_data, y_data)

        if res_method == "xy":
            residuals = get_residuals_eucl_squared(x_data, y_data, res.slope, res.intercept)
        elif res_method == "y":
            residuals = get_yresiduals_squared(x_data, y_data, res.slope, res.intercept)
        elif res_method == "x":
            residuals = get_xresiduals_squared(x_data, y_data, res.slope, res.intercept)
        else:
            raise ValueError(f"Invalid residual method: {res_method}, allowed: x, y, xy")

        residual_sum = np.sum(residuals)
        res_list.append((res, comb, residual_sum, residuals))

    best_res = min(res_list, key=lambda x: x[2])

    debug_data = None
    if debug:
        debug_data = [res_list, None, best_res[3]]
    return best_res[0], best_res[1], debug_data


if __name__ == "__main__":
    # choose_by_brute_force([1, 2, 5.5, 7], [2, 5, 6, 9], None, [1, 3, 5, 7])
    fit_by_dist([1.25, 0.75, 1.2, 0.8], [1.75, 1.25, 2.7, 1.3], [1.5, 1, 1.5, 1], [1, 2, 3, 4])
