import random
import time
from matplotlib.widgets import Button
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

_initialized = False
_rects = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_points = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_mpoints_lf = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_mpoints_rg = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_figure = None
_axes = None
_LANDSCAPE = True

_FIG_SIZE_LS = (12, 8)
_FIG_SIZE_PT = (14, 7)
_FIG_SIZE_LS_ONE = (12, 2)
_FIG_SIZE_PT_ONE = (4, 8)

_OFFSETS = {0: 219.8, 1: 977.3, 2: 1035.6, 3: 1819.8}
CELL_WIDTH = 42
CELL_HEIGHT = 13
_HALF_CW = CELL_WIDTH / 2
_HALF_CH = CELL_HEIGHT / 2


def _create_rectangles(landscape=True):
    global _initialized, _LANDSCAPE
    if _initialized:
        return

    _initialized = True
    _LANDSCAPE = landscape

    lay_off_ind = {0: 0, 1: 2, 2: 1, 3: 3}
    lay_off_left = {0: 0, 1: CELL_WIDTH * 0.5, 2: 0, 3: CELL_WIDTH * 0.5}
    for cham in [0, 2, 3]:
        offset = _OFFSETS[cham]
        for lay in range(4):
            ly_up = CELL_HEIGHT * (4 - lay) + offset
            ly_dw = CELL_HEIGHT * (4 - lay - 1) + offset
            ly_lf = lay_off_left[lay]
            for cell in range(16):
                index = lay_off_ind[lay] + cell * 4

                x = CELL_WIDTH * cell + ly_lf
                y = ly_dw
                xp = x + _HALF_CW
                yp = y + _HALF_CH
                xmp1 = xp - CELL_WIDTH * 0.25
                xmp2 = xp + CELL_WIDTH * 0.25
                ymp1, ymp2 = yp, yp
                cw, ch = CELL_WIDTH, CELL_HEIGHT

                if not landscape:
                    x, y = y, x
                    xp, yp = yp, xp
                    xmp1, ymp1 = ymp1, xmp1
                    xmp2, ymp2 = ymp2, xmp2
                    cw, ch = ch, cw

                edgecolor = "deeppink" if (index == 0) else "black"
                rect = Rectangle((x, y), width=cw, height=ch,
                                 edgecolor=edgecolor, fill=False, facecolor="lightseagreen")

                point = Circle((xp, yp), radius=1, color="red")

                point_lf = Circle((xmp1, ymp1), radius=0.5, color="lime", visible=False)
                point_rg = Circle((xmp2, ymp2), radius=0.5, color="lime", visible=False)

                _rects[cham][index] = rect
                _points[cham][index] = point
                _mpoints_lf[cham][index] = point_lf
                _mpoints_rg[cham][index] = point_rg


def _get_chambers_y_limits(chamber=None, landscape=True):
    if chamber is None:
        return 200, 1900

    min_y = _OFFSETS[chamber]
    max_y = _OFFSETS[chamber] + CELL_HEIGHT * 4
    more = 0 if landscape else 100

    return min_y - CELL_HEIGHT * 2 - more, max_y + CELL_HEIGHT * 2 + more


def _get_chambers_x_limits(chamber=None):
    return -10, 750


def create_canvas(chamber=None, landscape=True):
    _create_rectangles(landscape=landscape)
    global _axes, _figure
    if _axes:
        return

    if landscape:
        if chamber is not None:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_LS_ONE)
        else:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_LS)
        _axes.set_xlim(_get_chambers_x_limits(chamber))
        _axes.set_ylim(_get_chambers_y_limits(chamber, landscape=True))
    else:
        if chamber is not None:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_PT_ONE)
        else:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_PT)
        _axes.set_xlim(_get_chambers_y_limits(chamber, landscape=False))
        _axes.set_ylim(_get_chambers_x_limits(chamber))

    for cham in [0, 2, 3]:
        x = _get_chambers_x_limits(cham)[1] - 40
        y = _get_chambers_y_limits(cham)[0] + CELL_HEIGHT * 2
        if landscape:
            _axes.text(x, y, f"CH {cham}", rotation="vertical")
        else:
            _axes.text(y, x, f"CH {cham}", rotation="horizontal")

    for i in range(len(_rects)):
        for j in range(len(_rects[i])):
            if _rects[i][j]:
                _axes.add_patch(_rects[i][j])
                _axes.add_patch(_points[i][j])
                _axes.add_patch(_mpoints_lf[i][j])
                _axes.add_patch(_mpoints_rg[i][j])
    # plt.tight_layout()
    plt.tight_layout(pad=1.5)

    if not landscape:
        _axes.invert_yaxis()
    # plt.show(block=False)


def _reset_cells():
    for i in range(len(_rects)):
        for j in range(len(_rects[i])):
            if _rects[i][j]:
                _rects[i][j].set(fill=False)
                _mpoints_lf[i][j].set(visible=False)
                _mpoints_rg[i][j].set(visible=False)


def plot_event(chambers, cells, distances=None):
    create_canvas()
    _reset_cells()
    if distances is None:
        distances = [None] * len(chambers)

    for cham, cell, dist in zip(chambers, cells, distances):
        point = _points[cham][cell]
        _rects[cham][cell].set(fill=True)
        if dist:
            cx, cy = point.get_center()
            if _LANDSCAPE:
                _mpoints_lf[cham][cell].set(center=(cx - dist, cy), visible=True)
                _mpoints_rg[cham][cell].set(center=(cx + dist, cy), visible=True)
            else:
                _mpoints_lf[cham][cell].set(center=(cx, cy - dist), visible=True)
                _mpoints_rg[cham][cell].set(center=(cx, cy + dist), visible=True)

    # plt.show(block=False)
    # plt.pause(0.01)


class _InteractiveHelper:
    def __init__(self):
        self.index = 0
        self.running = False
        self.timer = False
        self.dfs = None
        self.add_info = None
        self.chamber_i = None
        self.chambers = [0, 2, 3]

    def load_data(self, dfs, add_info=None):
        self.reset()
        self.dfs = dfs
        self.add_info = add_info

    def reset(self):
        self.dfs = None
        self.index = 0
        self.running = False
        self.timer = None

    def has_next(self):
        return self.index < len(self.dfs) - 1

    def has_prev(self):
        return self.index > 0

    def get_next(self):
        self.index += 1
        return self.index

    def get_prev(self):
        self.index -= 1
        return self.index

    def get_title(self):
        text = f"ORBIT_ID: {self.dfs[self.index]['ORBIT'].to_numpy()[0]}, INDEX: {self.index}"
        if self.running:
            text += " AUTO: TRUE"
        if self.add_info is not None:
            text += f" INFO: {self.add_info[self.index]}"
        return text


_int_help = _InteractiveHelper()


# dfs: list of dataframes
def plot_interactive(dfs, landscape=True, add_info=None):
    create_canvas(landscape=landscape)
    _reset_cells()

    def next_ch(event):
        if _int_help.chamber_i is None:
            _int_help.chamber_i = 0

        elif _int_help.chamber_i < len(_int_help.chambers) - 1:
            _int_help.chamber_i += 1

        chamb_n = _int_help.chambers[_int_help.chamber_i]
        if landscape:
            _figure.set_figwidth(_FIG_SIZE_LS_ONE[0])
            _figure.set_figheight(_FIG_SIZE_LS_ONE[1])
            _axes.set_xlim(_get_chambers_x_limits(chamb_n))
            _axes.set_ylim(_get_chambers_y_limits(chamb_n))
        else:
            _figure.set_figwidth(_FIG_SIZE_PT_ONE[0])
            _figure.set_figheight(_FIG_SIZE_PT_ONE[1])
            _axes.set_xlim(_get_chambers_y_limits(chamb_n, landscape=False))
            _axes.set_ylim(_get_chambers_x_limits(chamb_n))
            _axes.invert_yaxis()

        plt.draw()

    def prev_ch(event):
        if _int_help.chamber_i is None:
            _int_help.chamber_i = 0

        elif _int_help.chamber_i > 0:
            _int_help.chamber_i -= 1

        chamb_n = _int_help.chambers[_int_help.chamber_i]
        if landscape:
            _figure.set_figwidth(_FIG_SIZE_LS_ONE[0])
            _figure.set_figheight(_FIG_SIZE_LS_ONE[1])
            _axes.set_xlim(_get_chambers_x_limits(chamb_n))
            _axes.set_ylim(_get_chambers_y_limits(chamb_n))
        else:
            _figure.set_figwidth(_FIG_SIZE_PT_ONE[0])
            _figure.set_figheight(_FIG_SIZE_PT_ONE[1])
            _axes.set_xlim(_get_chambers_y_limits(chamb_n, landscape=False))
            _axes.set_ylim(_get_chambers_x_limits(chamb_n))
            _axes.invert_yaxis()
        plt.draw()

    def reset_ch(event):
        if _int_help.chamber_i is not None:
            _int_help.chamber_i = None
            if landscape:
                _figure.set_figwidth(_FIG_SIZE_LS[0])
                _figure.set_figheight(_FIG_SIZE_LS[1])
                _axes.set_xlim(_get_chambers_x_limits())
                _axes.set_ylim(_get_chambers_y_limits())
            else:
                _figure.set_figwidth(_FIG_SIZE_PT[0])
                _figure.set_figheight(_FIG_SIZE_PT[1])
                _axes.set_xlim(_get_chambers_y_limits())
                _axes.set_ylim(_get_chambers_x_limits())
                _axes.invert_yaxis()

            plt.draw()

    def next_(event):
        if _int_help.has_next():
            index = _int_help.get_next()
            plot_event(dfs[index].CHAMBER, dfs[index].CELL, dfs[index].DISTANCE)
            _axes.set_title(_int_help.get_title())
            plt.draw()

        elif _int_help.running:
            _int_help.running = False
            _axes.set_title(_int_help.get_title())
            plt.draw()
            return False

    def previous(event):
        if _int_help.has_prev():
            index = _int_help.get_prev()
            plot_event(dfs[index].CHAMBER, dfs[index].CELL, dfs[index].DISTANCE)
            _axes.set_title(_int_help.get_title())
            plt.draw()

    def auto_go(event):
        if not _int_help.running:
            timer = _figure.canvas.new_timer(interval=800)
            timer.add_callback(next_, _axes)
            timer.start()

            _int_help.running = True
            _int_help.timer = timer
        else:
            _int_help.running = False
            _int_help.timer.stop()
            _axes.set_title(_int_help.get_title())
            plt.draw()

    _int_help.load_data(dfs, add_info=add_info)
    _axes.set_title(_int_help.get_title())
    plot_event(dfs[0].CHAMBER, dfs[0].CELL, dfs[0].DISTANCE)

    if landscape:
        axprev = _figure.add_axes([0.7, 0.75, 0.1, 0.075])
        axnext = _figure.add_axes([0.8, 0.75, 0.1, 0.075])
        axautogo = _figure.add_axes([0.9, 0.75, 0.1, 0.075])
        axchprev = _figure.add_axes([0.1, 0.75, 0.1, 0.075])
        axchnext = _figure.add_axes([0.2, 0.75, 0.1, 0.075])
        axchreset = _figure.add_axes([0.3, 0.75, 0.1, 0.075])
    else:
        h = 0.9
        p = (0.1, 0.03)
        axprev = _figure.add_axes([0.8, h + 0.03, *p])
        axnext = _figure.add_axes([0.8, h, *p])
        axautogo = _figure.add_axes([0.8, h - 0.03, *p])
        axchprev = _figure.add_axes([0.2, h + 0.03, *p])
        axchnext = _figure.add_axes([0.2, h, *p])
        axchreset = _figure.add_axes([0.2, h - 0.03, *p])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(previous)
    bautogo = Button(axautogo, 'Auto Go')
    bautogo.on_clicked(auto_go)

    bchnext = Button(axchprev, 'CH ^' if landscape else "CH >")
    bchnext.on_clicked(next_ch)
    bchprev = Button(axchnext, 'CH V' if landscape else "CH <")
    bchprev.on_clicked(prev_ch)
    bchreset = Button(axchreset, 'CH Reset')
    bchreset.on_clicked(reset_ch)

    plt.show()


if __name__ == "__main__":
    pass
    # create_canvas(chamber=None, landscape=True)
    # plot_event([0], [0], [10])
    # plt.show()
    # input()
    # plot_event([0, 0, 0, 0], [8, 12, 20, 26], [0, 10, 20, 15])
    # plot_interactive()
