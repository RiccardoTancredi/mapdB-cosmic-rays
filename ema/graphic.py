import pandas as pd
from matplotlib import _blocking_input
from matplotlib.widgets import Button
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.text import Text

_initialized = False
_rects = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_points = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_mpoints_lf = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_mpoints_rg = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_cell_numbers = [[None] * 64, [None] * 64, [None] * 64, [None] * 64]
_figure = None
_axes = None
_text = None
_consuming_axes = []
_allowed_axes = []

_FACECOLOR = "lightseagreen"

_DPI = 128
_FIG_SIZE_LS = (12, 8)
_FIG_SIZE_PT = (14, 7)
_FIG_SIZE_LS_ONE = (13, 4)
_FIG_SIZE_PT_ONE = (4, 8)

_OFFSETS = {0: 219.8, 1: 977.3, 2: 1035.6, 3: 1819.8}
CELL_WIDTH = 42
CELL_HEIGHT = 13
_HALF_CW = CELL_WIDTH / 2
_HALF_CH = CELL_HEIGHT / 2


def _create_rectangles(landscape=True):
    global _initialized
    if _initialized:
        return

    _initialized = True

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
                                 edgecolor=edgecolor, fill=False, facecolor=_FACECOLOR)

                point = Circle((xp, yp), radius=1, color="red", zorder=100)

                point_lf = Circle((xmp1, ymp1), radius=0.5, color="lime", visible=False, zorder=100)
                point_rg = Circle((xmp2, ymp2), radius=0.5, color="lime", visible=False, zorder=100)

                _rects[cham][index] = rect
                _points[cham][index] = point
                _mpoints_lf[cham][index] = point_lf
                _mpoints_rg[cham][index] = point_rg

                if landscape:
                    _cell_numbers[cham][index] = (xp - _HALF_CW, yp - _HALF_CH + 1, index)


def _get_chambers_y_limits(chamber=None, landscape=True):
    if chamber is None:
        return 200, 1900

    min_y = _OFFSETS[chamber]
    max_y = _OFFSETS[chamber] + CELL_HEIGHT * 4
    more = 75 if landscape else 75

    return min_y - CELL_HEIGHT * 2 - more, max_y + CELL_HEIGHT * 2 + more


def _get_chambers_x_limits(chamber=None, landscape=True):
    return -10, 750


def _set_canvas(chamber, landscape):
    if landscape:
        sizes = _FIG_SIZE_LS_ONE if chamber is not None else _FIG_SIZE_LS
        _figure.set_figwidth(sizes[0])
        _figure.set_figheight(sizes[1])
        _axes.set_xlim(_get_chambers_x_limits(chamber))
        _axes.set_ylim(_get_chambers_y_limits(chamber))
    else:
        sizes = _FIG_SIZE_PT_ONE if chamber is not None else _FIG_SIZE_PT
        _figure.set_figwidth(sizes[0])
        _figure.set_figheight(sizes[1])
        _axes.set_xlim(_get_chambers_y_limits(chamber, landscape=False))
        _axes.set_ylim(_get_chambers_x_limits(chamber))
        _axes.invert_yaxis()


def create_canvas(chamber=None, landscape=True):
    _create_rectangles(landscape=landscape)
    global _axes, _figure
    if _axes:
        return

    if landscape:
        if chamber is not None:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_LS_ONE, dpi=_DPI)
        else:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_LS, dpi=_DPI)
        _axes.set_xlim(_get_chambers_x_limits(chamber))
        _axes.set_ylim(_get_chambers_y_limits(chamber, landscape=True))
    else:
        if chamber is not None:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_PT_ONE, dpi=_DPI)
        else:
            _figure, _axes = plt.subplots(figsize=_FIG_SIZE_PT, dpi=_DPI)
        _axes.set_xlim(_get_chambers_y_limits(chamber, landscape=False))
        _axes.set_ylim(_get_chambers_x_limits(chamber))

    global _text
    _text = _axes.text(0, 0, s="")

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
                if landscape:
                    xt, yt, txt = _cell_numbers[i][j]
                    _axes.text(x=xt, y=yt, s=txt, size=5)
    # plt.tight_layout()
    plt.tight_layout(pad=1.5)

    if not landscape:
        _axes.invert_yaxis()
    # plt.show(block=False)


def _reset_cells():
    for i in range(len(_rects)):
        for j in range(len(_rects[i])):
            if _rects[i][j]:
                _rects[i][j].set(fill=False, facecolor=_FACECOLOR)
                _mpoints_lf[i][j].set(visible=False)
                _mpoints_rg[i][j].set(visible=False)

    if _axes.get_legend():
        _axes.get_legend().remove()
        _axes.set_prop_cycle(None)

    for line in list(_axes.lines):
        line.remove()


def _draw_event(chambers, cells, distances=None, regr_data=None, landscape=True, reset_canvas=True, facecolor=None):
    if reset_canvas:
        _reset_cells()

    kwargs = {}
    if facecolor is not None:
        kwargs["facecolor"] = facecolor

    for cham, cell in zip(chambers, cells):
        _rects[cham][cell].set(fill=True, **kwargs)

    if regr_data is not None:
        colors = ["blue", "green", "orange", "red"]
        for cham in [0, 2, 3]:
            if landscape:
                x_range = np.linspace(*_get_chambers_x_limits(cham), 50)
            else:
                x_range = np.linspace(*_get_chambers_y_limits(cham), 50)

            for i, (slope, intercept) in enumerate(regr_data[cham]):
                if landscape:
                    _axes.plot(x_range, x_range * slope + intercept, label=i, color=colors[i])
                else:
                    _axes.plot(x_range, x_range * 1 / slope - intercept / slope, label=i, color=colors[i])

    if len(_axes.lines) > 0:
        _axes.legend()

    if distances is not None:
        for cham, cell, dist in zip(chambers, cells, distances):
            point = _points[cham][cell]
            cx, cy = point.get_center()
            if landscape:
                _mpoints_lf[cham][cell].set(center=(cx - dist, cy), visible=True)
                _mpoints_rg[cham][cell].set(center=(cx + dist, cy), visible=True)
            else:
                _mpoints_lf[cham][cell].set(center=(cx, cy - dist), visible=True)
                _mpoints_rg[cham][cell].set(center=(cx, cy + dist), visible=True)


def write_text(text, x=None, y=None, center=None):
    if center:
        xlim = _axes.get_xlim()
        ylim = _axes.get_ylim()
        x = xlim[0] + (xlim[1] - xlim[0]) * 0.5 - len(text) * 3
        y = ylim[0] + (ylim[1] - ylim[0]) * 0.8
        # print(xlim, ylim)

    kwargs = {}
    if x is not None:
        kwargs["x"] = x
    if y is not None:
        kwargs["y"] = y
    _text.set(text=text, **kwargs)


def clear_text():
    write_text(text="")


# def _add_control_buttons():
#     if True:
#         axnext = _figure.add_axes([0.8, 0.75, 0.1, 0.075])
#
#     bnext = Button(axnext, 'Next')
#     bnext.on_clicked(lambda x: True)
#
#     _s.extend([bnext])
#     _consuming_axes.append(axnext)
#     # fig.axes


def plot_pattern_recognition_steps(pr_steps, landscape=True):
    chamber = None
    for lr, res, df, cell in pr_steps:
        print(list(df.CELL))
        for r in res:
            print(list(np.round(np.sqrt(r), 2)))
        chamber = int(df.CHAMBER.iloc[0])
        plot_event(df.CHAMBER, df.CELL, df.DISTANCE, focus_chamber=chamber, landscape=landscape)
        x = np.linspace(-10, 2500, 40)
        y = lr.slope * np.linspace(-10, 2500, 40) + lr.intercept
        _axes.plot(y, x)

        if cell:
            _rects[chamber][int(cell)].set(facecolor="lightcoral", fill=True)
        wait_for_event()


def plot_grouping_result(chamber, groups, index_good, text=None):
    if index_good is not None:
        plot_event(groups[index_good].CHAMBER, groups[index_good].CELL, groups[index_good].DISTANCE,
                   focus_chamber=chamber)
    else:
        plot_event([], [], None, focus_chamber=chamber)

    colors = ["gold", "violet", "royalblue", "coral", "silver", "bisque"]
    for i, group in enumerate(groups):
        if i == index_good:
            continue
        plot_event(group.CHAMBER, group.CELL, group.DISTANCE, focus_chamber=chamber, reset_canvas=False,
                   facecolor=colors[i])


def plot_event(chambers, cells, distances=None, regr_data=None, focus_chamber=None, landscape=True, facecolor=None,
               reset_canvas=True):
    create_canvas(landscape=landscape)
    _draw_event(chambers, cells, distances, regr_data, landscape=landscape, reset_canvas=reset_canvas,
                facecolor=facecolor)

    if focus_chamber is not None:
        _set_canvas(focus_chamber, landscape=landscape)


class _InteractiveHelper:
    def __init__(self):
        self.index = 0
        self.running = False
        self.timer = False
        self.dfs = None
        self.add_info = None
        self.regr_data = None
        self.chamber_i = None
        self.chambers = [0, 2, 3]

    def load_data(self, dfs, add_info=None, regr_data=None):
        self.reset()
        self.dfs = dfs
        self.add_info = add_info
        self.regr_data = regr_data

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
        try:
            text = f"ORBIT_ID: {self.dfs[self.index].ORBIT.to_numpy()[0]}, INDEX: {self.index}"
            if self.running:
                text += " AUTO: TRUE"
            if self.add_info is not None:
                text += f" INFO: {self.add_info[self.index]}"
        except Exception as e:
            text = "CIAO"
        return text


_int_help = _InteractiveHelper()

# dummy object to save the reference of the matplotlib buttons, otherwise they get garbage collected
_s = []


# def plot_simple_interactive(chambers, cells, distances=None, landscape=True):
#     df = pd.DataFrame(data=[chambers, cells, distances], columns=["CHAMBER", "CELL", "DISTANCE"])
#     plot_interactive([df], landscape=landscape)

def wait_for_event():
    event = None

    def handler(ev):
        nonlocal event
        event = ev
        for ax in _allowed_axes:
            if ax == ev.inaxes:
                _figure.canvas.stop_event_loop()
                return

    _blocking_input.blocking_input_loop(
        _figure, ["button_press_event", "key_press_event"], -1, handler)

    return None if event is None else event.name == "key_press_event"


def plot_interactive2(landscape=True):
    create_canvas(landscape=landscape)

    def ch0(event):
        _set_canvas(0, landscape)
        plt.draw()

    def ch2(event):
        _set_canvas(2, landscape)
        plt.draw()

    def ch3(event):
        _set_canvas(3, landscape)
        plt.draw()

    def reset_ch(event):
        _set_canvas(None, landscape)
        plt.draw()

    def next_(event):
        pass

    axnext = _figure.add_axes([0.8, 0.75, 0.1, 0.075])
    axch0 = _figure.add_axes([0.1, 0.85, 0.1, 0.075])
    axch2 = _figure.add_axes([0.2, 0.85, 0.1, 0.075])
    axch3 = _figure.add_axes([0.1, 0.75, 0.1, 0.075])
    axchreset = _figure.add_axes([0.2, 0.75, 0.1, 0.075])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_)

    bch0 = Button(axch0, 'CH 0')
    bch0.on_clicked(ch0)
    bch2 = Button(axch2, 'CH 2')
    bch2.on_clicked(ch2)
    bch3 = Button(axch3, 'CH 3')
    bch3.on_clicked(ch3)
    bchreset = Button(axchreset, 'CH Reset')
    bchreset.on_clicked(reset_ch)

    _s.extend([bnext, bch0, bch2, bch3, bchreset])
    _consuming_axes.extend([axch0, axch2, axch3, axchreset])
    _allowed_axes.append(axnext)


# dfs: list of dataframes
def plot_interactive(dfs, landscape=True, add_info=None, regr_data=None):
    regr_data = regr_data if regr_data is not None else [None] * len(dfs)
    create_canvas(landscape=landscape)
    _reset_cells()

    def next_ch(event):
        if _int_help.chamber_i is None:
            _int_help.chamber_i = 0

        elif _int_help.chamber_i < len(_int_help.chambers) - 1:
            _int_help.chamber_i += 1

        chamb_n = _int_help.chambers[_int_help.chamber_i]
        _set_canvas(chamb_n, landscape)

        plt.draw()

    def prev_ch(event):
        if _int_help.chamber_i is None:
            _int_help.chamber_i = 0

        elif _int_help.chamber_i > 0:
            _int_help.chamber_i -= 1

        chamb_n = _int_help.chambers[_int_help.chamber_i]
        _set_canvas(chamb_n, landscape)
        plt.draw()

    def reset_ch(event):
        if _int_help.chamber_i is not None:
            _int_help.chamber_i = None
            _set_canvas(None, landscape)
            plt.draw()

    def next_(event):
        if _int_help.has_next():
            index = _int_help.get_next()
            _draw_event(dfs[index].CHAMBER, dfs[index].CELL, dfs[index].DISTANCE, regr_data=regr_data[index],
                        landscape=landscape)
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
            _draw_event(dfs[index].CHAMBER, dfs[index].CELL, dfs[index].DISTANCE, regr_data=regr_data[index],
                        landscape=landscape)
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

    _int_help.load_data(dfs, add_info=add_info, regr_data=regr_data)
    _axes.set_title(_int_help.get_title())

    plot_event(dfs[0].CHAMBER, dfs[0].CELL, dfs[0].DISTANCE, regr_data=regr_data[0])

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

    _s.extend([bnext, bprev, bautogo, bchnext, bchprev, bchreset])
    # _consuming_axes.extend([axnext, axprev])


if __name__ == "__main__":
    pass
    # create_canvas(chamber=None, landscape=True)
    # plot_event([0], [0], [10])
    # plt.show()
    # input()
    # plot_event([0, 0, 0, 0], [8, 12, 20, 26], [0, 10, 20, 15])
    # plot_interactive()
