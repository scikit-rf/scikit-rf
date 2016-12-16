import bokeh

import bokeh.plotting as plotting
import bokeh.models as models

plotting.output_notebook()

from ..network import Network


lime_green = "#00FF00"
green = "#00AA00"
cyan = "#00FFFF"
blue = "#0000FF"
red = "#FF0000"
magenta = "#FF00FF"
yellow = "#FFFF00"
purple = "#990099"
orange = "#FFA500"


def trace_color_cycle(start = 0):
    """
    :start n: int
    :return:
    """
    count = start
    colors = [blue, red, purple, green]
    num = len(colors)
    while count < 1000:
        yield colors[count % num]
        count += 1


def plot_s_db(ntwk):
    """
    :type ntwk: Network
    :return: bokeh'bokeh.plotting.figure.Figure
    """

    colors = trace_color_cycle()

    fig = plotting.figure(
        title=ntwk.name,
        height=350, width=800,
        x_axis_label="frequency ({:s})".format(ntwk.frequency.unit),
        y_axis_label="decibels (dB)",
        tools="resize, pan, wheel_zoom, box_zoom, save, reset",
        toolbar_location="above",
        toolbar_sticky=True
    )

    labels = []
    glpyhs = []

    for t in range(ntwk.nports):
        for r in range(ntwk.nports):
            glpyhs.append(fig.line(ntwk.frequency.f_scaled, ntwk.s_db[:, r, t], line_color=next(colors)))
            labels.append("S{:d}{:d}".format(t + 1, r + 1))

    legend_items = []
    for label, glyph in zip(labels, glpyhs):
        legend_items.append((label, [glyph]))

    legend = models.Legend(items=legend_items, location=(0, -30))

    fig.add_layout(legend, 'right')

    plotting.show(fig)
    return fig


Network.plot_s_db = plot_s_db
