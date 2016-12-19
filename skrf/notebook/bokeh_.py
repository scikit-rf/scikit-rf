import copy

import bokeh
import bokeh.plotting as plotting
import bokeh.models as models

plotting.output_notebook()

from .. import network


lime_green = "#00FF00"
green = "#00AA00"
cyan = "#00FFFF"
blue = "#0000FF"
red = "#FF0000"
magenta = "#FF00FF"
yellow = "#FFFF00"
purple = "#990099"
orange = "#FFA500"

YLABELS = network.Y_LABEL_DICT
PRIMARY_PROPERTIES = network.PRIMARY_PROPERTIES
COMPONENT_FUNC_DICT = network.COMPONENT_FUNC_DICT


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


default_kwargs = {
    'primary_property': "s",
    'property_type': "db",
    'show': True,
    'fig': None,
    'subsets': {"s": {}, "y": {}, "a": {}, "z": {}}
}


def plot_rectangular(ntwk, **kwargs):
    """
    :type ntwk: Network
    :type primary_property: str
    :type property_type: str
    :type fig: bokeh.plotting.figure.Figure
    :type show: bool
    :return: bokeh.plotting.figure.Figure
    """

    fig = kwargs.get("fig", None)
    show = kwargs.get("show", True)\

    primary_property = kwargs.get("primary_property", "s")
    property_type = kwargs.get("property_type", "db")
    PROPERTY = primary_property + "_" + property_type

    colors = trace_color_cycle()

    if type(fig) is not bokeh.plotting.Figure:
        fig = plotting.figure(
            title=ntwk.name,
            height=350, width=800,
            x_axis_label="frequency ({:s})".format(ntwk.frequency.unit),
            y_axis_label=YLABELS[property_type],
            tools="resize, pan, wheel_zoom, box_zoom, save, reset",
            toolbar_location="above",
            toolbar_sticky=True
        )

    labels = []
    glpyhs = []

    for n in range(ntwk.nports):
        for m in range(ntwk.nports):
            X = ntwk.frequency.f_scaled
            Y = getattr(ntwk, PROPERTY)[:, m, n]
            glpyhs.append(fig.line(X, Y, line_color=next(colors)))
            labels.append("S{:d}{:d}".format(n + 1, m + 1))

    legend_items = []
    for label, glyph in zip(labels, glpyhs):
        legend_items.append((label, [glyph]))

    legend = models.Legend(items=legend_items, location=(0, -30))

    fig.add_layout(legend, 'right')

    if show: plotting.show(fig)

    return fig


for p in PRIMARY_PROPERTIES:
    for t in COMPONENT_FUNC_DICT.keys():
        attribute_name = "plot_{:s}_{:s}".format(p, t)

        def plot_function(p, t):
            def inner(ntwk, **kwargs):
                kwargs["primary_property"] = p
                kwargs["property_type"] = t
                plot_rectangular(ntwk, **kwargs)
            return inner

        setattr(network.Network, attribute_name, plot_function(p, t))
