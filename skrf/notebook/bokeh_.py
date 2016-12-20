from bokeh import models, plotting

from .utils import trace_color_cycle
from .. import network

plotting.output_notebook()

Y_LABEL_DICT = network.Y_LABEL_DICT  # type: dict
PRIMARY_PROPERTIES = network.PRIMARY_PROPERTIES  # type: dict
COMPONENT_FUNC_DICT = network.COMPONENT_FUNC_DICT  # type: dict

default_kwargs = {
    'primary_property': "s",
    'property_type': "db",
    'show': True,
    'fig': None
}


def plot_rectangular(ntwk, **kwargs):
    """
    :type ntwk: Network
    :return: plotting.figure.Figure
    """

    fig = kwargs.get("fig", None)
    show = kwargs.get("show", True)

    primary_property = kwargs.get("primary_property", "s")
    property_type = kwargs.get("property_type", "db")

    colors = trace_color_cycle()

    if type(fig) is not plotting.Figure:
        fig = plotting.figure(
            title=ntwk.name,
            height=350, width=800,
            x_axis_label="frequency ({:s})".format(ntwk.frequency.unit),
            y_axis_label=Y_LABEL_DICT[property_type],
            tools="resize, pan, wheel_zoom, box_zoom, save, reset",
            toolbar_location="above",
            toolbar_sticky=True
        )

    labels = []
    glpyhs = []

    for n in range(ntwk.nports):
        for m in range(ntwk.nports):
            x = ntwk.frequency.f_scaled
            y = getattr(ntwk, primary_property + "_" + property_type)[:, m, n]
            glpyhs.append(fig.line(x, y, line_color=next(colors)))
            labels.append("S{:d}{:d}".format(n + 1, m + 1))

    legend_items = []
    for label, glyph in zip(labels, glpyhs):
        legend_items.append((label, [glyph]))

    legend = models.Legend(items=legend_items, location=(0, -30))

    fig.add_layout(legend, 'right')

    if show:
        plotting.show(fig)

    return fig


def plot_polar():
    pass  # not native to bokeh, but I have seen some hacks to do this.  Smith chart may be tricky


def use_bokeh():
    for p in PRIMARY_PROPERTIES:
        for t in COMPONENT_FUNC_DICT.keys():
            attribute_name = "plot_{:s}_{:s}".format(p, t)

            def gen_plot_function(p, t):
                def plot_function(ntwk, **kwargs):
                    kwargs["primary_property"] = p
                    kwargs["property_type"] = t
                    plot_rectangular(ntwk, **kwargs)
                return plot_function

            setattr(network.Network, attribute_name, gen_plot_function(p, t))

use_bokeh()  # this function can be called again if we need to switch plotting engines for some reason
