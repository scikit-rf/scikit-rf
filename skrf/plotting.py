"""
plotting (:mod:`skrf.plotting`)
========================================


This module provides general plotting functions.

Plots and Charts
------------------

.. autosummary::
    :toctree: generated/

    smith
    plot_smith
    plot_rectangular
    plot_polar
    plot_complex_rectangular
    plot_complex_polar
    plot_it_all

    plot_minmax_bounds_component
    plot_minmax_bounds_s_db
    plot_minmax_bounds_s_db10
    plot_minmax_bounds_s_time_db

    plot_uncertainty_bounds_component
    plot_uncertainty_bounds_s_db
    plot_uncertainty_bounds_s_time_db

    plot_passivity
    plot_logsigma

    plot_contour

Backend selection
-----------------
``skrf.plotting.options.backend`` holds the global plotting backend:
``"matplotlib"`` (default) or ``"plotly"`` (needs
``pip install scikit-rf[plotly]``). The general-purpose plot methods also
take a per-call ``backend=`` argument.

With plotly they return a ``plotly.graph_objects.Figure``; pass it as ``fig``
to add traces to it. ``ax`` takes matplotlib Axes only, and ``fig`` plotly
Figures only. Neither selects the backend: one that does not belong to the
backend of the call raises a ``TypeError``. Likewise, keyword arguments go to the
library of the backend, so plotly takes plotly keywords such as ``line=dict(...)``,
not matplotlib ones such as ``linewidth``. ``label`` works with both, and names
the trace with plotly. Of the convenience functions below, `subplot_params`,
`shade_bands` and `signature` support plotly, as does `plot_vector`; the
others always use matplotlib.

Convenience plotting functions
-------------------------------
.. autosummary::
    :toctree: generated/

    stylely
    subplot_params
    shade_bands
    save_all_figs
    scale_frequency_ticks
    add_markers_to_lines
    legend_off
    func_on_all_figs
    scrape_legend
    signature

"""
from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from functools import partial, wraps
from logging import getLogger
from numbers import Number
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .constants import ErrorFunctionsT, NumberLike, PrimaryPropertiesT
    from .frequency import Frequency
    from .network import Network
    from .networkSet import NetworkSet

import warnings

import numpy as np

from . import mathFunctions as mf
from .util import now_string_2_dt

logger = getLogger(__name__)

SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'
SI_CONVERSION = {key: 10**((8-i)*3) for i, key in enumerate(SI_PREFIXES_ASCII)}


def plotting_available() -> bool:
    result = False
    with suppress(ImportError):
        import matplotlib  # noqa: F401
        result = True
    return result


_PLOTTING_BACKENDS = ('matplotlib', 'plotly')


def _validate_backend(backend: str) -> str:
    if backend not in _PLOTTING_BACKENDS:
        raise ValueError(f"Unknown plotting backend {backend!r}, choose one of {_PLOTTING_BACKENDS}")
    return backend


class _PlottingOptions:
    """Plotting settings: `backend` is 'matplotlib' (default) or 'plotly'."""

    def __init__(self):
        self._backend = 'matplotlib'

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = _validate_backend(value)


# global plotting settings
options = _PlottingOptions()


def _is_plotly_figure(ax) -> bool:
    # checked by name so that plotly is not imported
    return isinstance(ax, _PlotlyAxes) or any(c.__module__.startswith('plotly.') and c.__name__ == 'Figure'
                                              for c in type(ax).__mro__)


def _resolve_backend(backend: str | None = None, ax=None) -> str:
    """
    Backend for one plot call: `backend` if given, else the global option.

    Raises if `ax` does not belong to that backend.
    """
    backend = options.backend if backend is None else _validate_backend(backend)
    if ax is not None and (backend == 'plotly') != _is_plotly_figure(ax):
        raise TypeError(f"{type(ax).__name__} cannot be used with backend={backend!r}")
    return backend


_in_plot_call = ContextVar('_in_plot_call', default=False)
_MATPLOTLIB_KEYWORDS = {'alpha', 'c', 'color', 'linestyle', 'linewidth', 'ls', 'lw', 'markersize', 'ms'}


def _fig_kwarg(func: Callable) -> Callable:
    """
    Decorator adding the `fig` keyword, the plotly Figure to draw on, for an `ax` that takes matplotlib Axes only.

    It also rejects matplotlib keywords with the plotly backend. The decorated function gets `fig` as `ax`, and can
    pass it on as `ax`, with matplotlib keywords that are translated for plotly, to other decorated functions.
    """
    parameters = inspect.signature(func).parameters
    has_ax_position = 'ax' in parameters and parameters['ax'].kind is parameters['ax'].POSITIONAL_OR_KEYWORD
    ax_index = list(parameters).index('ax') if has_ax_position else None
    matplotlib_keywords = _MATPLOTLIB_KEYWORDS - set(parameters)

    @wraps(func)
    def wrapper(*args, fig=None, **kwargs):
        positional = has_ax_position and ax_index < len(args)
        ax = args[ax_index] if positional else kwargs.get('ax')
        if fig is not None:
            if ax is not None:
                raise TypeError("Pass either ax or fig, not both")
            if not _is_plotly_figure(fig):
                raise TypeError(f"fig must be a plotly Figure, not {type(fig).__name__}, pass matplotlib Axes as ax")
            if positional:
                args = (*args[:ax_index], fig, *args[ax_index + 1:])
            else:
                kwargs['ax'] = fig
        if not _in_plot_call.get() and not isinstance(ax, _PlotlyAxes):
            if fig is None and _is_plotly_figure(ax):
                raise TypeError("ax must be matplotlib Axes, pass a plotly Figure as fig")
            keywords = matplotlib_keywords & kwargs.keys()
            if isinstance(kwargs.get('marker'), str):
                keywords = keywords | {'marker'}
            if keywords and _resolve_backend(kwargs.get('backend'), ax if fig is None else fig) == 'plotly':
                raise TypeError(f"Matplotlib keywords {sorted(keywords)} are not supported by the plotly backend, "
                                "use plotly keywords such as line=dict(...) or marker=dict(...)")
        token = _in_plot_call.set(True)
        try:
            return func(*args, **kwargs)
        finally:
            _in_plot_call.reset(token)

    return wrapper


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError("The plotly backend requires plotly: pip install scikit-rf[plotly]") from err
    return go


def _get_axes(ax: Axes | None = None) -> Axes:
    """Return `ax`, or the current matplotlib axes if it is None."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RuntimeError("Plotting is not available") from err
    return plt.gca() if ax is None else ax


def figure(*args, **kwargs) -> Figure:
    """
    Wraps the matplotlib figure call and raises if not available.

    Raises
    ------
    RuntimeError
        When trying to get subplots without matplotlib installed.
    """

    try:
        import matplotlib.pyplot as plt
        return plt.figure(*args, **kwargs)
    except ImportError as err:
        raise RuntimeError("Plotting is not available") from err


def subplots(*args, **kwargs) -> tuple[Figure, np.ndarray]:
    """
    Wraps the matplotlib subplots call and raises if not available.

    Raises
    ------
    RuntimeError
        When trying to get subplots without matplotlib installed.
    """

    try:
        import matplotlib.pyplot as plt
        return plt.subplots(*args, **kwargs)
    except ImportError as err:
        raise RuntimeError("Plotting is not available") from err


def _get_label_str(netw: Network, param: str, m: int, n: int, usetex: bool | None = None) -> str:
    if usetex is None:
        import matplotlib.pyplot as plt
        usetex = plt.rcParams['text.usetex']

    label_string = ""
    if netw.name is not None:
        label_string += f"{netw.name}, "

    if usetex:
        label_string += f"${param}_{{{netw._fmt_trace_name(m,n)}}}$"
    else:
        label_string += f"{param}{netw._fmt_trace_name(m,n)}"
    return label_string


@_fig_kwarg
def scale_frequency_ticks(ax: Axes, funit: str, *, backend: str | None = None):
    """
    Scale frequency axis ticks.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib figure axe
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    funit : str
        frequency unit string as in :data:`~skrf.frequency.Frequency.unit`
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        plotly has no tick formatter, so the frequencies of the figure, which
        scikit-rf draws in its frequency unit, are converted to `funit`.

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Raises
    ------
    ValueError
        if invalid unit is passed, or with plotly if the figure has no frequency axis drawn by scikit-rf
    """
    if _resolve_backend(backend, ax) == 'plotly':
        from .constants import FREQ_UNITS

        units = {unit.lower(): unit for unit in FREQ_UNITS}
        if funit.lower() not in units:
            raise ValueError(f"invalid funit {funit}")
        ax = _plotly_figure(ax)
        meta = ax.layout.meta if isinstance(ax.layout.meta, dict) else {}
        if 'skrf_frequency_unit' not in meta:
            raise ValueError("the figure has no frequency axis drawn by scikit-rf")
        unit = units[funit.lower()]
        scale = FREQ_UNITS[units[meta['skrf_frequency_unit'].lower()]] / FREQ_UNITS[unit]
        for trace in ax.data:  # of the first subplot, the one the unit and the x-axis below belong to
            if getattr(trace, 'xaxis', None) in (None, 'x') and getattr(trace, 'x', None) is not None:
                trace.x = np.asarray(trace.x) * scale
                if getattr(trace, 'width', None) is not None:  # such as the violins of plot_violin
                    trace.width = np.multiply(trace.width, scale)
        for shape in ax.layout.shapes:  # such as the bands of shade_bands
            if shape.xref == 'x':
                shape.update(x0=shape.x0 * scale, x1=shape.x1 * scale)
        if ax.layout.xaxis.range is not None:
            log = ax.layout.xaxis.type == 'log'  # with its range in log10 units
            ax.update_xaxes(range=[x + np.log10(scale) if log else x * scale for x in ax.layout.xaxis.range])
        return ax.update_layout(meta={**meta, 'skrf_frequency_unit': unit}, xaxis_title_text=f'Frequency ({unit})')

    from matplotlib import ticker

    if funit.lower() == "hz":
        prefix = " "
        scale = 1
    elif len(funit) == 3:
        prefix = funit[0]
        scale = SI_CONVERSION[prefix]
    else:
        raise ValueError(f"invalid funit {funit}")
    ticks_x = ticker.FuncFormatter(lambda x, pos: f'{x * scale:g}')
    ax.xaxis.set_major_formatter(ticks_x)

@_fig_kwarg
def smith(smithR: Number = 1, chart_type: str = 'z', draw_labels: bool = False,
          border: bool = False, ax: Axes | None = None, ref_imm: float = 1.0,
          draw_vswr: list | bool | None = None, backend: str | None = None):
    """
    Plot the Smith chart of a given radius.

    The Smith chart is used to assist in solving problems with transmission lines
    and matching circuits. It can be used to simultaneously display multiple
    parameters including impedances, admittances, reflection coefficients,
    scattering parameters, noise figure circles, etc. [#]_

    Parameters
    ----------
    smithR : number, optional
        radius of smith chart. Default is 1.
    chart_type : str, optional
        Contour type. Default is 'z'. Possible values are:

        * *'z'* : lines of constant impedance
        * *'y'* : lines of constant admittance
        * *'zy'* : lines of constant impedance stronger than admittance
        * *'yz'* : lines of constant admittance stronger than impedance
    draw_labels : Boolean, optional
        annotate real and imaginary parts of impedance on the
        chart (only if smithR=1).
        Default is False.
    border : Boolean, optional.
        draw a rectangular border with axis ticks, around the perimeter
        of the figure. Not used if draw_labels = True.
        Default is False.
    ax : :class:`matplotlib.pyplot.Axes` or None, optional
        existing axes to draw smith chart on.
        Default is None (creates a new figure)
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    ref_imm : number, optional
        Reference immittance for center of Smith chart. Only changes
        labels, if printed.
        Default is 1.0.
    draw_vswr : list of numbers, Boolean or None, optional
        draw VSWR circles. If True, default values are used.
        Default is None.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        `border` is not used with plotly.

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Smith_chart

    """
    if _resolve_backend(backend, ax) == 'plotly':
        return _plotly_smith(smithR, chart_type, draw_labels, ax=_plotly_figure(ax, **_PLOTLY_SMITH_LAYOUT),
                             ref_imm=ref_imm, draw_vswr=draw_vswr)
    ax = _get_axes(ax)

    from matplotlib.patches import Circle

    colors = dict(superlight=dict(ec='whitesmoke', fc='none'), verylight=dict(ec='lightgrey', fc='none'),
                  light=dict(ec='grey', fc='none'), heavy=dict(ec='black', fc='none'))
    circles, labels = _smith_grid(smithR, chart_type, draw_labels, ref_imm, draw_vswr)

    # contour holds matplotlib instances of paths.Circle, which are the contours
    # on the smith chart
    contour = [Circle(center, radius, **colors[weight]) for center, radius, weight in circles]
    y_flip_sign = -1 if chart_type.startswith('y') else 1

    # clipping circle
    clipc = Circle( [0,0], smithR, ec='k',fc='None',visible=True)
    ax.add_patch( clipc)

    #draw x and y axis
    ax.axhline(0, color='k', lw=.1, clip_path=clipc)
    ax.axvline(1*y_flip_sign, color='k', clip_path=clipc)
    ax.grid(0)
    # Set axis limits by plotting white points so zooming works properly
    ax.plot(smithR*np.array([-1.1, 1.1]), smithR*np.array([-1.1, 1.1]), 'w.', markersize = 0)
    ax.axis('image') # Combination of 'equal' and 'tight'


    if not border:
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_color('none')


    if draw_labels:
        #Clear axis
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_color('none')

        # Make annotations only if the radius is 1
        if smithR == 1:
            #Make room for annotation
            ax.plot(np.array([-1.25, 1.25]), np.array([-1.1, 1.1]), 'w.', markersize = 0)
            ax.axis('image')

            for text, x, y, ha, va, kw in labels:
                ax.annotate(text, xy=(x, y), xytext=(x, y), ha=ha, va=va, **kw)

    # loop though contours and draw them on the given axes
    for currentContour in contour:
        cc=ax.add_patch(currentContour)
        cc.set_clip_path(clipc)


def _smith_grid(smithR: Number = 1, chart_type: str = 'z', draw_labels: bool = False,
                ref_imm: float = 1.0, draw_vswr: list | bool | None = None) -> tuple[list, list]:
    """
    Smith chart grid drawn by :func:`smith` and :func:`_plotly_smith`.

    Returns the circles as ``(center, radius, weight)`` in drawing order, where weight is
    'superlight', 'verylight', 'light' or 'heavy', and the labels as ``(text, x, y, ha, va, kwargs)``.
    """
    circles = []

    # these are hard-coded on purpose,as they should always be present
    rHeavyList = [0,1]
    xHeavyList = [1,-1]

    #TODO: fix this
    # these could be dynamically coded in the future, but work good'nuff for now
    if not draw_labels:
        rLightList = np.logspace(3,-5,9,base=.5)
        xLightList = np.hstack([np.logspace(2,-5,8,base=.5), -1*np.logspace(2,-5,8,base=.5)])
    else:
        rLightList = np.array( [ 0.2, 0.5, 1.0, 2.0, 5.0 ] )
        xLightList = np.array( [ 0.2, 0.5, 1.0, 2.0 , 5.0, -0.2, -0.5, -1.0, -2.0, -5.0 ] )

    # vswr lines
    if isinstance(draw_vswr, tuple | list):
        vswrVeryLightList = draw_vswr
    elif draw_vswr is True:
        # use the default I like
        vswrVeryLightList = [1.5, 2.0, 3.0, 5.0]
    else:
        vswrVeryLightList = []

    # cheap way to make a ok-looking smith chart at larger than 1 radii
    if smithR > 1:
        rMax = (1.+smithR)/(1.-smithR)
        rLightList = np.hstack([ np.linspace(0,rMax,11)  , rLightList ])

    if chart_type.startswith('y'):
        y_flip_sign = -1
    else:
        y_flip_sign = 1

    # draw impedance and/or admittance
    both_charts = chart_type in ('zy', 'yz')


    # loops through Verylight, Light and Heavy lists and lists circles
    # for analysis of this see R.M. Weikles Microwave II notes (from uva)

    # vswr circles verylight
    for vswr in vswrVeryLightList:
        radius = (vswr-1.0) / (vswr+1.0)
        circles.append(((0, 0), radius, 'verylight'))

    # impedance/admittance circles
    for r in rLightList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        if both_charts:
            circles.insert(0, ((-center[0], center[1]), radius, 'superlight'))
        circles.append((center, radius, 'light'))
    for x in xLightList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        if both_charts:
            circles.insert(0, ((-center[0], center[1]), radius, 'superlight'))
        circles.append((center, radius, 'light'))

    for r in rHeavyList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        circles.append((center, radius, 'heavy'))
    for x in xHeavyList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        circles.append((center, radius, 'heavy'))

    labels = []
    # Make annotations only if the radius is 1
    if draw_labels and smithR == 1:
        #Annotate real part
        for value in rLightList:
            # Set radius of real part's label; offset slightly left (Z
            # chart, y_flip_sign == 1) or right (Y chart, y_flip_sign == -1)
            # so label doesn't overlap chart's circles
            rho = (value - 1)/(value + 1) - y_flip_sign*0.01
            if y_flip_sign == 1:
                halignstyle = "right"
            else:
                halignstyle = "left"
            if y_flip_sign == -1:  # 'y' and 'yz' charts
                value = 1/value
            labels.append((str(value*ref_imm), rho*smithR, 0.01, halignstyle, "baseline", {}))

        #Annotate imaginary part
        radialScaleFactor = 1.01 # Scale radius of label position by this
                                 # factor. Making it >1 places the label
                                 # outside the Smith chart's circle
        for value in xLightList:
            #Transforms from complex to cartesian
            S = (1j*value - 1) / (1j*value + 1)
            S *= smithR * radialScaleFactor
            rhox = S.real
            rhoy = S.imag * y_flip_sign

            # Choose alignment anchor point based on label's value
            if ((value == 1.0) or (value == -1.0)):
                halignstyle = "center"
            elif (rhox < 0.0):
                halignstyle = "right"
            else:
                halignstyle = "left"

            if (rhoy < 0):
                valignstyle = "top"
            else:
                valignstyle = "bottom"
            if y_flip_sign == -1:  # 'y' and 'yz' charts
                value = 1/value
            #Annotate value
            labels.append((str(value*ref_imm) + 'j', rhox, rhoy, halignstyle, valignstyle, {}))

        #Annotate 0 and inf
        if y_flip_sign == 1:  # z and zy charts
            label_left, label_right = '0.0', r'$\infty$'
        else:  # y and yz charts
            label_left, label_right = r'$\infty$', '0.0'
        labels.append((label_left, -1.02, 0, "right", "center", {}))
        labels.append((label_right, radialScaleFactor, 0, "left", "center", {}))

        # annotate vswr circles
        for vswr in vswrVeryLightList:
            rhoy = (vswr-1.0) / (vswr+1.0)
            labels.append((str(vswr), 0, rhoy*smithR, "center", "bottom", dict(color='grey', size='smaller')))

    return circles, labels


@_fig_kwarg
def plot_rectangular(x: NumberLike, y: NumberLike,
                     x_label: str | None = None, y_label: str | None = None,
                     title: str | None = None, show_legend: bool = True,
                     axis: str = 'tight', ax: Axes | None = None,
                     *args, backend: str | None = None, **kwargs):
    r"""
    Plot rectangular data and optionally label axes.

    Parameters
    ----------
    x : array-like, of complex data
        data to plot
    y : array-like, of complex data
        data to plot
    x_label : string or None, optional.
        x-axis label. Default is None.
    y_label : string or None, optional.
        y-axis label. Default is None.
    title : string or None, optional.
        plot title. Default is None.
    show_legend : Boolean, optional.
        controls the drawing of the legend. Default is True.
    axis : str, optional
        whether or not to autoscale the axis. Default is 'tight'
    ax : :class:`matplotlib.axes.AxesSubplot` object or None, optional.
        axes to draw on. Default is None (creates a new figure)
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*args, \*\*kwargs : passed to pylab.plot (or :class:`plotly.graph_objects.Scatter`)

    Returns
    -------
    the list of matplotlib lines, or the plotly Figure with the plotly backend

    """
    if _resolve_backend(backend, ax) == 'plotly':
        return _plotly_rect(x, y, *args, x_label=x_label, y_label=y_label, title=title,
                            show_legend=show_legend, ax=_plotly_figure(ax), **kwargs)

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    my_plot = ax.plot(x, y, *args, **kwargs)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        # only show legend if they provide a label
        if 'label' in kwargs:
            ax.legend()

    if axis is not None:
        ax.autoscale(True, 'x', True)
        ax.autoscale(True, 'y', False)

    if plt.isinteractive():
        plt.draw()

    return my_plot


# plotly counterparts of the matplotlib plotting functions above. They only draw
# data that scikit-rf has already computed, and import plotly when called.

# layout of a new figure: a fixed size, as a notebook otherwise stretches a plotly figure across the whole cell,
# with less room above the plot than plotly's default, and the title centered over the plot like matplotlib
_PLOTLY_LAYOUT = dict(width=700, height=500, margin_t=60, title=dict(x=0.5, xanchor='center', xref='paper'))
# common matplotlib line keywords and their plotly equivalents
_PLOTLY_ALIASES = {'c': 'color', 'ls': 'linestyle', 'lw': 'linewidth', 'ms': 'markersize'}
_PLOTLY_COLORS = {'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow',
                  'k': 'black', 'w': 'white'}
_PLOTLY_LINESTYLES = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot', 'dashed': 'dash', 'dotted': 'dot'}
_PLOTLY_MARKERS = {'o': 'circle', '.': 'circle', 's': 'square', 'd': 'diamond', 'D': 'diamond', '+': 'cross',
                   '^': 'triangle-up', 'v': 'triangle-down', '*': 'star'}
_PLOTLY_NONE = ('', ' ', 'none', 'None')


def _plotly_figure(ax=None, **layout):
    """Return the plotly Figure `ax`, or a new Figure of the default size and `layout` if it is None."""
    go = _import_plotly()
    if ax is None:
        return go.Figure(layout={**_PLOTLY_LAYOUT, **layout})
    if isinstance(ax, _PlotlyAxes):
        return ax.fig
    if not isinstance(ax, go.Figure):
        raise TypeError(f"the plotly backend needs a plotly Figure as ax, not {type(ax).__name__}")
    return ax


def _plotly_kwargs(args: tuple, kwargs: dict) -> dict:
    """Translate the common matplotlib line keywords of skrf's own plots for go.Scatter; others are passed on."""
    if args:
        raise TypeError(f"Positional matplotlib format arguments {args!r} are not supported by the plotly "
                        "backend, use plotly keywords such as line=dict(...) or marker=dict(...)")
    kw = {_PLOTLY_ALIASES.get(k, k): v for k, v in kwargs.items()}
    color, linestyle, marker = kw.pop('color', None), kw.pop('linestyle', None), kw.pop('marker', None)
    line = dict(kw.pop('line', {}))
    markers = dict(marker) if isinstance(marker, dict) else {}  # a plotly marker dict is passed on
    if color is not None:
        line['color'] = markers['color'] = _plotly_color(color)
    if linestyle not in _PLOTLY_NONE + (None,):
        line['dash'] = _PLOTLY_LINESTYLES.get(linestyle, linestyle)
    if 'linewidth' in kw:
        line['width'] = kw.pop('linewidth')
    if isinstance(marker, str) and marker not in _PLOTLY_NONE:
        markers['symbol'] = _PLOTLY_MARKERS.get(marker, marker)
    if 'markersize' in kw:
        markers['size'] = kw.pop('markersize')
    if 'alpha' in kw:
        kw['opacity'] = kw.pop('alpha')
    if 'label' in kw:
        kw.setdefault('name', str(kw.pop('label')))
    mode = [] if linestyle in _PLOTLY_NONE else ['lines']
    if 'symbol' in markers or isinstance(marker, dict):
        mode.append('markers')
    kw.setdefault('mode', '+'.join(mode) or 'none')
    return {**kw, 'line': line, 'marker': markers}


def _plotly_color(color):
    """plotly color for a matplotlib color, including gray levels such as '0.5'."""
    if isinstance(color, str):
        try:
            return 'rgb({0},{0},{0})'.format(round(255 * float(color)))
        except ValueError:
            return _PLOTLY_COLORS.get(color, color)
    return color


def _plotly_frequency_x(fig, frequency: Frequency) -> tuple[np.ndarray, str]:
    """
    Frequency values and x-axis label for a trace on `fig`.

    All traces use the unit of the first one, which is kept in ``fig.layout.meta``.
    """
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    unit = meta.get('skrf_frequency_unit')
    if unit is None:
        unit = frequency.unit
        if fig.layout.meta is None or isinstance(fig.layout.meta, dict):  # leave other user metadata alone
            fig.update_layout(meta={**meta, 'skrf_frequency_unit': unit})
    return frequency.f / frequency.multiplier_dict[unit.lower()], f'Frequency ({unit})'


def _plotly_rect(x: NumberLike, y: NumberLike, *args,
                 x_label: str | None = None, y_label: str | None = None,
                 title: str | None = None, show_legend: bool = True,
                 axis: str = 'tight', ax=None, **kwargs):
    """plotly counterpart of :func:`plot_rectangular`, drawing on the plotly Figure `ax`."""
    trace = _plotly_kwargs(args, kwargs)
    trace.setdefault('showlegend', bool(show_legend and 'name' in trace))
    # the real part of complex data with a warning, like matplotlib, as plotly can't serialize it
    ax.add_trace(_import_plotly().Scatter(x=x, y=np.asarray(y, dtype=float), **trace))
    if x_label is not None:
        ax.update_xaxes(title_text=x_label)
    if y_label is not None:
        ax.update_yaxes(title_text=y_label)
    if title is not None:
        ax.update_layout(title_text=title)
    return ax


def _plotly_trace_color(fig) -> str:
    """Line color of the last trace of `fig`, set to the color for its index in the colorway of `fig` if unset."""
    trace = fig.data[-1]
    if trace.line.color is None:
        from plotly.colors import qualitative
        colorway = fig.layout.colorway or fig.layout.template.layout.colorway or qualitative.Plotly
        trace.line.color = colorway[(len(fig.data) - 1) % len(colorway)]
    return trace.line.color


def _plotly_fill_between(fig, frequency: Frequency, lower: NumberLike, upper: NumberLike,
                         color: str | None = None, alpha: float = .3, **kwargs):
    """plotly counterpart of ``ax.fill_between`` vs `frequency`, without a legend entry."""
    x, _ = _plotly_frequency_x(fig, frequency)
    fig.add_trace(_import_plotly().Scatter(x=np.concatenate([x, x[::-1]]),
                                           y=np.concatenate([upper, lower[::-1]]),
                                           fill='toself', fillcolor=color, opacity=alpha, mode='none',
                                           hoverinfo='skip', showlegend=False, **kwargs))


def _plotly_complex(z: NumberLike, x_label: str = 'Real', y_label: str = 'Imag',
                    title: str = 'Complex Plane', show_legend: bool = True,
                    axis: str = 'equal', ax=None, **kwargs):
    """plotly counterpart of :func:`plot_complex_rectangular`, with equal axis scales."""
    _plotly_rect(np.real(z), np.imag(z), x_label=x_label, y_label=y_label, title=title,
                 show_legend=show_legend, ax=ax, **kwargs)
    # pad the range of all traces by 5% like matplotlib, as plotly's autorange fits lines tightly
    for update, attr in ((ax.update_xaxes, 'x'), (ax.update_yaxes, 'y')):
        values = [getattr(t, attr, None) for t in ax.data]
        values = np.concatenate([np.asarray(v, dtype=float) for v in values if v is not None])
        lo, hi = np.nanmin(values), np.nanmax(values)
        update(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
    # shrink the plot area rather than widen the range, so it fits the data and zoom boxes match the drag
    ax.update_xaxes(constrain='domain')
    return ax.update_yaxes(scaleanchor='x', scaleratio=1, constrain='domain')


def _plotly_subplots(figs: list, cols: int, **kwargs):
    """Copy the traces, axes and annotations of the plotly Figures `figs` into a grid of subplots."""
    _import_plotly()
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=-(-len(figs) // cols), cols=cols, **kwargs).update_layout(_PLOTLY_LAYOUT)
    for i, sub in enumerate(figs):
        _plotly_add_panel(fig, sub, i // cols + 1, i % cols + 1, i)
    return fig


def _plotly_add_panel(fig, panel, row: int, col: int, k: int, show_legend: bool = True):
    """
    Copy the traces, axes and annotations of the plotly Figure `panel` into subplot `row`, `col` of `fig`.

    Its legend entries go to the `k`-th legend of `fig`, if `show_legend`, which is placed in the plot
    area of the subplot like matplotlib's legend of each Axes.
    """
    legend = f"legend{k + 1 if k else ''}"
    if k == 0 and fig.layout.meta is None:  # the frequency unit of the first subplot
        fig.update_layout(meta=panel.layout.meta)
    for trace in panel.data:
        trace.update(legend=legend, showlegend=bool(trace.showlegend) and show_legend)
    fig.add_traces(panel.data, rows=row, cols=col)
    subplot = fig.get_subplot(row, col)
    if hasattr(subplot, 'xaxis'):
        fig.update_xaxes(panel.layout.xaxis, row=row, col=col)
        fig.update_yaxes(panel.layout.yaxis, row=row, col=col, secondary_y=False)
        if panel.layout.yaxis.scaleanchor:
            fig.update_yaxes(scaleanchor=subplot.yaxis.anchor, row=row, col=col, secondary_y=False)
        for annotation in panel.layout.annotations:
            fig.add_annotation(annotation, row=row, col=col)
        (x0, x1), (y0, y1) = subplot.xaxis.domain, subplot.yaxis.domain
    else:  # a polar subplot
        fig.update_polars(panel.layout.polar, row=row, col=col)
        (x0, x1), (y0, y1) = subplot.domain.x, subplot.domain.y
    if not any(trace.showlegend for trace in panel.data):
        return
    margin = fig.layout.margin
    # plot area in pixels, with plotly's default size and margins where unset
    width = (x1 - x0) * ((fig.layout.width or 700) - (margin.l or 80) - (margin.r or 80))
    height = (y1 - y0) * ((fig.layout.height or 450) - (margin.t or 100) - (margin.b or 80))
    x, y = _plotly_legend_position(panel, width, height)
    fig.update_layout({legend: dict(
        x=x0 + x * (x1 - x0), y=y0 + y * (y1 - y0), xanchor={0: 'left', 1: 'right'}[x],
        yanchor={0: 'bottom', 1: 'top'}.get(y, 'middle'), font_size=_PLOTLY_LEGEND_FONT,
        bgcolor='rgba(255,255,255,0.8)', bordercolor='lightgrey', borderwidth=1)})


class _PlotlyLine:
    """A plotly trace as the matplotlib Line2D of :attr:`_PlotlyAxes.lines`."""

    def __init__(self, trace):
        self.trace = trace

    def get_xdata(self) -> np.ndarray:
        return np.asarray(self.trace.x)

    def get_ydata(self) -> np.ndarray:
        return np.asarray(self.trace.y)


class _PlotlyAxes:
    """
    A panel of a :class:`_PlotlyFigure`, with the matplotlib Axes methods used by the
    multi-panel figures of scikit-rf, so that the same code draws them with both backends.

    It draws on its own plotly Figure `fig`, and plot functions take it as a plotly `ax`.
    """

    def __init__(self, pos: tuple[int, int, int], polar: bool = False):
        self.pos, self.polar, self.title, self.has_legend, self.fig = pos, polar, None, False, _plotly_figure()
        self.twins = []  # from twinx(), drawn on a second y-axis of this panel

    @property
    def lines(self) -> list[_PlotlyLine]:
        return [_PlotlyLine(trace) for trace in self.fig.data]

    def plot(self, x, y=None, **kwargs):
        if y is None:
            x, y = np.arange(len(x)), x
        if str(kwargs.get('label')).startswith('_'):  # matplotlib's labels without a legend entry
            del kwargs['label']
        _plotly_rect(x, y, ax=self.fig, **kwargs)

    def set_title(self, title: str):
        self.title = title

    def set_xlabel(self, label: str):
        self.fig.update_xaxes(title_text=label)

    def set_ylabel(self, label: str):
        self.fig.update_yaxes(title_text=label)

    def set_xlim(self, lim):
        self.fig.update_xaxes(range=list(lim))

    def set_ylim(self, lim):
        self.fig.update_yaxes(range=list(lim))

    def legend(self, *args, **kwargs):
        self.has_legend = True  # placed where it covers the least data, see _plotly_legend_position

    def grid(self, *args, **kwargs):
        pass  # plotly draws a grid by default

    def twinx(self) -> _PlotlyAxes:
        self.twins.append(_PlotlyAxes(self.pos))
        return self.twins[-1]


# candidate legend positions in a plot area, as fractions of its width and height, in order of preference
_PLOTLY_LEGEND_SPOTS = ((1, 1), (0, 1), (1, 0), (0, 0), (1, .5), (0, .5))
_PLOTLY_LEGEND_FONT = 10


def _plotly_legend_position(fig, width: float, height: float) -> tuple[float, float]:
    """
    Position of the legend of the plotly Figure `fig`, with a plot area of `width` x `height` pixels,
    among :data:`_PLOTLY_LEGEND_SPOTS`, where it covers the fewest data points, like matplotlib's 'best'.
    """
    names = [str(trace.name) for trace in fig.data if trace.showlegend]
    # legend size estimated from the font size: about 0.6 em per character and 1.8 em per entry
    box_w = min(1., (40 + 0.6 * _PLOTLY_LEGEND_FONT * max(map(len, names), default=0)) / width)
    box_h = min(1., (10 + 1.8 * _PLOTLY_LEGEND_FONT * len(names)) / height)
    # the traces resampled along their length, so that lines with few points still count where they pass
    points = []
    for trace in fig.data:
        if getattr(trace, 'x', None) is None or getattr(trace, 'y', None) is None or len(trace.x) == 0                 or trace.meta == 'skrf_smith_chart':  # no data in a plot area (polar), or no data at all
            continue
        x, y = np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float)
        t = np.linspace(0, len(x) - 1, 200)
        points.append(np.column_stack([np.interp(t, np.arange(len(x)), x), np.interp(t, np.arange(len(y)), y)]))
    if not points:
        return _PLOTLY_LEGEND_SPOTS[0]
    points = np.concatenate(points)
    points = points[np.isfinite(points).all(axis=1)]
    # in fractions of the plot area, from the axis ranges if set, else the data range as plotly's autorange
    for i, axis in enumerate((fig.layout.xaxis, fig.layout.yaxis)):
        lo, hi = axis.range if axis.range is not None else (points[:, i].min(), points[:, i].max())
        points[:, i] = (points[:, i] - lo) / ((hi - lo) or 1)

    def covered(spot):
        x0, y0 = spot[0] * (1 - box_w), spot[1] * (1 - box_h)
        x, y = points[:, 0], points[:, 1]
        return np.sum((x >= x0) & (x <= x0 + box_w) & (y >= y0) & (y <= y0 + box_h))

    return min(_PLOTLY_LEGEND_SPOTS, key=covered)


class _PlotlyFigure:
    """
    plotly counterpart of a matplotlib Figure for :class:`_PlotlyAxes` panels.

    :meth:`to_plotly` copies the panels into subplots of the plotly Figure `fig`, or a new one.
    """

    def __init__(self, fig=None, figsize: tuple = (6.4, 4.8)):
        self.fig, self.figsize, self.axes, self.title = fig, figsize, [], None

    def add_subplot(self, rows: int, cols: int, index: int, projection: str | None = None) -> _PlotlyAxes:
        self.axes.append(_PlotlyAxes((rows, cols, index), polar=projection == 'polar'))
        return self.axes[-1]

    def suptitle(self, title: str):
        self.title = title

    def tight_layout(self):
        pass

    def to_plotly(self):
        """Copy the traces and axes of the panels into subplots, spanning cells as with matplotlib."""
        from math import lcm

        _import_plotly()
        from plotly.subplots import make_subplots

        rows, cols = lcm(*(ax.pos[0] for ax in self.axes)), lcm(*(ax.pos[1] for ax in self.axes))
        specs, cells = [[None] * cols for _ in range(rows)], {}
        for ax in self.axes:
            r, c, i = ax.pos
            row, col = (i - 1) // c * (rows // r), (i - 1) % c * (cols // c)
            spec = dict(type='polar') if ax.polar else dict(secondary_y=bool(ax.twins))
            specs[row][col], cells[row, col] = dict(rowspan=rows // r, colspan=cols // c, **spec), ax
        grid = dict(rows=rows, cols=cols, specs=specs,
                    subplot_titles=[cells[cell].title or '' for cell in sorted(cells)])
        fig = self.fig
        if fig is None:
            fig = make_subplots(**grid).update_layout(_PLOTLY_LAYOUT, width=100 * self.figsize[0],
                                                      height=100 * self.figsize[1])
        else:
            try:
                fig.set_subplots(**grid)
            except ValueError:  # a figure from an earlier call already has the subplots: draw over it
                pass
        for k, ((row, col), ax) in enumerate(sorted(cells.items())):
            _plotly_add_panel(fig, ax.fig, row + 1, col + 1, k, show_legend=ax.has_legend)
            for twin in ax.twins:  # on the second y-axis, in the legend of the panel
                for trace in twin.fig.data:
                    trace.update(legend=f"legend{k + 1 if k else ''}", showlegend=bool(trace.showlegend)
                                 and twin.has_legend)
                fig.add_traces(twin.fig.data, rows=row + 1, cols=col + 1, secondary_ys=[True] * len(twin.fig.data))
                fig.update_yaxes(twin.fig.layout.yaxis, row=row + 1, col=col + 1, secondary_y=True)
        if self.title is not None:
            fig.update_layout(title_text=self.title)
        return fig


def _figure(backend: str, fig=None, **kwargs):
    """
    :func:`figure` for `backend`, or with plotly a :class:`_PlotlyFigure` drawing into the plotly Figure `fig`.
    """
    if backend == 'plotly':
        return _PlotlyFigure(fig, kwargs.get('figsize', (6.4, 4.8)))
    return figure(**kwargs)


def _subplots(rows: int, cols: int, backend: str, fig=None, **kwargs):
    """:func:`subplots` for `backend`, see :func:`_figure`; the axes are squeezed like matplotlib's."""
    if backend == 'plotly':
        fig = _figure(backend, fig, **kwargs)
        axes = np.array([[fig.add_subplot(rows, cols, r * cols + c + 1) for c in range(cols)] for r in range(rows)])
        return fig, axes[0, 0] if axes.size == 1 else axes.squeeze()
    return subplots(rows, cols, **kwargs)


def _frequency_x(ax, frequency: Frequency) -> np.ndarray:
    """x values of `frequency` on `ax`: in Hz with matplotlib, whose ticks are scaled, else in the unit of `ax`."""
    return _plotly_frequency_x(_plotly_figure(ax), frequency)[0] if _is_plotly_figure(ax) else frequency.f


_PLOTLY_SMITH_COLORS = dict(superlight='whitesmoke', verylight='lightgrey', light='grey', heavy='black')
_PLOTLY_SMITH_LAYOUT = dict(height=550, margin=dict(l=20, r=20, t=30, b=20))  # a new figure around a square chart
_PLOTLY_ANCHORS = {'baseline': 'bottom', 'center': 'middle'}


def _plotly_smith(smithR: Number = 1, chart_type: str = 'z', draw_labels: bool = False,
                  ax=None, ref_imm: float = 1.0, draw_vswr: list | bool | None = None):
    """
    plotly counterpart of :func:`smith`.

    Each line weight is one trace, clipped to the chart radius, with ``meta='skrf_smith_chart'``.
    """
    go = _import_plotly()
    circles, labels = _smith_grid(smithR, chart_type, draw_labels, ref_imm, draw_vswr)
    y_flip_sign = -1 if chart_type.startswith('y') else 1
    unit_circle = np.exp(2j * np.pi * np.linspace(0, 1, 361))
    paths = {weight: [] for weight in _PLOTLY_SMITH_COLORS}
    for center, radius, weight in circles:
        paths[weight].append(complex(*center) + radius * unit_circle)
    # clipping circle and x and y axis
    line = np.linspace(-1, 1, 201) * smithR
    paths['heavy'] += [smithR * unit_circle, line, y_flip_sign + 1j * line]

    for weight, color in _PLOTLY_SMITH_COLORS.items():
        if not paths[weight]:
            continue
        p = np.concatenate([np.append(path, np.nan) for path in paths[weight]])
        p[np.abs(p) > smithR * (1 + 1e-9)] = np.nan
        p = p[~(np.isnan(p) & np.isnan(np.roll(p, 1)))]  # one gap between line pieces is enough
        ax.add_trace(go.Scatter(x=p.real, y=p.imag, mode='lines', line=dict(color=color, width=1),
                                hoverinfo='skip', showlegend=False, meta='skrf_smith_chart'))
    for text, x, y, ha, va, kw in labels:
        ax.add_annotation(x=x, y=y, text=text.replace(r'$\infty$', '∞'), showarrow=False,
                          xanchor=ha, yanchor=_PLOTLY_ANCHORS.get(va, va), font_color=kw.get('color'))

    ax.update_xaxes(range=[-1.1 * smithR, 1.1 * smithR], showgrid=False, zeroline=False, showticklabels=False,
                    constrain='domain')
    ax.update_yaxes(range=[-1.1 * smithR, 1.1 * smithR], showgrid=False, zeroline=False, showticklabels=False,
                    scaleanchor='x', scaleratio=1, constrain='domain')
    ax.update_layout(plot_bgcolor='white')
    return ax


@_fig_kwarg
def plot_polar(theta: NumberLike, r: NumberLike,
               x_label: str | None = None, y_label: str | None = None,
               title: str | None = None, show_legend: bool = True,
               axis_equal: bool = False, ax: Axes | None = None,
               *args, backend: str | None = None, **kwargs):
    r"""
    Plot polar data on a polar plot and optionally label axes.

    Parameters
    ----------
    theta : array-like
        angular data to plot
    r : array-like
        radial data to plot
    x_label : string or None, optional
        x-axis label. Default is None.
    y_label : string or None, optional.
        y-axis label. Default is None
    title : string or None, optional.
        plot title. Default is None.
    show_legend : Boolean, optional.
        controls the drawing of the legend. Default is True.
    ax : :class:`matplotlib.axes.AxesSubplot` object or None.
        axes to draw on. Default is None (creates a new figure).
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        With plotly, `theta` is in radians, `y_label` titles the radial axis, and
        `x_label` and `axis_equal` are not used.
    \*args, \*\*kwargs : passed to pylab.plot (or :class:`plotly.graph_objects.Scatterpolar`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    """
    if _resolve_backend(backend, ax) == 'plotly':
        ax = _plotly_figure(ax)
        trace = _plotly_kwargs(args, kwargs)
        trace.setdefault('showlegend', bool(show_legend and 'name' in trace))
        ax.add_trace(_import_plotly().Scatterpolar(r=r, theta=theta, thetaunit='radians', **trace))
        if y_label is not None:
            ax.update_polars(radialaxis_title_text=y_label)
        if title is not None:
            ax.update_layout(title_text=title)
        return ax

    import matplotlib.pyplot as plt

    if ax is None:
        # no Axes passed
        # if an existing (polar) plot is already present, grab and use its Axes
        # otherwise, create a new polar plot and use that Axes
        if not plt.get_fignums() or not plt.gcf().axes or plt.gca().name != 'polar':
            ax = plt.figure().add_subplot(projection='polar')
        else:
            ax = plt.gca()
    else:
        if ax.name != 'polar':
            # The projection of an existing axes can't be changed,
            # since specifying a projection when creating an axes determines the
            # axes class you get, which is different for each projection type.
            # So, passing a axe projection not polar is probably undesired
            warnings.warn(
                f"Projection of the Axes passed as `ax` is not 'polar' but is {ax.name}." +
                "See Matplotlib documentation to create a polar plot or call this function without the `ax` parameter."
                , stacklevel=2
            )

    ax.plot(theta, r, *args, **kwargs)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        # only show legend if they provide a label
        if 'label' in kwargs:
            ax.legend()

    if axis_equal:
        ax.axis('equal')

    if plt.isinteractive():
        plt.draw()


@_fig_kwarg
def plot_complex_rectangular(z: NumberLike,
                             x_label: str = 'Real', y_label: str = 'Imag',
                             title: str = 'Complex Plane', show_legend: bool = True,
                             axis: str = 'equal', ax: Axes | None = None,
                             backend: str | None = None, **kwargs):
    r"""
    Plot complex data on the complex plane.

    Parameters
    ----------
    z : array-like, of complex data
        data to plot
    x_label : string, optional.
        x-axis label. Default is 'Real'.
    y_label : string, optional.
        y-axis label. Default is 'Imag'.
    title : string, optional.
        plot title. Default is 'Complex Plane'
    show_legend : Boolean, optional.
        controls the drawing of the legend. Default is True.
    ax : :class:`matplotlib.axes.AxesSubplot` object or None.
        axes to draw on. Default is None (creates a new figure)
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*\*kwargs : passed to pylab.plot (or :class:`plotly.graph_objects.Scatter`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    """
    if _resolve_backend(backend, ax) == 'plotly':
        return _plotly_complex(z, x_label=x_label, y_label=y_label, title=title, show_legend=show_legend,
                               ax=_plotly_figure(ax), **kwargs)
    x = np.real(z)
    y = np.imag(z)
    plot_rectangular(x=x, y=y, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, backend='matplotlib', **kwargs)


@_fig_kwarg
def plot_complex_polar(z: NumberLike,
                       x_label: str | None = None, y_label: str | None = None,
                       title: str | None = None, show_legend: bool = True,
                       axis_equal: bool = False, ax: Axes | None = None,
                       backend: str | None = None, **kwargs):
    r"""
    Plot complex data in polar format.

    Parameters
    ----------
    z : array-like, of complex data
        data to plot
    x_label : string or None, optional
        x-axis label. Default is None.
    y_label : string or None, optional.
        y-axis label. Default is None
    title : string or None, optional.
        plot title. Default is None.
    show_legend : Boolean, optional.
        controls the drawing of the legend. Default is True.
    ax : :class:`matplotlib.axes.AxesSubplot` object or None.
        axes to draw on. Default is None (creates a new figure).
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        See :func:`plot_polar`.
    \*\*kwargs : passed to pylab.plot (or :class:`plotly.graph_objects.Scatterpolar`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    """
    theta = np.angle(z)
    r = np.abs(z)
    return plot_polar(theta=theta, r=r, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis_equal=axis_equal,
        ax=ax, backend=backend, **kwargs)


@_fig_kwarg
def plot_smith(s: NumberLike, smith_r: float = 1, chart_type: str = 'z',
               x_label: str = 'Real', y_label: str = 'Imaginary', title: str = 'Complex Plane',
               show_legend: bool = True, axis: str = 'equal', ax: Axes | None = None,
               force_chart: bool = False, draw_vswr: list | bool | None = None, draw_labels: bool = False,
               backend: str | None = None, **kwargs):
    r"""
    Plot complex data on smith chart.

    Parameters
    ------------
    s : complex array-like
        reflection-coefficient-like data to plot
    smith_r : number
        radius of smith chart
    chart_type : str in ['z','y']
        Contour type for chart.
        * *'z'* : lines of constant impedance
        * *'y'* : lines of constant admittance
    x_label : string, optional.
        x-axis label. Default is 'Real'.
    y_label : string, optional.
        y-axis label. Default is 'Imaginary'
    title : string, optional.
        plot title, Default is 'Complex Plane'.
    show_legend : Boolean, optional.
        controls the drawing of the legend. Default is True.
    axis_equal: Boolean, optional.
        sets axis to be equal increments. Default is 'equal'.
    ax : :class:`matplotlib.axes.AxesSubplot` object or None.
        axes to draw on. Default is None (creates a new figure).
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    force_chart : Boolean, optional.
        forces the re-drawing of smith chart. Default is False.
    draw_vswr : list of numbers, Boolean or None, optional
        draw VSWR circles. If True, default values are used.
        Default is None.
    draw_labels : Boolean
        annotate chart with impedance values
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*\*kwargs : passed to pylab.plot (or :class:`plotly.graph_objects.Scatter`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    """
    if _resolve_backend(backend, ax) == 'plotly':
        ax = _plotly_figure(ax, **_PLOTLY_SMITH_LAYOUT)
        if not force_chart and not any(trace.meta == 'skrf_smith_chart' for trace in ax.data):
            _plotly_smith(smith_r, chart_type, draw_labels, ax=ax, draw_vswr=draw_vswr)
        return _plotly_rect(np.real(s), np.imag(s), x_label=x_label, y_label=y_label, title=title,
                            show_legend=show_legend, ax=ax, **kwargs)

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # test if smith chart is already drawn
    if not force_chart:
        if len(ax.patches) == 0:
            smith(ax=ax, smithR = smith_r, chart_type=chart_type, draw_vswr=draw_vswr, draw_labels=draw_labels,
                  backend='matplotlib')

    plot_complex_rectangular(s, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, backend='matplotlib', **kwargs)

    ax.axis(smith_r*np.array([-1.1, 1.1, -1.1, 1.1]))
    if plt.isinteractive():
        plt.draw()


def subplot_params(ntwk: Network, param: str = 's', proj: str = 'db',
                   size_per_port: int = 4, newfig: bool = True,
                   add_titles: bool = True, keep_it_tight: bool = True,
                   subplot_kw: dict = None, *, backend: str | None = None,
                   **kwargs):
    """
    Plot all networks parameters individually on subplots.

    Parameters
    ----------
    ntwk : :class:`~skrf.network.Network`
        Network to get data from.
    param : str, optional
        Parameter to plot, by default 's'
    proj : str, optional
        Projection type, by default 'db'
    size_per_port : int, optional
        by default 4
    newfig : bool, optional
        by default True
    add_titles : bool, optional
        by default True
    keep_it_tight : bool, optional
        by default True
    subplot_kw : dict, optional
        by default {}
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        With plotly, `subplot_kw` is passed to :func:`plotly.subplots.make_subplots`,
        and `newfig` and `keep_it_tight` are not used.

    Returns
    -------
    f : :class:`matplotlib.pyplot.Figure`
        Matplotlib Figure
    ax : :class:`matplotlib.pyplot.Axes`
        Matplotlib Axes

    or the plotly Figure with the plotly backend

    """
    subplot_kw = subplot_kw if subplot_kw else {}
    if _resolve_backend(backend) == 'plotly':
        titles = [f"{param.upper()}{ntwk._fmt_trace_name(m, n)}" if add_titles else '' for m, n in ntwk.port_tuples]
        figs = [getattr(ntwk, f'plot_{param}_{proj}')(m=m, n=n, backend='plotly', **kwargs)
                for m, n in ntwk.port_tuples]
        fig = _plotly_subplots(figs, cols=ntwk.nports, subplot_titles=titles, **subplot_kw)
        # 100 pixels per inch of the matplotlib figure size
        return fig.update_layout(width=100 * size_per_port * ntwk.nports, height=100 * size_per_port * ntwk.nports)

    import matplotlib.pyplot as plt

    if newfig:
        f,axs= plt.subplots(ntwk.nports,ntwk.nports,
                            figsize =(size_per_port*ntwk.nports,
                                      size_per_port*ntwk.nports ),
                                      **subplot_kw)
    else:
        f = plt.gcf()
        axs = np.array(f.get_axes())

    for ports,ax in zip(ntwk.port_tuples, axs.flatten()):
        plot_func = ntwk.__getattribute__(f'plot_{param}_{proj}')
        plot_func(m=ports[0], n=ports[1], ax=ax, backend='matplotlib', **kwargs)
        if add_titles:
            ax.set_title(f"{param.upper()}{ntwk._fmt_trace_name(ports[0], ports[1])}")
    if keep_it_tight:
       plt.tight_layout()
    return f, axs


@_fig_kwarg
def shade_bands(edges: NumberLike, y_range: tuple | None = None,
                cmap: str | None = None, *, ax: Axes | None = None,
                backend: str | None = None, **kwargs):
    r"""
    Shades frequency bands.

    When plotting data over a set of frequency bands it is nice to
    have each band visually separated from the other. The kwarg `alpha`
    is useful, or `opacity` with plotly.

    Parameters
    ----------
    edges : array-like
        x-values separating regions of a given shade, in Hz for a plot vs frequency
    y_range : tuple or None, optional.
        y-values to shade in. Default is None, the whole height of the plot.
    cmap : str or None, optional.
        see matplotlib.cm  or matplotlib.colormaps for acceptable values,
        or with plotly a plotly colorscale name.
        Default is None: 'prism', or with plotly its qualitative colors.
    ax : :class:`matplotlib.axes.Axes`, optional
        plot to shade, default the current matplotlib axes.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*\*kwargs : key word arguments
        passed to `matplotlib.fill_between` (or the plotly Figure's ``add_shape``)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Examples
    --------
    >>> rf.plotting.shade_bands([325,500,750,1100], alpha=.2)
    """
    if _resolve_backend(backend, ax) == 'plotly':
        from plotly.colors import qualitative, sample_colorscale

        from .frequency import Frequency

        ax = _plotly_figure(ax)
        if cmap is None:
            colors = qualitative.Plotly * len(edges)
        else:
            colors = sample_colorscale(cmap, [k / len(edges) for k in range(len(edges) - 1)])
        # edges in Hz, as the x data of matplotlib plots vs frequency, drawn in the unit of the figure
        unit = (ax.layout.meta if isinstance(ax.layout.meta, dict) else {}).get('skrf_frequency_unit')
        scale = Frequency.multiplier_dict[unit.lower()] if unit else 1
        y = dict(y0=y_range[0], y1=y_range[1], yref='y') if y_range is not None else dict(y0=0, y1=1, yref='y domain')
        for k in range(len(edges) - 1):
            ax.add_shape(type='rect', x0=edges[k] / scale, x1=edges[k + 1] / scale, xref='x', **y,
                         fillcolor=colors[k], line_width=0, layer='below', **kwargs)
        return ax

    import matplotlib

    ax = _get_axes(ax)
    cmap = matplotlib.colormaps['prism' if cmap is None else cmap]
    if not isinstance(y_range, tuple | list) or (len(y_range) != 2):
        y_range=ax.get_ylim()
    axis = ax.axis()
    for k in range(len(edges)-1):
        ax.fill_between(
            [edges[k],edges[k+1]],
            y_range[0], y_range[1],
            color = cmap(1.0*k/len(edges)),
            **kwargs)
    ax.axis(axis)


def save_all_figs(dir: str = './', format: None | list[str] = None,
                  replace_spaces: bool = True, echo: bool| None = None, *, figs: list | None = None):
    """
    Save all open Figures to disk.

    Parameters
    ----------
    dir : string, optional.
        path to save figures into. Default is './'
    format : None or list of strings, optional.
        the types of formats to save figures as. The elements of this
        list are passed to :func:`matplotlib.pyplot.savefig`. This is a list so that
        you can save each figure in multiple formats. Default is None.
        A plotly figure is saved as 'html' by default; other formats, such as
        'png' or 'pdf', need the kaleido package.
    replace_spaces : bool, optional
        default is True.
    echo : bool, optional.
        True prints filenames as they are saved. Default is True.
    figs : list of matplotlib or plotly Figures, optional
        the figures to save, by default all open matplotlib figures.
        plotly keeps no list of open figures, so plotly figures are saved by passing them here.
    """
    if echo is not None:
        warnings.warn("`echo` parameter is deprecated and will be removed in future versions. "
                      "Use logging instead.", FutureWarning, stacklevel=2)

    if figs is None:
        import matplotlib.pyplot as plt

        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    if dir[-1] != '/':
        dir = dir + '/'
    for fig in figs:
        plotly = _is_plotly_figure(fig)
        fileName = (fig.layout.title.text or '') if plotly else fig.get_axes()[0].get_title()
        if replace_spaces:
            fileName = fileName.replace(' ','_')
        if fileName == '':
            fileName = 'unnamedPlot'
        for fmt in format if format is not None else [None]:
            if plotly:
                fmt = fmt or 'html'
                path = f"{dir}{fileName}.{fmt}"
                if fmt == 'html':
                    fig.write_html(path)
                else:
                    fig.write_image(path, format=fmt)
            else:
                path = dir + fileName + ('' if fmt is None else '.' + fmt)
                fig.savefig(path, format=fmt)
            logger.debug(f"Saved figure {path}")

saf = save_all_figs

@_fig_kwarg
def add_markers_to_lines(ax: Axes = None,
                         marker_list: list = None,
                         markevery: int = 10, *, backend: str | None = None):
    """
    Add markers to existing lings on a plot.

    Convenient if you have already have a plot made, but then
    need to add markers afterwards, so that it can be interpreted in
    black and white. The markevery argument makes the markers less
    frequent than the data, which is generally what you want.

    Parameters
    ----------
    ax : matplotlib.Axes or None, optional
        axis which to add markers to.
        Default is current axe gca()
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    marker_list : list of string, optional
        list of marker characters. Default is ['o', 'D', 's', '+', 'x'].
        see matplotlib.plot help for possible marker characters
    markevery : int, optional.
        markevery number of points with a marker.
        Default is 10.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    Returns
    -------
    None, or the plotly Figure with the plotly backend
    """
    marker_list = marker_list if marker_list else ['o', 'D', 's', '+', 'x']

    if _resolve_backend(backend, ax) == 'plotly':
        ax = _plotly_figure(ax)
        lines = [t for t in ax.data if t.type == 'scatter' and t.meta != 'skrf_smith_chart'
                 and 'lines' in (t.mode or 'lines')]
        if len(lines) > len(marker_list):
            marker_list *= 3
        for line, marker in zip(lines, marker_list):
            # marker.maxdisplayed thins the markers like markevery
            line.update(mode='lines+markers', marker_symbol=_PLOTLY_MARKERS.get(marker, marker),
                        marker_maxdisplayed=-(-len(line.x) // markevery))
        return ax

    ax = _get_axes(ax)
    lines = ax.get_lines()
    if len(lines) > len (marker_list ):
        marker_list *= 3
    [k[0].set_marker(k[1]) for k in zip(lines, marker_list)]
    [line.set_markevery(markevery) for line in lines]

@_fig_kwarg
def legend_off(ax: Axes = None, *, backend: str | None = None):
    """
    Turn off the legend for a given axes.

    If no axes is given then it will use current axes.

    Parameters
    ----------
    ax : matplotlib.Axes or None, optional
        axis to operate on.
        Default is None for current axe gca()
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    Returns
    -------
    None, or the plotly Figure with the plotly backend
    """
    if _resolve_backend(backend, ax) == 'plotly':
        return _plotly_figure(ax).update_layout(showlegend=False)
    _get_axes(ax).legend_.set_visible(0)

@_fig_kwarg
def scrape_legend(n: int | None = None,
                  ax: Axes = None, *, backend: str | None = None):
    """
    Scrape a legend with redundant labels.

    Given a legend of m entries of n groups, this will remove all but
    every m/nth entry. This is used when you plot many lines representing
    the same thing, and only want one label entry in the legend  for the
    whole ensemble of lines.

    Parameters
    ----------
    n : int or None, optional.
        Default is None.
    ax : matplotlib.Axes or None, optional
        axis to operate on.
        Default is None for current axe gca()
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    Returns
    -------
    None, or the plotly Figure with the plotly backend
    """
    plotly = _resolve_backend(backend, ax) == 'plotly'
    if plotly:
        ax = _plotly_figure(ax)
        handles = [trace for trace in ax.data if trace.showlegend is not False and trace.name is not None]
        labels = [trace.name for trace in handles]
    else:
        ax = _get_axes(ax)
        handles, labels = ax.get_legend_handles_labels()

    if n is None:
        n =len ( set(labels))

    if n>len(handles):
        raise ValueError('number of entries is too large')

    k_list = [int(k) for k in np.linspace(0,len(handles)-1,n)]
    if plotly:
        for k, trace in enumerate(handles):
            trace.showlegend = k in k_list
        return ax
    ax.legend([handles[k] for k in k_list], [labels[k] for k in k_list])


def func_on_all_figs(func: Callable, *args, figs: list | None = None, **kwargs):
    r"""
    Run a function after making all open figures current.

    Useful if you need to change the properties of many open figures
    at once, like turn off the grid.

    Parameters
    ----------
    func : function
        function to call
    figs : list of matplotlib or plotly Figures, optional
        the figures to run `func` on, by default all open matplotlib figures.
        plotly has no current figure, so for a plotly figure `func` is called
        with the figure as its first argument.
    \*args, \*\*kwargs : passed to func

    Examples
    --------
    >>> rf.plotting.func_on_all_figs(plt.grid, alpha=.3)
    >>> rf.plotting.func_on_all_figs(lambda fig: fig.update_xaxes(showgrid=False), figs=[fig])
    """
    if figs is None:
        import matplotlib.pyplot as plt

        figs = [plt.figure(fig_n) for fig_n in plt.get_fignums()]
    for fig in figs:
        if _is_plotly_figure(fig):
            func(fig, *args, **kwargs)
            continue
        import matplotlib.pyplot as plt

        plt.figure(fig.number)
        for ax_n in fig.axes:
            fig.add_axes(ax_n) # trick to make axes current
            func(*args, **kwargs)
            plt.draw()

foaf = func_on_all_figs


@_fig_kwarg
def plot_vector(a: complex, off: complex = 0+0j, *, ax: Axes | None = None,
                backend: str | None = None, **kwargs):
    """
    Plot a 2d vector.

    Parameters
    ----------
    a : complex
        the vector (real for X, imag for Y), drawn as an arrow from `off`.
    off : complex, optional
        start of the arrow, by default 0+0j
    ax : :class:`matplotlib.axes.Axes`, optional
        plot to draw on, default the current matplotlib axes.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    **kwargs :
        passed to :func:`matplotlib.pyplot.quiver`, or with plotly to the Figure's
        ``add_annotation``, such as `arrowcolor`.

    Returns
    -------
    quiver : matplotlib.pyplot.quiver, or the plotly Figure with the plotly backend
    """
    if _resolve_backend(backend, ax) == 'plotly':
        ax = _plotly_figure(ax)
        end = off + a
        ax.add_annotation(x=end.real, y=end.imag, ax=off.real, ay=off.imag, xref='x', yref='y', axref='x',
                          ayref='y', text='', showarrow=True, **{'arrowhead': 2, 'arrowwidth': 2, **kwargs})
        return ax

    return _get_axes(ax).quiver(off.real, off.imag, a.real, a.imag, scale_units='xy',
           angles='xy', scale=1, **kwargs)


def colors(*, backend: str | None = None) -> list[str]:
    """
    Return the list of colors of the rcParams color cycle.

    With plotly, the colors of plotly's default template, see :func:`stylely`.

    Parameters
    ----------
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    Returns
    -------
    colors : List[str]
    """
    if _resolve_backend(backend) == 'plotly':
        _import_plotly()
        import plotly.io as pio
        from plotly.colors import qualitative

        template = pio.templates[pio.templates.default] if pio.templates.default else None
        return list(template.layout.colorway or qualitative.Plotly) if template else list(qualitative.Plotly)

    import matplotlib.pyplot as plt

    return [c['color'] for c in plt.rcParams['axes.prop_cycle']]


def _start_plot(backend: str | None, kwargs: dict) -> str:
    """Resolve `backend`; with plotly, put the figure to draw on in ``kwargs['ax']``."""
    backend = _resolve_backend(backend, kwargs.get("ax"))
    if backend == "plotly":
        kwargs["ax"] = _plotly_figure(kwargs.get("ax"))
    return backend


## specific plotting functions
def plot(netw: Network, *args, **kw):
    """
    Plot something vs frequency
    """
    return netw.frequency.plot(*args, **kw)


@_fig_kwarg
def plot_passivity(netw: Network, port=None, label_prefix=None, *, backend: str | None = None, **kwargs):
    """
    Plot dB(diag(passivity metric)) vs frequency.

    Note
    ----
    This plot does not completely capture the passivity metric, which
    is a test for `unitary-ness` of the s-matrix. However, it may
    be used to display a measure of power dissipated in a network.

    See Also
    --------
    passivity
    """
    backend = _start_plot(backend, kwargs)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

    name = '' if netw.name is None else netw.name

    if port is None:
        ports = range(netw.nports)
    else:
        ports = [port]
    for k in ports:
        if label_prefix is None:
            label = name + ', port %i' % (k + 1)
        else:
            label = label_prefix + ', port %i' % (k + 1)
        netw.frequency.plot(mf.complex_2_db(netw.passivity[:, k, k]),
                            label=label, backend=backend,
                            **kwargs)

    if backend == "plotly":
        return kwargs["ax"]
    plt.legend()
    if plt.isinteractive():
        plt.draw()


@_fig_kwarg
def plot_reciprocity(netw: Network, db=False, *args, backend: str | None = None, **kwargs):
    """
    Plot reciprocity metric.

    See Also
    --------
    reciprocity
    """
    backend = _start_plot(backend, kwargs)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

    for m in range(netw.nports):
        for n in range(netw.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = f"ports {netw._fmt_trace_name(m, n)}"
                y = netw.reciprocity[:, m, n].flatten()
                y = mf.complex_2_db(y) if db else np.abs(y)
                netw.frequency.plot(y, *args, backend=backend, **kwargs)

    if backend == "plotly":
        return kwargs["ax"]
    plt.legend()
    if plt.isinteractive():
        plt.draw()


@_fig_kwarg
def plot_reciprocity2(netw: Network, db=False, *args, backend: str | None = None, **kwargs):
    """
    Plot reciprocity metric #2.

    This is distance of the determinant of the wave-cascading matrix
    from unity.

    .. math::

            abs(1 - S/S^T )



    See Also
    --------
    reciprocity
    """
    backend = _start_plot(backend, kwargs)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

    for m in range(netw.nports):
        for n in range(netw.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = f"ports {netw._fmt_trace_name(m, n)}"
                y = netw.reciprocity2[:, m, n].flatten()
                if db:
                    y = mf.complex_2_db(y)
                netw.frequency.plot(y, *args, backend=backend, **kwargs)

    if backend == "plotly":
        return kwargs["ax"]
    plt.legend()
    if plt.isinteractive():
        plt.draw()


@_fig_kwarg
def plot_s_error(netw: Network, other: Network,
                 error_function: ErrorFunctionsT = "average_l2_norm",
                 db: bool = False, *args, backend: str | None = None, **kwargs):
    r"""
    Plot the s-parameter error between two networks vs frequency.

    Parameters
    ----------
    netw : :class:`~skrf.network.Network`
        The reference network.
    other : :class:`~skrf.network.Network`
        The network to compare against ``netw``.
    error_function : str
        Error function name passed to :func:`~skrf.network.s_error`.
        One of ``average_l1_norm``, ``average_l2_norm``, ``maximum_l1_norm``,
        or ``average_normalized_l1_norm``. Default is ``average_l2_norm``.
    db : bool
        If True, plot :math:`20\log_{10}(\delta)`. Default is False.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``
    \*args, \*\*kwargs :
        Passed to :meth:`~skrf.frequency.Frequency.plot`.

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    See Also
    --------
    skrf.network.s_error
    skrf.network.Network.s_error
    """
    backend = _start_plot(backend, kwargs)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

    error = netw.s_error(other, error_function=error_function)
    if db:
        with np.errstate(divide='ignore'):
            error = 20 * np.log10(error)

    name_a = '' if netw.name is None else netw.name
    name_b = '' if other.name is None else other.name
    if 'label' not in kwargs:
        kwargs['label'] = f"{name_a} vs {name_b} ({error_function})"

    netw.frequency.plot(error, *args, backend=backend, **kwargs)

    if backend == "plotly":
        kwargs["ax"].update_yaxes(title_text=f"{'dB ' if db else ''}s-parameter error ({error_function})")
        return kwargs["ax"]
    plt.ylabel(f"{'dB ' if db else ''}s-parameter error ({error_function})")
    plt.legend()
    if plt.isinteractive():
        plt.draw()


def plot_s_db_time(netw: Network, *args, window: str | float | tuple[str, float]=('kaiser', 6),
        normalize: bool = True, center_to_dc: bool = None, **kwargs):
    return netw.windowed(window, normalize, center_to_dc).plot_s_time_db(*args,**kwargs)


# plotting
@_fig_kwarg
def plot_s_smith(netw: Network, m=None, n=None,r=1, ax=None, show_legend=True,\
        chart_type='z', draw_labels=False, label_axes=False, draw_vswr=None, *args,
        backend: str | None = None, **kwargs):
    r"""
    Plots the scattering parameter on a smith chart.

    Plots indices `m`, `n`, where `m` and `n` can be integers or
    lists of integers.


    Parameters
    ----------
    m : int, optional
            first index
    n : int, optional
            second index
    ax : matplotlib.Axes object, optional
            axes to plot on. in case you want to update an existing
            plot.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    show_legend : boolean, optional
            to turn legend show legend of not, optional
    chart_type : ['z','y']
        draw impedance or admittance contours
    draw_labels : Boolean
        annotate chart with impedance values
    label_axes : Boolean
        Label axis with titles `Real` and `Imaginary`
    border : Boolean
        draw rectangular border around image with ticks
    draw_vswr : list of numbers, Boolean or None
        draw VSWR circles. If True, default values are used.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    \*args : arguments, optional
            passed to the matplotlib.plot command
    \*\*kwargs : keyword arguments, optional
            passed to the matplotlib.plot command (or :class:`plotly.graph_objects.Scatter`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend


    See Also
    --------
    plot_vs_frequency_generic - generic plotting function
    smith -  draws a smith chart

    Examples
    --------
    >>> myntwk.plot_s_smith()
    >>> myntwk.plot_s_smith(m=0,n=1,color='b', marker='x')
    """
    # TODO: prevent this from re-drawing smith chart if one already
    # exists on current set of axes

    plotly = _resolve_backend(backend, ax) == 'plotly'
    if plotly:
        ax = _plotly_figure(ax, **_PLOTLY_SMITH_LAYOUT)
    else:
        # get current axis if user doesn't supply and axis
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()


    if m is None:
        M = range(netw.number_of_ports)
    else:
        M = [m]
    if n is None:
        N = range(netw.number_of_ports)
    else:
        N = [n]

    if 'label'  not in kwargs.keys():
        generate_label=True
    else:
        generate_label=False

    for m in M:
        for n in N:
            # set the legend label for this trace to the networks name if it
            # exists, and they didn't pass a name key in the kwargs
            if generate_label:
                kwargs['label'] = _get_label_str(netw, "S", m, n, usetex=False if plotly else None)

            if plotly:
                if not any(trace.meta == 'skrf_smith_chart' for trace in ax.data):
                    _plotly_smith(r, chart_type, draw_labels, ax=ax, draw_vswr=draw_vswr)
                _plotly_rect(netw.s[:,m,n].real, netw.s[:,m,n].imag, *args, show_legend=show_legend, ax=ax,
                             **kwargs)
                continue

            # plot the desired attribute vs frequency
            if len (ax.patches) == 0:
                smith(ax=ax, smithR = r, chart_type=chart_type, draw_labels=draw_labels, draw_vswr=draw_vswr,
                      backend='matplotlib')
            ax.plot(netw.s[:,m,n].real,  netw.s[:,m,n].imag, *args,**kwargs)

    if plotly:
        if label_axes:
            ax.update_xaxes(title_text='Real')
            ax.update_yaxes(title_text='Imaginary')
        return ax

    #draw legend
    if show_legend:
        ax.legend()
    ax.axis(np.array([-1.1,1.1,-1.1,1.1])*r)

    if label_axes:
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')


@_fig_kwarg
def plot_it_all(netw: Network, *args, **kwargs):
    r"""
    Plot dB, deg, smith, and complex in subplots.

    Plots the magnitude in dB in subplot 1, the phase in degrees in
    subplot 2, a smith chart in subplot 3, and a complex plot in
    subplot 4.

    Parameters
    ----------
    \*args : arguments, optional
            passed to the matplotlib.plot command
    \*\*kwargs : keyword arguments, optional
            passed to the matplotlib.plot command
            (or :class:`plotly.graph_objects.Scatter`)

    Returns
    -------
    None, or with the plotly `backend` a new plotly Figure with 2x2 subplots

    See Also
    --------
    plot_s_db - plot magnitude (in dB) of s-parameters vs frequency
    plot_s_deg - plot phase of s-parameters (in degrees) vs frequency
    plot_s_smith - plot complex s-parameters on smith chart
    plot_s_complex - plot complex s-parameters in the complex plane

    Examples
    --------
    >>> from skrf.data import ring_slot
    >>> ring_slot.plot_it_all()
    """
    kwargs['backend'] = _resolve_backend(kwargs.get('backend'), kwargs.get('ax'))
    if kwargs['backend'] == 'plotly':
        return _plotly_subplots([method(*args, **kwargs) for method in
                                 [netw.plot_s_db, netw.plot_s_deg, netw.plot_s_smith, netw.plot_s_complex]], cols=2)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.subplot(221)
    netw.plot_s_db(*args, **kwargs)
    plt.subplot(222)
    netw.plot_s_deg(*args, **kwargs)
    plt.subplot(223)
    netw.plot_s_smith(*args, **kwargs)
    plt.subplot(224)
    netw.plot_s_complex(*args, **kwargs)


# matplotlib style settings and the plotly template settings they become, see _plotly_template
_PLOTLY_STYLE_LAYOUT = {'axes.facecolor': 'plot_bgcolor', 'figure.facecolor': 'paper_bgcolor',
                        'font.size': 'font_size', 'axes.titlesize': 'title_font_size',
                        'legend.fontsize': 'legend_font_size'}
_PLOTLY_STYLE_AXES = {'axes.grid': 'showgrid', 'grid.color': 'gridcolor', 'axes.labelcolor': 'title_font_color',
                      'axes.labelsize': 'title_font_size', '{}tick.color': 'tickfont_color',
                      '{}tick.labelsize': 'tickfont_size'}
# matplotlib's relative font sizes
_PLOTLY_FONT_SCALES = {'xx-small': .579, 'x-small': .694, 'small': .833, 'medium': 1., 'large': 1.2,
                       'x-large': 1.44, 'xx-large': 1.728}


def _plotly_style_color(value: str) -> str:
    """plotly color for a color of a matplotlib style file, where hex colors are written without '#'."""
    if not isinstance(value, str):  # an rgb(a) tuple of `rc_dict`
        return 'rgb({},{},{})'.format(*(round(255 * c) for c in value[:3]))
    value = value.strip()
    return '#' + value if re.fullmatch('[0-9A-Fa-f]{6}', value) else _plotly_color(value)


def _plotly_template(rc: dict):
    """
    plotly template with the settings of the matplotlib style `rc` that plotly has, on top of plotly's default.

    `rc` maps the settings to their values, as written in a style file or as the types of rcParams.
    """
    go = _import_plotly()
    import plotly.io as pio

    font_size = float(rc.get('font.size', 10))

    def value(key: str, setting: str):
        if setting.endswith('color'):
            return _plotly_style_color(rc[key])
        v = str(rc[key]).strip()
        if setting == 'showgrid':
            return v.lower() == 'true'
        return font_size * _PLOTLY_FONT_SCALES[v] if v in _PLOTLY_FONT_SCALES else float(v)

    layout = {setting: value(key, setting) for key, setting in _PLOTLY_STYLE_LAYOUT.items() if key in rc}
    for axis in 'xy':
        for key, setting in _PLOTLY_STYLE_AXES.items():
            if key.format(axis) in rc:
                layout[f'{axis}axis_{setting}'] = value(key.format(axis), setting)
    if 'axes.prop_cycle' in rc:  # cycler('color', ['348ABD', ...])
        cycle = rc['axes.prop_cycle']
        cycle = re.findall("'([^']*)'", cycle)[1:] if isinstance(cycle, str) else cycle.by_key()['color']
        layout['colorway'] = [_plotly_style_color(c) for c in cycle]
    family = str(rc.get('font.family', '')).strip()
    if family in ('serif', 'sans-serif', 'monospace'):
        layout['font_family'] = ', '.join(re.findall("'([^']*)'", rc.get(f'font.{family}', '')) + [family])
    template = go.layout.Template(pio.templates['plotly'])
    template.layout.update(layout)
    if 'lines.linewidth' in rc:
        template.data.scatter = [go.Scatter(line_width=float(rc['lines.linewidth']))]
    return template


def stylely(rc_dict: dict = None, style_file: str = 'skrf.mplstyle'):
    """
    Loads the rc-params from the specified file (file must be located in skrf/data).

    The style applies to matplotlib and to plotly, whichever are installed. For plotly,
    the settings that plotly has make a plotly template named after `style_file`, such
    as 'skrf', which becomes plotly's default template.

    Parameters
    ----------
    rc_dict : dict, optional
        rcParams such as ``{'font.size': 12}`` over those of `style_file`, by default {}
    style_file : str, optional
        style file, by default 'skrf.mplstyle'
    """
    from .data import pwd  # delayed to solve circular import

    path, styled = os.path.join(pwd, style_file), False
    with suppress(ImportError):
        import matplotlib.pyplot as plt

        plt.style.use(path)
        plt.rcParams.update(rc_dict if rc_dict else {})
        styled = True
    with suppress(ImportError):
        import plotly.io as pio

        rc = {}
        with open(path) as f:
            for line in f:
                key, sep, value = line.split('#', 1)[0].partition(':')
                if sep and key.strip():
                    rc[key.strip()] = value.strip()
        name = os.path.splitext(style_file)[0]
        pio.templates[name] = _plotly_template({**rc, **(rc_dict or {})})
        pio.templates.default = name
        styled = True
    if not styled:
        warnings.warn("Could not import matplotlib or plotly to style", ImportWarning, stacklevel=2)

# Network Set Plotting Commands
@_fig_kwarg
def animate(self: NetworkSet, attr: str = 's_deg', ylims: tuple = (-5, 5),
            xlims: tuple | None = None, show: bool = True,
            savefigs: bool = False, dir_: str = '.', *args, **kwargs):
    r"""
    Animate a property of the networkset.

    This loops through all elements in the NetworkSet and calls
    a plotting attribute (ie Network.plot_`attr`), with given \*args
    and \*\*kwargs.

    Parameters
    ----------
    attr : str, optional
        plotting property of a Network (ie 's_db', 's_deg', etc)
        Default is 's_deg'
    ylims : tuple, optional
        passed to ylim. needed to have consistent y-limits across frames.
        Default is (-5 ,5).
    xlims : tuple or None, optional.
        passed to xlim. Default is None.
    show : bool, optional
        show each frame as its animated. Default is True.
    savefigs : bool, optional
        save each frame as a png. Default is False.
        With plotly, this needs the kaleido package.

    \*args, \*\*kwargs :
        passed to the Network plotting function, including ``backend``
        (default ``skrf.plotting.options.backend``)

    Returns
    -------
    None, or with the plotly backend a plotly Figure with one frame per
    Network, a slider and play/pause buttons. `show` is not used then.

    Note
    ----
    using `label=None` will speed up animation significantly,
    because it prevents the legend from drawing

    to create video paste this:

        !avconv -r 10 -i out_%5d.png  -vcodec huffyuv out.avi

    or (depending on your ffmpeg version)

        !ffmpeg -r 10 -i out_%5d.png  -vcodec huffyuv out.avi

    Examples
    --------
    >>> ns.animate('s_deg', ylims=(-5,5), label=None)

    """
    kwargs['backend'] = _resolve_backend(kwargs.get('backend'), kwargs.get('ax'))
    if kwargs['backend'] == 'plotly':
        if kwargs.pop('ax', None) is not None:
            raise TypeError("animate draws a new plotly Figure, fig is not supported")
        go = _import_plotly()
        unit = self[0].frequency.unit
        frames = []
        for idx, k in enumerate(self):
            ntwk = k.windowed() if 'time' in attr else k
            # every frame in the frequency unit of the first Network
            fig = ntwk.__getattribute__('plot_' + attr)(*args, ax=_plotly_figure(meta={'skrf_frequency_unit': unit}),
                                                        **kwargs)
            if ylims is not None:
                fig.update_yaxes(range=ylims)
            if xlims is not None:  # in Hz for frequency, like the matplotlib x data
                scale = 1 if 'time' in attr else ntwk.frequency.multiplier_dict[unit.lower()]
                fig.update_xaxes(range=[x / scale for x in xlims])
            if savefigs:
                fig.write_image(os.path.join(dir_, 'out_%.5i' % idx + '.png'))
            if idx == 0:
                first = fig
            frames.append(go.Frame(data=fig.data, name=str(idx)))
        first.frames = frames
        steps = [dict(method='animate', label=k.name or str(idx),
                      args=[[str(idx)], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
                 for idx, k in enumerate(self)]
        buttons = [dict(label='Play', method='animate',
                        args=[None, dict(fromcurrent=True, frame=dict(duration=500, redraw=True))]),
                   dict(label='Pause', method='animate',
                        args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])]
        return first.update_layout(
            # the slider under the x-axis title, and the buttons under the slider
            height=650, margin_b=200,
            sliders=[dict(steps=steps, x=0, len=1, y=0, yanchor='top', pad=dict(t=60))],
            updatemenus=[dict(type='buttons', direction='left', buttons=buttons, x=0, y=0,
                              xanchor='left', yanchor='top', pad=dict(t=150))])

    import matplotlib.pyplot as plt

    was_interactive = plt.isinteractive()
    plt.ioff()

    for idx, k in enumerate(self):
        plt.clf()
        if 'time' in attr:
            tmp_ntwk = k.windowed()
            tmp_ntwk.__getattribute__('plot_' + attr)(*args, **kwargs)
        else:
            k.__getattribute__('plot_' + attr)(*args, **kwargs)
        if ylims is not None:
            plt.ylim(ylims)
        if xlims is not None:
            plt.xlim(xlims)
        # rf.legend_off()
        plt.draw()
        if show:
            plt.show()
        if savefigs:
            fname = os.path.join(dir_, 'out_%.5i' % idx + '.png')
            plt.savefig(fname)

    if was_interactive:
        plt.ion()


#------------------------------
#
# NetworkSet plotting functions
#
#------------------------------

@_fig_kwarg
def plot_uncertainty_bounds_component(
        self: NetworkSet, attribute: PrimaryPropertiesT,
        m: int | None = None, n: int | None = None, *,
        type: str = 'shade', n_deviations: int = 3,
        alpha: float = .3, color_error: str | None = None,
        markevery_error: int = 20, ax: Axes = None,
        ppf: bool = None, kwargs_error: dict = None,
        backend: str | None = None, **kwargs):
    r"""
    Plot mean value of a NetworkSet with +/- uncertainty bounds in an Network's attribute.

    This is designed to represent uncertainty in a scalar component of the s-parameter.
    for example plotting the uncertainty in the magnitude would be expressed by,

    .. math::

        mean(|s|) \pm std(|s|)

    The order of mean and abs is important.


    Parameters
    ----------
    attribute : str
        attribute of Network type to analyze
    m : int or None
        first index of attribute matrix. Default is None (all)
    n : int or None
        second index of attribute matrix. Default is None (all)
    type : str
        ['shade' | 'bar'], type of plot to draw
    n_deviations : int
        number of std deviations to plot as bounds
    alpha : float
        passed to matplotlib.fill_between() command. [number, 0-1]
    color_error : str
        color of the +- std dev fill shading. Default is None.
    markevery_error : float
        tbd
    type : str
        if type=='bar', this controls frequency of error bars
    ax : matplotlib axes object
        Axes to plot on. Default is None.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    ppf : function
        post processing function. a function applied to the
        upper and lower bounds. Default is None
     kwargs_error : dict
         dictionary of kwargs to pass to the fill_between or
         errorbar plot command depending on value of type
         (to :class:`plotly.graph_objects.Scatter` with plotly).
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*\*kwargs :
        passed to Network.plot_s_re command used to plot mean response

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Note
    ----
    For phase uncertainty you probably want s_deg_unwrap, or
    similar. uncertainty for wrapped phase blows up at +-pi.

    """
    backend = _resolve_backend(backend, ax)
    plotly = backend == 'plotly'
    ax = _plotly_figure(ax) if plotly else _get_axes(ax)

    kwargs_error = kwargs_error if kwargs_error else {}

    if m is None:
        M = range(self[0].number_of_ports)
    else:
        M = [m]
    if n is None:
        N = range(self[0].number_of_ports)
    else:
        N = [n]

    for m in M:
        for n in N:

            plot_attribute = attribute

            ntwk_mean = self.__getattribute__('mean_'+attribute)
            ntwk_std = self.__getattribute__('std_'+attribute)
            ntwk_std.s = n_deviations * ntwk_std.s

            upper_bound = (ntwk_mean.s[:, m, n] + ntwk_std.s[:, m, n]).squeeze()
            lower_bound = (ntwk_mean.s[:, m, n] - ntwk_std.s[:, m, n]).squeeze()

            if ppf is not None:
                if type == 'bar':
                    raise NotImplementedError('the \'ppf\' options don\'t work correctly with the bar-type error plots')
                ntwk_mean.s = ppf(ntwk_mean.s)
                upper_bound = ppf(upper_bound)
                lower_bound = ppf(lower_bound)
                lower_bound[np.isnan(lower_bound)] = min(lower_bound)
                if ppf in [mf.magnitude_2_db, mf.mag_2_db]:  # fix of wrong ylabels due to usage of ppf for *_db plots
                    if attribute == 's_mag':
                        plot_attribute = 's_db'
                    elif attribute == 's_time_mag':
                        plot_attribute = 's_time_db'

            if type == 'shade':
                ntwk_mean.plot_s_re(ax=ax, m=m, n=n, backend=backend, **kwargs)
                if plotly:
                    _plotly_fill_between(ax, ntwk_mean.frequency, lower_bound.real, upper_bound.real,
                                         color_error or _plotly_trace_color(ax), alpha, **kwargs_error)
                    continue
                if color_error is None:
                    color_error = ax.get_lines()[-1].get_color()
                ax.fill_between(ntwk_mean.frequency.f,
                                lower_bound.real, upper_bound.real, alpha=alpha, color=color_error,
                                **kwargs_error)
                # ax.plot(ntwk_mean.frequency.f_scaled, ntwk_mean.s[:,m,n],*args,**kwargs)

            elif type == 'bar':
                ntwk_mean.plot_s_re(ax=ax, m=m, n=n, backend=backend, **kwargs)
                y = ntwk_mean.s_re[:, m, n].squeeze()[::markevery_error]
                yerr = ntwk_std.s_mag[:, m, n].squeeze()[::markevery_error]
                if plotly:
                    x, _ = _plotly_frequency_x(ax, ntwk_mean.frequency)
                    ax.add_trace(_import_plotly().Scatter(x=x[::markevery_error], y=y, mode='markers',
                                                          error_y=dict(array=yerr),
                                                          marker_color=color_error or _plotly_trace_color(ax),
                                                          showlegend=False, **kwargs_error))
                    continue
                if color_error is None:
                    color_error = ax.get_lines()[-1].get_color()
                ax.errorbar(ntwk_mean.frequency.f[::markevery_error], y, yerr=yerr,
                            color=color_error, **kwargs_error)

            else:
                raise(ValueError('incorrect plot type'))

            ax.set_ylabel(self[0].Y_LABEL_DICT.get(plot_attribute[2:], ''))  # use only the function of the attribute
            scale_frequency_ticks(ax, ntwk_mean.frequency.unit, backend='matplotlib')
            ax.axis('tight')

    if plotly:
        # the label of the last component, as matplotlib
        return ax.update_yaxes(title_text=self[0].Y_LABEL_DICT.get(plot_attribute[2:], ''))

@_fig_kwarg
def plot_minmax_bounds_component(self: NetworkSet, attribute: PrimaryPropertiesT, m: int = 0, n: int = 0,
                                 *, type: str = 'shade',
                                 alpha: float = .3, color_error: str | None = None,
                                 markevery_error: int = 20, ax: Axes = None,
                                 ppf: bool = None, kwargs_error: dict = None,
                                 backend: str | None = None, **kwargs):
    r"""
    Plots mean value of the NetworkSet with minimum and maximum bounds in an Network's attribute.

    This is designed to represent min/max in a scalar component of the s-parameter. For example
    plotting the min/max in the magnitude would be expressed by

    .. math::

        min(|s|)

        mean(|s|)

        max(|s|)

    The order of mean and abs is important.

    Parameters
    ----------
    attribute : str
        attribute of Network type to analyze
    m : int
        first index of attribute matrix
    n : int
        second index of attribute matrix
    type : str
        ['shade' | 'bar'], type of plot to draw
    alpha : float
        passed to matplotlib.fill_between() command. [number, 0-1]
    color_error : str
        color of the min/max fill shading. Default is None.
    markevery_error : float
        tbd
    type : str
        if type=='bar', this controls frequency of error bars
    ax : matplotlib axes object
        Axes to plot on. Default is None.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    ppf : function
        post processing function. a function applied to the
        upper and lower bounds. Default is None
     kwargs_error : dict
         dictionary of kwargs to pass to the fill_between or
         errorbar plot command depending on value of type
         (to :class:`plotly.graph_objects.Scatter` with plotly).
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*\*kwargs :
        passed to Network.plot_s_re command used to plot mean response

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Note
    ----
    For phase uncertainty you probably want s_deg_unwrap, or
    similar.  Uncertainty for wrapped phase blows up at +-pi.

    """
    backend = _resolve_backend(backend, ax)
    plotly = backend == 'plotly'
    ax = _plotly_figure(ax) if plotly else _get_axes(ax)

    kwargs_error = kwargs_error if kwargs_error else {}

    ntwk_mean = self.__getattribute__('mean_'+attribute)
    ntwk_std = self.__getattribute__('std_'+attribute)

    lower_bound = self.__getattribute__('min_'+attribute).s_re[:,m,n].squeeze()
    upper_bound = self.__getattribute__('max_'+attribute).s_re[:,m,n].squeeze()

    if ppf is not None:
        if type =='bar':
            raise NotImplementedError('the \'ppf\' options don\'t work correctly with the bar-type error plots')
        ntwk_mean.s = ppf(ntwk_mean.s)
        upper_bound = ppf(upper_bound)
        lower_bound = ppf(lower_bound)
        lower_bound[np.isnan(lower_bound)]=min(lower_bound)
        if ppf in [mf.magnitude_2_db, mf.mag_2_db]: # quickfix of wrong ylabels due to usage of ppf for *_db plots
            if attribute == 's_mag':
                attribute = 's_db'
            elif attribute == 's_time_mag':
                attribute = 's_time_db'

    if type == 'shade':
        ntwk_mean.plot_s_re(ax=ax,m=m,n=n, backend=backend, **kwargs)
        if plotly:
            _plotly_fill_between(ax, ntwk_mean.frequency, lower_bound, upper_bound,
                                 color_error or _plotly_trace_color(ax), alpha, **kwargs_error)
            return ax.update_yaxes(title_text=self[0].Y_LABEL_DICT.get(attribute[2:], ''))
        if color_error is None:
            color_error = ax.get_lines()[-1].get_color()
        ax.fill_between(ntwk_mean.frequency.f,
                        lower_bound, upper_bound, alpha=alpha, color=color_error,
                        **kwargs_error)
        #ax.plot(ntwk_mean.frequency.f_scaled,ntwk_mean.s[:,m,n],*args,**kwargs)
    elif type =='bar':
        raise (NotImplementedError)
        ntwk_mean.plot_s_re(ax=ax, m=m, n=n, backend=backend, **kwargs)
        if color_error is None:
            color_error = ax.get_lines()[-1].get_color()
        ax.errorbar(ntwk_mean.frequency.f[::markevery_error],
                    ntwk_mean.s_re[:,m,n].squeeze()[::markevery_error],
                    yerr=ntwk_std.s_mag[:,m,n].squeeze()[::markevery_error],
                    color=color_error,**kwargs_error)

    else:
        raise(ValueError('incorrect plot type'))

    ax.set_ylabel(self[0].Y_LABEL_DICT.get(attribute[2:], ''))  # use only the function of the attribute
    scale_frequency_ticks(ax, ntwk_mean.frequency.unit, backend='matplotlib')
    ax.axis('tight')

@_fig_kwarg
def plot_violin(self: NetworkSet, attribute: PrimaryPropertiesT, m: int = 0, n: int = 0,
                         *, widths: float = None, showmeans: bool = True,
                         showextrema: bool = True, showmedians: bool = False,
                         quantiles = None, points: int = 100, bw_method = None,
                         ax: Axes = None, backend: str | None = None, **kwargs
    ):
    r"""Plots the violin plot of the network set for the desired attribute.

    A violin plot provides the distribution of the attribute at each frequency point, and optionally the
    extrema, mean, and median. The plot becomes cluttered quickly with many frequencies, so reducing the number
    with :meth:`NetworkSet.interpolate_frequency` is recommended.

    Parameters
    ----------
    attribute : str
        attribute of Network type to analyze
    m : int
        first index of attribute matrix
    n : int
        second index of attribute matrix
    widths : float
        The maximum width of each violin in units of the positions axis.
        The default is 0.75 of the distance between the first two frequencies.
    showmeans : bool
        Whether to show the mean with a line.
    showextrema : bool
        Whether to show the extrema with a line.
    showmedians : bool
        Whether to show the median with a line.
    quantiles : ArrayLike
        If not None, set a list of floats in interval [0, 1] for each violin,
        which stands for the quantiles that will be rendered for that violin.
    points : int
        The number of points to evaluate each of the gaussian kernel density estimations at.
    bw_method : {'scott', 'silverman'} or float or callable, default: 'scott'
        _description_. Defaults to None.
    ax : matplotlib axes object
        Axes to plot on. Default is None.
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        With plotly, `showmedians` shows a box with the median, `points` and `showextrema` are ignored, and
        `quantiles` and `bw_method` are not supported.
    \*\*kwargs :
        passed to :meth:`matplotlib.pyplot.violinplot` (or :class:`plotly.graph_objects.Violin`)

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Note
    ----
    For phase plots you probably want s_deg_unwrap, or
    similar.  Uncertainty for wrapped phase blows up at +-pi.
    """
    plotly = _resolve_backend(backend, ax) == 'plotly'
    ax = _plotly_figure(ax) if plotly else _get_axes(ax)

    freq = self.ntwk_set[0].f

    # default widths to 3/4 distance between frequencies
    if not widths and len(freq) > 1:
        widths = (freq[1]-freq[0])*0.75
    elif not widths:
        widths = 0.5

    data = np.array([getattr(p, attribute)[:,m,n] for p in self.ntwk_set])

    if plotly:
        if quantiles is not None or bw_method is not None:
            raise NotImplementedError("quantiles and bw_method are not supported by the plotly backend")
        x, x_label = _plotly_frequency_x(ax, self.ntwk_set[0].frequency)
        x_per_hz = x[-1] / freq[-1] if freq[-1] else 1  # widths are in Hz, as freq
        ax.add_trace(_import_plotly().Violin(x=np.tile(x, len(data)), y=data.ravel(), width=widths * x_per_hz,
                                             spanmode='hard', meanline_visible=showmeans,
                                             box_visible=showmedians, points=False, **kwargs))
        ax.update_xaxes(title_text=x_label)
        return ax.update_yaxes(title_text=self[0].Y_LABEL_DICT.get(attribute[2:], ''))

    ax.violinplot(data, freq, widths=widths, showmeans=showmeans, showextrema=showextrema,
                  showmedians=showmedians, quantiles=quantiles, points=points, bw_method=bw_method, **kwargs)

    ax.set_xlabel(f'Frequency ({self.ntwk_set[0].frequency.unit})')
    ax.set_ylabel(self[0].Y_LABEL_DICT.get(attribute[2:], ''))  # use only the function of the attribute
    scale_frequency_ticks(ax, self.ntwk_set[0].frequency.unit, backend='matplotlib')
    ax.axis('tight')

def plot_uncertainty_bounds_s_db(self: NetworkSet, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute='s_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    See plot_uncertainty_bounds for help.

    """
    kwargs.update({'ppf':mf.magnitude_2_db})
    return self.plot_uncertainty_bounds_component("s_mag", *args,**kwargs)

def plot_minmax_bounds_s_db(self: NetworkSet, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    See plot_uncertainty_bounds for help.

    """
    kwargs.update({'ppf':mf.magnitude_2_db})
    return self.plot_minmax_bounds_component("s_mag", *args,**kwargs)

def plot_minmax_bounds_s_db10(self: NetworkSet, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'ppf':mf.mag_2_db10})
    return self.plot_minmax_bounds_component("s_mag", *args,**kwargs)

def plot_uncertainty_bounds_s_time_db(self: NetworkSet, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    See plot_uncertainty_bounds for help.

    """
    kwargs.update({'ppf':mf.magnitude_2_db})
    return self.plot_uncertainty_bounds_component("s_time_mag", *args,**kwargs)

def plot_minmax_bounds_s_time_db(self: NetworkSet, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    See plot_uncertainty_bounds for help.

    """
    kwargs.update({'ppf':mf.magnitude_2_db})
    return self.plot_minmax_bounds_component("s_time_mag", *args, **kwargs)

@_fig_kwarg
def plot_uncertainty_decomposition(self: NetworkSet, m: int = 0, n: int = 0, *,
                                   backend: str | None = None, **kwargs):
    r"""
    Plot the total and component-wise uncertainty.

    Parameters
    ----------
    m : int
        first s-parameters index
    n :
        second s-parameter index
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``
    \*\*kwargs :
        passed to :meth:`~skrf.network.Network.plot_s_mag`

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    """
    backend = _start_plot(backend, kwargs)
    if backend == "plotly":
        if self.name is not None:
            kwargs["ax"].update_layout(title_text=f"Uncertainty Decomposition: {self.name} "
                                                  f"S{self.ntwk_set[0]._fmt_trace_name(m, n)}")
    else:
        import matplotlib.pyplot as plt
        if self.name is not None:
            plt.title(f"Uncertainty Decomposition: {self.name} $S_{{{self.ntwk_set[0]._fmt_trace_name(m,n)}}}$")
    self.std_s.plot_s_mag(label='Distance', m=m,n=n, backend=backend, **kwargs)
    self.std_s_re.plot_s_mag(label='Real',  m=m,n=n, backend=backend, **kwargs)
    self.std_s_im.plot_s_mag(label='Imaginary',  m=m,n=n, backend=backend, **kwargs)
    self.std_s_mag.plot_s_mag(label='Magnitude',  m=m,n=n, backend=backend, **kwargs)
    self.std_s_arcl.plot_s_mag(label='Arc-length',  m=m,n=n, backend=backend, **kwargs)
    if backend == "plotly":
        return kwargs["ax"]

@_fig_kwarg
def plot_logsigma(self: NetworkSet, label_axis: bool = True, *args, backend: str | None = None, **kwargs):
    r"""
    Plot the uncertainty for the set in units of log-sigma.

    Log-sigma is the complex standard deviation, plotted in units
    of dB's.

    Parameters
    ----------
    label_axis : bool, optional
        Default is True.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``
    \*args, \*\*kwargs : arguments
        passed to self.std_s.plot_s_db()

    Returns
    -------
    None, or the plotly Figure with the plotly backend
    """
    if _resolve_backend(backend, kwargs.get("ax")) == "plotly":
        fig = self.std_s.plot_s_db(*args, backend="plotly", **kwargs)
        if label_axis:
            fig.update_yaxes(title_text='Standard Deviation(dB)')
        return fig

    import matplotlib.pyplot as plt

    self.std_s.plot_s_db(*args, backend="matplotlib", **kwargs)
    if label_axis:
        plt.ylabel('Standard Deviation(dB)')


def signature(self: NetworkSet, m: int = 0, n: int = 0, component: str = 's_mag',
              vmax: Number | None = None, vs_time: bool = False,
              cbar_label: str | None = None,
              *args, backend: str | None = None, **kwargs):
    r"""
    Visualization of a NetworkSet.

    Creates a colored image representing the some component
    of each Network in the  NetworkSet, vs frequency.

    Parameters
    ------------
    m : int, optional
        first s-parameters index. Default is 0.
    n : int, optional
        second s-parameter index. Default is 0.
    component : ['s_mag','s_db','s_deg' ..]
        scalar component of Network to visualize. should
        be a property of the Network object.
    vmax : number or None.
        sets upper limit of colorbar, if None, will be set to
        3*mean of the magnitude of the complex difference.
        Default is None.
    vs_time: Boolean, optional.
        if True, then we assume each Network.name was made with
        rf.now_string, and we make the y-axis a datetime axis.
        Default is False.
    cbar_label: String or None, optional
        label for the colorbar. Default is None
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
    \*args,\*\*kw : arguments, keyword arguments
        passed to :func:`~pylab.imshow`, or with plotly keyword arguments
        to :class:`plotly.graph_objects.Heatmap`

    Returns
    -------
    The matplotlib image, or the plotly Figure with the plotly backend
    """
    mat = np.array([self[k].__getattribute__(component)[:, m, n] \
                     for k in range(len(self))])

    if _resolve_backend(backend) == 'plotly':
        if args:
            raise TypeError(f"Positional imshow arguments {args!r} are not supported by the plotly backend")
        go = _import_plotly()
        freq = self[0].frequency
        y = [now_string_2_dt(k.name) for k in self] if vs_time else np.arange(len(self))
        fig = _plotly_figure()
        if vmax is not None:  # plotly ignores zmax without zmin
            kwargs.setdefault('zmin', np.nanmin(mat))
        fig.add_trace(go.Heatmap(z=mat, x=freq.f_scaled, y=y, zmax=vmax, colorbar=dict(title=cbar_label), **kwargs))
        # first Network at the top, like imshow
        return fig.update_layout(xaxis_title=f'Frequency ({freq.unit})', yaxis_title='Time' if vs_time else 'Network #',
                                 yaxis_autorange='reversed')

    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    # if vmax is None:
    #    vmax = 3*mat.mean()

    if vs_time:
        # create a datetime index
        dt_idx = [now_string_2_dt(k.name) for k in self]
        mpl_times = date2num(dt_idx)
        y_max = mpl_times[0]
        y_min = mpl_times[-1]

    else:
        y_min = len(self)
        y_max = 0

    # creates x and y scales
    freq = self[0].frequency
    extent = [freq.f_scaled[0], freq.f_scaled[-1], y_min, y_max]

    # set default imshow kwargs
    kw = {'extent': extent, 'aspect': 'auto', 'interpolation': 'nearest',
          'vmax': vmax}
    # update the users kwargs
    kw.update(kwargs)
    img = plt.imshow(mat, *args, **kw)

    if vs_time:
        ax = plt.gca()
        ax.yaxis_date()
        # date_format = plt.DateFormatter('%M:%S.%f')
        # ax.yaxis.set_major_formatter(date_format)
        # cbar.set_label('Magnitude (dB)')
        plt.ylabel('Time')
    else:
        plt.ylabel('Network #')

    plt.grid(0)
    freq.labelXAxis(backend='matplotlib')

    cbar = plt.colorbar()
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    return img

def plot_contour(freq: Frequency,
                 x: NumberLike, y: NumberLike, z: NumberLike,
                 min0max1: int, graph: bool = True,
                 cmap: str = 'plasma_r', title: str = '', *, backend: str | None = None,
                 **kwargs):
    r"""
    Create a contour plot.

    Parameters
    ----------
    freq : :skrf.Frequency:
        Frequency object.
    x : array
        x points
    y : array
        y points.
    z : array
        z points.
    min0max1 : int
        0 for min, 1 for max.
    graph : bool, optional
        plot graph if True. The default is True.
    cmap : str, optional
        Colormap label. The default is 'plasma_r'.
    title : str, optional
        Figure title. The default is ''.
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.
        With plotly, `cmap` is a plotly colorscale name.
    \*\*kwargs : dict
        Other parameters passed to `matplotlib.pyplot.subplots()`
        (or the layout of the plotly Figure).

    Returns
    -------
    GAMopt : :skrf.Network:
        Network
    VALopt : float
        min or max.

    """
    from matplotlib import tri

    from . import Network

    ri =  np.linspace(0,1, 50)
    ti =  np.linspace(0,2*np.pi, 150)
    Ri , Ti = np.meshgrid(ri, ti)
    xi = np.linspace(-1,1, 50)
    Xi, Yi = np.meshgrid(xi, xi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Zi = interpolator(Xi, Yi)
    if min0max1 == 1 :
        VALopt = np.max(z)
    else :
        VALopt = np.min(z)
    GAMopt = Network(f=[freq], s=x[z==VALopt] +1j*y[z==VALopt])

    if graph :
        an = np.linspace(0, 2*np.pi, 50)
        cs, sn = np.cos(an), np.sin(an)
        if _resolve_backend(backend) == 'plotly':
            go = _import_plotly()
            fig = _plotly_figure(**kwargs)
            fig.add_trace(go.Contour(x=xi, y=xi, z=Zi.filled(np.nan), ncontours=20, colorscale=cmap,
                                     line=dict(color='black', width=0.5)))
            for x_line, y_line, color in ((cs, sn, 'black'), (cs, sn*0, 'green'), ((1+cs)/2, sn/2, 'black')):
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color=color, width=0.5),
                                         hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='black', size=2),
                                     showlegend=False))
            fig.update_xaxes(range=[-1, 1], visible=False, constrain='domain')
            fig.update_yaxes(range=[-1, 1], visible=False, scaleanchor='x', constrain='domain')
            fig.update_layout(title_text=title, plot_bgcolor='white')
            fig.show()
            return GAMopt, VALopt

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(**kwargs)
        plt.plot(cs, sn, color='k', lw=0.25)
        plt.plot(cs, sn*0, color='g', lw=0.25)
        plt.plot((1+cs)/2, sn/2, color='k', lw=0.25)
        plt.axis('equal')
        ax.set_axis_off()
        ax.contour(Xi, Yi, Zi, levels=20, vmin=Zi.min(), vmax= Zi.max(), linewidths=0.5,  colors='k')
        cntr1 = ax.contourf(Xi, Yi, Zi, levels=20, vmin=Zi.min(), vmax= Zi.max(),cmap=cmap)
        fig.colorbar(cntr1, ax=ax)
        ax.plot(x, y, 'o', ms=0.3, color='k')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        plt.title(title)
        plt.show()
    return GAMopt, VALopt

@_fig_kwarg
def plot_prop_complex(netw: Network, prop_name: str,
                    m=None, n=None, ax=None,
                    show_legend=True, *, backend: str | None = None, **kwargs):
    r"""
    plot the Network attribute :attr:`{}` vs frequency.

    Parameters
    ----------
    attribute : string
        Network attribute to plot
    m : int, optional
        first index of s-parameter matrix, if None will use all
    n : int, optional
        second index of the s-parameter matrix, if None will use all
    ax : :class:`matplotlib.Axes` object, optional
        An existing Axes object to plot on
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    show_legend : Boolean
        draw legend or not
    y_label : string, optional
        the y-axis label
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    \*args,\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot`

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Note
    ----
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    --------
    >>> myntwk.plot_{}(m=1,n=0,color='r')
    """
    plotly = _resolve_backend(backend, ax) == 'plotly'
    if plotly:
        ax = _plotly_figure(ax)
    plot_z = _plotly_complex if plotly else partial(plot_complex_rectangular, backend='matplotlib')

    # create index lists, if not provided by user
    if m is None:
        M = range(netw.number_of_ports)
    else:
        M = [m]
    if n is None:
        N = range(netw.number_of_ports)
    else:
        N = [n]

    if 'label'  not in kwargs.keys():
        gen_label = True
    else:
        gen_label = False

    for m in M:
        for n in N:
            # set the legend label for this trace to the networks
            # name if it exists, and they didn't pass a name key in
            # the kwargs
            if gen_label:
                kwargs['label'] = _get_label_str(netw, prop_name[0].upper(), m, n,
                                                 usetex=False if plotly else None)

            # plot the desired attribute vs frequency
            plot_z(
                z=getattr(netw, prop_name)[:, m, n],
                show_legend=show_legend, ax=ax,
                **kwargs)

    return ax if plotly else None

@_fig_kwarg
def plot_prop_polar(netw: Network, prop_name: str,
                    m=None, n=None, ax=None,
                    show_legend=True, *, backend: str | None = None, **kwargs):

    r"""
    plot the Network attribute :attr:`{}` vs frequency.

    Parameters
    ----------
    attribute : string
        Network attribute to plot
    m : int, optional
        first index of s-parameter matrix, if None will use all
    n : int, optional
        second index of the s-parameter matrix, if None will use all
    ax : :class:`matplotlib.Axes` object, optional
        An existing Axes object to plot on
    fig : plotly Figure, optional
        figure to draw on with the plotly backend.
    show_legend : Boolean
        draw legend or not
    y_label : string, optional
        the y-axis label
    backend : {'matplotlib', 'plotly'}, optional
        plotting backend, default ``skrf.plotting.options.backend``.

    \*args,\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot`

    Returns
    -------
    None, or the plotly Figure with the plotly backend

    Note
    ----
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    --------
    >>> myntwk.plot_{}(m=1,n=0,color='r')
    """
    plotly = _resolve_backend(backend, ax) == 'plotly'
    if plotly:
        ax = _plotly_figure(ax)
    plot_z = partial(plot_complex_polar, backend='plotly' if plotly else 'matplotlib')

    # create index lists, if not provided by user
    if m is None:
        M = range(netw.number_of_ports)
    else:
        M = [m]
    if n is None:
        N = range(netw.number_of_ports)
    else:
        N = [n]

    if 'label'  not in kwargs.keys():
        gen_label = True
    else:
        gen_label = False

    for m in M:
        for n in N:
            # set the legend label for this trace to the networks
            # name if it exists, and they didn't pass a name key in
            # the kwargs
            if gen_label:
                kwargs['label'] = _get_label_str(netw, prop_name[0].upper(), m, n,
                                                 usetex=False if plotly else None)

            # plot the desired attribute vs frequency
            plot_z(
                z = getattr(netw,prop_name)[:,m,n],
                show_legend = show_legend, ax = ax,
                **kwargs)

    return ax if plotly else None
