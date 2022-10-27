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
    plot_v_frequency
    plot_it_all

    plot_minmax_bounds_component
    plot_minmax_bounds_s_db
    plot_minmax_bounds_s_db10
    plot_minmax_bounds_s_time_db

    plot_uncertainty_bounds_component
    plot_uncertainty_bounds_s
    plot_uncertainty_bounds_s_db
    plot_uncertainty_bounds_s_time_db

    plot_passivity
    plot_logsigma

    plot_circuit_graph

    plot_contour

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
from . constants import NumberLike
from numbers import Number
from typing import Callable, Tuple, Union, List
import os
import sys
import getpass
import warnings

import matplotlib as mpl
# if running on remote mode on a linux server which does not have a display (like Docker images for CI)
# seems to cause problems only in Python2, so let's try reconfiguring the backend only in that case
if os.name == 'posix' and not os.environ.get('DISPLAY', '') and mpl.rcParams['backend'] != 'agg' and sys.version.startswith('2'):
    print('No display found. Using non-interactive Agg backend.')
    # https://matplotlib.org/faq/usage_faq.html
    if getpass.getuser() != 'jovyan':  # only if not running on Binder
        mpl.use('Agg')
import numpy as npy
import matplotlib.pyplot as plt
from matplotlib import ticker, rcParams
from matplotlib.patches import Circle   # for drawing smith chart
from matplotlib.pyplot import quiver
from matplotlib.dates import date2num

from . import network, frequency, calibration, networkSet, circuit
from . import mathFunctions as mf
from . util import now_string_2_dt
import matplotlib.tri as tri

try:
    import networkx as nx
except ImportError as e:
    pass

SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'
SI_CONVERSION = {key: 10**((8-i)*3) for i, key in enumerate(SI_PREFIXES_ASCII)}


def scale_frequency_ticks(ax: plt.Axes, funit: str):
    """
    Scale frequency axis ticks.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib figure axe
    funit : str
        frequency unit string as in :data:`~skrf.frequency.Frequency.unit`

    Raises
    ------
    ValueError
        if invalid unit is passed
    """
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


def smith(smithR: Number = 1, chart_type: str = 'z', draw_labels: bool = False,
          border: bool = False, ax: Union[plt.Axes, None] = None, ref_imm: float = 1.0,
          draw_vswr: Union[List, bool, None] = None):
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
    ref_imm : number, optional
        Reference immittance for center of Smith chart. Only changes
        labels, if printed.
        Default is 1.0.
    draw_vswr : list of numbers, Boolean or None, optional
        draw VSWR circles. If True, default values are used.
        Default is None.

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Smith_chart

    """
    ##TODO: fix this function so it doesnt suck
    if ax is None:
        ax1 = plt.gca()
    else:
        ax1 = ax

    # contour holds matplotlib instances of: pathes.Circle, and lines.Line2D, which
    # are the contours on the smith chart
    contour = []

    # these are hard-coded on purpose,as they should always be present
    rHeavyList = [0,1]
    xHeavyList = [1,-1]

    #TODO: fix this
    # these could be dynamically coded in the future, but work good'nuff for now
    if not draw_labels:
        rLightList = npy.logspace(3,-5,9,base=.5)
        xLightList = npy.hstack([npy.logspace(2,-5,8,base=.5), -1*npy.logspace(2,-5,8,base=.5)])
    else:
        rLightList = npy.array( [ 0.2, 0.5, 1.0, 2.0, 5.0 ] )
        xLightList = npy.array( [ 0.2, 0.5, 1.0, 2.0 , 5.0, -0.2, -0.5, -1.0, -2.0, -5.0 ] )

    # vswr lines
    if isinstance(draw_vswr, (tuple,list)):
        vswrVeryLightList = draw_vswr
    elif draw_vswr is True:
        # use the default I like
        vswrVeryLightList = [1.5, 2.0, 3.0, 5.0]
    else:
        vswrVeryLightList = []

    # cheap way to make a ok-looking smith chart at larger than 1 radii
    if smithR > 1:
        rMax = (1.+smithR)/(1.-smithR)
        rLightList = npy.hstack([ npy.linspace(0,rMax,11)  , rLightList ])

    if chart_type.startswith('y'):
        y_flip_sign = -1
    else:
        y_flip_sign = 1

    # draw impedance and/or admittance
    both_charts = chart_type in ('zy', 'yz')


    # loops through Verylight, Light and Heavy lists and draws circles using patches
    # for analysis of this see R.M. Weikles Microwave II notes (from uva)

    superLightColor = dict(ec='whitesmoke', fc='none')
    veryLightColor = dict(ec='lightgrey', fc='none')
    lightColor = dict(ec='grey', fc='none')
    heavyColor = dict(ec='black', fc='none')

    # vswr circles verylight
    for vswr in vswrVeryLightList:
        radius = (vswr-1.0) / (vswr+1.0)
        contour.append( Circle((0, 0), radius, **veryLightColor))

    # impedance/admittance circles
    for r in rLightList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        if both_charts:
            contour.insert(0, Circle((-center[0], center[1]), radius, **superLightColor))
        contour.append(Circle(center, radius, **lightColor))
    for x in xLightList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        if both_charts:
            contour.insert(0, Circle( (-center[0], center[1]), radius, **superLightColor))
        contour.append(Circle(center, radius, **lightColor))

    for r in rHeavyList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        contour.append(Circle(center, radius, **heavyColor))
    for x in xHeavyList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        contour.append(Circle(center, radius, **heavyColor))

    # clipping circle
    clipc = Circle( [0,0], smithR, ec='k',fc='None',visible=True)
    ax1.add_patch( clipc)

    #draw x and y axis
    ax1.axhline(0, color='k', lw=.1, clip_path=clipc)
    ax1.axvline(1*y_flip_sign, color='k', clip_path=clipc)
    ax1.grid(0)
    # Set axis limits by plotting white points so zooming works properly
    ax1.plot(smithR*npy.array([-1.1, 1.1]), smithR*npy.array([-1.1, 1.1]), 'w.', markersize = 0)
    ax1.axis('image') # Combination of 'equal' and 'tight'


    if not border:
        ax1.yaxis.set_ticks([])
        ax1.xaxis.set_ticks([])
        for loc, spine in ax1.spines.items():
            spine.set_color('none')


    if draw_labels:
        #Clear axis
        ax1.yaxis.set_ticks([])
        ax1.xaxis.set_ticks([])
        for loc, spine in ax1.spines.items():
            spine.set_color('none')

        # Make annotations only if the radius is 1
        if smithR == 1:
            #Make room for annotation
            ax1.plot(npy.array([-1.25, 1.25]), npy.array([-1.1, 1.1]), 'w.', markersize = 0)
            ax1.axis('image')

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
                ax1.annotate(str(value*ref_imm), xy=(rho*smithR, 0.01),
                    xytext=(rho*smithR, 0.01), ha = halignstyle, va = "baseline")

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
                ax1.annotate(str(value*ref_imm) + 'j', xy=(rhox, rhoy),
                             xytext=(rhox, rhoy), ha = halignstyle, va = valignstyle)

            #Annotate 0 and inf
            if y_flip_sign == 1:  # z and zy charts
                label_left, label_right = '0.0', r'$\infty$'
            else:  # y and yz charts
                label_left, label_right = r'$\infty$', '0.0'
            ax1.annotate(label_left, xy=(-1.02, 0), xytext=(-1.02, 0),
                             ha = "right", va = "center")
            ax1.annotate(label_right, xy=(radialScaleFactor, 0), xytext=(radialScaleFactor, 0),
                             ha = "left", va = "center")

            # annotate vswr circles
            for vswr in vswrVeryLightList:
                rhoy = (vswr-1.0) / (vswr+1.0)

                ax1.annotate(str(vswr), xy=(0, rhoy*smithR),
                    xytext=(0, rhoy*smithR), ha="center", va="bottom",
                    color='grey', size='smaller')

    # loop though contours and draw them on the given axes
    for currentContour in contour:
        cc=ax1.add_patch(currentContour)
        cc.set_clip_path(clipc)


def plot_rectangular(x: NumberLike, y: NumberLike,
                     x_label: Union[str, None] = None, y_label: Union[str, None] = None,
                     title: Union[str, None] = None, show_legend: bool = True,
                     axis: str = 'tight', ax: Union[plt.Axes, None] = None,
                     *args, **kwargs):
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
    \*args, \*\*kwargs : passed to pylab.plot

    """
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


def plot_polar(theta: NumberLike, r: NumberLike,
               x_label: Union[str, None] = None, y_label: Union[str, None] = None,
               title: Union[str, None] = None, show_legend: bool = True,
               axis_equal: bool = False, ax: Union[plt.Axes, None] = None,
               *args, **kwargs):
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
    \*args, \*\*kwargs : passed to pylab.plot

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    """
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


def plot_complex_rectangular(z: NumberLike,
                             x_label: str = 'Real', y_label: str = 'Imag',
                             title: str = 'Complex Plane', show_legend: bool = True,
                             axis: str = 'equal', ax: Union[plt.Axes, None] = None,
                             *args, **kwargs):
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
    \*args, \*\*kwargs : passed to pylab.plot

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    """
    x = npy.real(z)
    y = npy.imag(z)
    plot_rectangular(x=x, y=y, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, *args, **kwargs)


def plot_complex_polar(z: NumberLike,
                       x_label: Union[str, None] = None, y_label: Union[str, None] = None,
                       title: Union[str, None] = None, show_legend: bool = True,
                       axis_equal: bool = False, ax: Union[plt.Axes, None] = None,
                       *args, **kwargs):
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
    \*args, \*\*kwargs : passed to pylab.plot

    See Also
    --------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    """
    theta = npy.angle(z)
    r = npy.abs(z)
    plot_polar(theta=theta, r=r, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis_equal=axis_equal,
        ax=ax, *args, **kwargs)


def plot_smith(s: NumberLike, smith_r: float = 1, chart_type: str = 'z',
               x_label: str = 'Real', y_label: str = 'Imaginary', title: str = 'Complex Plane',
               show_legend: bool = True, axis: str = 'equal', ax: Union[plt.Axes, None] = None,
               force_chart: bool = False, draw_vswr: Union[List, bool, None] = None,
               *args, **kwargs):
    r"""
    plot complex data on smith chart.

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
    force_chart : Boolean, optional.
        forces the re-drawing of smith chart. Default is False.
    draw_vswr : list of numbers, Boolean or None, optional
        draw VSWR circles. If True, default values are used.
        Default is None.
    \*args, \*\*kwargs : passed to pylab.plot

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    """

    if ax is None:
        ax = plt.gca()

    # test if smith chart is already drawn
    if not force_chart:
        if len(ax.patches) == 0:
            smith(ax=ax, smithR = smith_r, chart_type=chart_type, draw_vswr=draw_vswr)

    plot_complex_rectangular(s, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, *args, **kwargs)

    ax.axis(smith_r*npy.array([-1.1, 1.1, -1.1, 1.1]))
    if plt.isinteractive():
        plt.draw()


def subplot_params(ntwk, param: str = 's', proj: str = 'db',
                   size_per_port: int = 4, newfig: bool = True,
                   add_titles: bool = True, keep_it_tight: bool = True,
                   subplot_kw: dict = {},
                   *args, **kw):
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

    Returns
    -------
    f : :class:`matplotlib.pyplot.Figure`
        Matplotlib Figure
    ax : :class:`matplotlib.pyplot.Axes`
        Matplotlib Axes

    """
    if newfig:
        f,axs= plt.subplots(ntwk.nports,ntwk.nports,
                            figsize =(size_per_port*ntwk.nports,
                                      size_per_port*ntwk.nports ),
                                      **subplot_kw)
    else:
        f = plt.gcf()
        axs = npy.array(f.get_axes())

    for ports,ax in zip(ntwk.port_tuples, axs.flatten()):
        plot_func = ntwk.__getattribute__('plot_%s_%s'%(param, proj))
        plot_func(m=ports[0], n=ports[1], ax=ax,*args, **kw)
        if add_titles:
            ax.set_title('%s%i%i'%(param.upper(),ports[0]+1, ports[1]+1))
    if keep_it_tight:
       plt.tight_layout()
    return f, axs


def shade_bands(edges: NumberLike, y_range: Union[Tuple, None] = None,
                cmap: str = 'prism', **kwargs):
    r"""
    Shades frequency bands.

    When plotting data over a set of frequency bands it is nice to
    have each band visually separated from the other. The kwarg `alpha`
    is useful.

    Parameters
    ----------
    edges : array-like
        x-values separating regions of a given shade
    y_range : tuple or None, optional.
        y-values to shade in. Default is None.
    cmap : str, optional.
        see matplotlib.cm  or matplotlib.colormaps for acceptable values.
        Default is 'prism'.
    \*\*kwargs : key word arguments
        passed to `matplotlib.fill_between`

    Examples
    --------
    >>> rf.shade_bands([325,500,750,1100], alpha=.2)
    """
    cmap = plt.cm.get_cmap(cmap)
    y_range=plt.gca().get_ylim()
    axis = plt.axis()
    for k in range(len(edges)-1):
        plt.fill_between(
            [edges[k],edges[k+1]],
            y_range[0], y_range[1],
            color = cmap(1.0*k/len(edges)),
            **kwargs)
    plt.axis(axis)


def save_all_figs(dir: str = './', format: Union[None, List[str]] = None,
                  replace_spaces: bool = True, echo: bool = True):
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
    replace_spaces : bool, optional
        default is True.
    echo : bool, optional.
        True prints filenames as they are saved. Default is True.
    """
    if dir[-1] != '/':
        dir = dir + '/'
    for fignum in plt.get_fignums():
        fileName = plt.figure(fignum).get_axes()[0].get_title()
        if replace_spaces:
            fileName = fileName.replace(' ','_')
        if fileName == '':
            fileName = 'unnamedPlot'
        if format is None:
            plt.savefig(dir+fileName)
            if echo:
                print(dir+fileName)
        else:
            for fmt in format:
                plt.savefig(dir+fileName+'.'+fmt, format=fmt)
                if echo:
                    print(dir+fileName+'.'+fmt)
saf = save_all_figs


def add_markers_to_lines(ax: Union[plt.Axes, None] = None,
                         marker_list: List = ['o', 'D', 's', '+', 'x'],
                         markevery: int = 10):
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
    marker_list : list of string, optional
        list of marker characters. Default is ['o', 'D', 's', '+', 'x'].
        see matplotlib.plot help for possible marker characters
    markevery : int, optional.
        markevery number of points with a marker.
        Default is 10.

    """
    if ax is None:
        ax=plt.gca()
    lines = ax.get_lines()
    if len(lines) > len (marker_list ):
        marker_list *= 3
    [k[0].set_marker(k[1]) for k in zip(lines, marker_list)]
    [line.set_markevery(markevery) for line in lines]


def legend_off(ax: Union[plt.Axes, None] = None):
    """
    Turn off the legend for a given axes.

    if no axes is given then it will use current axes.

    Parameters
    ----------
    ax : matplotlib.Axes or None, optional
        axis to operate on.
        Default is None for current axe gca()
    """
    if ax is None:
        plt.gca().legend_.set_visible(0)
    else:
        ax.legend_.set_visible(0)


def scrape_legend(n: Union[int, None] = None,
                  ax: Union[plt.Axes, None] = None):
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
    """

    if ax is None:
        ax = plt.gca()

    handles, labels = ax.get_legend_handles_labels()

    if n is None:
        n =len ( set(labels))

    if n>len(handles):
        raise ValueError('number of entries is too large')

    k_list = [int(k) for k in npy.linspace(0,len(handles)-1,n)]
    ax.legend([handles[k] for k in k_list], [labels[k] for k in k_list])


def func_on_all_figs(func: Callable, *args, **kwargs):
    r"""
    Run a function after making all open figures current.

    useful if you need to change the properties of many open figures
    at once, like turn off the grid.

    Parameters
    ----------
    func : function
        function to call
    \*args, \*\*kwargs : passed to func

    Examples
    --------
    >>> rf.func_on_all_figs(grid, alpha=.3)
    """
    for fig_n in plt.get_fignums():
        fig = plt.figure(fig_n)
        for ax_n in fig.axes:
            fig.add_axes(ax_n) # trick to make axes current
            func(*args, **kwargs)
            plt.draw()

foaf = func_on_all_figs


def plot_vector(a: complex, off: complex = 0+0j, *args, **kwargs):
    """
    Plot a 2d vector.

    Parameters
    ----------
    a : complex
        complex coordinates (real for X, imag for Y) of the arrow location.
    off : complex, optional
        complex direction (real for U, imag for V) components
        of the arrow vectors, by default 0+0j

    Returns
    -------
    quiver : matplotlib.pyplot.quiver
    """
    return quiver(off.real, off.imag, a.real, a.imag, scale_units='xy',
           angles='xy', scale=1, *args, **kwargs)


def colors() -> List[str]:
    """
    Return the list of colors of the rcParams color cycle.

    Returns
    -------
    colors : List[str]
    """
    return [c['color'] for c in rcParams['axes.prop_cycle']]


PRIMARY_PROPERTIES = network.PRIMARY_PROPERTIES
COMPONENT_FUNC_DICT = network.COMPONENT_FUNC_DICT
Y_LABEL_DICT = network.Y_LABEL_DICT


# TODO: remove this as it takes up ~70% cpu time of this init
def setup_matplotlib_plotting():
    frequency.Frequency.labelXAxis = labelXAxis
    frequency.Frequency.plot = plot_v_frequency

    __generate_plot_functions(network.Network)
    network.Network.plot = plot
    network.Network.plot_passivity = plot_passivity
    network.Network.plot_reciprocity = plot_reciprocity
    network.Network.plot_reciprocity2 = plot_reciprocity2
    network.Network.plot_s_db_time = plot_s_db_time
    network.Network.plot_s_smith = plot_s_smith
    network.Network.plot_it_all = plot_it_all

    calibration.Calibration.plot_errors = plot_calibration_errors
    calibration.Calibration.plot_caled_ntwks = plot_caled_ntwks
    calibration.Calibration.plot_residuals = plot_residuals

    networkSet.NetworkSet.animate = animate
    networkSet.NetworkSet.plot_uncertainty_bounds_component = plot_uncertainty_bounds_component
    networkSet.NetworkSet.plot_minmax_bounds_component = plot_minmax_bounds_component
    networkSet.NetworkSet.plot_uncertainty_bounds_s_db = plot_uncertainty_bounds_s_db
    networkSet.NetworkSet.plot_minmax_bounds_s_db = plot_minmax_bounds_s_db
    networkSet.NetworkSet.plot_minmax_bounds_s_db10 = plot_minmax_bounds_s_db10
    networkSet.NetworkSet.plot_uncertainty_bounds_s_time_db = plot_uncertainty_bounds_s_time_db
    networkSet.NetworkSet.plot_minmax_bounds_s_time_db = plot_minmax_bounds_s_time_db
    networkSet.NetworkSet.plot_uncertainty_decomposition = plot_uncertainty_decomposition
    networkSet.NetworkSet.plot_uncertainty_bounds_s = plot_uncertainty_bounds_s
    networkSet.NetworkSet.plot_logsigma = plot_logsigma
    networkSet.NetworkSet.signature = signature

    circuit.Circuit.plot_graph = plot_circuit_graph

def __generate_plot_functions(self):
    """
    """
    for prop_name in PRIMARY_PROPERTIES:

        def plot_prop_polar(self,
                            m=None, n=None, ax=None,
                            show_legend=True, prop_name=prop_name, *args, **kwargs):

            # create index lists, if not provided by user
            if m is None:
                M = range(self.number_of_ports)
            else:
                M = [m]
            if n is None:
                N = range(self.number_of_ports)
            else:
                N = [n]

            if 'label'  not in kwargs.keys():
                gen_label = True
            else:
                gen_label = False

            # was_interactive = plt.isinteractive
            # if was_interactive:
            #     plt.interactive(False)

            for m in M:
                for n in N:
                    # set the legend label for this trace to the networks
                    # name if it exists, and they didn't pass a name key in
                    # the kwargs
                    if gen_label:
                        if self.name is None:
                            if plt.rcParams['text.usetex']:
                                label_string = '$%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = '%s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        else:
                            if plt.rcParams['text.usetex']:
                                label_string = self.name+', $%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = self.name+', %s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        kwargs['label'] = label_string

                    # plot the desired attribute vs frequency
                    plot_complex_polar(
                        z = getattr(self,prop_name)[:,m,n],
                         show_legend = show_legend, ax = ax,
                        *args, **kwargs)

            # if was_interactive:
            #     plt.interactive(True)
            #     plt.draw()
            #     plt.show()

        plot_prop_polar.__doc__ = r"""
plot the Network attribute :attr:`{}` vs frequency.

Parameters
----------
m : int, optional
    first index of s-parameter matrix, if None will use all
n : int, optional
    secon index of the s-parameter matrix, if None will use all
ax : :class:`matplotlib.Axes` object, optional
    An existing Axes object to plot on
show_legend : Boolean
    draw legend or not
attribute : string
    Network attribute to plot
y_label : string, optional
    the y-axis label

\*args,\**kwargs : arguments, keyword arguments
    passed to :func:`matplotlib.plot`

Note
----
This function is dynamically generated upon Network
initialization. This is accomplished by calling
:func:`plot_vs_frequency_generic`

Examples
--------
>>> myntwk.plot_{}(m=1,n=0,color='r')
""".format(prop_name, prop_name)
        # setattr(self.__class__,'plot_%s_polar'%(prop_name), \
        setattr(self, 'plot_%s_polar'%(prop_name), plot_prop_polar)

        def plot_prop_rect(self,
                           m=None, n=None, ax=None,
                           show_legend=True, prop_name=prop_name, *args, **kwargs):

            # create index lists, if not provided by user
            if m is None:
                M = range(self.number_of_ports)
            else:
                M = [m]
            if n is None:
                N = range(self.number_of_ports)
            else:
                N = [n]

            if 'label'  not in kwargs.keys():
                gen_label = True
            else:
                gen_label = False

            #was_interactive = plt.isinteractive
            #if was_interactive:
            #    plt.interactive(False)

            for m in M:
                for n in N:
                    # set the legend label for this trace to the networks
                    # name if it exists, and they didn't pass a name key in
                    # the kwargs
                    if gen_label:
                        if self.name is None:
                            if plt.rcParams['text.usetex']:
                                label_string = '$%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = '%s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        else:
                            if plt.rcParams['text.usetex']:
                                label_string = self.name+', $%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = self.name+', %s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        kwargs['label'] = label_string

                    # plot the desired attribute vs frequency
                    plot_complex_rectangular(
                        z=getattr(self, prop_name)[:, m, n],
                        show_legend=show_legend, ax=ax,
                        *args, **kwargs)

            #if was_interactive:
            #    plt.interactive(True)
            #    plt.draw()
            #    plt.show()

        plot_prop_rect.__doc__ = r"""
plot the Network attribute :attr:`{}` vs frequency.

Parameters
----------
m : int, optional
    first index of s-parameter matrix, if None will use all
n : int, optional
    secon index of the s-parameter matrix, if None will use all
ax : :class:`matplotlib.Axes` object, optional
    An existing Axes object to plot on
show_legend : Boolean
    draw legend or not
attribute : string
    Network attribute to plot
y_label : string, optional
    the y-axis label

\*args,\**kwargs : arguments, keyword arguments
    passed to :func:`matplotlib.plot`

Note
----
This function is dynamically generated upon Network
initialization. This is accomplished by calling
:func:`plot_vs_frequency_generic`

Examples
--------
>>> myntwk.plot_{}(m=1,n=0,color='r')
""".format(prop_name, prop_name)

        # setattr(self.__class__,'plot_%s_complex'%(prop_name), \
        setattr(self,'plot_%s_complex'%(prop_name), \
            plot_prop_rect)


        for func_name in COMPONENT_FUNC_DICT:
            attribute = f'{prop_name}_{func_name}'
            y_label = Y_LABEL_DICT[func_name]

            def plot_func(self,  m=None, n=None, ax=None,
                          show_legend=True, attribute=attribute,
                          y_label=y_label, logx=False, pad=0, window='hamming', z0=50, *args, **kwargs):

                # create index lists, if not provided by user
                if m is None:
                    M = range(self.number_of_ports)
                else:
                    M = [m]
                if n is None:
                    N = range(self.number_of_ports)
                else:
                    N = [n]

                if 'label'  not in kwargs.keys():
                    gen_label = True
                else:
                    gen_label = False

                #TODO: turn off interactive plotting for performance
                # this didnt work because it required a show()
                # to be called, which in turn, disrupted testCases
                #
                # was_interactive = plt.isinteractive
                # if was_interactive:
                #     plt.interactive(False)
                for m in M:
                    for n in N:
                        # set the legend label for this trace to the networks
                        # name if it exists, and they didn't pass a name key in
                        # the kwargs
                        if gen_label:
                            if self.name is None:
                                if plt.rcParams['text.usetex']:
                                    label_string = '$%s_{%i%i}$'%\
                                    (attribute[0].upper(),m+1,n+1)
                                else:
                                    label_string = '%s%i%i'%\
                                    (attribute[0].upper(),m+1,n+1)
                            else:
                                if plt.rcParams['text.usetex']:
                                    label_string = self.name+', $%s_{%i%i}$'%\
                                    (attribute[0].upper(),m+1,n+1)
                                else:
                                    label_string = self.name+', %s%i%i'%\
                                    (attribute[0].upper(),m+1,n+1)
                            kwargs['label'] = label_string

                        # quick and dirty way to plot step and impulse response
                        if 'time_impulse' in attribute:
                            xlabel = 'Time (ns)'
                            x,y = self.impulse_response(pad=pad, window=window)
                            # default is reflexion coefficient axis
                            if attribute[0].lower() == 'z':
                                # if they want impedance axis, give it to them
                                y_label = 'Z (ohm)'
                                y[x ==  1.] =  1. + 1e-12  # solve numerical singularity
                                y[x == -1.] = -1. + 1e-12  # solve numerical singularity
                                y = z0 * (1+y) / (1-y)
                            plot_rectangular(x=x * 1e9,
                                             y=y,
                                             x_label=xlabel,
                                             y_label=y_label,
                                             show_legend=show_legend, ax=ax,
                                             *args, **kwargs)
                        elif 'time_step' in attribute:
                            xlabel = 'Time (ns)'
                            x, y = self.step_response(pad=pad, window=window)
                            # default is reflexion coefficient axis
                            if attribute[0].lower() == 'z':
                                # if they want impedance axis, give it to them
                                y_label = 'Z (ohm)'
                                y[x ==  1.] =  1. + 1e-12  # solve numerical singularity
                                y[x == -1.] = -1. + 1e-12  # solve numerical singularity
                                y = z0 * (1+y) / (1-y)
                            plot_rectangular(x=x * 1e9,
                                             y=y,
                                             x_label=xlabel,
                                             y_label=y_label,
                                             show_legend=show_legend, ax=ax,
                                             *args, **kwargs)

                        else:
                            # plot the desired attribute vs frequency
                            if 'time' in attribute:
                                xlabel = 'Time (ns)'
                                x = self.frequency.t_ns

                            else:
                                xlabel = 'Frequency (%s)' % self.frequency.unit
                                # x = self.frequency.f_scaled
                                x = self.frequency.f  # always plot f, and then scale the ticks instead

                                # scale the ticklabels according to the frequency unit and set log-scale if desired:
                                if ax is None:
                                    ax = plt.gca()
                                if logx:
                                    ax.set_xscale('log')

                                scale_frequency_ticks(ax, self.frequency.unit)



                            plot_rectangular(x=x,
                                             y=getattr(self, attribute)[:, m, n],
                                             x_label=xlabel,
                                             y_label=y_label,
                                             show_legend=show_legend, ax=ax,
                                             *args, **kwargs)
                #if was_interactive:
                #    plt.interactive(True)
                #    plt.draw()
                #    #plt.show()

            plot_func.__doc__ = r"""
    plot the Network attribute :attr:`%s` vs frequency.

    Parameters
    ----------
    m : int, optional
        first index of s-parameter matrix, if None will use all
    n : int, optional
        secon index of the s-parameter matrix, if None will use all
    ax : :class:`matplotlib.Axes` object, optional
        An existing Axes object to plot on
    show_legend : Boolean
        draw legend or not
    attribute : string
        Network attribute to plot
    y_label : string, optional
        the y-axis label
    logx : Boolean, optional
        Enable logarithmic x-axis, default off

    \*args,\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot`

    Note
    ----
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    --------
    >>> myntwk.plot_%s(m=1,n=0,color='r')
    """%(attribute,attribute)

            # setattr(self.__class__,'plot_%s'%(attribute), \
            setattr(self,'plot_%s'%(attribute), \
                plot_func)


def labelXAxis(self, ax: Union[plt.Axes, None] = None):
    """
    Label the x-axis of a plot.

    Sets the labels of a plot using :func:`matplotlib.x_label` with
    string containing the frequency unit.

    Parameters
    ----------
    ax : :class:`matplotlib.Axes` or None, optional
            Axes on which to label the plot.
            Defaults is None, for the current axe
            returned by :func:`matplotlib.gca()`
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel('Frequency (%s)' % self.unit)


def plot_v_frequency(self, y: NumberLike, *args, **kwargs):
    """
    Plot something vs this frequency.

    This plots whatever is given vs. `self.f_scaled` and then
    calls `labelXAxis`.
    """

    try:
        if len(npy.shape(y)) > 2:
            # perhaps the dimensions are empty, try to squeeze it down
            y = y.squeeze()
            if len(npy.shape(y)) > 2:
                # the dimensions are full, so lets loop and plot each
                for m in range(npy.shape(y)[1]):
                    for n in range(npy.shape(y)[2]):
                        self.plot(y[:, m, n], *args, **kwargs)
                return
        if len(y) == len(self):
            pass
        else:

            raise IndexError(['thing to plot doesn\'t have same'
                              ' number of points as f'])
    except(TypeError):
        y = y * npy.ones(len(self))

    # plt.plot(self.f_scaled, y, *args, **kwargs)
    plt.plot(self.f, y, *args, **kwargs)
    ax = plt.gca()
    scale_frequency_ticks(ax, self.unit)
    plt.autoscale(axis='x', tight=True)
    self.labelXAxis()


## specific plotting functions
def plot(self, *args, **kw):
    """
    Plot something vs frequency
    """
    return self.frequency.plot(*args, **kw)


def plot_passivity(self, port=None, label_prefix=None, *args, **kwargs):
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
    name = '' if self.name is None else self.name

    if port is None:
        ports = range(self.nports)
    else:
        ports = [port]
    for k in ports:
        if label_prefix is None:
            label = name + ', port %i' % (k + 1)
        else:
            label = label_prefix + ', port %i' % (k + 1)
        self.frequency.plot(mf.complex_2_db(self.passivity[:, k, k]),
                            label=label,
                            *args, **kwargs)

    plt.legend()
    plt.draw()


def plot_reciprocity(self, db=False, *args, **kwargs):
    """
    Plot reciprocity metric.

    See Also
    --------
    reciprocity
    """
    for m in range(self.nports):
        for n in range(self.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = 'ports %i%i' % (m, n)
                y = self.reciprocity[:, m, n].flatten()
                if db:
                    y = mf.complex_2_db(y)
                self.frequency.plot(y, *args, **kwargs)

    plt.legend()
    plt.draw()


def plot_reciprocity2(self, db=False, *args, **kwargs):
    """
    Plot reciprocity metric #2.

    this is distance of the determinant of the wave-cascading matrix
    from unity.

    .. math::

            abs(1 - S/S^T )



    See Also
    --------
    reciprocity
    """
    for m in range(self.nports):
        for n in range(self.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = 'ports %i%i' % (m, n)
                y = self.reciprocity2[:, m, n].flatten()
                if db:
                    y = mf.complex_2_db(y)
                self.frequency.plot(y, *args, **kwargs)

    plt.legend()
    plt.draw()


def plot_s_db_time(self,center_to_dc=None,*args,**kwargs):
    return self.windowed(center_to_dc=center_to_dc).plot_s_time_db(*args,**kwargs)


# plotting
def plot_s_smith(self, m=None, n=None,r=1, ax=None, show_legend=True,\
        chart_type='z', draw_labels=False, label_axes=False, draw_vswr=None, *args,**kwargs):
    r"""
    plots the scattering parameter on a smith chart

    plots indices `m`, `n`, where `m` and `n` can be integers or
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

    \*args : arguments, optional
            passed to the matplotlib.plot command
    \*\*kwargs : keyword arguments, optional
            passed to the matplotlib.plot command


    See Also
    --------
    plot_vs_frequency_generic - generic plotting function
    smith -  draws a smith chart

    Examples
    --------
    >>> myntwk.plot_s_smith()
    >>> myntwk.plot_s_smith(m=0,n=1,color='b', marker='x')
    """
    # TODO: prevent this from re-drawing smith chart if one alread
    # exists on current set of axes

    # get current axis if user doesnt supply and axis
    if ax is None:
        ax = plt.gca()


    if m is None:
        M = range(self.number_of_ports)
    else:
        M = [m]
    if n is None:
        N = range(self.number_of_ports)
    else:
        N = [n]

    if 'label'  not in kwargs.keys():
        generate_label=True
    else:
        generate_label=False

    for m in M:
        for n in N:
            # set the legend label for this trace to the networks name if it
            # exists, and they didnt pass a name key in the kwargs
            if generate_label:
                if self.name is None:
                    if plt.rcParams['text.usetex']:
                        label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
                    else:
                        label_string = 'S'+repr(m+1) + repr(n+1)
                else:
                    if plt.rcParams['text.usetex']:
                        label_string = self.name+', $S_{'+repr(m+1) + \
                                repr(n+1)+'}$'
                    else:
                        label_string = self.name+', S'+repr(m+1) + repr(n+1)

                kwargs['label'] = label_string

            # plot the desired attribute vs frequency
            if len (ax.patches) == 0:
                smith(ax=ax, smithR = r, chart_type=chart_type, draw_labels=draw_labels, draw_vswr=draw_vswr)
            ax.plot(self.s[:,m,n].real,  self.s[:,m,n].imag, *args,**kwargs)

    #draw legend
    if show_legend:
        ax.legend()
    ax.axis(npy.array([-1.1,1.1,-1.1,1.1])*r)

    if label_axes:
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')


def plot_it_all(self, *args, **kwargs):
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
    plt.subplot(221)
    getattr(self,'plot_s_db')(*args, **kwargs)
    plt.subplot(222)
    getattr(self,'plot_s_deg')(*args, **kwargs)
    plt.subplot(223)
    getattr(self,'plot_s_smith')(*args, **kwargs)
    plt.subplot(224)
    getattr(self,'plot_s_complex')(*args, **kwargs)


def stylely(rc_dict: dict = {}, style_file: str = 'skrf.mplstyle'):
    """
    Loads the rc-params from the specified file (file must be located in skrf/data).

    Parameters
    ----------
    rc_dict : dict, optional
        rc dict passed to :func:`matplotlib.rc`, by default {}
    style_file : str, optional
        style file, by default 'skrf.mplstyle'
    """
    from .data import pwd # delayed to solve circular import
    mpl.style.use(os.path.join(pwd, style_file))
    mpl.rc(rc_dict)


def plot_calibration_errors(self, *args, **kwargs):
    """
    Plot biased, unbiased and total error in dB scaled.

    See Also
    --------
    biased_error
    unbiased_error
    total_error
    """
    port_list = self.biased_error.port_tuples
    for m,n in port_list:
        plt.figure()
        plt.title('S%i%i'%(m+1,n+1))
        self.unbiased_error.plot_s_db(m,n,**kwargs)
        self.biased_error.plot_s_db(m,n,**kwargs)
        self.total_error.plot_s_db(m,n,**kwargs)
        plt.ylim(-100,0)


def plot_caled_ntwks(self, attr: str = 's_smith', show_legend: bool = False, **kwargs):
    r"""
    Plot corrected calibration standards.

    Given that the calibration is overdetermined, this may be used
    as a heuristic verification of calibration quality.

    Parameters
    ----------
    attr : str
        Network property to plot, ie 's_db', 's_smith', etc.
        Default is 's_smith'
    show_legend : bool, optional
        draw a legend or not. Default is False.
    \*\*kwargs : kwargs
        passed to the plot method of Network
    """
    ns = networkSet.NetworkSet(self.caled_ntwks)
    kwargs.update({'show_legend':show_legend})

    if ns[0].nports ==1:
        ns.__getattribute__('plot_'+attr)(0,0, **kwargs)
    elif ns[0].nports ==2:
        plt.figure(figsize = (8,8))
        for k,mn in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
            plt.subplot(221+k)
            plt.title('S%i%i'%(mn[0]+1,mn[1]+1))
            ns.__getattribute__('plot_'+attr)(*mn, **kwargs)
    else:
        raise NotImplementedError
    plt.tight_layout()


def plot_residuals(self, attr: str = 's_db', **kwargs):
    r"""
    Plot residual networks.

    Given that the calibration is overdetermined, this may be used
    as a metric of the calibration's *goodness of fit*

    Parameters
    ----------
    attr : str, optional.
        Network property to plot, ie 's_db', 's_smith', etc.
        Default is 's_db'
    \*\*kwargs : kwargs
        passed to the plot method of Network

    See Also
    --------
    Calibration.residual_networks
    """

    networkSet.NetworkSet(self.residual_ntwks).__getattribute__('plot_'+attr)(**kwargs)


# Network Set Plotting Commands
def animate(self, attr: str = 's_deg', ylims: Tuple = (-5, 5),
            xlims: Union[Tuple, None] = None, show: bool = True,
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

    \*args, \*\*kwargs :
        passed to the Network plotting function

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
        plt.draw();
        if show:
            plt.show()
        if savefigs:
            fname = os.path.join(dir_, 'out_%.5i' % idx + '.png')
            plt.savefig(fname)

    if savefigs:
        print('\n\n')
    if was_interactive:
        plt.ion()


#------------------------------
#
# NetworkSet plotting functions
#
#------------------------------

def plot_uncertainty_bounds_component(
        self, attribute: str,
        m: Union[int, None] = None, n: Union[int, None] = None,
        type: str = 'shade', n_deviations: int = 3,
        alpha: float = .3, color_error: Union[str, None] = None,
        markevery_error: int = 20, ax: Union[plt.Axes, None] = None,
        ppf: bool = None, kwargs_error: dict = {},
        *args, **kwargs):
    r"""
    Plot mean value of a NetworkSet with +/- uncertainty bounds in an Network's attribute.

    This is designed to represent uncertainty in a scalar component of the s-parameter.
    for example plotting the uncertainty in the magnitude would be expressed by,

    .. math::

        mean(|s|) \pm std(|s|)

    the order of mean and abs is important.


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
    ax : matplotlib axe object
        Axes to plot on. Default is None.
    ppf : function
        post processing function. a function applied to the
        upper and lower bounds. Default is None
     kwargs_error : dict
         dictionary of kwargs to pass to the fill_between or
         errorbar plot command depending on value of type.
    \*args, \*\*kwargs :
        passed to Network.plot_s_re command used to plot mean response

    Note
    ----
    for phase uncertainty you probably want s_deg_unwrap, or
    similar. uncertainty for wrapped phase blows up at +-pi.

    """

    if ax is None:
        ax = plt.gca()

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
                lower_bound[npy.isnan(lower_bound)] = min(lower_bound)
                if ppf in [mf.magnitude_2_db, mf.mag_2_db]:  # quickfix of wrong ylabels due to usage of ppf for *_db plots
                    if attribute == 's_mag':
                        plot_attribute = 's_db'
                    elif attribute == 's_time_mag':
                        plot_attribute = 's_time_db'

            if type == 'shade':
                ntwk_mean.plot_s_re(ax=ax, m=m, n=n, *args, **kwargs)
                if color_error is None:
                    color_error = ax.get_lines()[-1].get_color()
                ax.fill_between(ntwk_mean.frequency.f,
                                lower_bound.real, upper_bound.real, alpha=alpha, color=color_error,
                                **kwargs_error)
                # ax.plot(ntwk_mean.frequency.f_scaled, ntwk_mean.s[:,m,n],*args,**kwargs)

            elif type == 'bar':
                ntwk_mean.plot_s_re(ax=ax, m=m, n=n, *args, **kwargs)
                if color_error is None:
                    color_error = ax.get_lines()[-1].get_color()
                ax.errorbar(ntwk_mean.frequency.f[::markevery_error],
                            ntwk_mean.s_re[:, m, n].squeeze()[::markevery_error],
                            yerr=ntwk_std.s_mag[:, m, n].squeeze()[::markevery_error],
                            color=color_error, **kwargs_error)

            else:
                raise(ValueError('incorrect plot type'))

            ax.set_ylabel(Y_LABEL_DICT.get(plot_attribute[2:], ''))  # use only the function of the attribute
            scale_frequency_ticks(ax, ntwk_mean.frequency.unit)
            ax.axis('tight')


def plot_minmax_bounds_component(self, attribute: str, m: int = 0, n: int = 0,
                                 type: str = 'shade', n_deviations: int = 3,
                                 alpha: float = .3, color_error: Union[str, None] = None,
                                 markevery_error: int = 20, ax: Union[plt.Axes, None] = None,
                                 ppf: bool = None, kwargs_error: dict = {},
                                 *args, **kwargs):
    r"""
    plots mean value of the NetworkSet with +/- uncertainty bounds in an Network's attribute.

    This is designed to represent uncertainty in a scalar component of the s-parameter. for example
    plotting the uncertainty in the magnitude would be expressed by

    .. math::

        mean(|s|) \pm std(|s|)

    the order of mean and abs is important.

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
    ax : matplotlib axe object
        Axes to plot on. Default is None.
    ppf : function
        post processing function. a function applied to the
        upper and lower bounds. Default is None
     kwargs_error : dict
         dictionary of kwargs to pass to the fill_between or
         errorbar plot command depending on value of type.
    \*args,  \*\*kwargs :
        passed to Network.plot_s_re command used to plot mean response

    Note
    ----
    for phase uncertainty you probably want s_deg_unwrap, or
    similar.  uncertainty for wrapped phase blows up at +-pi.

    """

    if ax is None:
        ax = plt.gca()

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
        lower_bound[npy.isnan(lower_bound)]=min(lower_bound)
        if ppf in [mf.magnitude_2_db, mf.mag_2_db]: # quickfix of wrong ylabels due to usage of ppf for *_db plots
            if attribute == 's_mag':
                attribute = 's_db'
            elif attribute == 's_time_mag':
                attribute = 's_time_db'

    if type == 'shade':
        ntwk_mean.plot_s_re(ax=ax,m=m,n=n,*args, **kwargs)
        if color_error is None:
            color_error = ax.get_lines()[-1].get_color()
        ax.fill_between(ntwk_mean.frequency.f,
                        lower_bound, upper_bound, alpha=alpha, color=color_error,
                        **kwargs_error)
        #ax.plot(ntwk_mean.frequency.f_scaled,ntwk_mean.s[:,m,n],*args,**kwargs)
    elif type =='bar':
        raise (NotImplementedError)
        ntwk_mean.plot_s_re(ax=ax, m=m, n=n, *args, **kwargs)
        if color_error is None:
            color_error = ax.get_lines()[-1].get_color()
        ax.errorbar(ntwk_mean.frequency.f[::markevery_error],
                    ntwk_mean.s_re[:,m,n].squeeze()[::markevery_error],
                    yerr=ntwk_std.s_mag[:,m,n].squeeze()[::markevery_error],
                    color=color_error,**kwargs_error)

    else:
        raise(ValueError('incorrect plot type'))

    ax.set_ylabel(Y_LABEL_DICT.get(attribute[2:], ''))  # use only the function of the attribute
    scale_frequency_ticks(ax, ntwk_mean.frequency.unit)
    ax.axis('tight')


def plot_uncertainty_bounds_s_db(self, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute='s_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'attribute':'s_mag','ppf':mf.magnitude_2_db})
    self.plot_uncertainty_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_db(self, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'attribute':'s_mag','ppf':mf.magnitude_2_db})
    self.plot_minmax_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_db10(self, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'attribute':'s_mag','ppf':mf.mag_2_db10})
    self.plot_minmax_bounds_component(*args,**kwargs)

def plot_uncertainty_bounds_s_time_db(self, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'attribute':'s_time_mag','ppf':mf.magnitude_2_db})
    self.plot_uncertainty_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_time_db(self, *args, **kwargs):
    """
    Call ``plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)``.

    see plot_uncertainty_bounds for help

    """
    kwargs.update({'attribute':'s_time_mag','ppf':mf.magnitude_2_db})
    self.plot_minmax_bounds_component(*args, **kwargs)

def plot_uncertainty_decomposition(self, m: int = 0, n: int = 0):
    """
    Plot the total and component-wise uncertainty.

    Parameters
    ----------
    m : int
        first s-parameters index
    n :
        second s-parameter index

    """
    if self.name is not None:
        plt.title(r'Uncertainty Decomposition: %s $S_{%i%i}$'%(self.name,m,n))
    self.std_s.plot_s_mag(label='Distance', m=m,n=n)
    self.std_s_re.plot_s_mag(label='Real',  m=m,n=n)
    self.std_s_im.plot_s_mag(label='Imaginary',  m=m,n=n)
    self.std_s_mag.plot_s_mag(label='Magnitude',  m=m,n=n)
    self.std_s_arcl.plot_s_mag(label='Arc-length',  m=m,n=n)

def plot_uncertainty_bounds_s(self, multiplier: float = 200, *args, **kwargs):
    """
    Plot complex uncertainty bounds plot on smith chart.

    This function plots the complex uncertainty of a NetworkSet
    as circles on the smith chart. At each frequency a circle
    with radii proportional to the complex standard deviation
    of the set at that frequency is drawn. Due to the fact that
    the `markersize` argument is in pixels, the radii can scaled by
    the input argument  `multiplier`.

    default kwargs are
        {
        'marker':'o',
        'color':'b',
        'mew':0,
        'ls':'',
        'alpha':.1,
        'label':None,
        }

    Parameters
    ----------
    multiplier : float
        controls the circle sizes, by multiples of the standard
        deviation.

    """
    default_kwargs = {
        'marker':'o',
        'color':'b',
        'mew':0,
        'ls':'',
        'alpha':.1,
        'label':None,
        }
    default_kwargs.update(**kwargs)

    if plt.isinteractive():
        was_interactive = True
        plt.interactive(0)
    else:
        was_interactive = False

    [self.mean_s[k].plot_s_smith(*args, ms = self.std_s[k].s_mag*multiplier, **default_kwargs) for k in range(len(self[0]))]

    if was_interactive:
        plt.interactive(1)
    plt.draw()
    plt.show()

def plot_logsigma(self, label_axis: bool = True, *args,**kwargs):
    r"""
    Plot the uncertainty for the set in units of log-sigma.

    Log-sigma is the complex standard deviation, plotted in units
    of dB's.

    Parameters
    ----------
    label_axis : bool, optional
        Default is True.
    \*args, \*\*kwargs : arguments
        passed to self.std_s.plot_s_db()
    """
    self.std_s.plot_s_db(*args,**kwargs)
    if label_axis:
        plt.ylabel('Standard Deviation(dB)')


def signature(self, m: int = 0, n: int = 0, component: str = 's_mag',
              vmax: Union[Number, None] = None, vs_time: bool = False,
              cbar_label: Union[str, None] = None,
              *args, **kwargs):
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
    \*args,\*\*kw : arguments, keyword arguments
        passed to :func:`~pylab.imshow`
    """

    mat = npy.array([self[k].__getattribute__(component)[:, m, n] \
                     for k in range(len(self))])

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
    freq.labelXAxis()

    cbar = plt.colorbar()
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    return img

def plot_circuit_graph(self, **kwargs):
    """
    Plot the graph of the circuit using networkx drawing capabilities.

    Customisation options with default values:
    ::
        'network_shape': 's'
        'network_color': 'gray'
        'network_size', 300
        'network_fontsize': 7
        'inter_shape': 'o'
        'inter_color': 'lightblue'
        'inter_size', 300
        'port_shape': '>'
        'port_color': 'red'
        'port_size', 300
        'port_fontsize': 7
        'edges_fontsize': 5
        'network_labels': False
        'edge_labels': False
        'inter_labels': False
        'port_labels': False
        'label_shift_x': 0
        'label_shift_y': 0

    """
    # Get the circuit graph. Will raise an error if the networkx package
    # is not installed.
    G = self.G

    # default values
    network_labels = kwargs.pop('network_labels', False)
    network_shape = kwargs.pop('network_shape', 's')
    network_color = kwargs.pop('network_color', 'gray')
    network_fontsize = kwargs.pop('network_fontsize', 7)
    network_size = kwargs.pop('network_size', 300)
    inter_labels = kwargs.pop('inter_labels', False)
    inter_shape = kwargs.pop('inter_shape', 'o')
    inter_color = kwargs.pop('inter_color', 'lightblue')
    inter_size = kwargs.pop('inter_size', 300)
    port_labels = kwargs.pop('port_labels', False)
    port_shape = kwargs.pop('port_shape', '>')
    port_color = kwargs.pop('port_color', 'red')
    port_size = kwargs.pop('port_size', 300)
    port_fontsize = kwargs.pop('port_fontsize', 7)
    edge_labels = kwargs.pop('edge_labels', False)
    edge_fontsize = kwargs.pop('edge_fontsize', 5)
    label_shift_x = kwargs.pop('label_shift_x', 0)
    label_shift_y = kwargs.pop('label_shift_y', 0)


    # sort between network nodes and port nodes
    all_ntw_names = [ntw.name for ntw in self.networks_list()]
    port_names = [ntw_name for ntw_name in all_ntw_names if 'port' in ntw_name]
    ntw_names = [ntw_name for ntw_name in all_ntw_names if 'port' not in ntw_name]
    # generate connecting nodes names
    int_names = ['X'+str(k) for k in range(self.connections_nb)]

    fig, ax = plt.subplots(figsize=(10,8))

    pos = nx.spring_layout(G)

    # draw Networks
    nx.draw_networkx_nodes(G, pos, port_names, ax=ax,
                           node_size=port_size,
                           node_color=port_color, node_shape=port_shape)
    nx.draw_networkx_nodes(G, pos, ntw_names, ax=ax,
                           node_size=network_size,
                           node_color=network_color, node_shape=network_shape)
    # draw intersections
    nx.draw_networkx_nodes(G, pos, int_names, ax=ax,
                           node_size=inter_size,
                           node_color=inter_color, node_shape=inter_shape)
    # labels shifts
    pos_labels = {}
    for node, coords in pos.items():
        pos_labels[node] = (coords[0] + label_shift_x,
                            coords[1] + label_shift_y)

    # network labels
    if network_labels:
        network_labels = {lab:lab for lab in ntw_names}

        nx.draw_networkx_labels(G, pos_labels, labels=network_labels,
                                font_size=network_fontsize, ax=ax)

    # intersection labels
    if inter_labels:
        inter_labels = {'X'+str(k):'X'+str(k) for k in range(self.connections_nb)}

        nx.draw_networkx_labels(G, pos_labels, labels=inter_labels,
                                font_size=network_fontsize, ax=ax)

    # port labels
    if port_labels:
        port_labels = {lab:lab for lab in port_names}

        nx.draw_networkx_labels(G, pos_labels, labels=port_labels,
                                font_size=port_fontsize, ax=ax)

    # draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)
    if edge_labels:
        edge_labels = self.edge_labels
        nx.draw_networkx_edge_labels(G, pos,
                                      edge_labels=edge_labels, label_pos=0.5,
                                      font_size=edge_fontsize, ax=ax)
    # remove x and y axis and labels
    plt.axis('off')
    plt.tight_layout()


def plot_contour(freq: frequency.Frequency,
                 x: NumberLike, y: NumberLike, z: NumberLike,
                 min0max1: int, graph: bool = True,
                 cmap: str = 'plasma_r', title: str = '',
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
    \*\*kwargs : dict
        Other parameters passed to `matplotlib.plot()`.

    Returns
    -------
    GAMopt : :skrf.Network:
        Network
    VALopt : float
        min or max.

    """
    ri =  npy.linspace(0,1, 50);
    ti =  npy.linspace(0,2*npy.pi, 150);
    Ri , Ti = npy.meshgrid(ri, ti)
    xi = npy.linspace(-1,1, 50);
    Xi, Yi = npy.meshgrid(xi, xi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Zi = interpolator(Xi, Yi)
    if min0max1 == 1 :
        VALopt = npy.max(z)
    else :
        VALopt = npy.min(z)
    GAMopt = network.Network(f=[freq], s=x[z==VALopt] +1j*y[z==VALopt])

    if graph :
        fig, ax = plt.subplots(**kwargs)
        an = npy.linspace(0, 2*npy.pi, 50)
        cs, sn = npy.cos(an), npy.sin(an)
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
