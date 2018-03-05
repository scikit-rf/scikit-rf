'''
.. module:: skrf.plotting
========================================
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

Misc Functions
-----------------

.. autosummary::
    :toctree: generated/

    save_all_figs
    add_markers_to_lines
    legend_off
    func_on_all_figs
    scrape_legend

'''
import os

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plb
import numpy as npy
from matplotlib.patches import Circle   # for drawing smith chart
from matplotlib.pyplot import quiver
from matplotlib import rcParams
from matplotlib.dates import date2num

from . import network, frequency, calibration, networkSet
from . import mathFunctions as mf
from . util import now_string_2_dt

#from matplotlib.lines import Line2D            # for drawing smith chart

SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'
SI_CONVERSION = dict([(key, 10**((8-i)*3)) for i, key in enumerate(SI_PREFIXES_ASCII)])


def scale_frequency_ticks(ax, funit):
    if funit.lower() == "hz":
        prefix = " "
        scale = 1
    elif len(funit) == 3:
        prefix = funit[0]
        scale = SI_CONVERSION[prefix]
    else:
        raise ValueError("invalid funit {}".format(funit))
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale))
    ax.xaxis.set_major_formatter(ticks_x)


def smith(smithR=1, chart_type = 'z', draw_labels = False, border=False,
    ax=None, ref_imm = 1.0, draw_vswr=None):
    '''
    plots the smith chart of a given radius

    Parameters
    -----------
    smithR : number
            radius of smith chart
    chart_type : ['z','y','zy', 'yz']
            Contour type. Possible values are
             * *'z'* : lines of constant impedance
             * *'y'* : lines of constant admittance
             * *'zy'* : lines of constant impedance stronger than admittance
             * *'yz'* : lines of constant admittance stronger than impedance
    draw_labels : Boolean
             annotate real and imaginary parts of impedance on the
             chart (only if smithR=1)
    border : Boolean
        draw a rectangular border with axis ticks, around the perimeter
        of the figure. Not used if draw_labels = True

    ax : matplotlib.axes object
            existing axes to draw smith chart on

    ref_imm : number
            Reference immittance for center of Smith chart. Only changes
            labels, if printed.

    draw_vswr : list of numbers, Boolean or None
        draw VSWR circles. If True, default values are used.

    '''
    ##TODO: fix this function so it doesnt suck
    if ax == None:
        ax1 = plb.gca()
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

    # vswr circules verylight
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
        if smithR is 1:
            #Make room for annotation
            ax1.plot(npy.array([-1.25, 1.25]), npy.array([-1.1, 1.1]), 'w.', markersize = 0)
            ax1.axis('image')

            #Annotate real part
            for value in rLightList:
                # Set radius of real part's label; offset slightly left (Z
                # chart, y_flip_sign == 1) or right (Y chart, y_flip_sign == -1)
                # so label doesn't overlap chart's circles
                rho = (value - 1)/(value + 1) - y_flip_sign*0.01
                if y_flip_sign is 1:
                    halignstyle = "right"
                else:
                    halignstyle = "left"
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
                #Annotate value
                ax1.annotate(str(value*ref_imm) + 'j', xy=(rhox, rhoy),
                             xytext=(rhox, rhoy), ha = halignstyle, va = valignstyle)

            #Annotate 0 and inf
            ax1.annotate('0.0', xy=(-1.02, 0), xytext=(-1.02, 0),
                         ha = "right", va = "center")
            ax1.annotate('$\infty$', xy=(radialScaleFactor, 0), xytext=(radialScaleFactor, 0),
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


def plot_rectangular(x, y, x_label=None, y_label=None, title=None,
                     show_legend=True, axis='tight', ax=None, *args, **kwargs):
    '''
    plots rectangular data and optionally label axes.

    Parameters
    ------------
    x : array-like, of complex data
        data to plot
    y : array-like, of complex data
        data to plot
    x_label : string
        x-axis label
    y_label : string
        y-axis label
    title : string
        plot title
    show_legend : Boolean
        controls the drawing of the legend
    axis : str
        whether or not to autoscale the axis
    ax : :class:`matplotlib.axes.AxesSubplot` object
        axes to draw on
    *args, **kwargs : passed to pylab.plot

    '''
    if ax is None:
        ax = plb.gca()

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

    if plb.isinteractive():
        plb.draw()

    return my_plot


def plot_polar(theta, r, x_label=None, y_label=None, title=None,
    show_legend=True, axis_equal=False, ax=None, *args, **kwargs):
    '''
    plots polar data on a polar plot and optionally label axes.

    Parameters
    ------------
    theta : array-like
        data to plot
    r : array-like

    x_label : string
        x-axis label
    y_label : string
        y-axis label
    title : string
        plot title
    show_legend : Boolean
        controls the drawing of the legend
    ax : :class:`matplotlib.axes.AxesSubplot` object
        axes to draw on
    *args,**kwargs : passed to pylab.plot

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    '''
    if ax is None:
        ax = plb.gca(polar=True)

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

    if plb.isinteractive():
        plb.draw()

def plot_complex_rectangular(z, x_label='Real', y_label='Imag',
    title='Complex Plane', show_legend=True, axis='equal', ax=None,
    *args, **kwargs):
    '''
    plot complex data on the complex plane

    Parameters
    ------------
    z : array-like, of complex data
        data to plot
    x_label : string
        x-axis label
    y_label : string
        y-axis label
    title : string
        plot title
    show_legend : Boolean
        controls the drawing of the legend
    ax : :class:`matplotlib.axes.AxesSubplot` object
        axes to draw on
    *args,**kwargs : passed to pylab.plot

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart

    '''
    x = npy.real(z)
    y = npy.imag(z)
    plot_rectangular(x=x, y=y, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, *args, **kwargs)


def plot_complex_polar(z, x_label=None, y_label=None,
    title=None, show_legend=True, axis_equal=False, ax=None,
    *args, **kwargs):
    '''
    plot complex data in polar format.

    Parameters
    ------------
    z : array-like, of complex data
        data to plot
    x_label : string
        x-axis label
    y_label : string
        y-axis label
    title : string
        plot title
    show_legend : Boolean
        controls the drawing of the legend
    ax : :class:`matplotlib.axes.AxesSubplot` object
        axes to draw on
    *args,**kwargs : passed to pylab.plot

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    '''
    theta = npy.angle(z)
    r = npy.abs(z)
    plot_polar(theta=theta, r=r, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis_equal=axis_equal,
        ax=ax, *args, **kwargs)


def plot_smith(s, smith_r=1, chart_type='z', x_label='Real',
    y_label='Imaginary', title='Complex Plane', show_legend=True,
    axis='equal', ax=None, force_chart = False, draw_vswr=None, *args, **kwargs):
    '''
    plot complex data on smith chart

    Parameters
    ------------
    s : complex array-like
        reflection-coeffient-like data to plot
    smith_r : number
        radius of smith chart
    chart_type : ['z','y']
        Contour type for chart.
         * *'z'* : lines of constant impedance
         * *'y'* : lines of constant admittance
    x_label : string
        x-axis label
    y_label : string
        y-axis label
    title : string
        plot title
    show_legend : Boolean
        controls the drawing of the legend
    axis_equal: Boolean
        sets axis to be equal increments (calls axis('equal'))
    force_chart : Boolean
        forces the re-drawing of smith chart
    ax : :class:`matplotlib.axes.AxesSubplot` object
        axes to draw on
    *args,**kwargs : passed to pylab.plot

    See Also
    ----------
    plot_rectangular : plots rectangular data
    plot_complex_rectangular : plot complex data on complex plane
    plot_polar : plot polar data
    plot_complex_polar : plot complex data on polar plane
    plot_smith : plot complex data on smith chart
    '''

    if ax is None:
        ax = plb.gca()

    # test if smith chart is already drawn
    if not force_chart:
        if len(ax.patches) == 0:
            smith(ax=ax, smithR = smith_r, chart_type=chart_type, draw_vswr=draw_vswr)

    plot_complex_rectangular(s, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, *args, **kwargs)

    ax.axis(smith_r*npy.array([-1.1, 1.1, -1.1, 1.1]))
    if plb.isinteractive():
        plb.draw()


def subplot_params(ntwk, param='s', proj='db', size_per_port=4, newfig=True,
                   add_titles=True, keep_it_tight=True,  subplot_kw={}, *args, **kw):
    '''
    Plot all networks parameters individually on subplots

    Parameters
    --------------


    '''
    if newfig:
        f,axs= plb.subplots(ntwk.nports,ntwk.nports,
                            figsize =(size_per_port*ntwk.nports,
                                      size_per_port*ntwk.nports ),
                                      **subplot_kw)
    else:
        f = plb.gcf()
        axs = npy.array(f.get_axes())

    for ports,ax in zip(ntwk.port_tuples, axs.flatten()):
        plot_func = ntwk.__getattribute__('plot_%s_%s'%(param, proj))
        plot_func(m=ports[0], n=ports[1], ax=ax,*args, **kw)
        if add_titles:
            ax.set_title('%s%i%i'%(param.upper(),ports[0]+1, ports[1]+1))
    if keep_it_tight:
       plb.tight_layout()
    return f,axs

def shade_bands(edges, y_range=None,cmap='prism', **kwargs):
    '''
    Shades frequency bands.

    when plotting data over a set of frequency bands it is nice to
    have each band visually separated from the other. The kwarg `alpha`
    is useful.

    Parameters
    --------------
    edges : array-like
        x-values separating regions of a given shade
    y_range : tuple
        y-values to shade in
    cmap : str
        see matplotlib.cm  or matplotlib.colormaps for acceptable values
    \*\* : key word arguments
        passed to `matplotlib.fill_between`

    Examples
    -----------
    >>> rf.shade_bands([325,500,750,1100], alpha=.2)
    '''
    cmap = plb.cm.get_cmap(cmap)
    y_range=plb.gca().get_ylim()
    axis = plb.axis()
    for k in range(len(edges)-1):
        plb.fill_between(
            [edges[k],edges[k+1]],
            y_range[0], y_range[1],
            color = cmap(1.0*k/len(edges)),
            **kwargs)
    plb.axis(axis)

def save_all_figs(dir = './', format=None, replace_spaces = True, echo = True):
    '''
    Save all open Figures to disk.

    Parameters
    ------------
    dir : string
            path to save figures into
    format : None, or list of strings
            the types of formats to save figures as. The elements of this
            list are passed to :matplotlib:`savefig`. This is a list so that
            you can save each figure in multiple formats.
    echo : bool
            True prints filenames as they are saved
    '''
    if dir[-1] != '/':
        dir = dir + '/'
    for fignum in plb.get_fignums():
        fileName = plb.figure(fignum).get_axes()[0].get_title()
        if replace_spaces:
            fileName = fileName.replace(' ','_')
        if fileName == '':
            fileName = 'unnamedPlot'
        if format is None:
            plb.savefig(dir+fileName)
            if echo:
                print((dir+fileName))
        else:
            for fmt in format:
                plb.savefig(dir+fileName+'.'+fmt, format=fmt)
                if echo:
                    print((dir+fileName+'.'+fmt))
saf = save_all_figs

def add_markers_to_lines(ax=None,marker_list=['o','D','s','+','x'], markevery=10):
    '''
    adds markers to existing lings on a plot

    this is convinient if you have already have a plot made, but then
    need to add markers afterwards, so that it can be interpreted in
    black and white. The markevery argument makes the markers less
    frequent than the data, which is generally what you want.

    Parameters
    -----------
    ax : matplotlib.Axes
        axis which to add markers to, defaults to gca()
    marker_list : list of marker characters
        see matplotlib.plot help for possible marker characters
    markevery : int
        markevery number of points with a marker.

    '''
    if ax is None:
        ax=plb.gca()
    lines = ax.get_lines()
    if len(lines) > len (marker_list ):
        marker_list *= 3
    [k[0].set_marker(k[1]) for k in zip(lines, marker_list)]
    [line.set_markevery(markevery) for line in lines]

def legend_off(ax=None):
    '''
    turn off the legend for a given axes.

    if no axes is given then it will use current axes.

    Parameters
    -----------
    ax : matplotlib.Axes object
        axes to operate on
    '''
    if ax is None:
        plb.gca().legend_.set_visible(0)
    else:
        ax.legend_.set_visible(0)

def scrape_legend(n=None, ax=None):
    '''
    scrapes a legend with redundant labels

    Given a legend of m entries of n groups, this will remove all but
    every m/nth entry. This is used when you plot many lines representing
    the same thing, and only want one label entry in the legend  for the
    whole ensemble of lines

    '''

    if ax is None:
        ax = plb.gca()

    handles, labels = ax.get_legend_handles_labels()

    if n is None:
        n =len ( set(labels))

    if n>len(handles):
        raise ValueError('number of entries is too large')

    k_list = [int(k) for k in npy.linspace(0,len(handles)-1,n)]
    ax.legend([handles[k] for k in k_list], [labels[k] for k in k_list])

def func_on_all_figs(func, *args, **kwargs):
    '''
    runs a function after making all open figures current.

    useful if you need to change the properties of many open figures
    at once, like turn off the grid.

    Parameters
    ----------
    func : function
        function to call
    \*args, \*\*kwargs : pased to func

    Examples
    ----------
    >>> rf.func_on_all_figs(grid,alpha=.3)
    '''
    for fig_n in plb.get_fignums():
        fig = plb.figure(fig_n)
        for ax_n in fig.axes:
            fig.add_axes(ax_n) # trick to make axes current
            func(*args, **kwargs)
            plb.draw()

foaf = func_on_all_figs

def plot_vector(a, off=0+0j, *args, **kwargs):
    '''
    plot a 2d vector
    '''
    return quiver(off.real,off.imag,a.real,a.imag,scale_units ='xy',
           angles='xy',scale=1, *args, **kwargs)


def colors():
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


def __generate_plot_functions(self):
    '''
    '''
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

            # was_interactive = plb.isinteractive
            # if was_interactive:
            #     plb.interactive(False)

            for m in M:
                for n in N:
                    # set the legend label for this trace to the networks
                    # name if it exists, and they didn't pass a name key in
                    # the kwargs
                    if gen_label:
                        if self.name is None:
                            if plb.rcParams['text.usetex']:
                                label_string = '$%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = '%s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        else:
                            if plb.rcParams['text.usetex']:
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
            #     plb.interactive(True)
            #     plb.draw()
            #     plb.show()

        plot_prop_polar.__doc__ = '''
plot the Network attribute :attr:`%s` vs frequency.

Parameters
-----------
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

\*args,\\**kwargs : arguments, keyword arguments
    passed to :func:`matplotlib.plot`

Notes
-------
This function is dynamically generated upon Network
initialization. This is accomplished by calling
:func:`plot_vs_frequency_generic`

Examples
------------
>>> myntwk.plot_%s(m=1,n=0,color='r')
''' % (prop_name, prop_name)
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

            #was_interactive = plb.isinteractive
            #if was_interactive:
            #    plb.interactive(False)

            for m in M:
                for n in N:
                    # set the legend label for this trace to the networks
                    # name if it exists, and they didn't pass a name key in
                    # the kwargs
                    if gen_label:
                        if self.name is None:
                            if plb.rcParams['text.usetex']:
                                label_string = '$%s_{%i%i}$'%\
                                (prop_name[0].upper(),m+1,n+1)
                            else:
                                label_string = '%s%i%i'%\
                                (prop_name[0].upper(),m+1,n+1)
                        else:
                            if plb.rcParams['text.usetex']:
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
            #    plb.interactive(True)
            #    plb.draw()
            #    plb.show()

        plot_prop_rect.__doc__ = '''
plot the Network attribute :attr:`%s` vs frequency.

Parameters
-----------
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

\*args,\\**kwargs : arguments, keyword arguments
    passed to :func:`matplotlib.plot`

Notes
-------
This function is dynamically generated upon Network
initialization. This is accomplished by calling
:func:`plot_vs_frequency_generic`

Examples
------------
>>> myntwk.plot_%s(m=1,n=0,color='r')
''' % (prop_name, prop_name)

        # setattr(self.__class__,'plot_%s_complex'%(prop_name), \
        setattr(self,'plot_%s_complex'%(prop_name), \
            plot_prop_rect)


        for func_name in COMPONENT_FUNC_DICT:
            attribute = '%s_%s' % (prop_name, func_name)
            y_label = Y_LABEL_DICT[func_name]

            def plot_func(self,  m=None, n=None, ax=None,
                          show_legend=True, attribute=attribute,
                          y_label=y_label, pad=0, window='hamming', z0=50, *args, **kwargs):

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
                # was_interactive = plb.isinteractive
                # if was_interactive:
                #     plb.interactive(False)
                for m in M:
                    for n in N:
                        # set the legend label for this trace to the networks
                        # name if it exists, and they didn't pass a name key in
                        # the kwargs
                        if gen_label:
                            if self.name is None:
                                if plb.rcParams['text.usetex']:
                                    label_string = '$%s_{%i%i}$'%\
                                    (attribute[0].upper(),m+1,n+1)
                                else:
                                    label_string = '%s%i%i'%\
                                    (attribute[0].upper(),m+1,n+1)
                            else:
                                if plb.rcParams['text.usetex']:
                                    label_string = self.name+', $%s_{%i%i}$'%\
                                    (attribute[0].upper(),m+1,n+1)
                                else:
                                    label_string = self.name+', %s%i%i'%\
                                    (attribute[0].upper(),m+1,n+1)
                            kwargs['label'] = label_string

                        # quick and dirty way to plot step and impulse response
                        if 'time_impulse' in attribute:
                            xlabel = 'Time (s)'
                            x,y = self.impulse_response(pad=pad, window=window)
                            # default is reflexion coefficient axis
                            if attribute[0].lower() == 'z':
                                # if they want impedance axis, give it to them
                                y_label = 'Z (ohm)'
                                y[x ==  1.] =  1. + 1e-12  # solve numerical singularity
                                y[x == -1.] = -1. + 1e-12  # solve numerical singularity
                                y = z0 * (1+y) / (1-y)
                            plot_rectangular(x=x,
                                             y=y,
                                             x_label=xlabel,
                                             y_label=y_label,
                                             show_legend=show_legend, ax=ax,
                                             *args, **kwargs)
                        elif 'time_step' in attribute:
                            xlabel = 'Time (s)'
                            x, y = self.step_response(pad=pad, window=window)
                            # default is reflexion coefficient axis
                            if attribute[0].lower() == 'z':
                                # if they want impedance axis, give it to them
                                y_label = 'Z (ohm)'
                                y[x ==  1.] =  1. + 1e-12  # solve numerical singularity
                                y[x == -1.] = -1. + 1e-12  # solve numerical singularity
                                y = z0 * (1+y) / (1-y)
                            plot_rectangular(x=x,
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

                                # scale the ticklabels according to the frequency unit:
                                if ax is None:
                                    ax = plb.gca()
                                scale_frequency_ticks(ax, self.frequency.unit)

                            plot_rectangular(x=x,
                                             y=getattr(self, attribute)[:, m, n],
                                             x_label=xlabel,
                                             y_label=y_label,
                                             show_legend=show_legend, ax=ax,
                                             *args, **kwargs)
                #if was_interactive:
                #    plb.interactive(True)
                #    plb.draw()
                #    #plb.show()

            plot_func.__doc__ = '''
    plot the Network attribute :attr:`%s` vs frequency.

    Parameters
    -----------
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

    \*args,\\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot`

    Notes
    -------
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    ------------
    >>> myntwk.plot_%s(m=1,n=0,color='r')
    '''%(attribute,attribute)

            # setattr(self.__class__,'plot_%s'%(attribute), \
            setattr(self,'plot_%s'%(attribute), \
                plot_func)


def labelXAxis(self, ax=None):
    '''
    Label the x-axis of a plot.

    Sets the labels of a plot using :func:`matplotlib.x_label` with
    string containing the frequency unit.

    Parameters
    ---------------
    ax : :class:`matplotlib.Axes`, optional
            Axes on which to label the plot, defaults what is
            returned by :func:`matplotlib.gca()`
    '''
    if ax is None:
        ax = plb.gca()
    ax.set_xlabel('Frequency (%s)' % self.unit)


def plot_v_frequency(self, y, *args, **kwargs):
    '''
    Plot something vs this frequency

    This plots whatever is given vs. `self.f_scaled` and then
    calls `labelXAxis`.
    '''

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

    # plb.plot(self.f_scaled, y, *args, **kwargs)
    plb.plot(self.f, y, *args, **kwargs)
    ax = plb.gca()
    scale_frequency_ticks(ax, self.unit)
    plb.autoscale(axis='x', tight=True)
    self.labelXAxis()


## specific ploting functions
def plot(self, *args, **kw):
    '''
    plot somthing vs frequency
    '''
    return self.frequency.plot(*args, **kw)


def plot_passivity(self, port=None, label_prefix=None, *args, **kwargs):
    '''
    Plot dB(diag(passivity metric)) vs frequency

    Notes
    -------
    This plot does not completely capture the passivity metric, which
    is a test for `unitary-ness` of the s-matrix. However, it may
    be  used to display a measure of power dissipated in a network.

    See Also
    -----------
    passivity
    '''
    name = '' if self.name is None else self.name

    if port is None:
        ports = range(self.nports)
    else:
        ports = [port]
    for k in ports:
        if label_prefix == None:
            label = name + ', port %i' % (k + 1)
        else:
            label = label_prefix + ', port %i' % (k + 1)
        self.frequency.plot(mf.complex_2_db(self.passivity[:, k, k]),
                            label=label,
                            *args, **kwargs)

    plb.legend()
    plb.draw()


def plot_reciprocity(self, db=False, *args, **kwargs):
    '''
    Plot reciprocity metric

    See Also
    -----------
    reciprocity
    '''
    for m in range(self.nports):
        for n in range(self.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = 'ports %i%i' % (m, n)
                y = self.reciprocity[:, m, n].flatten()
                if db:
                    y = mf.complex_2_db(y)
                self.frequency.plot(y, *args, **kwargs)

    plb.legend()
    plb.draw()


def plot_reciprocity2(self, db=False, *args, **kwargs):
    '''
    Plot reciprocity metric #2

    this is distance of the determinant of the wave-cascading matrix
    from unity.

    .. math::

            abs(1 - S/S^T )



    See Also
    -----------
    reciprocity
    '''
    for m in range(self.nports):
        for n in range(self.nports):
            if m > n:
                if 'label' not in kwargs.keys():
                    kwargs['label'] = 'ports %i%i' % (m, n)
                y = self.reciprocity2[:, m, n].flatten()
                if db:
                    y = mf.complex_2_db(y)
                self.frequency.plot(y, *args, **kwargs)

    plb.legend()
    plb.draw()


def plot_s_db_time(self,*args,**kwargs):
    return self.windowed().plot_s_time_db(*args,**kwargs)


# plotting
def plot_s_smith(self,m=None, n=None,r=1,ax = None, show_legend=True,\
        chart_type='z', draw_labels=False, label_axes=False, draw_vswr=None, *args,**kwargs):
    '''
    plots the scattering parameter on a smith chart

    plots indices `m`, `n`, where `m` and `n` can be integers or
    lists of integers.


    Parameters
    -----------
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
        draw impedance or addmitance contours
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
    ---------
    >>> myntwk.plot_s_smith()
    >>> myntwk.plot_s_smith(m=0,n=1,color='b', marker='x')
    '''
    # TODO: prevent this from re-drawing smith chart if one alread
    # exists on current set of axes

    # get current axis if user doesnt supply and axis
    if ax is None:
        ax = plb.gca()


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
                    if plb.rcParams['text.usetex']:
                        label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
                    else:
                        label_string = 'S'+repr(m+1) + repr(n+1)
                else:
                    if plb.rcParams['text.usetex']:
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


def plot_it_all(self,*args, **kwargs):
    '''
    Plots dB, deg, smith, and complex in subplots

    Plots the magnitude in dB in subplot 1, the phase in degrees in
    subplot 2, a smith chart in subplot 3, and a complex plot in
    subplot 4.

    Parameters
    -----------
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
    ---------
    >>> from skrf.data import ring_slot
    >>> ring_slot.plot_it_all()
    '''
    plb.subplot(221)
    getattr(self,'plot_s_db')(*args, **kwargs)
    plb.subplot(222)
    getattr(self,'plot_s_deg')(*args, **kwargs)
    plb.subplot(223)
    getattr(self,'plot_s_smith')(*args, **kwargs)
    plb.subplot(224)
    getattr(self,'plot_s_complex')(*args, **kwargs)


def stylely(rc_dict={}, style_file = 'skrf.mplstyle'):
    '''
    loads the rc-params from the specified file (file must be located in skrf/data)
    '''

    from skrf.data import pwd # delayed to solve circular import
    rc = mpl.rc_params_from_file(os.path.join(pwd, style_file))
    mpl.rcParams.update(rc)
    mpl.rcParams.update(rc_dict)


def plot_calibration_errors(self, *args, **kwargs):
    '''
    Plots biased, unbiased and total error in dB scaled

    See Also
    ---------
    biased_error
    unbiased_error
    total_error
    '''
    port_list = self.biased_error.port_tuples
    for m,n in port_list:
        plb.figure()
        plb.title('S%i%i'%(m+1,n+1))
        self.unbiased_error.plot_s_db(m,n,**kwargs)
        self.biased_error.plot_s_db(m,n,**kwargs)
        self.total_error.plot_s_db(m,n,**kwargs)
        plb.ylim(-100,0)


def plot_caled_ntwks(self, attr='s_smith', show_legend=False,**kwargs):
    '''
    Plots corrected calibration standards

    Given that the calibration is overdetermined, this may be used
    as a heuristic verification of calibration quality.

    Parameters
    ------------------
    attr : str
        Network property to plot, ie 's_db', 's_smith', etc
    show_legend : bool
        draw a legend or not
    \\*\\*kwargs : kwargs
        passed to the plot method of Network
    '''
    ns = networkSet.NetworkSet(self.caled_ntwks)
    kwargs.update({'show_legend':show_legend})

    if ns[0].nports ==1:
        ns.__getattribute__('plot_'+attr)(0,0, **kwargs)
    elif ns[0].nports ==2:
        plb.figure(figsize = (8,8))
        for k,mn in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
            plb.subplot(221+k)
            plb.title('S%i%i'%(mn[0]+1,mn[1]+1))
            ns.__getattribute__('plot_'+attr)(*mn, **kwargs)
    else:
        raise NotImplementedError
    plb.tight_layout()


def plot_residuals(self, attr='s_db', **kwargs):
    '''
    Plot residual networks.

    Given that the calibration is overdetermined, this may be used
    as a metric of the calibration's *goodness of fit*

    Parameters
    ------------------
    attr : str
        Network property to plot, ie 's_db', 's_smith', etc
    \\*\\*kwargs : kwargs
        passed to the plot method of Network

    See Also
    --------
    Calibration.residual_networks
    '''

    networkSet.NetworkSet(self.residual_ntwks).__getattribute__('plot_'+attr)(**kwargs)


# Network Set Plotting Commands
def animate(self, attr='s_deg', ylims=(-5, 5), xlims=None, show=True,
            savefigs=False, dir_='.', *args, **kwargs):
    '''
    animate a property of the networkset

    This loops through all elements in the NetworkSet and calls
    a plotting attribute (ie Network.plot_`attr`), with given \*args
    and \*\*kwargs.

    Parameters
    --------------
    attr : str
        plotting property of a Network (ie 's_db', 's_deg', etc)
    ylims : tuple
        passed to ylim. needed to have consistent y-limits accross frames
    xlims : tuple
        passed to xlim
    show : bool
        show each frame as its animated
    savefigs : bool
        save each frame as a png

    \*args, \*\*kwargs :
        passed to the Network plotting function

    Notes
    --------
    using `label=None` will speed up animation significantly,
    because it prevents the legend from drawing

    to create video paste this:

        !avconv -r 10 -i out_%5d.png  -vcodec huffyuv out.avi

    or (depending on your ffmpeg version)

        !ffmpeg -r 10 -i out_%5d.png  -vcodec huffyuv out.avi

    Examples
    ------------
    >>>ns.animate('s_deg', ylims=(-5,5),label=None)

    '''
    was_interactive = plb.isinteractive()
    plb.ioff()

    for idx, k in enumerate(self):
        plb.clf()
        if 'time' in attr:
            tmp_ntwk = k.windowed()
            tmp_ntwk.__getattribute__('plot_' + attr)(*args, **kwargs)
        else:
            k.__getattribute__('plot_' + attr)(*args, **kwargs)
        if ylims is not None:
            plb.ylim(ylims)
        if xlims is not None:
            plb.xlim(xlims)
        # rf.legend_off()
        plb.draw();
        if show:
            plb.show()
        if savefigs:
            fname = os.path.join(dir_, 'out_%.5i' % idx + '.png')
            plb.savefig(fname)

    if savefigs:
        print('\n\n')
    if was_interactive:
        plb.ion()


def plot_uncertainty_bounds_component(
        self, attribute, m=None, n=None,
        type='shade', n_deviations=3, alpha=.3, color_error=None, markevery_error=20,
        ax=None, ppf=None, kwargs_error={}, *args, **kwargs):
    '''
    plots mean value of the NetworkSet with +- uncertainty bounds
    in an Network's attribute. This is designed to represent
    uncertainty in a scalar component of the s-parameter. for example
    plotting the uncertainty in the magnitude would be expressed by,

            mean(abs(s)) +- std(abs(s))

    the order of mean and abs is important.


    takes:
            attribute: attribute of Network type to analyze [string]
            m: first index of attribute matrix [int]
            n: second index of attribute matrix [int]
            type: ['shade' | 'bar'], type of plot to draw
            n_deviations: number of std deviations to plot as bounds [number]
            alpha: passed to matplotlib.fill_between() command. [number, 0-1]
            color_error: color of the +- std dev fill shading
            markevery_error: if type=='bar', this controls frequency
                    of error bars
            ax: Axes to plot on
            ppf: post processing function. a function applied to the
                    upper and lower bounds
            *args,**kwargs: passed to Network.plot_s_re command used
                    to plot mean response
            kwargs_error: dictionary of kwargs to pass to the fill_between
                    or errorbar plot command depending on value of type.

    returns:
            None


    Note:
            for phase uncertainty you probably want s_deg_unwrap, or
            similar. uncerainty for wrapped phase blows up at +-pi.

    '''

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

            ax = plb.gca()

            ntwk_mean = self.__getattribute__('mean_'+attribute)
            ntwk_std = self.__getattribute__('std_'+attribute)
            ntwk_std.s = n_deviations * ntwk_std.s

            upper_bound = (ntwk_mean.s[:, m, n] + ntwk_std.s[:, m, n]).squeeze()
            lower_bound = (ntwk_mean.s[:, m, n] - ntwk_std.s[:, m, n]).squeeze()

            if ppf is not None:
                if type =='bar':
                    raise NotImplementedError('the \'ppf\' options don\'t work correctly with the bar-type error plots')
                ntwk_mean.s = ppf(ntwk_mean.s)
                upper_bound = ppf(upper_bound)
                lower_bound = ppf(lower_bound)
                lower_bound[npy.isnan(lower_bound)]=min(lower_bound)
                if ppf in [mf.magnitude_2_db, mf.mag_2_db]:  # quickfix of wrong ylabels due to usage of ppf for *_db plots
                    if attribute is 's_mag':
                        plot_attribute = 's_db'
                    elif attribute is 's_time_mag':
                        plot_attribute = 's_time_db'

            if type == 'shade':
                ntwk_mean.plot_s_re(ax=ax, m=m, n=n, *args, **kwargs)
                if color_error is None:
                    color_error = ax.get_lines()[-1].get_color()
                ax.fill_between(ntwk_mean.frequency.f,
                                lower_bound, upper_bound, alpha=alpha, color=color_error,
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

            ax.set_ylabel(Y_LABEL_DICT.get(plot_attribute[2:],''))  # use only the function of the attribute
            scale_frequency_ticks(ax, ntwk_mean.frequency.unit)
            ax.axis('tight')


def plot_minmax_bounds_component(self, attribute, m=0, n=0,
                                 type='shade', alpha=.3, color_error=None, markevery_error=20,
                                 ax=None, ppf=None, kwargs_error={}, *args, **kwargs):
    '''
    plots mean value of the NetworkSet with +- uncertainty bounds
    in an Network's attribute. This is designed to represent
    uncertainty in a scalar component of the s-parameter. for example
    plotting the uncertainty in the magnitude would be expressed by,

            mean(abs(s)) +- std(abs(s))

    the order of mean and abs is important.


    takes:
            attribute: attribute of Network type to analyze [string]
            m: first index of attribute matrix [int]
            n: second index of attribute matrix [int]
            type: ['shade' | 'bar'], type of plot to draw
            n_deviations: number of std deviations to plot as bounds [number]
            alpha: passed to matplotlib.fill_between() command. [number, 0-1]
            color_error: color of the +- std dev fill shading
            markevery_error: if type=='bar', this controls frequency
                    of error bars
            ax: Axes to plot on
            ppf: post processing function. a function applied to the
                    upper and low
            *args,**kwargs: passed to Network.plot_s_re command used
                    to plot mean response
            kwargs_error: dictionary of kwargs to pass to the fill_between
                    or errorbar plot command depending on value of type.

    returns:
            None


    Note:
            for phase uncertainty you probably want s_deg_unwrap, or
            similar.  uncertainty for wrapped phase blows up at +-pi.

    '''

    if ax is None:
        ax = plb.gca()

    ntwk_mean = self.__getattribute__('mean_'+attribute)

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
            if attribute is 's_mag':
                attribute = 's_db'
            elif attribute is 's_time_mag':
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
    '''
    this just calls
            plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)
    see plot_uncertainty_bounds for help

    '''
    kwargs.update({'attribute':'s_mag','ppf':mf.magnitude_2_db})
    self.plot_uncertainty_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_db(self,*args, **kwargs):
    '''
    this just calls
            plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)
    see plot_uncertainty_bounds for help

    '''
    kwargs.update({'attribute':'s_mag','ppf':mf.magnitude_2_db})
    self.plot_minmax_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_db10(self,*args, **kwargs):
    '''
    this just calls
            plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)
    see plot_uncertainty_bounds for help

    '''
    kwargs.update({'attribute':'s_mag','ppf':mf.mag_2_db10})
    self.plot_minmax_bounds_component(*args,**kwargs)

def plot_uncertainty_bounds_s_time_db(self,*args, **kwargs):
    '''
    this just calls
            plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)
    see plot_uncertainty_bounds for help

    '''
    kwargs.update({'attribute':'s_time_mag','ppf':mf.magnitude_2_db})
    self.plot_uncertainty_bounds_component(*args,**kwargs)

def plot_minmax_bounds_s_time_db(self,*args, **kwargs):
    '''
    this just calls
            plot_uncertainty_bounds(attribute= 's_mag','ppf':mf.magnitude_2_db*args,**kwargs)
    see plot_uncertainty_bounds for help

    '''
    kwargs.update({'attribute':'s_time_mag','ppf':mf.magnitude_2_db})
    self.plot_minmax_bounds_component(*args,**kwargs)

def plot_uncertainty_decomposition(self, m=0,n=0):
    '''
    plots the total and  component-wise uncertainty

    Parameters
    --------------
    m : int
        first s-parameters index
    n :
        second s-parameter index

    '''
    if self.name is not None:
        plb.title(r'Uncertainty Decomposition: %s $S_{%i%i}$'%(self.name,m,n))
    self.std_s.plot_s_mag(label='Distance', m=m,n=n)
    self.std_s_re.plot_s_mag(label='Real',  m=m,n=n)
    self.std_s_im.plot_s_mag(label='Imaginary',  m=m,n=n)
    self.std_s_mag.plot_s_mag(label='Magnitude',  m=m,n=n)
    self.std_s_arcl.plot_s_mag(label='Arc-length',  m=m,n=n)

def plot_uncertainty_bounds_s(self, multiplier =200, *args, **kwargs):
    '''
    Plots complex uncertainty bounds plot on smith chart.

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
    -------------
    multipliter : float
        controls the circle sizes, by multiples of the standard
        deviation.



    '''
    default_kwargs = {
        'marker':'o',
        'color':'b',
        'mew':0,
        'ls':'',
        'alpha':.1,
        'label':None,
        }
    default_kwargs.update(**kwargs)



    if plb.isinteractive():
        was_interactive = True
        plb.interactive(0)
    else:
        was_interactive = False

    [self.mean_s[k].plot_s_smith(*args, ms = self.std_s[k].s_mag*multiplier, **default_kwargs) for k in range(len(self[0]))]

    if was_interactive:
        plb.interactive(1)
    plb.draw()
    plb.show()

def plot_logsigma(self, label_axis=True, *args,**kwargs):
        '''
        plots the uncertainty for the set in units of log-sigma.
        Log-sigma is the complex standard deviation, plotted in units
        of dB's.

        Parameters
        ------------
        \\*args, \\*\\*kwargs : arguments
            passed to self.std_s.plot_s_db()
        '''
        self.std_s.plot_s_db(*args,**kwargs)
        if label_axis:
            plb.ylabel('Standard Deviation(dB)')

def signature(self, m=0, n=0, component='s_mag',
              vmax=None, vs_time=False, cbar_label=None,
              *args, **kwargs):
    '''
    Visualization of a NetworkSet.

    Creates a colored image representing the some component
    of each Network in the  NetworkSet, vs frequency.


    Parameters
    ------------
    m : int
        first s-parameters index
    n : int
        second s-parameter index
    component : ['s_mag','s_db','s_deg' ..]
        scalar component of Network to visualize. should
        be a property of the Network object.
    vmax : number
        sets upper limit of colorbar, if None, will be set to
        3*mean of the magnitude of the complex difference
    vs_time: Boolean
        if True, then we assume each Network.name was made with
        rf.now_string, and we make the y-axis a datetime axis
    cbar_label: String
        label for the colorbar

    \*args,\*\*kw : arguments, keyword arguments
        passed to :func:`~pylab.imshow`


    '''

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
    img = plb.imshow(mat, *args, **kw)

    if vs_time:
        ax = plb.gca()
        ax.yaxis_date()
        # date_format = plb.DateFormatter('%M:%S.%f')
        # ax.yaxis.set_major_formatter(date_format)
        # cbar.set_label('Magntidue (dB)')
        plb.ylabel('Time')
    else:
        plb.ylabel('Network #')

    plb.grid(0)
    freq.labelXAxis()

    cbar = plb.colorbar()
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    return img
