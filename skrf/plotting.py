

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

import matplotlib.pyplot as plb
import numpy as npy
from matplotlib.patches import Circle   # for drawing smith chart
from matplotlib.pyplot import quiver
from matplotlib import rcParams
#from matplotlib.lines import Line2D            # for drawing smith chart



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
    return rcParams['axes.color_cycle']
