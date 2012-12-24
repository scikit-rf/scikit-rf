
#       plotting.py
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

'''
.. module:: skrf.plotting
========================================
plotting (:mod:`skrf.plotting`)
========================================


This module provides general plotting functions.

Charts
-------

.. autosummary::
    :toctree: generated/

    smith
    plot_smith
    plot_rectangular
    plot_polar
    plot_complex_rectangular 
    plot_complex_polar
    

'''
import pylab as plb
import numpy as npy
from matplotlib.patches import Circle   # for drawing smith chart
#from matplotlib.lines import Line2D            # for drawing smith chart



def smith(smithR=1, chart_type = 'z', draw_labels = False, ax=None):
    '''
    plots the smith chart of a given radius

    Parameters
    -----------
    smithR : number
            radius of smith chart
    chart_type : ['z','y']
            Contour type. Possible values are
             * *'z'* : lines of constant impedance
             * *'y'* : lines of constant admittance
    draw_labels : Boolean
             annotate real and imaginary parts of impedance on the 
             chart (only if smithR=1)
    ax : matplotlib.axes object
            existing axes to draw smith chart on


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
        rLightList = plb.logspace(3,-5,9,base=.5)
        xLightList = plb.hstack([plb.logspace(2,-5,8,base=.5), -1*plb.logspace(2,-5,8,base=.5)])
    else:
        rLightList = plb.array( [ 0.2, 0.5, 1.0, 2.0, 5.0 ] )
        xLightList = plb.array( [ 0.2, 0.5, 1.0, 2.0 , 5.0, -0.2, -0.5, -1.0, -2.0, -5.0 ] )

    # cheap way to make a ok-looking smith chart at larger than 1 radii
    if smithR > 1:
        rMax = (1.+smithR)/(1.-smithR)
        rLightList = plb.hstack([ plb.linspace(0,rMax,11)  , rLightList ])

    if chart_type is 'y':
        y_flip_sign = -1
    else:
        y_flip_sign = 1
    # loops through Light and Heavy lists and draws circles using patches
    # for analysis of this see R.M. Weikles Microwave II notes (from uva)
    for r in rLightList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        contour.append( Circle( center, radius, ec='grey',fc = 'none'))
    for x in xLightList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        contour.append( Circle( center, radius, ec='grey',fc = 'none'))

    for r in rHeavyList:
        center = (r/(1.+r)*y_flip_sign,0 )
        radius = 1./(1+r)
        contour.append( Circle( center, radius, ec= 'black', fc = 'none'))
    for x in xHeavyList:
        center = (1*y_flip_sign,1./x)
        radius = 1./x
        contour.append( Circle( center, radius, ec='black',fc = 'none'))

    # clipping circle
    clipc = Circle( [0,0], smithR, ec='k',fc='None',visible=True)
    ax1.add_patch( clipc)

    #draw x and y axis
    ax1.axhline(0, color='k', lw=.1, clip_path=clipc)
    ax1.axvline(1*y_flip_sign, color='k', clip_path=clipc)
    ax1.grid(0)
    #set axis limits
    ax1.axis('equal')
    ax1.axis(smithR*npy.array([-1., 1., -1., 1.]))     

    if draw_labels:
        #Clear axis
        ax1.yaxis.set_ticks([])
        ax1.xaxis.set_ticks([])
        for loc, spine in ax1.spines.iteritems():
            spine.set_color('none')

        #Will make annotations only if the radius is 1 and it is the impedance smith chart
        if smithR is 1 and y_flip_sign is 1:
            #Make room for annotation
            ax1.axis(smithR*npy.array([-1., 1., -1.2, 1.2]))

            #Annotate real part
            for value in rLightList:
                rho = (value - 1)/(value + 1)
                ax1.annotate(str(value), xy=((rho-0.12)*smithR, 0.01*smithR), \
                    xytext=((rho-0.12)*smithR, 0.01*smithR))

            #Annotate imaginary part
            deltax = plb.array([-0.17, -0.14, -0.06,  0., 0.02, -0.2, -0.2, -0.08, 0., 0.03])
            deltay = plb.array([0., 0.03, 0.01, 0.02, 0., -0.02, -0.06, -0.09, -0.08, -0.05])
            for value, dx, dy in zip(xLightList, deltax, deltay):
                #Transforms from complex to cartesian and adds a delta to x and y values
                rhox = (-value**2 + 1)/(-value**2 - 1) * smithR * y_flip_sign + dx
                rhoy = (-2*value)/(-value**2 - 1) * smithR + dy
                #Annotate value
                ax1.annotate(str(value) + 'j', xy=(rhox, rhoy), xytext=(rhox, rhoy))

            #Annotate 0 and inf
            ax1.annotate('0.0', xy=(-1.15, -0.02), xytext=(-1.15, -0.02))
            ax1.annotate('$\infty$', xy=(1.02, -0.02), xytext=(1.02, -0.02))

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

    ax.plot(x, y, *args, **kwargs)

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
        ax.axis(axis)
        
    if plb.isinteractive():
        plb.draw()

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

def plot_smith(z, smith_r=1, chart_type='z', x_label='Real',
    y_label='Imag', title='Complex Plane', show_legend=True,
    axis='equal', ax=None, force_chart = False, *args, **kwargs):
    '''
    plot complex data on smith chart

    Parameters
    ------------
    z : array-like, of complex data
        data to plot
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
            smith(ax=ax, smithR = smith_r, chart_type=chart_type)

    plot_complex_rectangular(z, x_label=x_label, y_label=y_label,
        title=title, show_legend=show_legend, axis=axis,
        ax=ax, *args, **kwargs)

    ax.axis(smith_r*npy.array([-1., 1., -1., 1.]))
    if plb.isinteractive():
        plb.draw()



def shade_bands(edges, y_range=[-1e5,1e5],cmap='prism', **kwargs):
    '''
    Shades frequency bands.
    
    when plotting data over a set of frequency bands it is nice to 
    have each band visually seperated from the other. The kwarg `alpha`
    is useful.
    
    Parameters 
    --------------
    edges : array-like
        x-values seperating regions of a given shade
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
    for k in range(len(edges)-1):
        plb.fill_between(
            [edges[k],edges[k+1]], 
            y_range[0], y_range[1], 
            color = cmap(1.0*k/len(edges)),
            **kwargs)
