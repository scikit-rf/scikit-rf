
#
#       helper.py
#
#       Copyright 2012 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
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

.. currentmodule:: skrf.convenience
========================================
convenience (:mod:`skrf.convenience`)
========================================

Holds functions that are general conveniences.


Plotting 
------------
.. autosummary::
   :toctree: generated/

   save_all_figs
   add_markers_to_lines
   legend_off
   func_on_all_figs
   

General 
------------
.. autosummary::
   :toctree: generated/

   now_string
   find_nearest
   find_nearest_index


Shorthand Names 
----------------

Below is a list of shorthand object names which can be use to save some 
typing. These names are defined in the main __init__ module. but listing
them here makes more sense. 


============ ================
Shorthand    Full Object Name   
============ ================
F            :class:`~skrf.frequency.Frequency`
N            :class:`~skrf.network.Network`
NS           :class:`~skrf.networkSet.NetworkSet`
M            :class:`~skrf.media.media.Media`
C            :class:`~skrf.calibration.calibration.Calibration`
============ ================

The following are shorthand names for commonly used, but unfortunately
longwinded functions.

============ ================
Shorthand    Full Object Name   
============ ================
lat          :func:`~skrf.network.load_all_touchstones`
saf          :func:`~skrf.convenience.save_all_figs`
============ ================
 



References
-------------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
'''



import mathFunctions as mf

import warnings
import os
import cPickle as pickle
import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime



# globals 



# Ploting
def save_all_figs(dir = './', format=['eps','pdf','svg','png']):
    '''
    Save all open Figures to disk.

    Parameters
    ------------
    dir : string
            path to save figures into
    format : list of strings
            the types of formats to save figures as. The elements of this
            list are passed to :matplotlib:`savefig`. This is a list so that
            you can save each figure in multiple formats.
    '''
    if dir[-1] != '/':
        dir = dir + '/'
    for fignum in plb.get_fignums():
        fileName = plb.figure(fignum).get_axes()[0].get_title()
        if fileName == '':
            fileName = 'unamedPlot'
        for fmt in format:
            plb.savefig(dir+fileName+'.'+fmt, format=fmt)
            print (dir+fileName+'.'+fmt)

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

def plot_complex(z,*args, **kwargs):
    '''
    plots a complex array or list in real vs imaginary.
    '''
    plb.plot(npy.array(z).real,npy.array(z).imag, *args, **kwargs)


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

# other
def now_string():
    '''
    returns a unique sortable string, representing the current time
    
    nice for generating date-time stamps to be used in file-names 
    
    '''
    return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')

def find_nearest(array,value):
    '''
    find nearest value in array.
    
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in 
    value : element of the array
        value to search for 
    
    Returns
    --------
    found_value : an element of the array 
        the value that is numerically closest to `value`
    
    '''
    idx=(npy.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    '''
    find nearest value in array.
    
    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in 
    value : element of the array
        value to search for 
    
    Returns
    --------
    found_index : int 
        the index at which the  numerically closest element to `value`
        was found at
    
    
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    '''
    return (npy.abs(array-value)).argmin()

# file IO

def get_fid(file, *args, **kwargs):
    '''
    Returns a file object, given a filename or file object
    
    Useful  when you want to allow the arguments of a function to
    be either files or filenames
    
    Parameters
    -------------
    file : str or file-object
        file to open 
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load
    '''
    if isinstance(file, basestring):
        return open(file, *args, **kwargs)
    else:
        return file
    
def get_extn(filename):
    '''
    Get the extension from a filename.
    
    The extension is defined as everything passed the last '.'.
    Returns None if it aint got one
    
    Parameters
    ------------
    filename : string 
        the filename 
    
    Returns
    --------
    ext : string, None
        either the extension (not including '.') or None if there 
        isnt one
        

    '''
    ext = os.path.splitext(filename)[-1]
    if len(ext)==0: 
        return None
    else:
        return ext[1:]




