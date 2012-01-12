
#
#       convenience.py
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen 
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

.. currentmodule:: mwavepy.convenience
========================================
convenience (:mod:`mwavepy.convenience`)
========================================

Holds pre-initialized  objects's and functions that are general 
conveniences.


Functions
------------
.. autosummary::
   :toctree: generated/

   save_all_figs
   add_markers_to_lines 
   legend_off
   now_string
   find_nearest
   find_nearest_index
   

Pre-initialized Objects 
--------------------------

:class:`~mwavepy.frequency.Frequency` Objects
==============================================
These are predefined :class:`~mwavepy.frequency.Frequency` objects 
that correspond to standard waveguide bands. This information is taken 
from the VDI Application Note 1002 [#]_ . The naming convenction is 
f_wr# where '#' is the band number.


=======================  ===============================================
Object Name              Description
=======================  ===============================================
f_wr10                   WR-10, 75-110 GHz
f_wr3                    WR-3, 220-325 GHz
f_wr2p2                  WR-2.2, 330-500 GHz
f_wr1p5                  WR-1.5, 500-750 GHz
f_wr1                    WR-1, 750-1100 GHz
=======================  ===============================================


:class:`~mwavepy.media.media.Media` Objects
==============================================
These are predefined :class:`~mwavepy.media.media.Media` objects 
that represent Standardized transmission line media's. This information

Rectangular Waveguide Media's 
++++++++++++++++++++++++++++++++++

:class:`~mwavepy.media.rectangularWaveguide.RectangularWaveguide` 
Objects for standard bands.

=======================  ===============================================
Object Name              Description
=======================  ===============================================
wr10                     WR-10, 75-110 GHz
wr3                      WR-3, 220-325 GHz
wr2p2                    WR-2.2, 330-500 GHz
wr1p5                    WR-1.5, 500-750 GHz
wr1                      WR-1, 750-1100 GHz
=======================  ===============================================
  

	
References
-------------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
'''


from network import *
from frequency import Frequency
from media import RectangularWaveguide


import os
import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime


# pre-initialized classes

#frequency bands 
f_wr10 	= Frequency(75,110,201, 'ghz')
f_wr3  	= Frequency(220,325,201, 'ghz')
f_wr2p2 = Frequency(330,500,201, 'ghz')
f_wr1p5 = Frequency(500,750,201, 'ghz')
f_wr1	= Frequency(750,1100,201, 'ghz')

# rectangular waveguides
wr10 	= RectangularWaveguide(Frequency(75,110,201, 'ghz'), 100*mil,z0=50)
wr3 	= RectangularWaveguide(Frequency(220,325,201, 'ghz'), 30*mil,z0=50)
wr2p2	= RectangularWaveguide(Frequency(330,500,201, 'ghz'), 22*mil,z0=50)
wr1p5 	= RectangularWaveguide(Frequency(500,750,201, 'ghz'), 15*mil,z0=50)
wr1 	= RectangularWaveguide(Frequency(750,1100,201, 'ghz'), 10*mil,z0=50)



## Functions
# Ploting
def save_all_figs(dir = './', format=['eps','pdf','png']):
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
	if ax is None:
		ax=plb.gca()
	lines = ax.get_lines()
	if len(lines) > len (marker_list ):
		marker_list *= 3
	[k[0].set_marker(k[1]) for k in zip(lines, marker_list)]
	[line.set_markevery(markevery) for line in lines]
	
def legend_off(ax=None):
	'''
	turn off the legend for a given axes. if no axes is given then 
	it will use current axes. 
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
	
	
# other
def now_string():
	return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')

def find_nearest(array,value):
	'''
	find nearest value in array.
	taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
	
	'''
	idx=(npy.abs(array-value)).argmin()
	return array[idx]

def find_nearest_index(array,value):
	'''
	find nearest value in array.
	taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
	
	'''
	return (npy.abs(array-value)).argmin()
