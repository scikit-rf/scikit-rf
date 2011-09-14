
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
Holds pre-initialized  class's to provide convenience. Also provides
some functions, which cant be categorized as anything better than 
general conviniencies.
	
	Pre-initialized classes include:
		Frequency Objects for standard frequency bands
		Media objects for standard waveguide bands, 
		
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
	if dir[-1] != '/':
		dir = dir + '/'
	for fignum in plb.get_fignums():
		plb.figure(fignum)
		fileName = plb.gca().get_title()
		if fileName == '':
				fileName = 'unamedPlot'
		for fmt in format:
			plb.savefig(dir+fileName+'.'+fmt, format=fmt)
			print (dir+fileName+'.'+fmt)

def legend_off(ax=None):
	'''
	turn off the legend for a given axes. if no axes is given then 
	it will use current axes. 
	'''
	if ax is None:
		plb.gca().legend_.set_visible(0)
	else:
		ax.lengend_.set_visible(0)

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
