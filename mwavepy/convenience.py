
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
Holds pre-initialized  class's and functions which provide convenience.
	Pre-initialized classes include:
		Standard waveguide bands, which are prefixed by, 
			- Freqency Objects (f_*)
			- RectangularWaveguideTE10 Objects	(wr_*)
			- WorkingBand Objects	(wb_*)
'''


from network import *
from transmissionLine import RectangularWaveguideTE10
from frequency import Frequency
from workingBand import WorkingBand


import os
import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime

## sets of Frequency,RectangularWaveguideTE10, and WorkingBand objects,
## which correspond to designated waveguide bands

# WR-1.5 
f_wr1p5 = Frequency(500,750,201, 'ghz')
wg_wr1p5 = RectangularWaveguideTE10(1.5*10*mil)
wb_wr1p5 = WorkingBand(frequency = f_wr1p5, tline = wg_wr1p5)

# WR-3
f_wr3 = Frequency(500,750,201, 'ghz')
wg_wr3 = RectangularWaveguideTE10(3*10*mil)
wb_wr1p5 = WorkingBand(frequency = f_wr3, tline = wg_wr3)




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
	
	
# Touchstone manipulation	
def load_all_touchstones(dir = '.', contains=None, f_unit=None):
	'''
	loads all touchtone files in a given dir 
	
	takes:
		dir  - the path to the dir, passed as a string (defalut is cwd)
		contains - string which filename must contain to be loaded, not 
			used if None.(default None)
	returns:
		ntwkDict - a Dictonary with keys equal to the file name (without
			a suffix), and values equal to the corresponding ntwk types
	
		
	'''
	ntwkDict = {}

	for f in os.listdir (dir):
		if contains is not None and contains not in f:
			continue
			
		# TODO: make this s?p with reg ex
		if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
			name = f[:-4]
			ntwkDict[name]=(Network(dir +'/'+f))
			if f_unit is not None: ntwkDict[name].frequency.unit=f_unit
		
	return ntwkDict	

def write_dict_of_networks(ntwkDict, dir='.'):
	'''
	writes a dictionary of networks to a given directory
	'''
	for ntwkKey in ntwkDict:
		ntwkDict[ntwkKey].write_touchstone(filename = dir+'/'+ntwkKey)

def now_string():
	return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')


def csv_2_touchstone(filename):
	'''
	converts a csv file saved from a Rhode swarz and possibly other 
	
	takes:
		filename: name of file
	returns:
		Network object
	'''
		
	ntwk = Network(name=filename[:-4])
	try: 
		data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
			usecols=range(9))
		s11 = data[:,1] +1j*data[:,2]	
		s21 = data[:,3] +1j*data[:,4]	
		s12 = data[:,5] +1j*data[:,6]	
		s22 = data[:,7] +1j*data[:,8]	
		ntwk.s = npy.array([[s11, s21],[s12,s22]]).transpose().reshape(-1,2,2)
	except(IndexError):
		data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
			usecols=range(3))		
		ntwk.s = data[:,1] +1j*data[:,2]
	
	ntwk.frequency.f = data[:,0]
	ntwk.frequency.unit='ghz'
	
	return ntwk
