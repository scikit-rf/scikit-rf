
#       frequency.py
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
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
Provides the Frequency class, and related functions
'''

from pylab import linspace, gca
from numpy import pi

class Frequency(object):
	'''
	represents a frequency band. 
	
	attributes:
		start: starting frequency  (in Hz)
		stop: stoping frequency  (in Hz)
		npoints: number of points, an int
		unit: unit which to scale a formated axis, when accesssed. see
			formattedAxis
		
	frequently many calcluations are made in a given band , so this class 
	is used in other classes so user doesnt have to continually supply 
	frequency info.
	'''
	unit_dict = {\
		'hz':'Hz',\
		'mhz':'MHz',\
		'ghz':'GHz'\
		}
	multiplier_dict={
		'hz':1,\
		'mhz':1e6,\
		'ghz':1e9\
		}
	def __init__(self,start, stop, npoints, unit='hz', sweep_type='lin'):
		'''
		takes:
			start: start of band.  units of unit, defaults is  Hz
			stop: end of band. units of unit, defaults is  Hz
			npoints: number of points in the band. 
			unit: unit you want the band in for plots. a string. can be:
				'hz', 'mhz','ghz', 
		
		example:
			wr1p5band = frequencyBand(500,750,401, 'ghz')
			
		note: unit sets the property freqMultiplier, which is used 
		to scale the frequency when f_scaled is referenced.
			
		'''
		self._unit = unit.lower()
		self.start =  self.multiplier * start
		self.stop = self.multiplier * stop
		self.npoints = npoints
		self.sweep_type = sweep_type
		
	@classmethod
	def from_f(cls,f, *args,**kwargs):
		'''
		alternative constructor from a frequency vector,
		takes:
			f: frequency array (default in Hz) 
		returns:
			mwavepy.Frequency object
		'''
		return cls(start=f[0], stop=f[-1],npoints = len(f), *args, **kwargs)
	
	
	def __eq__(self, other):
		return (list(self.f) == list(other.f))	
	def __ne__(self,other):
		return (not self.__eq__(other))
		
	@property
	def center(self):
		return self.start + (self.stop-self.start)/2.
	@property
	def	f(self):
		'''
		returns a frequency vector  in Hz 
		'''
		return linspace(self.start,self.stop,self.npoints)
		#return self._f
	@f.setter
	def f(self,new_f):
		'''
		sets the frequency object by passing a vector in Hz
		'''
		#self._f = new_f
		self.start = new_f[0]
		self.stop = new_f[-1]
		self.npoints = len(new_f)
		
	@property
	def	f_scaled(self):
		'''
		returns a frequency vector in units of self.unit 
		'''
		return self.f/self.multiplier
	@property
	def w(self):
		'''
		angular frequency in radians
		'''
		return 2*pi*self.f
	@property
	def unit(self):
		'''
		The unit to format the frequency axis in. see formatedAxis
		'''
		return self.unit_dict[self._unit]
	@unit.setter
	def unit(self,unit):
		self._unit = unit.lower()
	
	@property
	def multiplier(self):
		'''
		multiplier for formating axis
		'''
		return self.multiplier_dict[self._unit]
	
	
	
	def labelXAxis(self, ax=None):
		if ax is None:
			ax = gca()
		ax.set_xlabel('Frequency [%s]' % self.unit )
	
def f_2_frequency(f):
	'''
	convienience function
	converts a frequency vector to a Frequency object 
	
	!depricated, use classmethod from_f instead. 
	'''
	return Frequency(start=f[0], stop=f[-1],npoints = len(f), unit='hz')
