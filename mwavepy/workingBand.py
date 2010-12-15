
#       workingBand.py
#       
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
Contains WorkingBand class. 
'''

from copy import copy
import numpy as npy
from scipy import stats

from frequency import Frequency
from network import Network
from transmissionLine.functions import electrical_length


class WorkingBand(object):
	## TODO: put docstrings in standard format of takes/returns
	'''
	A WorkingBand is an high-level object which exists solely to make 
	 working with and creation of Networks within the same band,
	 more concise and convenient. 
	
	A WorkingBand object has two properties: 
		frequency information (Frequency object)
		transmission line information	(transmission line-like object)
		

	the methods of WorkingBand simply calls functions from createNetwork,
	but it saves the user the hassle of repetitously providing a
	tline and frequency type for every network creation. 	

	note: frequency and tline classes are copied, so they are passed
	by value and not by-reference.
	'''
	def __init__(self, frequency, tline):
		self.frequency = frequency 
		self.tline = tline


	## PROPERTIES	
	@property
	def frequency(self):
		return self._frequency
	@frequency.setter
	def frequency(self,new_frequency):
		self._frequency= copy( new_frequency)
	@property
	def tline(self):
		return self._tline
	@tline.setter
	def tline(self,new_tline):
		self._tline = copy(new_tline)
		




	## Network creation
	def short(self,nports=1,**kwargs):
		'''
		creates a Network for a short  transmission line (Gamma0=-1) 
		
		takes:
			nports: number of ports. creates a short on all ports,
				default is 1
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, a short
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = npy.zeros((self.frequency.npoints,nports, nports),dtype=complex)
		for f in range(self.frequency.npoints):
			result.s[f,:,:] = -1.0*npy.eye(nports, dtype=complex)
		return result


	def match(self,nports=1, **kwargs):
		'''
		creates a Network for a perfect matched transmission line (Gamma0=0) 
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, representing a perfect match
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s =  npy.zeros((self.frequency.npoints,nports, nports),\
			dtype=complex)
		return result

	def open(self,nports=1, **kwargs):
		'''
		creates a Network for a 'open' transmission line (Gamma0=1) 
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, an open
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = npy.zeros((self.frequency.npoints,nports, nports),dtype=complex)
		for f in range(self.frequency.npoints):
			result.s[f,:,:] = 1.0*npy.eye(nports, dtype=complex)
		return result

	def load(self,Gamma0,**kwargs):
		'''
		creates a Network for a Load termianting a transmission line 
		
		takes:
			Gamma0: reflection coefficient of load (not in db)
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, where  S = Gamma0*ones(...)
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = Gamma0*npy.ones(self.frequency.npoints, dtype=complex)
		return result
	
	def line(self,d,**kwargs):
		'''
		creates a Network for a section of transmission line
		
		takes:
			d: the length (in meters)
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 2-port Network class, representing a transmission line of length d
	
		note: the only function called from the tline class is
		propagation_constant(f,d), where f is frequency in Hz and d is
		distance in meters. so you can use any class  which provides this
		and it  will work .
		'''

		result = Network(**kwargs)
		result.frequency = self.frequency
		
		f= self.frequency.f
		
		# propagation constant function
		gamma = self.tline.propagation_constant
		
		# calculate the electrical length
		theta = electrical_length(gamma=gamma, f= f, d = d)
		
		s11 = npy.zeros(self.frequency.npoints, dtype=complex)
		s21 = npy.exp(1j* theta)
		result.s = npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
		return result
	def thru(self, **kwargs):
		'''
		creates a Network for a thru
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 2-port Network class, representing a thru

		note:
			this just calls self.line(0)
		'''
		return self.line(0,**kwargs)
		
	def delay_short(self,d,**kwargs):
		'''
		creates a Network for a delayed short transmission line
		
		takes:
			d: the length (in meters)
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, representing a shorted transmission
			line of length d
		'''
		return self.line(d,**kwargs) ** self.short(**kwargs)

	def delay_load(self,d,Gamma0,**kwargs):
		'''
		creates a Network for a delayed short transmission line
		
		takes:
			d: the length (in meters)
			Gamma0: reflection coefficient of load (not in dB)
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, representing a loaded transmission
			line of length d
		'''
		return self.line(d,**kwargs) ** self.load(Gamma0,**kwargs)
	





	
		
	## Noise Networks
	def white_gaussian_polar(self,phase_dev, mag_dev,n_ports=1,**kwargs):
		'''
		creates a complex zero-mean gaussian white-noise signal of given
		standard deviations for phase and magnitude

		takes:
			phase_mag: standard deviation of magnitude
			phase_dev: standard deviation of phase
			n_ports: number of ports. defualt to 1
			**kwargs: passed to Network() initializer
		returns:
			result: Network type 
		'''
		shape = (self.frequency.npoints, n_ports,n_ports)
		phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = shape)
		mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = shape)

		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = mag_rv*npy.exp(1j*phase_rv)
		return result

	## OTHER METHODS
	def guess_length_of_delay_short(self, aNtwk):
		'''
		guess length of physical length of a Delay Short given by aNtwk
		
		takes:
			aNtwk: a mwavepy.ntwk type . (note: if this is a measurment 
				it needs to be normalized to the reference plane)
			tline: transmission line class of the medium. needed for the 
				calculation of propagation constant
				
		
		'''
		beta = npy.real(self.tline.propagation_constant(self.frequency.f))
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]
	def two_port_reflect(self, ntwk1, ntwk2, **kwargs):
		'''
		generates a two-port reflective (S21=S12=0) network,from the
		responses of 2 one-port networks

		takes:
			ntwk1: Network type, seen from port 1
			ntwk2: Network type, seen from port 2
		returns:
			result: two-port reflective Network type

		
		example:
			wb.two_port_reflect(wb.short(), wb.match())
		'''
		result = self.match(nports=2,**kwargs)
		for f in range(self.frequency.npoints):
			result.s[f,0,0] = ntwk1.s[f,0,0]
			result.s[f,1,1] = ntwk2.s[f,0,0]
		return result
