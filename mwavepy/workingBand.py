'''
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

from copy import copy
import numpy as npy

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
	def short(self,**kwargs):
		'''
		creates a Network for a short  transmission line (Gamma0=-1) 
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, a short
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = -1.0*npy.ones(self.frequency.npoints, dtype=complex)
		return result


	def match(self,**kwargs):
		'''
		creates a Network for a perfect matched transmission line (Gamma0=0) 
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, representing a perfect match
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s =  npy.zeros(self.frequency.npoints, dtype=complex)
		return result

	def open(self,**kwargs):
		'''
		creates a Network for a 'open' transmission line (Gamma0=1) 
		
		takes:
			**kwargs: key word arguments passed to Network Constructor
		returns:
			a 1-port Network class, an open
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = 1.0*npy.ones(self.frequency.npoints, dtype=complex)
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
		
		# calculate a propagation constant
		gamma = self.tline.propagation_constant(f=f)
		
		# calculate the electrical length
		theta = electrical_length(gamma=gamma, f= f, d = d)
		
		s11 = npy.zeros(self.frequency.npoints, dtype=complex)
		s21 = npy.exp(1j* theta)
		result.s = npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
		return result
	
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
		return line(d,**kwargs) ** load(Gamma0,**kwargs)

	def delay_load(self,d,Gamma0,**kwargs):
		'''
		creates a delayed load of length 'd'  and refelction coefficient
		Gamma0, Network object

		takes:
			d: length of delay [m]
			Gamma0: reflection coefficient [not dB]
		returns:
			Network object
		'''
		ntwk = createNetwork.delay_load(d=d,tline=self.tline, \
			f = self.f,Gamma0=Gamma0,**kwargs)
		ntwk.frequency = self.frequency
		return ntwk





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
		beta = npy.real(self.tline.propagation_constant())
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((-2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]
	
