
#       distributedCircuit.py
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
A transmission line defined in terms of distributed circuit components
'''

from copy import deepcopy
from scipy.constants import  epsilon_0, mu_0, c,pi, mil
import numpy as npy
from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape

from ..tlineFunctions import electrical_length
from .media import Media	
# used as substitutes to handle mathematical singularities.
INF = 1e99
ONE = 1.0 + 1/1e14


class DistributedCircuit(Media):
	'''
	A TEM transmission line, defined in terms of  distributed impedance
	 and admittance values. This class takes the following information,
	
		distributed Capacitance, C
		distributed Inductance, I
		distributed Resistance, R
		distributed Conductance, G
		
	from these the following quantities may be calculated, which
	are functions of angular frequency (w):
	
		distributed Impedance,  Z'(w) = wR + jwI
		distributed Admittance, Y'(w) = wG + jwC
	
	from these we can calculate properties which define their wave 
	behavior:
		
		characteristic Impedance, Z0(w) = sqrt(Z(w)/Y'(w))		[ohms]
		propagation Constant,	gamma(w) = sqrt(Z(w)*Y'(w))	[none]
		
	given the following definitions, the components of propagation 
	constant are interpreted as follows:
		
		positive real(gamma) = attenuation
		positive imag(gamma) = forward propagation 

	'''
	## CONSTRUCTOR
	def __init__(self, frequency,  C, I, R, G,*args, **kwargs):
		'''
		generic, distributed circuit model transmission line constructor.
		
		takes:
			frequency: [mwavepy.Frequency object]
			C: distributed_capacitance [real float, or vector]
			I: distributed_inductance [real float, or vector]
			R: distributed_resistance [real float, or vector]
			G: distributed_conductance [real float, or vector]
		
		returns:
			mwavepy.Media object
		
		notes:
			C,I,R,G can all be vectors as long as they are the same length
			
			can be constructed from a Media instance too, see the 
			classmethod from_Media()
		
			see class help for details on the class structure.
	
		'''
		
		self.frequency = deepcopy(frequency)
		self.C, self.I, self.R, self.G = C,I,R,G

		# for unambiguousness  
		self.distributed_resistance = self.R
		self.distributed_capacitance = self.C
		self.distributed_inductance = self.I
		self.distributed_conductance = self.G
		
		Media.__init__(self,\
			frequency = frequency,\
			propagation_constant = self.gamma, \
			characteristic_impedance = self.Z0,\
			*args, **kwargs)
	
	def __str__(self):
		f=self.frequency
		output =  \
			'Distributed Circuit Media.  %i-%i %s.  %i points'%\
			(f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
			'\nI\'= %.2f, C\'= %.2f,R\'= %.2f, G\'= %.2f, '% \
			(self.I, self.C,self.R, self.G)
		return output
		
	def __repr__(self):
		return self.__str__()
	
	
	@classmethod
	def from_Media(cls, my_media, *args, **kwargs):
		'''
		initializer which creates  DistributedCircuit from a Media 
		instance
		'''
		
		w  =  my_media.frequency.w
		gamma = my_media.propagation_constant
		Z0 = my_media.characteristic_impedance
		
		Y = gamma/Z0
		Z = gamma*Z0
		G,C = real(Y)/w, imag(Y)/w
		R,I = real(Z)/w, imag(Z)/w
		return cls(my_media.frequency, C=C, I=I, R=R, G=G, *args, **kwargs)
	
	
	@property	
	def Z(self):
		'''
		distributed Impedance, ohms/m.
		
		 Z'(w) = wR + jwI
		
		'''
		w  = 2*npy.pi * self.frequency.f
		return w*self.R + 1j*w*self.I
	
	@property
	def Y(self):
		'''
		distributed Admittance,in ohms^-1 /m
		
		 Y'(w) = wG + jwC
		
		'''
		
		w = 2*npy.pi*self.frequency.f
		return w*self.G + 1j*w*self.C
	
	
	def Z0(self):
		'''
		The characteristic impedance in ohms
		
		'''
		
		return sqrt(self.Z/self.Y)
	
	
	def gamma(self):
		'''
		possibly complex propagation constant, [rad/m]
			gamma = sqrt(Z'*Y')
		
		note:
		the components of propagation constant are interpreted as follows:
		
		positive real(gamma) = attenuation
		positive imag(gamma) = forward propagation 
		'''
		return sqrt(self.Z*self.Y)
	
	



