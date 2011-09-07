
#       freeSpace.py
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
contains coplanar waveguide class (CPW)
'''
import numpy as npy
from numpy import pi, sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape
from scipy.constants import  epsilon_0, mu_0, c,pi, mil,pi
from genericTEM import GenericTEM

class CPW(GenericTEM):
	'''
	Coplanar waveguide
	
	
	'''
	def __init__(self, w,s, ep_r=1, mu_r=1):
		self.w = w
		self.s = s
		self.ep_r  = ep_r
		self.mu_r  = mu_r
		
		self.characteristic_impedance = self.Z0
		self.propagation_constant = self.gamma
	@property	
	def ep_re(self):
		'''
		effective permativity of cpw line
		
		ep_re = (ep_r + 1)/2
		'''
		return (self.ep_r +1)/2.
	
	def Z0(self,f=None):
		'''
		The  characteristic impedance at a given  frequency.
			
		returns:
			Z0: characteristic impedance  in ohms
		
		'''
		
				
		k = 1.0*self.w/(self.w+2*self.s)
		#ratio of elliptical integrals of the 1st kind
		if (0<= k <=1/sqrt(2)):
			K_ratio =  (pi/(log(2*(1.+sqrt(k))/(1.-sqrt(k)))))
		elif (1/sqrt(2) <= k <=1): 
			((log(2*(1.+sqrt(k))/(1.-sqrt(k))))/pi)
		else:
			raise(ValueError('Pitch is out of range'))	
		
		return 30 * pi/sqrt(self.ep_re) * K_ratio
		
		
	def gamma(self,f):
		'''
		the propagation constant 
					
		takes:
			f: frequency [Hz]
			
		returns:
			gamma: possibly complex propagation constant, [rad/m]
			
			
		note:
		the components of propagation constant are interpreted as follows:
		
		positive real(gamma) = attenuation
		positive imag(gamma) = forward propagation 
		'''
		
		f = array(f)
		return 1j* 2*pi*f * sqrt(self.ep_re*epsilon_0 * self.mu_r*mu_0)

	def electrical_length(self, f,d,deg=False):
		'''
		convenience function for this class. the real function for this 
		is defined in transmissionLine.functions, under the same name.
		'''
		return electrical_length(gamma = self.gamma,f=f,d=d,deg=deg)
