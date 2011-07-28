
#       genericTEM.py
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
general class for TEM transmission lines
'''


from scipy.constants import  epsilon_0, mu_0, c,pi, mil
import numpy as npy
from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape

from functions import * 
	
# used as substitutes to handle mathematical singularities.
INF = 1e99
ONE = 1.0 + 1/1e14
# TEM transmission lines
class GenericTEM(object):
	'''
	This is a general super-class for TEM transmission lines. The 
	structure behind the methods dependencies is a results of the 
	physics. a brief summary is given below. 
	
	
	
	a TEM transmission line is defined by its:
	
		distributed Capacitance, C'
		distributed Inductance, I'
		distributed Resistance, R'
		distributed Conductance, G'
		
	from these the following quantities may be calculated, which
	are functions of angular frequency (w):
	
		distributed Impedance,  Z'(w) = R' + jwI'
		distributed Admittance, Y'(w) = G' + jwC'
	
	from these we can get to properties which define their wave behavoir
		
		characteristic Impedance, Z0(w) = sqrt(Z'(w)/Y'(w))		[ohms]
		propagation Constant,	gamma(w) = sqrt(Z'(w)*Y'(w))	
		
	
	and then finnally produce methods which we use 
		
		electrical Length
		input Impedance
		relfection Coefficient
		
	
	'''
	## CONSTRUCTOR
	def __init__(self, \
		distributed_capacitance,	distributed_inductance,\
		distributed_resistance, distributed_conductance):
		'''
		constructor.
		
		takes:
			distributed_capacitance: C'
			distributed_inductance: I'
			distributed_resistance: R'
			distributed_conductance: G'
			
				
		'''
		
		self.distributed_capacitance = distributed_capacitance
		self.distributed_inductance = distributed_inductance
		self.distributed_resistance = distributed_resistance
		self.distributed_conductance = distributed_conductance
		
		# for convinience 
		self.z0 = self.characteristic_impedance
		self.gamma = self.propagation_constant
	
	## METHODS
	def distributed_impedance(self,f):
		'''
		distributed Impedance,  Z'(w) = R' + jwI'
		
		takes:
			f: frequency [Hz]
		'''
		omega  = 2*npy.pi * array(f)
		return self.distributed_resistance+1j*omega*self.distributed_inductance
	
	def distributed_admittance(self,f):
		'''
		distributed Admittance, Y'(w) = G' + jwC'
		
		takes:
			f: frequency [Hz]
		'''
		omega = 2*npy.pi*array(f)
		return self.distributed_conductance+1j*omega*self.distributed_capacitance
	
	def characteristic_impedance(self,f):
		'''
		
		The  characteristic impedance at a given angular frequency.
			Z0(w) = sqrt(Z'(w)/Y'(w))
		takes:
			f:  frequency
		returns:
			Z0: characteristic impedance  ohms
		
		'''
		f = array(f)
		return sqrt(self.distributed_impedance(f)/\
			self.distributed_admittance(f))
	
	def propagation_constant(self,f):
		'''
		the propagation constant 
			gamma(w) = sqrt(Z'(w)*Y'(w))
		
		
		takes:
			f: frequency [Hz]
			
		returns:
			gamma: possibly complex propagation constant, [jrad/m+]
		'''
		f = array(f)
		return 1j*sqrt(self.distributed_impedance(f)*\
			self.distributed_admittance(f))



	
	def electrical_length(self, f,d,deg=False):
		return electrical_length( \
			gamma = self.propagation_constant,f=f,d=d,deg=deg)


