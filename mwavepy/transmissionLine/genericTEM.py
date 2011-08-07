
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
	structure behind the methods dependencies is a results of  
	physics. a brief summary is given below. 
	
	
	A TEM transmission line can be described by a characterisitc 
	impedance and propagation constant, or by distributed impedance and 
	admittance. This description will be in terms of distributed 
	circuit quantities, given:
	
		distributed Capacitance, C'
		distributed Inductance, I'
		distributed Resistance, R'
		distributed Conductance, G'
		
	from these the following quantities may be calculated, which
	are functions of angular frequency (w):
	
		distributed Impedance,  Z'(w) = wR' + jwI'
		distributed Admittance, Y'(w) = wG' + jwC'
	
	from these we can calculate properties which define their wave 
	behavior:
		
		characteristic Impedance, Z0(w) = sqrt(Z'(w)/Y'(w))		[ohms]
		propagation Constant,	gamma(w) = sqrt(Z'(w)*Y'(w))	[none]
		
	given the following definitions, the components of propagation 
	constant are interpreted as follows:
		
		positive real(gamma) = attenuation
		positive imag(gamma) = forward propagation 
	
	this sign convention means that the transmission gain through a
	distance, d, is given by, 
		
		S21 = exp(-gamma*d)
		
	and then finally these all produce methods which we use 
		
		electrical Length (theta) 
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
			distributed_capacitance: C' [real float]
			distributed_inductance: I' [real float]
			distributed_resistance: R' [real float]
			distributed_conductance: G' [real float]
			
		note:
			see class help for details on the class structure.
			
			if you want to initialize this class in terms of propagation
		constant and characteristic impedance, instead of distributed 
		circuit quantities, then use the conversion function provided
		in 	transmissionLine.functions. its confusingly called, 
			propagation_impedance_2_distributed_circuit()
			
			
				
		'''
		
		self.distributed_capacitance = distributed_capacitance
		self.distributed_inductance = distributed_inductance
		self.distributed_resistance = distributed_resistance
		self.distributed_conductance = distributed_conductance
		
		# for convenience 
		self.z0 = self.characteristic_impedance
		self.gamma = self.propagation_constant
	
	## METHODS
	def distributed_impedance(self,f):
		'''
		distributed Impedance,  Z'(w) = wR' + jwI'
		
		takes:
			f: frequency [Hz]
		'''
		omega  = 2*npy.pi * array(f)
		return omega*self.distributed_resistance + \
			1j*omega*self.distributed_inductance
	
	def distributed_admittance(self,f):
		'''
		distributed Admittance, Y'(w) = wG' + jwC'
		
		takes:
			f: frequency [Hz]
		'''
		omega = 2*npy.pi*array(f)
		return omega*self.distributed_conductance + \
			1j*omega*self.distributed_capacitance
	
	def characteristic_impedance(self,f):
		'''
		
		The  characteristic impedance at a given angular frequency.
			Z0(w) = sqrt(Z'(w)/Y'(w))
		takes:
			f:  frequency
		returns:
			Z0: characteristic impedance  in ohms
		
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
			gamma: possibly complex propagation constant, [rad/m]
		'''
		f = array(f)
		return sqrt(self.distributed_impedance(f)*\
			self.distributed_admittance(f))

	def electrical_length(self, f,d,deg=False):
		'''
		convenience function for this class. the real function for this 
		is defined in transmissionLine.functions, under the same name.
		'''
		return electrical_length( \
			gamma = self.propagation_constant,f=f,d=d,deg=deg)


