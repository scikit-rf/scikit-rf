
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

from ..transmissionLine.functions import electrical_length
from .media import Media	
# used as substitutes to handle mathematical singularities.
INF = 1e99
ONE = 1.0 + 1/1e14

# TEM transmission lines
class TEM(Media):
	'''
	== Intro ==
	This is a general super-class for TEM transmission lines. The 
	structure behind the methods dependencies is a result of  
	physics. a brief summary is given below. 
	
	== Class Structure ==
	This class can support two models for TEM transmission lines:
		1)simple media: in which the distributed circuit quantities are 
			NOT functions of frequency,
		2)not-simple media:  in which the distributed circuit quantities
			ART functions of frequency
	
	1) The simple media can be constructed with scalar values for the 
	distributed circuit quantities. then all methods for transmission
	line properties( Z0, gamma) will take frequency as an argument. 
	
	2) The not-simple media must be constructed with array's for 
	distributed circuit quanties and a frequency array. alternativly, 
	you can construct this type of tline from propagation constant,  
	characterisitc impedance, and frequency information, through use of
	the class method; from_gamma_Z0().
	
	== Physics ==
	A TEM transmission line can be described by a characterisitc 
	impedance and propagation constant, or by distributed impedance and 
	admittance. This description will be in terms of distributed 
	circuit quantities, given:
	
		distributed Capacitance, C
		distributed Inductance, I
		distributed Resistance, R
		distributed Conductance, G
		
	from these the following quantities may be calculated, which
	are functions of angular frequency (w):
	
		distributed Impedance,  Z(w) = wR + jwI
		distributed Admittance, Y'(w) = wG + jwC
	
	from these we can calculate properties which define their wave 
	behavior:
		
		characteristic Impedance, Z0(w) = sqrt(Z(w)/Y'(w))		[ohms]
		propagation Constant,	gamma(w) = sqrt(Z(w)*Y'(w))	[none]
		
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
	def __init__(self, frequency,  C, I, R, G):
		'''
		TEM transmission line constructor.
		
		takes:
			frequency: [mwavepy.Frequency object]
			C: distributed_capacitance [real float]
			I: distributed_inductance [real float]
			R: distributed_resistance [real float]
			G: distributed_conductance [real float]
			
		
		notes:
			can be constructed from propagation constant, and 
		characteristic impedance as well, through the class method; 
		from_gamma_Z0, like so,
			my_tline = GenericTEM.from_gamma_Z0(....)
		note that this requires frequency information.
		
			see class help for details on the class structure.
	
		'''
		
		self.C, self.I, self.R, self.G = C,I,R,G

		
		# for unambiguousness  
		self.distributed_resistance = self.R
		self.distributed_capacitance = self.C
		self.distributed_inductance = self.I
		self.distributed_conductance = self.G
		Media.__init__(self, frequency)
		
		self.propagation_constant = self.gamma
		self.characteristic_impedance = self.Z0
	@classmethod
	def from_gamma_Z0(cls, frequency, gamma, Z0):
		'''
		initializer which creates a transmission line from propagation 
		constant, characteristic impedance, and frequency information
		'''
		w  = 2*npy.pi * array(frequency.f)
		Y = gamma/Z0
		Z = gamma*Z0
		G,C = real(Y)/w, imag(Y)/w
		R,I = real(Z)/w, imag(Z)/w
		return cls(frequency, C=C, I=I, R=R,G=G)
	
	
	@property	
	def Z(self):
		'''
		distributed Impedance, ohms/m.
		
		 Z'(w) = wR + jwI
		
		'''
		w  = 2*npy.pi * array(self.frequency.f)
		return w*self.R + 1j*w*self.I
	@property
	def Y(self):
		'''
		distributed Admittance,in ohms^-1 /m
		
		 Y'(w) = wG + jwC
		
		'''
		
		w = 2*npy.pi*array(self.frequency.f)
		return w*self.G + 1j*w*self.C
	@property
	def Z0(self):
		'''
		The  characteristic impedance in ohms
		
		'''
		
		return sqrt(self.Z/self.Y)
	@property
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
	
	



