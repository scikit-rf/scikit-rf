'''
#       transmissionLine.py
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
try:
	from scipy.constants import  epsilon_0, mu_0, c,pi, mil
	from scipy import signal
	
except:
	raise ImportError ('Depedent Packages not found. Please install: scipy')
try:
	import numpy as npy
	from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape
except:
	raise ImportError ('Depedent Packages not found. Please install: numpy')	

import mwavepy as mv
	


class transmissionLine:
	'''
	general super-class for TEM transmission lines
	'''
	fBand = None
	def __init__(self, \
		distributedCapacitance,	distributedInductance,\
		distributedResistance, distributedConductance, fBand=None ):
		
		self.distributedCapacitance = distributedCapacitance
		self.distributedInductance = distributedInductance
		self.distributedResistance = distributedResistance
		self.distributedConductance = distributedConductance

		self.fBand = fBand
		
	def distributedImpedance(self,omega):
		omega = array(omega)
		return self.distributedResistance+1j*omega*self.distributedInductance
	
	def distributedAdmittance(self,omega):
		omega= array(omega)
		return self.distributedConductance+1j*omega*self.distributedCapacitance
	# could put a test for losslessness here and choose whether to make this
	# a funtion of omega or not.
	def characteristicImpedance(self,omega):
		return sqrt(self.distributedImpedance(omega)/self.distributedAdmittance(omega))
	
	def propagationConstant(self,omega):
		
		return sqrt(self.distributedImpedance(omega)*self.distributedAdmittance(omega))
	
	@classmethod
	def electricalLength(self, l , f=None, gamma=None,deg=False):
		'''
		calculates the electrical length of a section of transmission line.
	
		takes:
			l - length of line in meters
			f: frequency at which to calculate, array-like or float
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.  
			
		returns:
			electricalLength: electrical length in radians or degrees, 
				if deg =True
		note:
			you can pass a function on the fly, like  
			electricalLength(freqVector, l, beta = lambda omega: omega/c )
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if f is None:
			if  self.fBand is None:
				raise ValueError('please supply frequency information')
			else:
				f = self.fBand.axis
				
		if deg==False:
			return  gamma(2*pi*f ) *l 
		elif deg ==True:
			return  rad2deg(gamma(2*pi*f ) *l )
	
	
	@classmethod
	def reflectionCoefficient(self, l,f,zl,z0=None, gamma=None):
		'''
		calculates the reflection coefficient for a given load 
		takes:
			l: distance of transmission line to load, in meters (float)
			f: frequency at which to calculate, array-like or float
			zl: load impedance. may be a function of omega (2*pi*f), or 
				a number 
			z0 - characteristic impedance may be a function of omega 
				(2*pi*f), or a number 
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if z0 is None:
			z0 = self.characteristicImpedance
		
		try:
			zl = zl(2*pi*f)
		except TypeError:
			pass
		try:
			z0 = z0(2*pi*f)
		except TypeError:
			pass
		
		try : 
			if len(z0) != len(zl): 
				raise IndexError('len(zl) != len(z0)')
		except (TypeError):
			# zl and z0 might just be numbers, which dont have len
			pass
		# flexible way to typecast ints, or arrays
		zl = 1.0*(zl)
		z0 = 1.0*(z0)
		l = 1.0* (l)
		
		theta = self.electricalLength(l,f, gamma=gamma)
		
		if isinstance(zl,npy.ndarray):
			# handle the limit of open circuit. for arrays
			zl[(zl==npy.inf)]=1e100
			gammaAt0 = (zl-z0)/(zl+z0)
		else:
			if zl == inf:
				gammaAt0 = 1
			else: 
				gammaAt0 = (zl-z0)/(zl+z0)
		
		gammaAtL =gammaAt0 * npy.exp(-2j*theta)
		return gammaAtL
	
	@classmethod
	def inputImpedance(self, l,f, zl,z0=None,gamma=None):
		'''
		returns the input impedance of a transmission line of character impedance z0 and electrical length el, terminated with a load impedance zl. 
		takes:
			l: distance from load, in meters
			f: frequency at which to calculate, array-like or float 
			zl: load impedance. may be a function of omega (2*pi*f), or 
				a number 
			z0 - characteristic impedance may be a function of omega 
				(2*pi*f), or a number
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.
		returns:
			input impedance ( in general complex)
			
		note:
			this can also be calculated in terms of reflectionCoefficient
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if z0 is None:
			z0 = self.characteristicImpedance
		
		try:
			zl = zl(2*pi*f)
		except TypeError:
			pass
		try:
			z0 = z0(2*pi*f)
		except TypeError:
			pass
			
		try : 
			if len(z0) != len(zl): 
				raise IndexError('len(zl) != len(z0)')
		except (TypeError):
			# zl and z0 might just be numbers, which dont have len
			pass
		
		
		
		theta = propagationConstant(l,2*pi*f, gamma=gamma)
		
		if zl == inf:
			return -1j*z0*1./(tan(theta))
		elif zl == 0:
			return 1j*z0*tan(theta)
		else:
			return z0 *	(zl + 1j*z0 * tan(theta)) /\
						(z0 + 1j*zl * tan(theta))
	

	
	
	
	def createNtwk_delayShort(self,l,f=None, gamma=None, **kwargs ):
		'''
		generate the reflection coefficient for a  delayed short of length l 
		
		takes:
			l - length of delay, in meters
			f: frequency axis. if self.fBand exists then this 
				can left as None
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m. can be omited.
			kwargs: passed to ntwk constructor
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if f is None:
			if  self.fBand is None:
				raise ValueError('please supply frequency information')
			else:
				f = self.fBand.axis
		
			
		
		s = -1*exp(-1j* 2*self.electricalLength(l,f,gamma))
		print shape (s)
		return mv.ntwk(data=s,paramType='s',freq=f,**kwargs)


class freespace(transmissionLine):
	'''
	represents a plane-wave in freespace, defined by [possibly complex]
	values of relative permativity and relative permeability
	'''
	def __init__(self, relativePermativity=1, relativePermeability=1,fBand=None):
		transmissionLine.__init__(self,\
			distributedCapacitance = real(epsilon_0*relativePermativity),\
			distributedResistance = imag(epsilon_0*relativePermativity),\
			distributedInductance = real(mu_0*relativePermeability),\
			distributedConductance = imag(mu_0*relativePermeability),\
			fBand = fBand
			)
		
class coax(transmissionLine):
	def __init__(self, innerRadius, outerRadius, surfaceResistance=0, relativePermativity=1, relativePermeability=1,fBand=None):
		# changing variables just for readablility
		a = innerRadius
		b = outerRadius
		eR = relativePermativity
		uR = relativePermeability
		Rs = surfaceResistance
		
		transmissionLine.__init__(self,\
			distributedCapacitance = 2*pi*real(epsilon_0*eR)/log(b/a),\
			distributedResistance = Rs/(2*pi) * (1/a + 1/b),\
			distributedInductance = muR*mu_0/(2*pi) * log(b/a),\
			distributedConductance = 2*pi*omega*imag(epsilon_0*eR)/log(b/a),\
			fBand = fBand
			)


		
