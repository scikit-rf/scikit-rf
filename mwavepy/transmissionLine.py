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
	#from scipy import signal
	
except:
	raise ImportError ('Depedent Packages not found. Please install: scipy')
try:
	import numpy as npy
	from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape
except:
	raise ImportError ('Depedent Packages not found. Please install: numpy')	

import mwavepy as mv
	
# used as substitutes to handle mathematical singularities.
INF = 1e99
ONE = 1.0 + 1/1e14
# TEM transmission lines
class TransmissionLine(object):
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
		distributedCapacitance,	distributedInductance,\
		distributedResistance, distributedConductance):
		'''
		constructor.
		
		takes:
			distributedCapacitance, C'
			distributedInductance, I'
			distributedResistance, R'
			distributedConductance, G'
			fBand: a mwavepy.fb.frequencyBand object. this provides all 
				the	methods of the transmissionLine, with frequency info.
				this is solely for convinience, because frequently many
				calculations are done over a given range of frequencies.
				
		'''
		
		self.distributedCapacitance = distributedCapacitance
		self.distributedInductance = distributedInductance
		self.distributedResistance = distributedResistance
		self.distributedConductance = distributedConductance
		
		# for convinience 
		self.z0 = self.characteristicImpedance
		self.gamma = self.propagationConstant
	
	## METHODS
	def distributedImpedance(self,omega):
		'''
		distributed Impedance,  Z'(w) = R' + jwI'
		
		'''
		omega = array(omega)
		return self.distributedResistance+1j*omega*self.distributedInductance
	
	def distributedAdmittance(self,omega):
		'''
		distributed Admittance, Y'(w) = G' + jwC'
		'''
		omega = array(omega)
		return self.distributedConductance+1j*omega*self.distributedCapacitance
	
	def characteristicImpedance(self,omega):
		'''
		
		The  characteristic impedance at a given angular frequency.
			Z0(w) = sqrt(Z'(w)/Y'(w))
		takes:
			omega: radian angular frequency
		returns:
			Z0: characteristic impedance  ohms
		
		'''
		omega = array(omega)
		return sqrt(self.distributedImpedance(omega)/\
			self.distributedAdmittance(omega))
	
	def propagationConstant(self,omega):
		'''
		the propagation constant 
			gamma(w) = sqrt(Z'(w)*Y'(w))
		
		takes: 
			omega
			
		returns:
			gamma: possibly complex propagation constant, [jrad/m+]
		'''
		omega = array(omega)
		return 1j*sqrt(self.distributedImpedance(omega)*\
			self.distributedAdmittance(omega))



	
	def electricalLength(self, f,d,deg=False):
		return electricalLength( \
			gamma = self.propagationConstant,f=f,d=d,deg=deg)
	
## FUNCTIONS
def electricalLength(gamma, f , d, deg=False):
	'''
	calculates the electrical length of a section of transmission line.

	takes:
		gamma: propagation constant, (a function)
		l: length of line. in meters
		f: frequency at which to calculate. array-like or float. if
			left as None and self.fBand exists, it will use that.
		deg: return in degrees or not. boolean.
	
	returns:
		theta: electrical length in radians or degrees, 
			depending on  value of deg.
	'''
	
	# typecast to a 1D array
	f = array(f, dtype=float).reshape(-1)
	d = array(d, dtype=float).reshape(-1)
			
	if deg == False:
		return  gamma(2*pi*f )*d
	elif deg == True:
		return  mf.radian_2_degree(gamma(2*pi*f ) *l )


def inputImpedance2ReflectionCoefficient(z0, zl):
	'''
	calculates the reflection coefficient for a given input impedance 
	takes:
		
		zl: input (load) impedance.  
		z0 - characteristic impedance. 
	'''
	# typecast to a complex 1D array. this makes everything easier	
	z0 = array(z0, dtype=complex).reshape(-1)
	zl = array(zl, dtype=complex).reshape(-1)
	
	# handle singularity  by numerically representing inf as big number
	zl[(zl==npy.inf)] = INF

	return ((zl -z0 )/(zl+z0))

def reflectionCoefficient2InputImpedance(z0,Gamma):
	'''
	calculates the input impedance given a reflection coefficient and 
	characterisitc impedance of the medium
	takes:
		
		Gamma: reflection coefficient
		z0 - characteristic impedance. 
	'''
	# typecast to a complex 1D array. this makes everything easier	
	Gamma = array(Gamma, dtype=complex).reshape(-1)
	z0 = array(z0, dtype=complex).reshape(-1)
	
	#handle singularity by numerically representing inf as close to 1
	Gamma[(Gamma == 1)] = ONE
	
	return z0*((1.0+Gamma )/(1.0-Gamma))
	
def reflectionCoefficientAtTheta(Gamma0,theta):
	'''
	reflection coefficient at electrical length theta
	takes:
		Gamma0: reflection coefficient at theta=0
		theta: electrical length, (may be complex)
	returns:
		Gamma_in
		
	note: 
		 = Gamma0 * exp(-2j* theta)
	'''
	Gamma = array(Gamma, dtype=complex).reshape(-1)
	theta = array(theta, dtype=complex).reshape(-1)
	return Gamma0 * exp(-2j* theta)

def inputImpedanceAtTheta(z0,zl, theta):
	'''
	input impedance of load impedance zl at electrical length theta, 
	given characteristic impedance z0.
	
	takes:
		z0 - characteristic impedance. 
		zl: load impedance
		theta: electrical length of the line, (may be complex) 
	'''
	Gamma0 = inputImpedance2ReflectionCoefficient(z0=z0,zl=zl)
	Gamma_in = reflectionCoefficientAtTheta(Gamma0=Gamma0, theta=theta)
	return reflectionCoefficient2InputImpedance(z0=z0, Gamma=Gamma_in)
	
def inputImpedance2ReflectionCoefficientAtTheta(z0, zl, theta):
	Gamma0 = inputImpedance2ReflectionCoefficient(z0=z0,zl=zl)
	Gamma_in = reflectionCoefficientAtTheta(Gamma0=Gamma0, theta=theta)
	return Gamma_in

def reflectionCoefficient2InputImpedanceAtTheta(z0, Gamma0, theta):
	'''
	calculates the input impedance at electrical length theta, given a
	reflection coefficient and characterisitc impedance of the medium
	takes:
		z0 - characteristic impedance. 
		Gamma: reflection coefficient
		theta: electrical length of the line, (may be complex) 
	returns 
		zin: input impedance at theta
	'''
	Gamma_in = reflectionCoefficientAtTheta(Gamma0=Gamma0, theta=theta)
	zin = reflectionCoefficient2InputImpedance(z0=z0,Gamma=Gamma)
	return zin
# short hand convinience. 
# admitantly these follow no logical naming scheme, but they closely 
# correspond to common symbolic conventions
theta = electricalLength


zl_2_Gamma0 = inputImpedance2ReflectionCoefficient
Gamma0_2_zl = reflectionCoefficient2InputImpedance

zl_2_zin = inputImpedanceAtTheta
zl_2_Gamma_in = inputImpedance2ReflectionCoefficientAtTheta
Gamma0_2_Gamma_in = reflectionCoefficientAtTheta
Gamma0_2_zin = reflectionCoefficient2InputImpedanceAtTheta



class Freespace(TransmissionLine):
	'''
	Represents a plane-wave in freespace, defined by [possibly complex]
	values of relative permativity and relative permeability.
	
	The field properties of space are related to the transmission line
	model given in circuit theory by:
			
		distributedCapacitance = real(epsilon_0*relativePermativity)
		distributedResistance = imag(epsilon_0*relativePermativity)
		distributedInductance = real(mu_0*relativePermeability)
		distributedConductance = imag(mu_0*relativePermeability)
	
	
	'''
	def __init__(self, relativePermativity=1, relativePermeability=1):
		TransmissionLine.__init__(self,\
			distributedCapacitance = real(epsilon_0*relativePermativity),\
			distributedResistance = imag(epsilon_0*relativePermativity),\
			distributedInductance = real(mu_0*relativePermeability),\
			distributedConductance = imag(mu_0*relativePermeability),\
			)

class FreespaceWithAttenuation(Freespace):
	def __init__(self, relativePermativity=1, relativePermeability=1, \
		loss = None, fBand=None):
		transmissionLine.__init__(self,\
			distributedCapacitance = real(epsilon_0*relativePermativity),\
			distributedResistance = imag(epsilon_0*relativePermativity),\
			distributedInductance = real(mu_0*relativePermeability),\
			distributedConductance = imag(mu_0*relativePermeability),\
			fBand = fBand
			)
		self.surfaceConductivity = loss
	
	
	def createDelayShort(self,l,f=None, gamma=None, numPoints = None, **kwargs ):
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
			1 port mwavepy.ntwk  instance representing a delay  short of
			length l 
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if f is None:
			if  self.fBand is None:
				raise ValueError('please supply frequency information')
			else:
				f = self.fBand.axis
		
			
		
		s = -1*exp(-1j* 2*self.electricalLength(l,f,gamma))
		outputNtwk = mv.ntwk(data=s,paramType='s',freq=f,**kwargs)
		outputNtwk.attenuate(self.surfaceConductivity*l)
		return outputNtwk

class FreespacePointSource( Freespace):
	'''
	represents a point source in freespace, defined by [possibly complex]
	values of relative permativity and relative permeability.
	
	technically this is not a TEM wave, because a point source is an 
	infinite combination of plane waves. therefore, to simulate a 
	1/r**2 loss, a trick is included in the electrical length calculation
	to produce a similar loss characteristic,
	
	the actual trick is 
		exp ( gamma(2*pi*f )*l + 1j*log(1./(1-l)**2 
	'''
	def __init__(self, relativePermativity=1, relativePermeability=1,fBand=None):
		freespace.__init__(self, \
			relativePermativity=relativePermativity, \
			relativePermeability=relativePermeability,\
			fBand=fBand)
	
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
			return  gamma(2*pi*f)*l - 1j*log(1./(1-l)**2) 
		elif deg ==True:
			return  mv.rad2deg(gamma(2*pi*f )*l - 1j*log(1./(1-l)**2)  )		
class Coax(TransmissionLine):
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


		
# quasi, or non- TEM lines
class Microstrip:
	def __init__(self):
		raise NotImplementedError
		return None
	def eEffMicrostrip(w,h,epR):
		'''
		The above formulas are in Transmission Line Design Handbook by Brian C Wadell, Artech House 1991. The main formula is attributable to Harold A. Wheeler and was published in, "Transmission-line properties of a strip on a dielectric sheet on a plane", IEEE Tran. Microwave Theory Tech., vol. MTT-25, pplb. 631-647, Aug. 1977. The effective dielectric constant formula is from: M. V. Schneider, "Microstrip lines for microwave integrated circuits," Bell Syst Tech. J., vol. 48, pplb. 1422-1444, 1969.
		'''
		
		if w < h:
			return (epR+1.)/2 + (epR-1)/2 *(1/sqrt(1+12*h/w) + .04*(1-w/h)**2)
		else:
			return (epR+1.)/2 + (epR-1)/2 *(1/sqrt(1+12*h/w))
		
		
		
	
	def betaMicrostrip(w,h,epR):
		return lambda omega: omega/c * sqrt(eEffMicrostrip(w,h,epR))
		
		
	def impedanceMicrostrip(w,h,epR):
		'''
		taken from pozar
		'''
		eEff = eEffMicrostrip(w,h,epR)
		if w/h < 1:
			return 60/sqrt(eEff) * npy.ln( 8*h/w + w/(4*h))
		else:
			return 120*pi/ ( sqrt(eEff)* w/h+1.393+.667*npy.ln(w/h+1.444) )

class CoplanarWaveguide:
	def __init__(self):
		raise NotImplementedError
		return None

		
