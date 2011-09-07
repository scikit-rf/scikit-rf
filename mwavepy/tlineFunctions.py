
#       tlinefunctions.py
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
transmission line theory related functions 
'''

import numpy as npy
from numpy import pi, sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape

from scipy.constants import mu_0
import mathFunctions as mf

INF = 1e99
ONE = 1.0 + 1/1e14


def skin_depth(f,rho, mu_r):
	'''
	
	the skin depth for a material. see www.microwaves101.com for more info. 
	
	takes:
		f: frequency, in Hz
		rho: bulk resistivity of material, in ohm*m
		mu_r: relative permiability of material
		
	returns:
		skin depth: in m
	
	'''
	return sqrt(rho/(pi*f*mu_r*mu_0))

def surface_resistivity(f,rho,mu_r):
	'''
	surface resistivity. see www.microwaves101.com for more info. 
	
	takes:
		f: frequency, in Hz
		rho: bulk resistivity of material, in ohm*m
		mu_r: relative permiability of material
	
	returns:
		surface resistivity: ohms/square
		
		
	'''
	return rho/skin_depth(rho=rho,f = f, mu_r=mu_r)
	
def distributed_circuit_2_propagation_impedance( distributed_admittance,\
	distributed_impedance):
	'''
	
	converts complex distributed impedance and admittance to propagation 
	constant and characteristic impedance.
	
	takes:
		distributed_admittance: what it says [complex number or array]
		distributed_impedance: what it says [complex number or array]
	
	returns:
		propagation_constant: what it says [complex number or array]
		characteristic_impedance: what it says [complex number or array]
	'''
	propagation_constant = \
		sqrt(distributed_impedance*distributed_admittance)
	characteristic_impedance = \
		sqrt(distributed_impedance/distributed_admittance)
	return (propagation_constant, characteristic_impedance)

def propagation_impedance_2_distributed_circuit(propagation_constant, \
	characteristic_impedance):
	'''
	converts complex propagation constant and characteristic impedance 
	to distributed impedance and admittance.
	
	takes:
		propagation_constant: what it says [complex number or array]
		characteristic_impedance: what it says [complex number or array]
			
	returns:
		distributed_admittance: what it says [complex number or array]
		distributed_impedance: what it says [complex number or array]
	'''
	distributed_admittance = propagation_constant/characteristic_impedance
	distributed_impedance = propagation_constant*characteristic_impedance
	return (distributed_admittance,distributed_impedance)

def electrical_length(gamma, f , d, deg=False):
	'''
	calculates the electrical length of a section of transmission line.

	takes:
		gamma: propagation constant function [function], 
			(a function which takes frequency in hz )
		l: length of line. in meters
		f: frequency at which to calculate. [array-like or float]. 
		deg: return in degrees or not. [boolean].
	
	returns:
		theta: electrical length in radians or degrees, 
			depending on  value of deg.
	'''
	
	# typecast to a 1D array
	f = array(f, dtype=float).reshape(-1)
	d = array(d, dtype=float).reshape(-1)
			
	if deg == False:
		return  gamma(f)*d
	elif deg == True:
		return  mf.radian_2_degree(gamma(f)*d )

def electrical_length_2_distance(theta, gamma, f0,deg=True):
	'''
	convert electrical length to a physical distance.
	
	takes:
		theta: electrical length
		gamma: propagation constant function [function]
		f0:	frequency of interest [number]
		deg: is theta in degrees [Boolean]
	returns:
		d: physical distance
		
	note: the gamma function must take a single variable, that is 
	frequency and return complex propagation constant such that the 
	propagating part is positive imag part.  
	'''
	if deg == True:
		theta = mf.degree_2_radian(theta)
	return theta/imag(gamma(f0))

def input_impedance_2_reflection_coefficient(z0, zl):
	'''
	calculates the reflection coefficient for a given input impedance 
	takes:
		
		zl: input (load) impedance [number of array].  
		z0 - characteristic impedance[number of array].
		
	note:
		input data is typecasted to 1D complex array
	'''
	# typecast to a complex 1D array. this makes everything easier	
	z0 = array(z0, dtype=complex).reshape(-1)
	zl = array(zl, dtype=complex).reshape(-1)
	
	# handle singularity  by numerically representing inf as big number
	zl[(zl==npy.inf)] = INF

	return ((zl -z0 )/(zl+z0))

def reflection_coefficient_2_input_impedance(z0,Gamma):
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
	
def reflection_coefficient_at_theta(Gamma0,theta):
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
	Gamma0 = array(Gamma0, dtype=complex).reshape(-1)
	theta = array(theta, dtype=complex).reshape(-1)
	return Gamma0 * exp(2j* theta)

def input_impedance_at_theta(z0,zl, theta):
	'''
	input impedance of load impedance zl at electrical length theta, 
	given characteristic impedance z0.
	
	takes:
		z0 - characteristic impedance. 
		zl: load impedance
		theta: electrical length of the line, (may be complex) 
	'''
	Gamma = input_impedance_2_reflection_coefficient(z0=z0,zl=zl)
	Gamma_in = reflection_coefficient_at_theta(Gamma=Gamma, theta=theta)
	return reflection_coefficient_2_input_impedance(z0=z0, Gamma=Gamma_in)
	
def input_impedance_2_reflection_coefficient_at_theta(z0, zl, theta):
	Gamma0 = input_impedance_2_reflection_coefficient(z0=z0,zl=zl)
	Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
	return Gamma_in

def reflection_coefficient_2_input_impedance_at_theta(z0, Gamma0, theta):
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
	Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
	zin = reflection_coefficient_2_input_impedance(z0=z0,Gamma=Gamma_in)
	return zin
# short hand convinience. 
# admitantly these follow no logical naming scheme, but they closely 
# correspond to common symbolic conventions, and are convenient
theta = electrical_length
distance_2_electrical_length = electrical_length

zl_2_Gamma0 = input_impedance_2_reflection_coefficient
Gamma0_2_zl = reflection_coefficient_2_input_impedance

zl_2_zin = input_impedance_at_theta
zl_2_Gamma_in = input_impedance_2_reflection_coefficient_at_theta

Gamma0_2_Gamma_in = reflection_coefficient_at_theta
Gamma0_2_zin = reflection_coefficient_2_input_impedance_at_theta

