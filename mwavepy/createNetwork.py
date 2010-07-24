'''
#       createNetwork.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       
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
from  mwavepy1.network import Network
import numpy as npy
def short(frequency, **kwargs):
	'''
	creates a Network for a short  transmission line (Gamma0=-1) 
	
	takes:
		frequency: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a transmission line of length d
	'''
	frequency = npy.array(frequency, dtype=float).reshape(-1)
	
	npoints = len(frequency)	
	result = Network(**kwargs)
	result.f = frequency
	result.s = -1.0*npy.ones(npoints, dtype=complex)
	
	return result

def open(frequency, **kwargs):
	'''
	creates a Network for a 'open'   transmission line (Gamma0=+1) 
	
	takes:
		frequency: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a transmission line of length d
	'''
	frequency = npy.array(frequency, dtype=float).reshape(-1)
	
	npoints = len(frequency)	
	result = Network(**kwargs)
	result.f = frequency
	result.s = 1.0*npy.ones(npoints, dtype=complex)
	
	return result
	
def line(d, tline, frequency, **kwargs):
	'''
	creates a Network for a section of transmission line
	
	takes:
		d: the length (in meters)
		tline: TransmissionLine class
		frequency: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 2-port Network class, representing a transmission line of length d
	'''
	frequency = npy.array(frequency, dtype=float).reshape(-1)
	npoints = len(frequency)
	d = 1.0*d 
	
	result = Network(**kwargs)
	result.f = frequency
	
	s11 = npy.zeros(npoints, dtype=complex)
	s21 = npy.exp(1j* tline.electrical_length(f=result.f,d=d))
	
	result.s = npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
	return result

def delay_short(d, tline,frequency, **kwargs):
	'''
	creates a Network for a delayed short transmission line
	
	takes:
		d: the length (in meters)
		tline: TransmissionLine class
		frequency: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a transmission line of length d
	'''
	frequency = npy.array(frequency, dtype=float)
	a_line = line(d,tline,frequency,**kwargs)
	a_short = short(frequency)
	return a_line ** a_short
	

def impedance_step():
	raise NotImplementedError

def transformer():
	raise NotImplementedError
	
def shunt_inductance():
	raise NotImplementedError


