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
from  network import Network
import numpy as npy
def short(f, **kwargs):
	'''
	creates a Network for a short  transmission line (Gamma0=-1) 
	
	takes:
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a transmission line of length d
	'''
	f = npy.array(f, dtype=float).reshape(-1)
	
	npoints = len(f)	
	result = Network(**kwargs)
	result.f = f
	result.s = -1.0*npy.ones(npoints, dtype=complex)
	
	return result

def match(f, **kwargs):
	'''
	creates a Network for a perfect match  transmission line (Gamma0=0) 
	
	takes:
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a perfect match
	'''
	f = npy.array(f, dtype=float).reshape(-1)
	
	npoints = len(f)	
	result = Network(**kwargs)
	result.f = f
	result.s = npy.zeros(npoints, dtype=complex)
	
	return result

def load(Gamma0, f, **kwargs):
	'''
	creates a Network for a simple load termination  
	
	takes:
		Gamma0: reflection coefficient of the load (not in dB)
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a perfect match
	'''
	f = npy.array(f, dtype=float).reshape(-1)
	
	npoints = len(f)	
	result = Network(**kwargs)
	result.f = f
	result.s = Gamma0 * -1*npy.ones(npoints, dtype=complex)
	return result


def open(f, **kwargs):
	'''
	creates a Network for a 'open'   transmission line (Gamma0=+1) 
	
	takes:
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a transmission line of length d
	'''
	f = npy.array(f, dtype=float).reshape(-1)
	
	npoints = len(f)	
	result = Network(**kwargs)
	result.f = f
	result.s = 1.0*npy.ones(npoints, dtype=complex)
	
	return result
	
def line(d, tline, f, **kwargs):
	'''
	creates a Network for a section of transmission line
	
	takes:
		d: the length (in meters)
		tline: TransmissionLine class
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 2-port Network class, representing a transmission line of length d

	note: the only function called from teh tline class is
	electrical_length(f,d). any class which provides this will work .
	'''
	f = npy.array(f, dtype=float).reshape(-1)
	npoints = len(f)
	d = 1.0*d 
	
	result = Network(**kwargs)
	result.f = f
	
	s11 = npy.zeros(npoints, dtype=complex)
	s21 = npy.exp(1j* tline.electrical_length(f=result.f,d=d))
	
	result.s = npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
	return result

def delay_short(d, tline,f, **kwargs):
	'''
	creates a Network for a delayed short transmission line
	
	takes:
		d: the length (in meters)
		tline: TransmissionLine class
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a shorted transmission
		line of length d
	'''
	f = npy.array(f, dtype=float)
	a_line = line(d,tline,f,**kwargs)
	a_short = short(f)
	return a_line ** a_short

def delay_load(d, tline, Gamma0,f, **kwargs):
	'''
	creates a Network for a delayed short transmission line
	
	takes:
		d: the length (in meters)
		tline: TransmissionLine class
		Gamma0: reflection coefficient at load (not in db)
		f: frequency vector (in Hz)
		**kwargs: key word arguments passed to Network Constructor
	returns:
		a 1-port Network class, representing a load terminated
		transmission line of length d
	'''
	f = npy.array(f, dtype=float)
	a_line = line(d,tline,f,**kwargs)
	a_load = load(Gamma0,f)
	return a_line ** a_load

def impedance_step():
	raise NotImplementedError

def transformer():
	raise NotImplementedError
	
def shunt_inductance():
	raise NotImplementedError


