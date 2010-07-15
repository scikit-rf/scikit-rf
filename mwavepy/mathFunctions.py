'''
#       mathFunctions.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen 
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
import numpy as npy
from numpy import pi
## simple conversions
def complex_2_magnitude(input):
	'''
	returns the magnitude of a complex number. 
	'''
	return abs(input)

def complex_2_db(input):
	'''
	returns the magnitude in dB of a complex number. 
	
	returns:
		20*log10(|z|)
	where z is a complex number
	'''
	return magnitude_2_db(npy.abs( input))

def complex_2_radian(input):
	'''
	returns the angle complex number in radians. 

	'''
	return npy.angle(input)

def complex_2_degree(input):
	'''
	returns the angle complex number in radians. 

	'''
	return npy.angle(input, deg=True)

def magnitude_2_db(input):
	'''
	converts magnitude to db 
	
	 db is given by 
		20*log10(|z|)
	where z is a complex number
	'''
	return  20*npy.log10(input)
	
def db_2_magnitude(input):
	'''
	converts db to normal magnitude
	
	returns:
		10**((z)/20.)
	where z is a complex number
	'''
	return 10**((input)/20.)

def db_2_np(x):
	'''
	converts a value in nepers to dB
	'''	
	return (log(10)/20) * x
def np_2_db(x):
	'''
	converts a value in dB to neper's
	'''
	return 20/log(10) * x

def radian_2_degree(rad):
	return (rad)*180/pi
	
def degree_2rad_(deg):
	return (deg)*pi/180
	




# old functions just for reference
def complex2Scalar(input):
	input= array(input)
	output = []
	for k in input:
		output.append(real(k))
		output.append(imag(k))
	return array(output).flatten()
	
def scalar2Complex(input):
	input= array(input)
	output = []
	
	for k in range(0,len(input),2):
		output.append(input[k] + 1j*input[k+1])
	return array(output).flatten()


def complex2dB(complx):
	dB = 20 * npy.log10(npy.abs( (npy.real(complx) + 1j*npy.imag(complx) )))
	return dB



def complex2ReIm(complx):
	return npy.real(complx), npy.imag(complx)

def complex2MagPhase(complx,deg=False):
	return npy.abs(complx), npy.angle(complx,deg=deg)












############### Ploting ############### 
