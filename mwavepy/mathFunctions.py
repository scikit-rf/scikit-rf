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

## simple conversions
def complex_2_magnitude(input):
	'''
	returns the magnitude of a complex number. 
	'''
	return abs(input)












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



def mag2dB(mag):
	return  20*npy.log10(mag)
	
def dB2Mag(dB):
	return 10**((dB)/20.)

def dB2np(x):
	'''
	converts a value in nepers to dB
	'''	
	return (log(10)/20) * x
def np2dB(x):
	'''
	converts a value in dB to neper's
	'''
	return 20/log(10) * x

def rad2deg(rad):
	return (rad)*180/pi
	
def deg2rad(deg):
	return (deg)*pi/180
	








############### Ploting ############### 
