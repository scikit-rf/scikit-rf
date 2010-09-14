
'''
#       calibrationFunctions.py
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


def abc_2_coefs_dict(abc):
	'''
	converts an abc ndarry to a dictionarry containing the error 
	coefficients.
	
	takes: 
		abc : Nx3 numpy.ndarray, which holds the complex calibration 
			coefficients. the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
	returns:
		coefsDict: dictionary containing the following
			'directivity':e00
			'reflection tracking':e01e10
			'source match':e11
	note: 
		e00 = directivity error
		e10e01 = reflection tracking error
		e11 = source match error	
	'''
	a,b,c = abc[:,0], abc[:,1],abc[:,2]
	e01e10 = a+b*c
	e00 = b
	e11 = c
	coefsDict = {'directivity':e00, 'reflection tracking':e01e10, \
		'source match':e11}
	return coefsDict
