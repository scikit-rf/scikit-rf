
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
import numpy as npy
from network import Network
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

def guess_length_of_delay_short( aNtwk,tline):
		'''
		guess length of physical length of a Delay Short given by aNtwk
		
		takes:
			aNtwk: a mwavepy.ntwk type . (note: if this is a measurment 
				it needs to be normalized to the short plane
			tline: transmission line class of the medium. needed for the 
				calculation of propagation constant
				
		
		'''
		#TODO: re-write this and document better
		
		beta = real(tline.beta())
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((-2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		#print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]



def error_dict_2_network(coefs, frequency=None, is_reciprocal=False, **kwargs):
		'''
		convert a dictionary holding standard error terms to a Network
		object. 
		
		takes:
		
		returns:
		

		'''
		
		if len (coefs.keys()) == 3:
			# ASSERT: we have one port data
			ntwk = Network(**kwargs)
			
			if frequency is not None:
				ntwk.frequency = frequency
				
			if is_reciprocal:
				#TODO: make this better and maybe have a phase continuity
				# functionality
				tracking  = coefs['reflection tracking'] 
				s12 = npy.sqrt(tracking)
				s21 = npy.sqrt(tracking)
				
			else:
				s21 = coefs['reflection tracking'] 
				s12 = npy.ones(len(s21), dtype=complex)
			
			s11 = coefs['directivity'] 
			s22 = coefs['source match']
			ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
			return ntwk
		else:
			raise NotImplementedError('sorry')
