'''
#       network.py
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

import  mwavepy1.mathFunctions as mf


class Network(object):
	def __init__(self):
		self._s = None
		self._f = None
		self._z0 = None


## PRIMARY PROPERTIES
	# s-parameter matrix
	@property
	def s(self):
		'''
		The scattering parameter matrix 
		'''
		return self._s
	
	@s.setter
	def s(self, input_s_matrix):
		self._s = input_s_matrix
	
	
	# frequency information
	@property
	def f(self):
		''' the frequency vector for the network, in Hz. '''
		raise NotImplementedError
		
	@f.setter
	def f(self):
		raise NotImplementedError
	
	# characteristic impedance
	@property
	def z0(self):
		''' the characteristic impedance of the network.'''
		raise NotImplementedError
	
	@z0.setter
	def z0(self):
		raise NotImplementedError
	
## SECONDARY PROPERTIES

# s-parameters convinience properties	
	@property
	def s_manitude(self):
		'''
		returns the magnitude of the s-parameters.
		'''
		return mf.complex_2_magnitude(self.s)
	
	@property
	def s_db(self):
		'''
		returns the magnitude of the s-parameters, in dB
		
		note:
			dB is calculated by 
				20*log10(|s|)
		'''
		return mf.complex_2_db(self.s)
		
	@property
	def s_deg(self):
		'''
		returns the phase of the s-parameters, in radians
		'''
		return mf.complex_2_degree(self.s)
		
	@property
	def s_rad(self):
		'''
		returns the phase of the s-parameters, in radians.
		'''
		return mf.complex_2_radian(self.s)
	
	@property
	def s_deg_unwrap(self):
		'''
		returns the unwrapped phase of the s-paramerts, in degrees
		'''
		return mf.rad_2_degree(self.s_rad_unwrap)
	
	@property
	def s_rad_unwrap(self):
		'''
		returns the unwrapped phase of the s-parameters, in radians.
		'''
		return npy.unwrap(mf.complex_2_radian(self.s))



## CLASS METHODS
	def method_of_network(self):
		'''help on this method'''
		raise NotImplementedError



## FUNCTIONS
def cascade():
	raise NotImplementedError

def de_embed():
	raise NotImplementedError

def divide():
	raise NotImplementedError
