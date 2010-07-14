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


## PROPERTIES
	# s-parameter matrix
	@property
	def s(self):
		return self._s
	
	@s.setter
	def s(self, input_s_matrix):
		self._s = input_s_matrix
	
	
	# frequency information
	@property
	def f(self):
		''' this is how to use this'''
		raise NotImplementedError
		
	@f.setter
	def f(self):
		raise NotImplementedError
	
	# characteristic impedance
	@property
	def z0(self):
		''' this is how to use this'''
		raise NotImplementedError
	
	@z0.setter
	def z0(self):
		raise NotImplementedError
	
	
	@property
	def s_manitude(self):
		'''
		help on s_magnitude
		'''
		return mf.complex_2_magnitude(self.s)
	

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
