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

class Network(object):

## PROPERTIES
	# s-parameter matrix
	def __set_s(self):
		raise NotImplementedError
	def __get_s(self):
		raise NotImplementedError
	s = property(__get_s, __set_s,doc =\
	'''This is how to use this.
	and there is more info''')
	
	
	# frequency information
	def __set_f(self):
		raise NotImplementedError
	def __get_f(self):
		raise NotImplementedError
	f = property(__get_f, __set_f, doc= \
	''' this is how to use this''')
	
	
	def __set_z0(self):
		raise NotImplementedError
	def __get_z0(self):
		raise NotImplementedError
	z0 = property(__get_z0, __set_z0, doc= \
	''' this is how to use this''')


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
