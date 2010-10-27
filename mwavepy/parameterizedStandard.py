

#       parametertizedStandard.py
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
provides Parameterized Standard class.  
'''
import numpy as npy
from copy import copy

from discontinuities.variationalMethods import translation_offset

class ParameterizedStandard(object):
	def __init__(self, function=None, parameters={}, **kwargs):
		'''
		takes:
			parameters: an dictionary holding an list of parameters,
				which will be the dependent variables to optimize.
				these are passed to the network creating function. 
		'''
		self.kwargs = kwargs
		self.parameters = parameters
		self.function = function

	
	@property
	def parameter_keys(self):
		'''
		returns a list of parameter dictionary keys in alphabetical order
		'''
		keys = self.parameters.keys()
		keys.sort()
		return keys

	@property	
	def parameter_array(self):
		'''
		returns an array of the parameters, in alphabetacally order of
		parameters.keys
		'''
		return npy.array([self.parameters[k] for  k in self.parameter_keys])

	@parameter_array.setter	
	def parameter_array(self,input_array):
		'''
		fills the parameter dictionary from an array of values. the
		dictionary is filled in alphabetical order. 
		'''
		counter = 0
		for k in self.parameter_keys:
			self.parameters[k]= input_array[counter]
			counter+=1
	@property
	def number_of_parameters(self):
		return len(self.parameter_keys)
	
	@property
	def s(self):
		return self.network.s

	@property
	def network(self):
		tmp_args = copy(self.kwargs)
		tmp_args.update(self.parameters)
		return self.function(**tmp_args)


# pre-defined parametrized standards

def ps_translation_missalignment(wg, freq):
	parameters = {'delta_a':0, 'delta_b':0}
	kwargs = {'wg':wg,'freq':freq}
	function = translation_offset
	return ParameterizedStandard(function = function, \
		parameters = parameters, **kwargs)

def ps_parameterless(ideal_network):
	return ParameterizedStandard(function = lambda: ideal_network) 
	
def delay_short_unknown_delay():
	return 0
	
