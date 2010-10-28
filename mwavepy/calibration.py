
#       calibration.py
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
Contains the Calibration class, and supporting functions
'''
import numpy as npy
import os 
from copy import copy

from calibrationAlgorithms import *
from frequency import *
from network import *
from convenience import *

## main class
class Calibration(object):
	'''
	Represents a calibration instance, a generic class to hold sets
	of measurements, ideals, and calibration results.

	all calibration algorithms are in calibrationAlgorithms.py, and are
	referenced by the dictionary in this object called
	'calibration_algorihtm_dict'
	'''
	calibration_algorithm_dict={\
		'one port': one_port,\
		'one port xds':xds,\
		'one port parameterized':parameterized_self_calibration,\
		}
	
	def __init__(self,frequency , type, name=None,is_reciprocal=False,\
		**kwargs):
		'''
		Calibration initializer
		
		takes:
			frequency: a Frequency object over which the calibration
				is defined

			type: string representing what type of calibration is to be
				performed. supported types at the moment are:

				'one port':	standard one-port cal. if more than
					2 measurement/ideal pairs are given it will
					calculate the least squares solution.

				'one port xds': self-calibration of a unknown-length
					delay-shorts.

			**kwargs: key-word arguments passed to teh calibration
				algorithm.

			name: name of calibration, just a handy identifing string
			is_reciprocal: enables the reciprocity assumption on 
				calculated error network
		'''
		self.frequency = copy(frequency)
		# a dictionary holding key word arguments to pass to whatever
		# calibration function we are going to call
		self.kwargs = kwargs
		self.type = type
		self.name = name
		self.is_reciprocal = is_reciprocal

	## properties
	@property
	def type (self):
		'''
		string representing what type of calibration is to be
		performed. supported types at the moment are:

		'one port':	standard one-port cal. if more than
			2 measurement/ideal pairs are given it will
			calculate the least squares solution.

		'one port xds': self-calibration of a unknown-length
			delay-shorts.

		note:
		algorithms referenced by  calibration_algorithm_dict 
		'''
		return self._type

	@type.setter
	def type(self, new_type):
		if new_type not in self.calibration_algorithm_dict.keys():
			raise ValueError('incorrect calibration type')
		self._type = new_type

	@property
	def output_from_cal(self):
		'''
		a dictionary holding all of the output from the calibration
		algorithm
		'''
		
		return self._output_from_cal
	
	@property	
	def coefs(self):
		'''
		coefs: a dictionary holding the calibration coefficients

		for one port cal's
			'directivity':e00
			'reflection tracking':e01e10
			'source match':e11
		'''
		
		try:
			return self._output_from_cal['error coefficients']
		except(AttributeError):
			self.run()
			return self._output_from_cal['error coefficients']

	@property
	def error_ntwk(self):
		'''
		a Network type which represents the error network being
		calibrated out.
		'''
		try:
			return self._error_ntwk
		except(AttributeError):
			self.run()
			return self._error_ntwk

	##  methods for manual control of internal calculations
	def run(self):
		'''
		runs the calibration algorihtm.
		
		 this is automatically called the
		first time	any dependent property is referenced (like error_ntwk)
		, but only the first time. if you change something and want to
		re-run the calibration use this.  
		'''
		self._output_from_cal = self.calibration_algorithm_dict[self.type]\
			(**self.kwargs)
		self._error_ntwk = error_dict_2_network(self.coefs, \
			frequency=self.frequency, is_reciprocal=self.is_reciprocal)
	

	## methods 
	def apply_cal(self,input_ntwk):
		'''
		apply the current calibration to a measurement.

		takes:
			input_ntwk: the measurement to apply the calibration to, a
				Network type.
		returns:
			caled: the calibrated measurement, a Network type.
		'''
		caled =  input_ntwk//self.error_ntwk
		caled.name = input_ntwk.name
		return caled 

	def apply_cal_to_all_in_dir(self, dir, contains=None, f_unit = 'ghz'):
		'''
		convience function to apply calibration to an entire directory
		of measurements, and return a dictionary of the calibrated
		results, optionally the user can 'grep' the direction
		by using the contains switch.

		takes:
			dir: directory of measurements (string)
			contains: will only load measurements who's filename contains
				this string.
			f_unit: frequency unit, to use for all networks. see
				frequency.Frequency.unit for info.
		returns:
			ntwkDict: a dictionary of calibrated measurements, the keys
				are the filenames.
		'''
		ntwkDict = load_all_touchstones(dir=dir, contains=contains,\
			f_unit=f_unit)

		for ntwkKey in ntwkDict:
			ntwkDict[ntwkKey] = self.apply_cal(ntwkDict[ntwkKey])
		
		return ntwkDict
		
	
	#def plot_error_coefs(self):
		



## Functions
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

