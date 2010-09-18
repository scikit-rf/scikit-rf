
'''
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
from calibrationAlgorithms import *
from calibrationFunctions import *
from frequency import *
from network import *
import numpy as npy
import os 



class Calibration(object):
	'''
	represents a calibration instance.

	a generic class, to hold sets of measurements, ideals, and calibration
	results.



	all calibration algorithms are in calibrationAlgorithms.py
	'''
	calibration_algorithm_dict={\
		'one port': one_port,\
		'one port xds':xds,\
		'one port xds_xdl':xds_xdl,\
		}
	
	def __init__(self,frequency , type, name=None,  **kwargs):
		self.frequency = frequency
		# a dictionary holding key word arguments to pass to whatever
		# calibration function we are going to call
		self.kwargs = kwargs
		self.type = type
		self.name = name
	@property
	def type (self):
		return self._type
	@type.setter
	def type(self, new_type):
		if new_type not in self.calibration_algorithm_dict.keys():
			raise ValueError('incorrect calibration type')
		self._type = new_type

	@property
	def output_from_cal(self):
		'''
		a dictionary holding the output from the calibration algorithm
		'''
		
		return self._output_from_cal
	
	@property	
	def coefs(self):
		'''
		coefs: a dictionary holding the calibration coefficients

		for one port 
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
		try:
			return self._error_ntwk
		except(AttributeError):
			self.run()
			return self._error_ntwk
		


	def run(self):
		self._output_from_cal = self.calibration_algorithm_dict[self.type](**self.kwargs)
		self._error_ntwk = self.calculate_error_ntwk()

	def apply_cal(self,input_ntwk):
		caled =  input_ntwk//self.error_ntwk
		caled.name = input_ntwk.name
		return caled 

	def apply_cal_to_all_in_dir(self, dir, contains=None, f_unit = 'ghz'):
		ntwkDict = load_all_touchstones(dir=dir, contains=contains,\
			f_unit=f_unit)

		for ntwkKey in ntwkDict:
			ntwkDict[ntwkKey] = self.apply_cal(ntwkDict[ntwkKey])
		
		return ntwkDict
		
	def calculate_error_ntwk(self):
		if len (self.coefs.keys()) == 3:
			# ASSERT: we have one port data
			ntwk = Network()
			ntwk.frequency = self.frequency
			
			s12 = npy.ones(self.frequency.npoints, dtype=complex)
			s21 = self.coefs['reflection tracking'] 
			s11 = self.coefs['directivity'] 
			s22 = self.coefs['source match']
			ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
			return ntwk
		else:
			raise NotImplementedError('sorry')
	#def plot_error_coefs(self):
		
