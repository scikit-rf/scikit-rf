
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

class a():
    def __init__(self,**kwargs):
        self.kwargs = kwargs


class Calibration(object):
	
	calibration_algorithm_dict={'one port': one_port}
	
	def __init__(self,f , type, ideals, measured, **kwargs):
		self.f = f
		self.frequency = f_2_frequency(f)
		self.ideals = ideals
		self.measured = measured
		# a dictionary holding key word arguments to pass to whatever
		# calibration function we are going to call
		self.kwargs = kwargs
		self.type = type
	
	@property
	def type (self):
		return self._type
	@type.setter
	def type(self, new_type):
		if new_type not in self.calibration_algorithm_dict.keys():
			raise ValueError('incorrect calibration type')
		self._type = new_type
	
	@property	
	def coefs(self):
		'''
		coefs: a dictionary holding the calibration coefficients
		'''
		
		try:
			return self._coefs
		except(AttributeError):
			self.run()
			return self._coefs
	
	def run(self):
		self.output_from_cal = calibration_algorithm_dict[self.type](**self.kwargs)
		self._coefs = output_from_cal['error coefficients']

