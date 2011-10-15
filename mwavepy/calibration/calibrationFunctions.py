
#       calibrationFunctions.py
#       
#       
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
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
Functions which operate on or pertain to Calibration Objects
'''
from itertools import product
from calibration import Calibration


def CartesianProductCalibrationEnsemble( ideals, measured):
	'''
	Creates an ensemble of calibration instances. the set  of 
	measurement lists used in the ensemble is the Cartesian Product
	of all instances of each standard.
	
	'ideals' must be a list of Networks whose names correspond 
	uniquely to the corresponding Network in the measured list. 
	'''
	# this creates a 2D nested list. first level is a the standard, 
	# the second level is the instance of the standard
	measured_iterable = \
		[[ measure for measure in measured \
			if ideal.name in measure.name] for ideal in ideals]
	measured_product = product(*measured_iterable)
	
	return [Calibration(ideals =ideals, measured = list(product_element))\
		for product_element in measured_product]

def MultipleConnectionCalibration( ideals, measured):
	'''
	
	'ideals' must be a list of Networks whose names correspond 
	uniquely to the corresponding Network in the measured list. 
	'''
	# this creates a 2D nested list. first level is a the standard, 
	# the second level is the instance of the standard
	measured_iterable = \
		[[ measure for measure in measured \
			if ideal.name in measure.name] for ideal in ideals]
	measured_product = product(*measured_iterable)
	
	return [Calibration(ideals =ideals, measured = list(product_element))\
		for product_element in measured_product]
