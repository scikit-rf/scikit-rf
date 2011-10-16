
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


def cartesian_product_calibration_ensemble( ideals, measured):
	'''
	Creates an ensemble of calibration instances. the set  of 
	measurement lists used in the ensemble is the Cartesian Product
	of all instances of each standard.
	
	'ideals' must be a list of Networks whose names correspond 
	uniquely to the corresponding Network in the measured list. 
	
	takes:
		ideals: list of ideal Networks
		measured: list of measured Networks
	
	returns:
		cal_ensemble: a list of Calibration instances.
	'''
	# this creates a 2D nested list. first level is a the standard, 
	# the second level is the instance of the standard
	measured_iterable = \
		[[ measure for measure in measured \
			if ideal.name in measure.name] for ideal in ideals]
	measured_product = product(*measured_iterable)
	
	return [Calibration(ideals =ideals, measured = list(product_element))\
		for product_element in measured_product]

def calibration_from_names( ideals, measured):
	'''
	Creates a Calibration from unordered and possibly different length 
	lists of ideal and measured Networks. The elements of each list 
	are aligned to each other based on their names. the ideal Network's
	must have uniquely identifying names which are sub-strings of the 
	measured Network's names. 
		you can have more than one measured Network corresponding to a
	single ideal, in which case this will use multiple copies of that 
	ideal network.
	
	takes:
		ideals: list of ideal Networks
		measured: list of measured Networks
	
	returns:
		cal: Calibration instance
	'''
	# this creates a 2D nested list. first level is a the standard, 
	# the second level is the instance of the standard
	ideals_new = \
		[ ideal for ideal in ideals for measure in measured if ideal.name in measure.name]
	
	return Calibration(ideals = ideals_new, measured = measured)
