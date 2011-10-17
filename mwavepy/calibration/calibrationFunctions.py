
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
from itertools import product, combinations
from calibration import Calibration


def cartesian_product_calibration_ensemble( ideals, measured, *args, **kwargs):
	'''
	This function is used for calculating calibration uncertainty due 
	to un-biased, non-systematic errors. 
	
	It creates an ensemble of calibration instances. the set  of 
	measurement lists used in the ensemble is the Cartesian Product
	of all instances of each measured standard.
	
	The idea is that if you have multiple measurements of each standard,
	then the multiple calibrations can be made, by generating all possible
	combinations of measurements.  This yields a simple way to estimate 
	calibration uncertainty. This is similiar to a technique described 
	in 'De-embeding and	Un-terminating' by Penfield and Baurer. 
	
	
		
	takes:
		ideals: list of ideal Networks
		measured: list of measured Networks
		*args,**kwargs: passed to Calibration initializer
	
	returns:
		cal_ensemble: a list of Calibration instances.
		
		
	you can use the output to estimate uncertainty by calibrating a DUT 
	with all calibrations, and then running statistics on the resultant
	set of Networks. for example
	
	import mwavepy as mv
	# define you lists of ideals and measured networks
	cal_ensemble = \
		mv.cartesian_product_calibration_ensemble( ideals, measured)
	dut = mv.Network('dut.s1p')
	network_ensemble = [cal.apply_cal(dut) for cal in cal_ensemble]
	mv.plot_uncertainty_mag(network_ensemble)
	[network.plot_s_smith() for network in network_ensemble]
	'''
	measured_iterable = \
		[[ measure for measure in measured \
			if ideal.name in measure.name] for ideal in ideals]
	measured_product = product(*measured_iterable)
	
	return [Calibration(ideals =ideals, measured = list(product_element)\
		*args, **kwargs)\
		for product_element in measured_product]

def subset_calibration_ensemble( ideals, measured, n,  *args, **kwargs):
	'''
	This is similiar to a technique described in 'De-embeding and
	Un-terminating' by Penfield and Baurer. 
	
	
		
	takes:
		ideals: list of ideal Networks
		measured: list of measured Networks
		*args,**kwargs: passed to Calibration initializer
	
	returns:
		cal_ensemble: a list of Calibration instances.
		
		

	'''
	if n >= len(ideals):
		raise ValueError('n must be larger than # of standards')
	
	ideal_subsets = \
		[ ideal_subset for ideal_subset in combinations(ideals,n)]
	measured_subsets = \
		[ measured_subset for measured_subset in combinations(measured,n)]

	return 	[Calibration(ideals = list(k[0]), measured=list(k[1]),\
		*args, **kwargs) for k in zip(ideal_subsets, measured_subsets)]

