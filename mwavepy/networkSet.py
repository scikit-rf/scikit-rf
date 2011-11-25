
#       networkSet.py
#       
#       
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
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
Provides the Network Set class, used for statistics and ploting for 
network sets.
'''

from Network import func_on_networks as fon
from Network import average as network_average


import numpy as npy


class NetworkSet(object):
	'''
	A class to hold a set of Networks, to calculate statistics on the set 
	'''
	def __init__(self, ntwk_set):
		'''
		input:
			ntwk_set: a list of Network's.
		returns:
			a NetworkSet type
		'''
		self.ntwk_set = ntwk_set
	
	@property 
	def mean(self):
		'''
		returns the complex mean network for the set 
		'''
		return network_average(self.ntwk_set)
	
	@property 
	def mean_s_mag(self):
		'''
		returns the mean of the magnitude  for the set 
		'''
		return fon(self.ntwk_set,npy.mean,'s_mag')
	
	@property
	def std(self):
		'''
		return complex standard deviation (mean distance)
		'''
		return fon(self.ntwk_set,npy.std,'s')

	@property
	def std_s_mag(self):
		'''
		return complex standard deviation (mean distance)
		'''
		return fon(self.ntwk_set,npy.std,'s_mag')
