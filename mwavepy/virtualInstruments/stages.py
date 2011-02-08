
#	   stages.py
#
#	This file holds  stage objects
#
#	   Copyright 2010  lihan chen, alex arsenovic <arsenovic@virginia.edu>
#	   
#	   This program is free software; you can redistribute it and/or modify
#	   it under the terms of the GNU General Public License as published by
#	   the Free Software Foundation; either version 2 of the License, or
#	   (at your option) any later version.
#	   
#	   This program is distributed in the hope that it will be useful,
#	   but WITHOUT ANY WARRANTY; without even the implied warranty of
#	   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	   GNU General Public License for more details.
#	   
#	   You should have received a copy of the GNU General Public License
#	   along with this program; if not, write to the Free Software
#	   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#	   MA 02110-1301, USA.

'''
holds class's for objects for stages
'''
import numpy as npy
import visa
from visa import GpibInstrument


class ESP300(GpibInstrument):
	'''
	Newport ESP300 Stage Controller
	'''
	def __init__(self, address=1, axis =  **kwargs):
		GpibInstrument.__init__(self,'GPIB::'+str(address),**kwargs)

		axis= 


	@property
	def velocity():
		raise NotImplementedError
	@velocity.setter
	def velocity(input):
		raise NotImplementedError

	@property
	def acceleration():
		raise NotImplementedError
	@acceleration.setter
	def acceleration(input):
		raise NotImplementedError

	@property
	def deceleration():
		raise NotImplementedError
	@deceleration.setter
	def deceleration(input):
		raise NotImplementedError

	@property
	def position_relative():
		raise NotImplementedError
	@position_relative.setter
	def position_relative(input):
		raise NotImplementedError

	@property
	def position_absolute():
		raise NotImplementedError
	@position_absolute.setter
	def position_absolute(input):
		raise NotImplementedError

	@property
	def position():
		raise NotImplementedError

	@property
	def wait_for_stop():
		raise NotImplementedError
