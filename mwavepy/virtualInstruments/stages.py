
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
	UNIT_DICT = {\
		'enoder count':0,\
		'motor step':1,\
		'millimeter':2,\
		'micrometer':3,\
		'inches':4,\
		'milli inches':5,\
		'micro inches':6,\
		'degree':7,\
		'gradient':8,\
		'radian':9,\
		'milliradian':10,\
		'microradian':11,\
		}
	'''
	Newport ESP300 Stage Controller

	all axis control commands are sent to the number axis given by the
	local variable self.current_axis. so here is an example usage

	esp= ESP300()
	esp.current_axis=1
	esp.units= 'millimeter'
	esp.position = 10
	print esp.position
	'''
	def __init__(self, address=1, current_axis=1, **kwargs):
		GpibInstrument.__init__(self,address,**kwargs)
		self.current_axis = current_axis

	@property
	def current_axis(self):
		return self._current_axis
	@current_axis.setter
	def current_axis(self, input):
		self._current_axis = input


	@property
	def velocity(self):
		command_string = 'VA'
		return (self.ask('%i%s?'%(self.current_axis,command_string)))
	@velocity.setter
	def velocity(self,input):
		command_string = 'VA'
		self.write('%i%s%f'%(self.current_axis,command_string,input))

	@property
	def acceleration(self):
		command_string = 'AC'
		return (self.ask('%i%s?'%(self.current_axis,command_string)))
	@acceleration.setter
	def acceleration(self,input):
		command_string = 'AC'
		self.write('%i%s%f'%(self.current_axis,command_string,input))

	@property
	def deceleration(self):
		command_string = 'AG'
		return (self.ask('%i%s?'%(self.current_axis,command_string)))
	@deceleration.setter
	def deceleration(self,input):
		command_string = 'AG'
		self.write('%i%s%f'%(self.current_axis,command_string,input))

	@property
	def position_relative(self):
		raise NotImplementedError('See position property for reading position')
	@position_relative.setter
	def position_relative(self,input):
		command_string = 'PR'
		self.write('%i%s%f'%(self.current_axis,command_string,input))

	@property
	def position(self):
		command_string = 'TP'
		return (self.ask('%i%s'%(self.current_axis,command_string)))
	@position.setter
	def position(self,input):
		'''
		set the position of current axis to input 
		'''
		command_string = 'PA'
		self.write('%i%s%f'%(self.current_axis,command_string,input))

	@property
	def home(self):
		raise NotImplementedError
	@home.setter
	def home(self, input):
		command_string = 'DH'
		self.write('%i%s%f'%(self.current_axis,command_string,input))


	@property
	def units(self):
		raise NotImplementedError('I dont know how to read units')
	@unit.setter
	def units(self, input):
		'''
		 set axis units for all commands.
		 takes:
			input: a string, describing the units here are a list of
				possibilities. 
				 'enoder count'
				'motor step'
				'millimeter'
				'micrometer'
				'inches'
				'milli inches'
				'micro inches'
				'degree'
				'gradient'
				'radian'
				'milliradian'
				'microradian'
		'''
		command_string = 'SN'
		self.write('%i%s%i'%(self.current_axis,command_string, self.UNIT_DICT[input]))

	@property
	def error_message(self):
		return (self.ask('TB?'))

	@property
	def motor_on(self):
		command_string = 'MO'
		return (self.ask('%i%s?'%(self.current_axis,command_string)))
	@motor_on.setter
	def motor_on(self,input):
		if input:
			command_string = 'MO'
			self.write('%i%s'%(self.current_axis,command_string,input))
		if not input:
			command_string = 'MF'
			self.write('%i%s'%(self.current_axis,command_string,input))
		
	def send_stop(self):
		command_string = 'ST'
		self.write('%i%s'%(self.current_axis,command_string))

	def wait_for_stop(self, input=0):
		command_string = 'WS'
		self.write('%i%s%i'%(self.current_axis,command_string, input))
		
	
