from time import sleep 
from datetime import datetime
import pylab as plb

from .futekLoadCell import *
from .stages import ESP300
from .vna import ZVA40_alex

class LifeTimeProbeTester(object):
	def __init__(self, stage=None, vna=None, load_cell=None, \
		down_direction=-1, step_increment =.01, contact_force=.1,\
		time_delay=0,raiseup_overshoot=.1,file_dir = './'):

		if stage is None: self.stage = ESP300()
		else: self.stage = stage

		if vna is None: self.vna = ZVA40_alex()
		else: self.vna = vna

		if load_cell is None: self.load_cell = Futek_USB210_socket()
		else: self.load_cell = load_cell
		
		self.down_direction = down_direction
		self.step_increment = step_increment
		self.contact_force = contact_force
		self.time_delay = time_delay
		self.raiseup_overshoot = raiseup_overshoot
		self.file_dir=file_dir
		
		self.zero()
		self.stage.motor_on = True
		self.force_history = []
		self.position_history = []

	@property
	def data(self):
		return self.read_loadcell_and_stage_position()

	def move_toward(self,value):
		self.stage.position_relative = self.down_direction*value

	def move_apart(self,value):
		self.stage.position_relative = -1*	self.down_direction*value		

	def clear_history(self):
		self.force_history = []
		self.position_history = []

	def read_loadcell(self):
		current_force = self.load_cell.data - self.zero_force 
		self.force_history.append(current_force)
		return current_force

	def read_stage_position(self):
		current_position = self.stage.position
		self.position_history.append(current_position)
		return current_position

	def read_loadcell_and_stage_position(self):
		return self.read_stage_position(),self.read_loadcell() 
		
	def zero(self):
		self.zero_force = self.load_cell.data

	def contact(self):
		measured_force =  self.data[0]
		while measured_force < self.contact_force:
			self.move_toward(self.step_increment)
			measured_force, measured_position = self.data
			print measured_position, measured_force
			sleep(self.time_delay)

	def uncontact(self):
		measured_force =  self.data[0]
		while measured_force > self.zero_force:
			self.move_apart(self.step_increment)
			measured_force, measured_position = self.data
			print measured_position, measured_force
			sleep(self.time_delay)

	def raiseup(self):
		self.stage.position_relative = -1*self.down_direction*self.raiseup_overshoot

	def lowerdown(self):
		self.stage.position_relative = self.down_direction*self.raiseup_overshoot*.7
		
	def record_network(self,filename=None):
		ntwk = self.vna.ch1.one_port
		if filename is None:
			filename = datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')
		ntwk.write_touchstone(self.file_dir+filename)

	
	def cycle_and_record_touchstone(self):
		self.touchdown()
		self.record_network()
		self.raiseup()
		sleep(1)
	def plot_data(self,**kwargs):
		plb.plot(self.position_history, self.force_history,**kwargs)
	def close(self):
		for k in [self.vna, self.stage, self.load_cell]:
			k.close()
