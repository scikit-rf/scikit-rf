from time import sleep 
from datetime import datetime
from .futekLoadCell import *
from .stages import ESP300
from .vna import ZVA40_alex

class LifeTimeProbeTester(object):
	def __init__(self, stage=None, vna=None, load_cell=None, \
		down_direction=-1, step_increment =.001, contact_force=.1,\
		time_delay=.01,raiseup_overshoot=.1,file_dir = './'):

		if stage is None: self.stage = ESP300()
		else: self.stage = stage

		if vna is None: self.vna = ZVA40_alex()
		else: self.vna = vna

		if load_cell is None: self.load_cell = Futek_USB210()
		else: self.load_cell = load_cell
		
		self.down_direction = down_direction
		self.step_increment = step_increment
		self.contact_force = contact_force
		self.time_delay = time_delay
		self.raiseup_overshoot = raiseup_overshoot
		self.file_dir=file_dir

		self.zero_force = self.load_cell_data
		

	def touchdown(self):
		measured_force = self.load_cell.data
		while measured_force < self.contact_force:
			self.stage.position_relative = self.down_direction*self.step_increment
			self.measured_force = self.load_cell.data
			sleep(time_delay)

	def raiseup(self):
		measured_force = self.load_cell.data
		while measured_force > self.zero_force:
			self.stage.position_relative = -1*self.down_direction*self.step_increment
			self.measured_force = self.load_cell.data
			sleep(time_delay)
		self.stage.position_relative = -1*self.down_direction*self.raiseup_overshoot
	
	def record_network(self,filename=None):
		ntwk = self.vna.one_port
		if filename is None:
			date_string = datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')
		ntwk.write_touchstone(self.file_dir+filename)
	
	def cycle_and_record_touchstone(self):
		self.touchown()
		self.record_network()
		self.raiseup()
