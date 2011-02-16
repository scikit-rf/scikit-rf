
from .futekLoadCell import *
from .stages import ESP300
from .vna import ZVA40_alex

class LifeTimeProbeTester(object):
	def __init__(self):
		self.stage = ESP300()
		self.vna = ZVA40_alex()
		self.load_cell = Futek_USB210()

		self.down_direction = -1
		self.step_increment = .001

	@property
	def contact_force(self):
		return self._contact_force
	@contact_force.setter
	def contact_force(self,value):
		self._contact_force = value

	def touchdown(self):
		measured_force = self.load_cell.data
		while measured_force < contact_force:
			self.stage.position_relative = self.down_direction*self.step_increment
			self.measured_force = self.load_cell.data
