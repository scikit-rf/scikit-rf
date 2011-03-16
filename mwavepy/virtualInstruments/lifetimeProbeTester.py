from time import sleep 
from datetime import datetime
import pylab as plb
import numpy as npy

from .futekLoadCell import *
from .stages import ESP300
from .vna import ZVA40_alex
import mwavepy as mv

class LifeTimeProbeTester(object):
	'''
		Object for CPW probe landing with loadcell force feedback
		support and VNA data retrieval.
	'''
	def __init__(self, stage=None, vna=None, load_cell=None, \
		down_direction=-1, step_increment =.001, contact_force=5,\
		delay=.5,raiseup_overshoot=.1,uncontact_gap = .005,\
		raiseup_velocity=10, zero_force_threshold=.05, \
		read_networks=False, file_dir = './'):
		'''
		takes:
			stage: a ESP300 object [None]
			vna: a ZVA_alex object [None]
			load_cell: a Futek_USB210_socket object [None]
			down_direction:
			step_increment:
			contact_force:
			delay: time delay passed to stage object, int[1]
			raisup_overshoot:
			file_dir:
		'''
		if stage is None: self.stage = ESP300()
		else: self.stage = stage

		if vna is None: self.vna = ZVA40_alex()
		else: self.vna = vna

		if load_cell is None: self.load_cell = Futek_USB210_socket()
		else: self.load_cell = load_cell
		
		self.down_direction = down_direction
		self.step_increment = step_increment
		self.contact_force = contact_force
		self.stage.delay = delay
		self.raiseup_overshoot = raiseup_overshoot
		self.file_dir=file_dir
		self.uncontact_gap =uncontact_gap
		self.raiseup_velocity = raiseup_velocity
		self.zero_force_threshold =zero_force_threshold
		self.read_networks = read_networks
		
		
		self.zero_force()
		self.zero_position()
		self.stage.motor_on = True
		self.force_history = []
		self.position_history = []
		self.ntwk_history = []
	
	@property
	def data(self):
		if self.read_networks:
			self.read_network()
		return self.read_loadcell_and_stage_position()
	
	@property
	def history(self):
		return ( self.position_history, self.force_history, \
			self.ntwk_history)
	
	def save_history(self, filename='force_vs_position.txt'):
		data= (npy.vstack((self.position_history, self.force_history)).T)
		npy.savetxt(filename, data)
		for ntwk in self.ntwk_history:
			ntwk.write_touchstone()
	
	def move_toward(self,value):
		self.stage.position_relative = self.down_direction*value

	def move_apart(self,value):
		self.stage.position_relative = -1*	self.down_direction*value		

	def clear_history(self):
		self.force_history = []
		self.position_history = []
		self.ntwk_history = []
		
	def read_loadcell(self):
		current_force = self.load_cell.data - self._zero_force 
		self.force_history.append(current_force)
		return current_force

	def read_stage_position(self):
		current_position = self.stage.position - self._zero_position
		self.position_history.append(current_position)
		return current_position

	def read_loadcell_and_stage_position(self):
		return self.read_stage_position(),self.read_loadcell() 
	
	def read_network(self,name=None):
		ntwk = self.vna.ch1.one_port
		if name is None:
			name = datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')
		print ('reading %s'%name)
		ntwk.name= name
		self.ntwk_history.append(ntwk)	
	def zero_force(self):
		self._zero_force = self.load_cell.data
	
	def zero_position(self):
		self._zero_position = self.stage.position
	
	def zero(self):
		self.zero_force()
		self.zero_position()
	
	def contact(self):
		print ('position\tforce')
		measured_position,measured_force = self.data
		print ('%f\t%f'% (measured_position, measured_force))
		while measured_force < self.contact_force:
			self.move_toward(self.step_increment)
			measured_position,measured_force = self.data
			print ('%f\t%f'% (measured_position, measured_force))
		print ('Contact!')
	
	def uncontact(self):
		print ('position\tforce')
		measured_position,measured_force = self.data
		print ('%f\t%f'% (measured_position, measured_force))
		while measured_force  > self.zero_force_threshold :
			self.move_apart(self.step_increment)
			measured_position,measured_force = self.data
			print ('%f\t%f'% (measured_position, measured_force))
		self.move_apart(self.uncontact_gap)
		print('Un-contacted.')	
	
	def raiseup(self):
		tmp_velocity = self.stage.velocity
		self.stage.velocity = self.raiseup_velocity
		self.move_apart (self.raiseup_overshoot)
		self.stage.velocity = tmp_velocity

	def lowerdown(self):
		tmp_velocity = self.stage.velocity
		self.stage.velocity = self.raiseup_velocity
		self.move_toward( self.raiseup_overshoot*.9)
		self.stage.velocity = tmp_velocity
		

	
	
	def cycle_and_record_touchstone(self):
		self.raiseup()
		self.record_network()
		self.lowerdown()
		self.contact()
		self.record_network()
		self.uncontact()
		
	def plot_data(self,**kwargs):
		plb.plot(self.position_history, self.force_history,**kwargs)
		plb.xlabel('Position [mm]')
		plb.ylabel('Force[mN]')
		
	def plot_electrical_data(self, dir='./',f_index=None, **kwargs):
		ntwks = mv.load_all_touchstones(dir=dir)
		keys = ntwks.keys()
		keys.sort()
		phase_at_f = [ntwks[key].s_deg[f_index,0,0] for key in keys]
		freq = ntwks[keys[0]].frequency
		
		if f_index is None:
			f_index = [int(freq.npoints/2)]
		for a_f_index in f_index:
			f = freq.f_scaled[a_f_index]
			f_unit = freq.unit
			plb.figure()
			plb.plot(npy.array(self.position_history)*1e3, phase_at_f, label='f=%i%s'%(f,f_unit),**kwargs)
			plb.xlabel('Position[um]')
			plb.ylabel('Phase [deg]')
			plb.legend()
			
			plb.figure()
			plb.plot(self.force_history, phase_at_f,label='f=%i%s'%(f,f_unit),**kwargs)
			plb.ylabel('Phase [deg]')
			plb.xlabel('Force[mN]')
			plb.legend()
		
	def close(self):
		for k in [self.vna, self.stage, self.load_cell]:
			k.close()
