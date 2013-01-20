from time import sleep
from datetime import datetime
import pylab as plb
import numpy as npy

from .futekLoadCell import *
from .stages import ESP300
from .vna import ZVA40_alex
import skrf as rf

class LifeTimeProbeTester(object):
    '''
            Object for CPW probe landing with loadcell force feedback
            support and VNA data retrieval.
    '''
    def __init__(self, stage=None, vna=None, load_cell=None, \
            down_direction=-1, step_increment =.001, contact_force=5,\
            delay=.5,raiseup_overshoot=.1,uncontact_gap = .005,\
            raiseup_velocity=10, zero_force_threshold=.05, \
            read_networks=False, file_dir = './', position_upper_limit=-10,cpw_line_spaceing = .145):
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
        if stage is None:
            self.stage = ESP300()
            self.stage2 = ESP300(current_axis=2)
        else:
            self.stage = stage


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
        self.position_upper_limit= position_upper_limit
        self.cpw_line_spacing = .145# inmm

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
        value = abs(value)
        if self.down_direction == 1:
            if self.stage.position + self.down_direction * value > \
            self.position_upper_limit:
                raise(ValueError('Safety Stop: Upper limit reached on stage'))
        elif self.down_direction == -1:
            if self.stage.position + self.down_direction * value < \
            self.position_upper_limit:
                raise(ValueError('Safety Stop: Upper limit reached on stage'))



        self.stage.position_relative = self.down_direction*value

    def move_toward_fast(self,value):
        tmp_velocity = self.stage.velocity
        self.stage.velocity = self.raiseup_velocity
        self.move_toward( value)
        self.stage.velocity = tmp_velocity


    def move_apart(self,value):
        self.stage.position_relative = -1*      self.down_direction*value
    def move_apart_fast(self,value):
        tmp_velocity = self.stage.velocity
        self.stage.velocity = self.raiseup_velocity
        self.move_apart( value)
        self.stage.velocity = tmp_velocity

    def move_left(self,n=1):
        self.stage2.position_relative= n*self.cpw_line_spacing
    def move_right(self,n=1):
        self.stage2.position_relative= -n*self.cpw_line_spacing
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

    def read_network(self,name=None, write=False):
        ntwk = self.vna.ch1.one_port
        if name is None:
            name = datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')
        print ('reading %s'%name)
        ntwk.name= name
        self.ntwk_history.append(ntwk)
        if write:
            ntwk.write_touchstone()

    def zero_force(self):
        self._zero_force = self.load_cell.data

    def zero_position(self):
        self._zero_position = self.stage.position

    def zero(self):
        self.zero_force()
        self.zero_position()
    def set_position_upper_limit(self, distance=.1):
        self.contact_sloppy()
        self.position_upper_limit = self.stage.position +\
                distance*self.down_direction
        print ('new hardware limit set to %f'%self.position_upper_limit)
        self.uncontact_sloppy()

    def contact(self):
        print ('position\tforce')
        measured_position,measured_force = self.data
        print ('%f\t%f'% (measured_position, measured_force))
        while measured_force < self.contact_force:
            self.move_toward(self.step_increment)
            measured_position,measured_force = self.data
            print ('%f\t%f'% (measured_position, measured_force))
        print ('Contact!')

    def contact_sloppy(self):
        print ('position\tforce')
        tmp_delay = self.stage.delay
        tmp_step_increment = self.step_increment
        self.stage.delay = .1
        self.step_increment =  .004# self.step_increment * 4
        measured_position,measured_force = self.data
        print ('%f\t%f'% (measured_position, measured_force))
        while measured_force < self.contact_force:
            self.move_toward(self.step_increment)
            measured_position,measured_force = self.data
            print ('%f\t%f'% (measured_position, measured_force))
        print ('Contact!')
        self.stage.delay = tmp_delay
        self.step_increment = tmp_step_increment

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

    def uncontact_sloppy(self):
        print ('position\tforce')
        tmp_delay = self.stage.delay
        tmp_step_increment = self.step_increment
        self.stage.delay = .1
        self.step_increment = .004#self.step_increment * 4
        measured_position,measured_force = self.data
        print ('%f\t%f'% (measured_position, measured_force))
        while measured_force  > self.zero_force_threshold :
            self.move_apart(self.step_increment)
            measured_position,measured_force = self.data
            print ('%f\t%f'% (measured_position, measured_force))
        self.move_apart(self.uncontact_gap)
        print ('Un-Contact.')

        self.stage.delay = tmp_delay
        self.step_increment = tmp_step_increment

    def raiseup(self):
        self.move_apart_fast(self.raiseup_overshoot)

    def lowerdown(self):
        self.move_toward_fast(self.raiseup_overshoot*.9)

    def cycle_and_record_touchstone(self):
        self.raiseup()
        self.record_network()
        self.lowerdown()
        self.contact()
        self.record_network()
        self.uncontact()

    def plot_data(self,**kwargs):
        plb.plot(npy.array(self.position_history)*1e3, self.force_history,**kwargs)
        plb.title('Force vs. Position')
        plb.xlabel('Position [um]')
        plb.ylabel('Force[mN]')

    def plot_electrical_data(self,f_index=None, **kwargs):
        freq = self.ntwk_history[0].frequency
        if f_index is None:
            #f_index = [int(freq.npoints/2)]
            f_index = int(freq.npoints/2)

        phase_at_f = npy.array([ntwk.s_deg[f_index,0,0] for ntwk in self.ntwk_history ])

        f = freq.f_scaled[f_index]
        f_unit = freq.unit

        plb.figure(2)
        plb.title('Phase vs Position')
        plb.plot(npy.array(self.position_history)*1e3, phase_at_f, label='f=%i%s'%(f,f_unit),**kwargs)
        plb.xlabel('Position[um]')
        plb.ylabel('Phase [deg]')
        plb.legend()

        plb.figure(3)
        plb.title('Phase vs Force')
        plb.plot(npy.array(self.position_history[1:])*1e3, npy.diff(phase_at_f), label='f=%i%s'%(f,f_unit),**kwargs)
        plb.xlabel('Position[um]')
        plb.ylabel('Phase Difference [deg]')
        plb.legend()

        plb.figure(4)
        plb.title('Phase Change vs Force')
        plb.plot(self.force_history, phase_at_f,label='f=%i%s'%(f,f_unit),**kwargs)
        plb.ylabel('Phase [deg]')
        plb.xlabel('Force[mN]')
        plb.legend()

        plb.figure(5)
        plb.title('Phase Change vs Position')
        plb.plot(self.force_history[1:], npy.diff(phase_at_f),label='f=%i%s'%(f,f_unit),**kwargs)
        plb.ylabel('Phase Difference[deg]')
        plb.xlabel('Force[mN]')
        plb.legend()
    def monitor(self):
        while (1):
            ax = plb.gca()
            a.read_loadcell_and_stage_position()
            ax.clear()
            ax.plot(range(len(a.force_history)), a.force_history, marker='o')
            plb.draw()

    def beat_probe_up(self,total_beatings=1000, beatings_per_spot=100, go_left=True):
        slide_counter = 0
        for current_beating in range(total_beatings):
            if current_beating%beatings_per_spot == 0:
                slide_counter+=1
                if go_left:
                    self.move_left()
                else:
                    self.move_right()
            self.contact_force -=5
            self.contact_sloppy()
            self.contact_force +=5
            self.contact()
            self.uncontact_sloppy()
            print('landing #%i'%current_beating)


        if go_left:
            self.move_right(slide_counter)
        else:
            self.move_left(slide_counter)


    def close(self):
        for k in [self.vna, self.stage, self.load_cell]:
            k.close()
