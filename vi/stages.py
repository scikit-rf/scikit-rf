

'''

.. module:: skrf.vi.stages
++++++++++++++++++++++++++++++++++++++++++++++++++++
Stages  (:mod:`skrf.vi.stages`)
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autosummary::
    :toctree: generated/

    ESP300

'''
from time import sleep
import numpy as npy

from . ivihack import Driver


class ESP300(Driver):
    '''
    Newport Universal Motion Controller/Driver Model ESP300

    all axis control commands are sent to the number axis given by the
    local variable self.current_axis. An example usage ::

        from skrf.vi.stages import ESP300
        esp = ESP300()
        esp.current_axis = 1
        esp.position = 10
        print(esp.position)
    '''
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

    def __init__(self, address=1, current_axis=1,\
            always_wait_for_stop=True,delay=0,**kwargs):
        '''
        Initializer

        Parameters
        -------------
        address :   int
            Gpib address
        current_axis :   int
            number of current axis
        always_wait_for_stop :   Boolean
            wait for stage to stop before
            returning control to calling program
        \*\*kwargs :
            passed to GpibInstrument initializer
        '''

        if isinstance(address,int):
            resource = 'GPIB::%i::INSTR'%address
        else:
            resource = address

        Driver.__init__(self,resource = resource, **kwargs)
        self.current_axis = current_axis
        self.always_wait_for_stop = always_wait_for_stop
        self.delay=delay
    @property
    def current_axis(self):
        '''
        current axis used in all subsequent commands
        '''
        return self._current_axis
    @current_axis.setter
    def current_axis(self, input):
        '''
        takes:
                input:  desired current axis number, int []
        '''
        self._current_axis = input


    @property
    def velocity(self):
        '''
        the velocity of current axis
        '''
        command_string = 'VA'
        return (float(self.ask('%i%s?'%(self.current_axis,command_string))))
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
        if self.always_wait_for_stop:
            self.wait_for_stop()

    @property
    def position(self):
        command_string = 'TP'
        return float(self.ask('%i%s'%(self.current_axis,command_string)))
    @position.setter
    def position(self,input):
        '''
        set the position of current axis to input
        '''
        command_string = 'PA'
        self.write('%i%s%f'%(self.current_axis,command_string,input))
        if self.always_wait_for_stop:
            self.wait_for_stop()
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
    @units.setter
    def units(self, input):
        '''
         set axis units for all commands.
         takes:
                input: a string, describing the units here are a list of
                        possibilities.
                        'encoder count'
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
            self.write('%i%s'%(self.current_axis,command_string))
        if not input:
            command_string = 'MF'
            self.write('%i%s'%(self.current_axis,command_string))

    def send_stop(self):
        command_string = 'ST'
        self.write('%i%s'%(self.current_axis,command_string))

    def wait_for_stop(self):
        command_string = 'WS'
        self.write('%i%s%i'%(self.current_axis,command_string, 1e3*self.delay))
        tmp = self.position
