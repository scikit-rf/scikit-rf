'''
This module have eritten as base class for remote control of R&S instruments. 
It based on vna_pyvisa.py of scikit-rf packege. Code was written and tested with PyVisa 1.8.
'''

import re

import numpy as np
from time import sleep
from skrf import Network, Frequency
from skrf import mathFunctions as mf
from skrf.calibration.calibration import Calibration, SOLT, OnePort, \
                                      convert_pnacoefs_2_skrf,\
                                      convert_skrfcoefs_2_pna
import visa
import pyvisa.resources.messagebased


class viRs:
    """
    The base class for remote control Rohde&Schwarz instruments.
    Class contains basic T&M instrumrnts properties and comands.

    """
    
    def __init__(self, address, channel=1, timeout=10, echo=False,
                 front_panel_lockout=False, **kwargs):
        '''
        Constructor

        Parameters
        -------------
        address : int or str
            GPIB address , or resource string
        channel : int
            set active channel. Most commands operate on the active channel
        timeout : number
            GPIB command timeout in seconds.
        echo : Boolean
            echo  all strings passed to the write command to stdout.
            useful for troubleshooting
        front_panel_lockout : Boolean
            lockout front panel during operation.
        \*\*kwargs :
            passed to  `visa.Driver.__init__`  - in work

        '''
#        visa.__init__(self,resource = resource, **kwargs)
        
        #Initiate for GPIB device
        if isinstance(address,int):
            resource = 'GPIB::%i::INSTR' % address
        else:
            resource = address
        
        rm = visa.ResourceManager()
        self.resource = rm.open_resource(resource)  # type: pyvisa.resources.messagebased.MessageBasedResource
        
        
        if "socket" in resource.lower():
            #Didn't tested, just copied.
            self.resource.read_termination = "\n"
            self.resource.write_termination = "\n"

        if "gpib" in resource.lower():
            #Didn't tested, just copied.
            self.resource.control_ren(2)

        self.channel = channel
        self.port = 1
        self.echo = echo
        self.timeout = timeout
        if not front_panel_lockout:
            pass  # self.gtl()
       
        
    '''I don't work how does code comennted below work exacttlly.
       Plans fore future!
    '''
    #def __enter__(self):
    #    return self
    
    #def __exit__(self, exc_type, exc_val, exc_tb):
    #    self.resource.close()
    
    @property
    def timeout(self):
        return self.resource.timeout / 1000.

    @timeout.setter
    def timeout(self, val):
        self.resource.timeout = val * 1000.
    
    @property
    def idn(self):
        '''
        Identifying string for the instrument
        '''
        return self.resource.query('*IDN?')

    def opc(self):
        '''
        Ask for indication that operations complete
        '''
        return self.resource.query('*OPC?')

    def wai(self):
        """
        Write *WAI to R&S instrumets.
        """
        self.write("*WAI")
        
    #@property
    def reset(self):
        '''
        reset
        '''
        self.resource.write('*RST;')
        if self.echo:
            print('*RST;')

    def write(self, msg, *args, **kwargs):
        '''
        Write a msg to the instrument.
        '''
        if self.echo:
            print(msg)
        return self.resource.write(msg, *args, **kwargs)


    def read(self, msg, *args, **kwargs):
        '''
        Read a msg from the instrument.
        '''
        if self.echo:
            print(msg)
        return self.resource.query(msg, *args, **kwargs)


    def query(self, msg, *args, **kwargs):
        '''
        query()
        data = hmp.query('MEAS:CURR?')
        
        Queru - write and read a msg to the instrument.
        '''
        if self.echo:
            print(msg)
        return self.resource.query(msg, *args, **kwargs)

    def query_ascii_values(self, msg, *args, **kwargs):
        '''
        Write a msg to the instrument.
        '''
        if self.echo:
            print(msg)
        return self.resource.query_ascii_values(msg, *args, **kwargs)


    def open(self, *args, **kwargs):
        '''
        open()
        Open existig VISA resource.
        See olso close()
        '''
        if self.echo:
            print(str(self)+'.open()')
        #return self.resource.query(msg, *args, **kwargs)

    def close(self, *args, **kwargs):
        '''
        close()
        Close existig VISA resource.zv
        See olso open()
        '''
        if self.echo:
            print(str(self)+'.close()')
        return self.resource.close()
