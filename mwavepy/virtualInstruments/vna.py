'''#       vna.py
#
#	This file holds all VNA models
#
#       Copyright 2010  lihan chen, alex arsenovic <arsenovic@virginia.edu>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
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

import numpy as npy
import visa
from visa import instrument

from ..frequency import *
from ..network import * 

class ZVA40_alex(object, instrument):
    '''
     zva40
    '''
    def __init__(self, address=20, channel=1,**kwargs):
	instrument.__init__('GPIB::'+str(address),**kwargs)
	self.channel=channel

    
    @property
    def f(self):
	return npy.array(self.ask_for_values( \
	    'CALCulate'+repr(self.channel)+':DATA:STIMulus?'))

    @property
    def frequency(self):
	return f_2_frequency(self.f)
    @property
    def s(self):
	return self.get_s()



    @property
    def continuous_sweep(self):
	raise NotImplementedError
	#somthing like this
	return self.ask('initiate:continuous ON')
    @continuous_sweep.setter
    def continuous_sweep(self, input):
	if input==True:
	    self.write('initiate:continuous ON')
	elif input==False:
	    self.write('initiate:continuous OFF')
	else:
	    raise ValueError ('input should be True or False')

    def get_data(self, dataFormat='sData'):
	'''
	takes:
		dataFormat: possible values are 'SDAT','FDAT','MDAT'
	returns:
		data, depends on what you asked for
		
	note:
		FDATa: Formatted trace data, according to the selected trace
		format (CALCulate<Chn>:FORMat). 1 value per trace point for
		Cartesian diagrams, 2 values for polar diagrams.
		
		sDATa: Unformatted trace data: Real and imaginary part of 
		each measurement point. 2 values per trace point 
		irrespective of the selected trace format. The trace 
		mathematics is not taken into account.
		
		MDATa: Unformatted trace data (see SDATa) after evaluation
		of the trace mathematics.

	'''
	
	if dataFormat[:4].upper() not in ['SDAT','FDAT','MDAT']:
	    raise ValueError('dataFormat not acceptable. see help')
	else:
	    return self.ask_for_values('CALCulate'+repr(self.channel)+':DATA? '+dataFormat[:4])
    
    def get_s(self,mn=None):
	'''
	    mn: index, like 11, 22, 12, if None then gets current
	'''
	# use param input to possibly change s-parameter
	rawData = npy.array(self.getData(dataFormat='SData',\
	    Ch=self.channel),dtype=complex)
	complexData = rawData[0::2]  + 1j*rawData[1::2]
	return complexData

    def get_network(self, number_of_ports=1,**kwargs):
	'''
	'''
	ntwk = Network(**kwargs)

	if number_of_ports ==1:
	    ntwk.s = self.s
	    ntwk.frequency = self.frequency
	else:
	    raise NotImplementedError
	    
class ZVA40(object):
    '''
    Created on Aug 3, 2010
    @author: Lihan
    
    This class is create to remote control Rohde & Schwarz by using pyvisa.
    For detail about visa please refer to:
	    http://pyvisa.sourceforge.net/
	    
    After installing the pyvisa and necessary driver (GPIB to USB driver, for instance), please follow the pyvisa manual to set up the module
    
    This class only has several methods. You can add as many methods as possible by reading the Network Analyzer manual
    Here is an example
    
    In the manual,
    
	    "CALC:DATA? FDAT"
	    
	This is the SCPI command
	
	    "Query the response values of the created trace. In the FDATa setting, N
	    comma-separated ASCII values are returned."
	    
	This descripes the function of the SCPI command above
	
	Since this command returns ASCII values, so we can use ask_for_values method in pyvisa
	
	temp=vna.ask_for_values('CALCulate1:DATA? SDATa')
	
	vna is a pyvisa.instrument instance    
	
	
    '''
    def __init__(self,address=20, **kwargs):
	    self.vna=visa.instrument('GPIB::'+str(address),**kwargs)
	    self.spara=npy.array([],dtype=complex)
	    
    def continuousOFF(self):
	    self.vna.write('initiate:continuous OFF')
	    

    def continuousON(self):
	    self.vna.write('initiate:continuous ON')
	    
    def displayON(self):
	    self.vna.write('system:display:update ON')
	    
    def setFreqBand(self,StartFreq,StopFreq):
	    '''
	    Set the frequency band in GHz
	    setFreqBand(500,750)
	    Start frequency 500GHz, Stop frequency 750GHz
	    '''
	    self.freqGHz=npy.linspace(StartFreq, StopFreq, 401)
	    self.vna.write('FREQ:STAR '+'StartFreq'+'GHz')
	    self.vna.write('FREQ:STOP '+'StopFreq'+'GHz')
	    
    def sweep(self):
	    '''
	    Initiate a sweep under continuous OFF mode
	    '''
	    self.vna.write('initiate;*WAI')
	    
    def getData(self):
	    '''
	    Get data from current trace
	    '''
	    temp=self.vna.ask_for_values('CALCulate1:DATA? SDATa')
	    temp=npy.array(temp)
	    temp.shape=(-1,2)
	    self.spara=temp[:,0]+1j*temp[:,1]
	    self.spara.shape=(-1,1,1)            #this array shape is compatible to Network Class
	    return self.spara
    
    def measure(self):
	    '''
	    Take one-port measurement
	    1.turn continuous mode off
	    2.initiate a single sweep
	    3.get the measurement data
	    4.turn continuous mode on
	    '''
	    self.continuousOFF()
	    self.sweep()
	    temp=self.getData()
	    self.continuousON()
	    return temp
    
    def saveSpara(self,fileName):
	    '''
	    Take one-port measurement and save the data as touchstone file, .s1p
	    '''
	    temp=self.spara
	    formatedData=npy.array([self.freqGHz[:],temp[:,0,0].real,temp[:,0,0].imag],dtype=float)
	    fid = open(fileName+'.s1p', 'w')
	    fid.write("# GHz S RI R 50\n")
	    npy.savetxt(fid,formatedData,fmt='%10.5f')
	    fid.close()
