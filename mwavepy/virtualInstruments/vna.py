#	   vna.py
#
#	This file holds all VNA models
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
holds class's for VNA virtual instruments
'''
import numpy as npy
import visa
from visa import GpibInstrument

from ..frequency import *
from ..network import * 

class PNAX(GpibInstrument):
	'''
	Agilent PNAX
	'''
	def __init__(self, address=16, channel=1,**kwargs):
		GpibInstrument.__init__(self,'GPIB::'+str(address),**kwargs)
		self.channel=channel
		self.write('calc:par:sel CH1_S11_1')
	
	@property
	def continuous(self):
		raise NotImplementedError
	
	@continuous.setter
	def continuous(self, mode):
		self.write('initiate:continuous '+ mode)
	
	@property
	def frequency(self, unit='ghz'):
		freq=Frequency( float(self.ask('sens:FREQ:STAR?')),
			float(self.ask('sens:FREQ:STOP?')),\
			int(self.ask('sens:sweep:POIN?')),'hz')
		freq.unit = unit
		return freq
	
	
	@property
	def network(self):
		'''
		Initiates a sweep and returns a  Network type represting the
		data.
		
		if you are taking multiple sweeps, and want the sweep timing to
		work, put the turn continuous mode off. like pnax.continuous='off'
		'''
		self.write('init:imm')
		self.write('*wai')
		s = npy.array(self.ask_for_values('CALCulate1:DATA? SDATa'))
		s.shape=(-1,2)
		s =  s[:,0]+1j*s[:,1]
		ntwk = Network()
		ntwk.s = s
		ntwk.frequency= self.frequency 
		return ntwk
		
		
class ZVA40(object):
	'''
	Created on Aug 3, 2010
	@author: Lihan
	
	This class is create to remote control Rohde & Schwarz by using pyvisa.
	For detail about visa please refer to:
		http://pyvisa.sourceforge.net/
		
	After installing the pyvisa and necessary driver (GPIB to USB driver,
	 for instance), please follow the pyvisa manual to set up the module
	
	This class only has several methods. You can add as many methods
	as possible by reading the Network Analyzer manual
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
		self.spara.shape=(-1,1,1)			#this array shape is compatible to Network Class
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


class HP8510C(GpibInstrument):
	'''
	good ole 8510
	'''
	def __init__(self, address=16,**kwargs):
		GpibInstrument.__init__(self,'GPIB::'+str(address),**kwargs)
		self.write('FORM4;')
	

	@property
	def error(self):
		return self.ask('OUTPERRO')
	@property
	def continuous(self):
		raise NotImplementedError
	
	@continuous.setter
	def continuous(self, choice):
		if choice:
			self.write('CONT;')
		elif not choice:
			self.write('SING;')
		else:
			raise(ValueError('takes a boolean'))
	@property
	def averaging(self):
		'''
		averaging factor
		'''
		raise NotImplementedError
	
	@averaging.setter
	def averaging(self, factor ):
		self.write('AVERON %i;'%factor )
			
	@property
	def frequency(self, unit='ghz'):
		freq=Frequency( float(self.ask('star;outpacti;')),
			float(self.ask('stop;outpacti;')),\
			int(float(self.ask('poin;outpacti;'))),'hz')
		freq.unit = unit
		return freq
	
	
	@property
	def one_port(self):
		'''
		Initiates a sweep and returns a  Network type represting the
		data.
		
		if you are taking multiple sweeps, and want the sweep timing to
		work, put the turn continuous mode off. like pnax.continuous='off'
		'''
		self.continuous = False
		s = npy.array(self.ask_for_values('OUTPDATA'))
		s.shape=(-1,2)
		s =  s[:,0]+1j*s[:,1]
		ntwk = Network()
		ntwk.s = s
		ntwk.frequency= self.frequency 
		return ntwk

	@property
	def two_port(self):
		'''
		Initiates a sweep and returns a  Network type represting the
		data.
		
		if you are taking multiple sweeps, and want the sweep timing to
		work, put the turn continuous mode off. like pnax.continuous='off'
		'''
		self.continuous = False
		s_dict={}
		for param in ['s11','s12','s21','s22']:
			self.write(param+';')
			s = npy.array(self.ask_for_values('OUTPDATA;'))
			s.shape=(-1,2)
			s_dict[param] =  s[:,0]+1j*s[:,1]
		ntwk = Network()
		ntwk.s = npy.array([[s_dict['s11'], s_dict['s12']],\
			[s_dict['s21'], s_dict['s22']]]).transpose().reshape(-1,2,2)
		ntwk.frequency= self.frequency 
		return ntwk
	##properties for the super lazy
	@property
	def s11(self):
		self.write('s11;')
		ntwk =  self.one_port
		ntwk.name = 'S11'
		return ntwk
	@property
	def s22(self):
		self.write('s22;')
		ntwk =  self.one_port
		ntwk.name = 'S22'
		return ntwk
	@property
	def s12(self):
		self.write('s12;')
		ntwk =  self.one_port
		ntwk.name = 'S12'
		return ntwk
	@property
	def s21(self):
		self.write('s21;')
		ntwk =  self.one_port
		ntwk.name = 'S21'
		return ntwk
