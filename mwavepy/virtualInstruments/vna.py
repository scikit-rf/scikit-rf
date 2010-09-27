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

import numpy as np
import visa

class ZVA40(object):
	def __init__(self,address=20, **kwargs):
		self.vna=visa.instrument('GPIB::'+str(address),**kwargs)
		self.spara=np.array([],dtype=complex)
		
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
		self.freqGHz=np.linspace(StartFreq, StopFreq, 401)
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
		temp=np.array(temp)
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
		formatedData=np.array([self.freqGHz[:],temp[:,0,0].real,temp[:,0,0].imag],dtype=float)
		fid = open(fileName+'.s1p', 'w')
		fid.write("# GHz S RI R 50\n")
		np.savetxt(fid,formatedData,fmt='%10.5f')
		fid.close()
