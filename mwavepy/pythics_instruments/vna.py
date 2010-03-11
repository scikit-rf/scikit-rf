'''
#       vna.py
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
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

# Builtin imports
from time import sleep 

# Depencies
try:
	import pythics.libinstrument
except:
	raise ImportError ('Depedent Packages not found. Please install pythics')
try:
	import numpy as npy
except:
	raise ImportError ('Depedent Packages not found. Please install: numpy')
	
# local imports
import mwavepy as m	


def realTimeDb():
	clf()
	myvna.getNtwk().plotdB()
	ax = gca()
	trace = ax.get_lines()[1]
	
	myvna.setSweepType(continuousOn =False)
		
	for k in range(2):
		myvna.startSweep(wait=False, opc=True)
		aNtwk = myvna.getNtwk()
		trace.set_ydata(aNtwk.sdB)
		draw()

# see http://pyvisa.sourceforge.net/pyvisa/node10.html for GPIB methods
# see http://sigma.ucsd.edu/Facilities/manuals/8510C.pdf
GHz= 1e9

class rszva40(pythics.libinstrument.GPIBInstrument):
	def __init__(self, *args, **kwargs):
		'''
		default constructor see pyvisa for more info.
		
		example:
			myvna = rszva("GPIB::20", timeout=10)
		where:
			"GPIB::20", where 16 is the GPIB address number
			timeout=10, timeout of GPIB commands, in seconds
		
		see http://pyvisa.sourceforge.net/pyvisa/node10.html for GPIB 
		methods

		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)

	
	
	def getNtwk(self, nPorts=1, Ch=1, calOff = False,**kwargs):
		if calOff:
			self.correctionOn(False)
		else:
			self.correctionOn(True)
			
		frequencyAxis = self.getFrequencyAxis(Ch=Ch)
		sParameter = self.getSParameter()
		return m.ntwk(data=sParameter, paramType='s', freq = frequencyAxis,\
			freqMultiplier=1e9, freqUnit='GHz',**kwargs)
	
	
	def correctionOn(self,value = True,Ch=1):
		if value:
			self.write('SENS'+repr(Ch)+':CORR ON')
		else:
			self.write('SENS'+repr(Ch)+':CORR OFF')
		return None
		
	def getCalCoefs(self):
		'''
		
		
		returns: 
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
				

		'''
		self.setSweepType(continuousOn=False)
		e00 = npy.array(self.ask_for_values('SENS1:DATA? \'SCORR1\''),dtype=complex)
		e11 = npy.array(self.ask_for_values('SENS1:DATA? \'SCORR2\''),dtype=complex)
		e10e01 = npy.array(self.ask_for_values('SENS1:DATA? \'SCORR3\''),dtype=complex)
		
		a = e10e01-e00*e11
		b = e00
		c = e11
		
		abc = npy.hstack((a,b,c))
		
	
	def getSParameter(self,param=None,Ch=1, **kwargs):
		# use param input to possibly change s-parameter
		rawData = npy.array(self.getData(dataFormat='SData',Ch=Ch),dtype=complex)
		complexData = rawData[0::2]  + 1j*rawData[1::2]
		return complexData
		
	
	
	def getData(self, dataFormat='sData',Ch=1):
		'''
		takes:
			dataFormat can be ['SDAT','FDAT','MDAT']:
		returns:
			data
			
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
			return self.ask_for_values('CALCulate'+repr(Ch)+':DATA? '+dataFormat[:4])
	
	def getFrequencyAxis(self,Ch=1):
		return npy.array(self.ask_for_values('CALCulate'+repr(Ch)+':DATA:STIMulus?'))
		
	
	def changeFormat(self, newFormat=None, Ch=1):
		'''
		changes format type of active data trace.
		
		takes:
			newFormat: SCPI command for new format type. can be any one
				in formatsList  = ['MLIN','MLOG','PHAS','UPH','POL','SMIT',
				'ISM','GDEL','REAL','IMAG','SWR']
		returns:
			None:
			
		'''
		formatsList  = ['MLIN','MLOG','PHAS','UPH','POL','SMIT','ISM','GDEL','REAL','IMAG','SWR']
		if newFormat == None:
			return self.ask('CALCulate' + repr(Ch)+':FORM?')
		else:
			self.write('CALCulate' + repr(Ch) +':FORM ' + newFormat)
		return None
		
	
		
	def reset(self):
		self.write('*RST')	 
		return None
		
	def wait(self):
		'''
		Wait to continue;
		WAIt to continue prevents servicing of the subsequent commands
		no query until all preceding commands have been executed and all
		signals have settled (see also command synchronization and *OPC).

		'''
		self.write('*WAIt')	 
		return None
		
	def opc(self):
		return self.ask('*OPC')
	def clear(self):
		self.write('*CLS')
	
	def setSweepType(self, continuousOn=True,Ch=1):
		if continuousOn == True:
			self.write('INITiate'+repr(Ch)+':CONTinuous ON')
		elif continuousOn == False:
			self.write('INITiate'+repr(Ch)+':CONTinuous OFF')	
		else:
			raise ValueError('continousOn must be a boolean.')
		return None
	
	def setSweepScope(self,value,Ch=1):
		'''
		takes:
			value: scope type for sweep [acceptable values:
			'ALL','SING','SINGLE']
		returns:
			None
		
		
		'''
		if value in ['ALL','SING','SINGLE']:
			self.write('INITiate'+repr(Ch)+':SCOPe ' + value)
		else:
			raise ValueError('bad value: see help for acceptable values ')
		return None
		
	def startSweep(self,Ch=1,wait=False,opc=True):
		if wait == True and opc==False:
			self.write('INITiate'+repr(Ch)+':IMMediate; *WAIt')
		
		elif wait == False and opc==True:
			if not self.ask('INITiate'+repr(Ch)+':IMMediate; *OPC?'):
				raise GeneralError('didnt recieve OPC answer')
			
		elif wait == False and opc==False:
			self.write('INITiate'+repr(Ch)+':IMMediate;')
		
		else:
			raise ValueError('cant do both wait and opc. or bad input.')
		
		return None	
		
		
	
class hp8720c(pythics.libinstrument.GPIBInstrument):
	'''
	Virtual Instrument for HP8720C model VNA.
	'''
	def __init__(self, *args, **kwargs):
		'''
		default constructor see pyvisa for more info.
		
		example:
			myvna = hp8720c("GPIB::16", timeout=10)
		where:
			"GPIB::16", where 16 is the GPIB address number
			timeout=10, timeout of GPIB commands, in seconds
		
		see http://pyvisa.sourceforge.net/pyvisa/node10.html for GPIB 
		methods

		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)
	
		
	def setS(self, sParam):
		'''
		makes the vna switch the active s-parameter to  the desiredS s-parameterS.
		
		sParam- a string. may be either s11,s12,s21, or s22
		
		'''	
		self.write(input + ';')
		# learn how to call these methods from within methodss waitOPC()
		return None
				
	
	def getData(self):
		'''
		returns output from 'OUTPDATA' in array format
		'''
		data = npy.array(self.ask_for_values("OUTPDATA;"))
		return data
		
	def getS(self,sParam=None, opc=True):
		'''
		returns an s-parameter type, which represents current S-param data,
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		if sParam != None:
			self.setS(sParam)
			
		if opc:
			self.ask("OPC?;SING;")
			
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		data = self.getData()
		#seperate into real and imageinary parts
		sReal = data[0:len(data):2]
		sImag = data[1:len(data):2]
		
		s = sReal +1j*sImag
		return s
	
	def getOnePort(self, sParam=None, opc=True):
		'''
		gets a 1-port s-parameter ntwk
		
		takes: 
			sParam - string of which s-parameter to get 
		returns:
			mwavepy.ntwk 1-port object
		'''	
		freq= getFrequencyAxis()
		if sParame != None:
			s = getS(sParam = sParam, opc=opc)
		else:
			s = getS( opc=opc)
		return m.ntwk(data=s, paramType='s',freq=freq)
		
	def getTwoPort(self):
		'''
		returns an mwavepy.ntwk 2-port type, which represents current 
		DUT
		'''
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		
		freq = self.getFrequencyAxis()
		s11 = self.getS('S11')	
		s21 = self.getS('S21')
		s12 = self.getS('S12')
		s22 = self.getS('S22')
		
		s = npy.array([[s11, s12],[s21, s22]])
		# reshape to kx2x2
		s = s.transpose().reshape(-1,2,2)
		# call s-parameter constructor
		return m.ntwk(data=s, freq=freq, paramType='s')
		
	def getFrequencyAxis(self, freqUnit=1e9):
		'''
		returns the current frequency axis. optionally scales freq 
		by 1/freqUnit
		TODO: should ask vna for freqUnit
		'''
		fStart = float(self.ask("star?"))/freqUnit
		fStop = float(self.ask("stop?"))/freqUnit
		fPoints = int(float(self.ask("poin?")))
		# setup frequency list, in units of freqUnit
		freqs = npy.linspace(fStart,fStop,fPoints)
		return freqs
	
	
	def getError(self):
		'''
		returns string, which contains error codes in the instrument
		'''
		error = self.ask("OUTPERRO")
		return error
	
	def putInContSweepMode(self):
		''' 
		place machine back in continuous sweep, and local operation
		'''
		self.write("CONT;")	
		return None
		
	def putInSingSweepMode(self):
		''' 
		place machine back in continuous sweep, and local operation
		'''
		self.write("SING;")	
		return None
	
	def setDataFormatAscii(self):
		'''
		TODO: verify what this does
		set output data in ascii and re/im pairs
		'''
		self.write("FORM4;")
		return None
		
	def putInHold(self):
		'''
		put machine in hold mode (not workable from front panel)
		'''
		self.write("HOLD;")
		return None
		
	def waitOPC(self):
		'''
		waits for VNA to respond. usefule for timing things.  puts vna into single sweep mode. 
		ex: 
			myVNA.__setS('s11')
		'''
		self.ask("OPC?;SING;")#wait for vna to switch
	
	
	
	
class hp8510c(pythics.libinstrument.GPIBInstrument):
	'''
	Virtual Instrument for HP8720C model VNA.
	
	see http://sigma.ucsd.edu/Facilities/manuals/8510C.pdf
	'''
	def __init__(self, *args, **kwargs):
		'''
		default constructor. common options are 
			"GPIB::$addressNumber"
			timeout=
		
		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)
	
	
		
	def setS(self, sParam):
		'''
		sets the vna to measure the desired s-parameter.
		
		takes:
			sParam - a string. may be either s11,s12,s21, or s22
			sing - boolean, if true, uses the a "SING;" command for 
				which waits for 1 sweep to finish before any command can
				run. 
		
		'''	
		self.write(sParam )
		
		# learn how to call these methods from within methodss waitOPC()
		return None
				
	
	def getData(self):
		'''
		returns output from 'OUTPDATA' in array format
		'''
		data = npy.array(self.ask_for_values("OUTPDATA;"))
		return data
		
	def getS(self, sParam=None, sing=True):
		'''
		
		
		'''
		freq = self.getFrequencyAxis()
		if sParam != None:
			self.setS(sParam)
		if sing == True:
			self.write('sing;')
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		data = self.getData()
		#seperate into real and imageinary parts
		sReal = data[0:len(data):2]
		sImag = data[1:len(data):2]
		# call s-parameter constructor
		return m.ntwk(data=data, freq=freq, paramType='s')
		
	def getTwoPort(self, freqArray=[],z0=50):
		'''
		returns an two-port type, which represents current DUT
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		freqArray = self.getFrequencyAxis()
		self.setS('s11') 
		s11 = self.getS(freqArray,z0)	
		self.setS('s21') 
		s21 = self.getS(freqArray,z0)
		self.setS('s12') 
		s12 = self.getS(freqArray,z0)
		self.setS('s22') 
		s22 = self.getS(freqArray,z0)
		# call s-parameter constructor
		return m.twoPort (s11,s12,s21,s22)
		
	def getFrequencyAxis(self, freqUnit=1e9):
		'''
		returns the current frequency axis.
		optionally scales freq by 1/freqUnit
		'''
		fStart = float(self.ask("star;outpacti;"))/freqUnit
		fStop = float(self.ask("stop;outpacti;"))/freqUnit
		fPoints = int(float(self.ask("poin;outpacti;")))
		# setup frequency list, in units of freqUnit
		freqs = npy.linspace(fStart,fStop,fPoints)
		return freqs
	
	
	def getError(self):
		'''
		returns string, which contains error codes in the instrument
		'''
		error = self.ask("OUTPERRO")
		return error
	
	def putInContSweepMode(self):
		''' 
		place machine back in continuous sweep, and local operation
		'''
		self.write("CONT;")	
		return None
		
	def putInSingSweepMode(self):
		''' 
		place machine back in continuous sweep, and local operation
		'''
		self.write("SING;")	
		return None
	
	def setDataFormatAscii(self):
		'''
		TODO: verify what this does
		set output data in ascii and re/im pairs
		'''
		self.write("FORM4;")
		return None
		
	def putInHold(self):
		'''
		put machine in hold mode (not workable from front panel)
		'''
		self.write("HOLD;")
		return None
		

	
	
	
