'''
#       vna.py
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
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

#import visa
import pythics.libinstrument
import mwavepy as m	
import numpy as n
from time import sleep 

# see http://pyvisa.sourceforge.net/pyvisa/node10.html for GPIB methods
GHz= 1e9

class hp8720c(pythics.libinstrument.GPIBInstrument):
	'''
	Virtual Instrument for HP8720C model VNA.
	'''
	def __init__(self, *args, **kwargs):
		'''
		default constructor. common options are 
			"GPIB::$addressNumber"
			timeout=$GPIBTimeout
		
		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)
	
	
		
	def setS(self, input,opc=True):
		'''
		makes the vna measure the desiredS s-parameterS.
		desiredS - a string. may be either s11,s12,s21, or s22
		opc=True/False  sends a "OPC?" to vna after setting s-parame, 
			which waits for it to switch, before continuing (doesnt work on 8510)
		
		'''	
		self.write(input + ';')
		if opc:
			self.ask("OPC?;SING;")
		
		# learn how to call these methods from within methodss waitOPC()
		return None
				
	
	def getData(self):
		'''
		returns output from 'OUTPDATA' in array format
		'''
		data = n.array(self.ask_for_values("OUTPDATA;"))
		return data
		
	def getS(self,sParam=None):
		'''
		returns an s-parameter type, which represents current S-param data,
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		if sParam != None:
			setS(sParam)
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		data = self.getData()
		#seperate into real and imageinary parts
		sReal = data[0:len(data):2]
		sImag = data[1:len(data):2]
		
		s = sReal +1j*sImag
		return s
	
	def getOnePort(self):
		'''
		TODO: could ask for current s-param and put it in the name field. 
		'''
		freq= getFrequencyAxis()
		s = getS()
		return m.ntwk(data=s, paramType='s',freq=freq)
		
	def getTwoPort(self):
		'''
		returns an two-port type, which represents current DUT
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
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
		TODO: should ask vna for freqUnit
		'''
		fStart = float(self.ask("star?"))/freqUnit
		fStop = float(self.ask("stop?"))/freqUnit
		fPoints = int(float(self.ask("poin?")))
		# setup frequency list, in units of freqUnit
		freqs = n.linspace(fStart,fStop,fPoints)
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
	'''
	def __init__(self, *args, **kwargs):
		'''
		default constructor. common options are 
			"GPIB::$addressNumber"
			timeout=
		
		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)
	
	
		
	def setS(self, input):
		'''
		makes the vna measure the desiredS s-parameterS.
		desiredS - a string. may be either s11,s12,s21, or s22
		uses the a "SING;" command for timing
		
		'''	
		self.write(input + ';sing;')
		
		# learn how to call these methods from within methodss waitOPC()
		return None
				
	
	def getData(self):
		'''
		returns output from 'OUTPDATA' in array format
		'''
		data = n.array(self.ask_for_values("OUTPDATA;"))
		return data
		
	def getS(self, freqArray=[],z0=50):
		'''
		returns an s-parameter type, which represents current S-param data,
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		freqArray = self.getFrequencyAxis()
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		data = self.getData()
		#seperate into real and imageinary parts
		sReal = data[0:len(data):2]
		sImag = data[1:len(data):2]
		# call s-parameter constructor
		return m.s('re',freqArray,'GHz', sReal,sImag,z0)
		
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
		freqs = n.linspace(fStart,fStop,fPoints)
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
		

	
	
	
