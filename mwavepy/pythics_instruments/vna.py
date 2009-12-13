import visa
import pythics.libinstrument
import mwavepy as m	
import numpy as n


# see http://pyvisa.sourceforge.net/pyvisa/node10.html for GPIB methods
GHz= 1e9

class hp8720c(pythics.libinstrument.GPIBInstrument):
	def __init__(self, *args, **kwargs):
		'''
		default constructor. 
		'''
		pythics.libinstrument.GPIBInstrument.__init__(self, *args, **kwargs)
		# DO WE NEED TO INITIALIZE SOME DEFAULT VALUES?
		
	
	
		
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
		
	def getS(self, freqArray=[],z0=50):
		'''
		returns an s-parameter type, which represents current S-param data,
		
		freqArray - is the frequency array assigned to the s-parameter
		z0 - is the characteristic impedance to assign to the s-parameter
		'''
		# Long sweeps may create time outs, see pyvisa page for solving this
		# using contructor  options
		data = self.getData()
		#seperate into real and imageinary parts
		sReal = data[0:len(data):2]
		sImag = data[1:len(data):2]
		# call s-parameter constructor
		return m.s('re',freqArray,'GHz', sReal,sImag,z0)
		
	def getFrequencyAxis(self, freqUnit=1e9):
		'''
		returns the current frequency axis.
		optionally scales freq by 1/freqUnit
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
	
	
	
	#doesnt belong in instrument VI
	
