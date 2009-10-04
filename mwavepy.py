'''
Filename: mwavepy.py
Author: alex arsenovic
Data: 04/04/09
Requires: numpy, matplotlib (aka pylab)

Summary: This is a collection of datatypes and functions which aid in microwave engineering.
there are a few classes such as s, z which represent scattering and impedance parameters. There
is also a twoPort and onePort class, which represent networks. The main use of these functions is 
for analyzing data gotten from network analyzers or simulation programs. The standard format for 
such parameters is Touchtone file format. There is a function which reads a standard formated 
touchtone file and creates an object representing the data within the file. 


Examples: 

Loading a touchtone file:
	import mwavepy as m
	myTwoPort = m.loadTouchton('myTouchToneFile.s2p')

Plotting some data:
	import pylab as p
	p.figure()
	myTwoPort.plotAllS()
	p.figure()
	myTwoPort.plotZin1()

Accessing parameter functions:
	p.figure()
	myTwoPort.s11.plotdB()
	p.figure()
	myTwoPort.s11.plotPhase()
	p.figure()
	myTwoPort.s11.plotSmith()
	
	Most of these functions have not been rigidly tested. use with caution
'''

import numpy as n
import pylab as p
from scipy import constants as const
from scipy import signal
import os # for fileIO

# most of these functions have not been rigidly tested. use with caution

#TODO: this could be structured as a generic 2-port to n-port, with type of S, Z, Y, or ABCD
# each netowrk could have difference function depending on the type. as of now it is still structured
# around s-parameter as the base type
#
# to make ploting faster there should be 
#	if p.isinteractive()
#		p.ioff()
#		do the ploting
#		p.draw()
#		p.ion()




##------- objects ---------
class s:
	''' represents a s-parameter. has the following fields:
			freq, freqUnit, re, im, dB, mag, deg, complex, z0
	 	TODO: fix constructor to allow more versitile input check for
			vectors being the same length
	'''	
	def __init__(self, format=None, freq=[],freqUnit='GHz', input1=[],input2=[], z0=50):
		''' The format variable, is a string which determines what the
			input vectors are assigned to. Values for format:
			'dB' - input1 = mag(dB) , input2 = phase(degrees)
			're' - input1 = real, input2 = imaginary 
			
			note: dB is 20*log10(mag)
		'''
		
		if format == 'db':
			self.freq= freq
			self.freqUnit = freqUnit
			self.dB = input1
			self.deg = input2
			self.re, self.im  = dBDeg2ReIm(self.dB,self.deg)
			self.mag = dB2Mag(self.dB)
			self.complex = self.re + 1j*self.im
			self.z0 = z0
		elif format == 'ma':
			self.freq= freq
			self.freqUnit = freqUnit
			self.mag = input1
			self.deg = input2
			self.dB = mag2dB(self.mag)
			self.re, self.im  = magDeg2ReIm(self.mag,self.deg)
			self.complex = self.re + 1j*self.im
			self.z0 = z0	
		elif ( (format == 're') or (format == 'ri')):
			self.freq= freq
			self.freqUnit = freqUnit
			self.re = input1
			self.im = input2
			self.dB, self.deg = reIm2dBDeg(self.re,self.im)
			self.mag = dB2Mag(self.dB)
			self.complex = self.re + 1j*self.im
			self.z0 = z0
		else:
			# not type passed we dont know what to do
			self.freq = freq
			self.freqUnit = freqUnit
			self.re = input1
			self.im = input1
			self.dB = input1
			self.mag = input1
			self.deg = input1
			self.complex = input1
			self.z0 = z0
		#elif format == ma:
	
	def plotdB(self):
		''' Plot the S-parameter mag in log mode. 
		'''
		p.plot(self.freq, self.dB)
		p.xlabel('Frequency (' + self.freqUnit +')') 
		p.ylabel('Magnitude (dB)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		
	def plotPhase(self):
		''' Plot the S-parameters phase mode. 
		'''
		p.plot(self.freq, self.deg)
		p.xlabel('Frequency (' + self.freqUnit +')') 
		p.ylabel('Phase (deg)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
	
	def plotSmith(self, radius=1, res=1000):
		''' Plot the S-parameters on a smith chart.
		can be passed the smith radius and resolution of smith chart circles 
		'''
		p.hold(1)
		smith(radius,res)
		p.plot(self.re, self.im)
	
	def plotZ0(self):
		# could check for complex port impedance
		p.plot(self.freq, self.z0)
		p.xlabel('Frequency (' + self.freqUnit +')') 
		p.ylabel('Impedance (Ohms)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('Characterisctic Impedance')

		
	
class z:
	''' represents a z-parameter. has the following fields:
			freq, re, im, mag, deg, z0
	 	TODO: fix constructor to allow more versitile input check for
			vectors being the same length
	'''	
	def __init__(self, format=None, freq=[], freqUnit='GHz', input1=[],input2=[], z0=50):
		''' The format variable, is a string which determines what the
			input vectors are assigned to. Values for format:
				're' - input1 = real, input2 = imaginary 

		'''
		# TODO: do this with a case 
		if format == 'db':
			print('ERROR: this format is not supported for a parameter of this type, defaulting to MA format.') 
			self.freq= freq
			self.freqUnit = freqUnit
			self.mag = input1
			self.deg = input2
			self.re, self.im  = magDeg2ReIm(self.mag,self.deg)
			self.complex = self.re + 1j*self.im
			self.z0 = z0	
		elif format == 'ma':
			self.freq= freq
			self.freqUnit = freqUnit
			self.mag = input1
			self.deg = input2
			self.re, self.im  = magDeg2ReIm(self.mag,self.deg)
			self.complex = self.re + 1j*self.im
			self.z0 = z0	
		elif format == 're':
			self.freq= freq
			self.freqUnit = freqUnit
			self.re = input1
			self.im = input2
			self.mag, self.deg = reIm2MagPhase(self.re,self.im)
			self.complex = self.re + 1j*self.im
			self.z0 = z0
		else:
			# no type passed we dont know what to do
			self.freq = freq
			self.freqUnit = freqUnit
			self.re = input1
			self.im = input1
			self.mag = input1
			self.deg = input1
			self.complex = input1
			self.z0 = z0
		
	def plotReIm(self):
		''' Plot the S-parameter mag in log mode. 
		'''
		p.plot(self.freq, self.re)
		p.plot(self.freq, self.im)
		p.xlabel('Frequency (' + self.freqUnit +')')
		p.ylabel('Impedance')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])


class twoPort:
	''' represents a two port network characterized by its s-parameters
		has the following fields:
			freq, 	s11, s12, s21, s22, z11, z12, z21, z22, zin1, zin2, swr1, swr2
		and the folling functions:
			plotAllS, plotReturnLoss, plotTransmission, plotSwr1, plotSwr2, plotZin1, 
			plotZin2, plotZ01, plotZ02
		note:
			the s-parameter fields and z-parameter fields have their own functions as well.
			many of twoPorts functions just calls these.  
	'''
	#TODO: generalize this constructor so we can call it from S, Z,Y, or ABCD
	def __init__ (self, s11=s(), s21=s(), s12= s(), s22 = s()):
		self.freq = s11.freq # frequencies must be same for all s-param
		self.freqUnit = s11.freqUnit
		self.s11 = s11	
		self.s12 = s12
		self.s21 = s21
		self.s22 = s22
		
		# set the z-parameters, see 'Microwave Engineering'  by Pozar, section 4.4 for details
		# BUG: i dont know what to do for the z0 when translating to z-parameters, i just guessed
		z11Complex = s11.z0 *( (1+s11.complex)*(1-s22.complex) + s12.complex*s21.complex) / \
							( (1-s11.complex)*(1-s22.complex) - s12.complex*s21.complex)
		z12Complex = s12.z0 *			(2*s12.complex) / \
								( (1-s11.complex)*(1-s22.complex) - s12.complex*s21.complex)
		z21Complex = s21.z0 *			(2*s21.complex) / \
								( (1-s11.complex)*(1-s22.complex) - s12.complex*s21.complex)
		
		z22Complex = s22.z0 *	( (1-s11.complex)*(1+s22.complex) + s12.complex*s21.complex) / \
								( (1-s11.complex)*(1-s22.complex) - s12.complex*s21.complex)
		
		self.z11 = z('re', self.freq, n.real(z11Complex), n.imag(z11Complex), s11.z0)
		self.z12 = z('re', self.freq, n.real(z12Complex), n.imag(z12Complex), s12.z0)
		self.z21 = z('re', self.freq, n.real(z21Complex), n.imag(z21Complex), s21.z0)
		self.z22 = z('re', self.freq, n.real(z22Complex), n.imag(z22Complex), s22.z0)
		#TODO: self.y's and self.abcd's
		
		
		# these might be better as a property of the s-parameters
		# input impedance (NOT z11 of impedance matrix) of the 2 ports
		self.zin1 = s11.z0 * (1 + s11.complex) / (1 - s11.complex)
		self.zin2 = s22.z0 * (1 + s22.complex) / (1 - s22.complex)
		
		# standing wave ratio
		self.swr1 = (1 + n.abs(s11.complex)) / (1 - n.abs(s11.complex))
		self.swr2 = (1 + n.abs(s22.complex)) / (1 - n.abs(s22.complex))
				
	def plotReturnLoss(self):
		self.s11.plotdB()
		self.s22.plotdB()
		p.legend(('S11','S22'))
		p.title('Return Loss')
		p.xlabel('Frequency (' + self.freqUnit +')')
		
	def plotTransmission(self):
		self.s12.plotdB()
		self.s21.plotdB()
		p.legend(('S12','S21'))
		p.title('Transmission')
		p.xlabel('Frequency (' + self.freqUnit +')')
	
	def plotAllS(self):
		
		self.s11.plotdB()
		self.s12.plotdB()
		self.s21.plotdB()
		self.s22.plotdB()
		p.legend(('S11','S12','S21','S22'))
		p.xlabel('Frequency (' + self.freqUnit +')')
		
		
	def plotZin1(self):
		p.plot(self.freq, n.real(self.zin1), label='Real')
		p.plot(self.freq, n.imag(self.zin1), label='Imaginary')
		p.xlabel('Frequency (' + self.freqUnit +')')
		p.ylabel('Impedance (Ohms)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('Input Impedance, Port 1')
		
	def plotZin2(self):
		p.plot(self.freq, n.real(self.zin2),label='Real')
		p.plot(self.freq, n.imag(self.zin2), label='Imaginary')
		p.xlabel('Frequency (' + self.freqUnit +')')
		p.ylabel('Impedance (Ohms)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('Input Impedance, Port 1')
	
	def plotSwr1(self):
		p.plot(self.freq, self.swr1)
		p.xlabel('Frequency (' + self.freqUnit +')') 
		p.ylabel('SWR')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('SWR, Port 1')
	
	def plotSwr2(self):
		p.plot(self.freq, self.swr2)
		p.xlabel('Frequency (' + self.freqUnit +')') 
		p.ylabel('SWR')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('SWR, Port 1')

	def plotZ01(self):
		self.s11.plotZ0()
		
	def plotZ02(self):
		self.s22.plotZ0()
	def writeTouchtone(self, fileName='twoPort.s2p',format='MA'):
		if os.access(fileName,1):
			# TODO: prompt for file removal
			os.remove(fileName)
		f=open(fileName,"w")
		
		# write header file. note: the #  line is NOT a comment it is essential and it must be exactly
		# this way for puff to read it
		#TODO check format string for acceptable values
		f.write("# GHz S " + format + " R " + self.s11.z0 +" \n! EEsoF format of Network Analyzer Results\n")
		f.write ("!freq\t")
		for p in range(size(arrayOfS)):
			f.write( arrayOfS[p].name + "Mag\t" + arrayOfS[p].name + "Phase\t")
		
		# loop through frequency points and write out s-parmeter info in Mag Phase
		for k in range(0,fPoints):	
			f.write("\n"+repr(freqs[k]))
			for kk in range(size(arrayOfS)):
				f.write("\t"+repr( arrayOfS[kk].mag[k]) +"\t" + repr(arrayOfS[kk].phase[k]))
			
		
		f.close()




class onePort:
	''' represents a one port network characterized by its s-parameters
		has the following fields:
			freq, s11 
	'''
	def __init__ (self, s11=s()):
		self.freq = s11.freq
		self.freqUnit = s11.freqUnit
		self.s11 = s11	
		
		# these might be better as a property of the s-parameters
		# input impedance (NOT z11 of impedance matrix) of the 2 ports
		self.zin = s11.z0 * (1 + s11.complex) / (1 - s11.complex)





class wr:
	'''
	waveguide class, following WR naming convention, 
		ie WR75 = wr(75)
	wr has following fields
		a, b, fco, fStart, fStop, fCenter
	'''
	def __init__(self, a):
		'''
		takes one argument, "a" in tenths of an inch. 
		which is the number	following WR, ie WR75  has a=.75*inch
		'''
		self.a = a *1e-2 * const.inch
		self.fco = const.c/(2.*self.a)
		self.fStart = 1.2 * self.fco
		self.fStop = 1.9*self.fco
		self.fCenter = 1.55*self.fco
	
		
	def printSpecs(self):
		'''
		print the useful fields of a WR 
		'''
		print ' ----- WR%i Specs -------' % (self.a * 1e2/const.inch)
		print 'f cut-off:	%.1f GHz' % (self.fco *1e-9)
		print 'band start:	%.1f GHz' %(self.fStart *1e-9)
		print 'band stop:	%.1f GHz' %(self.fStop *1e-9)
		print 'band center:	%.1f GHz' %(self.fCenter *1e-9)
	

##------- networks --------
def seriesTwoPort(twoPortA ,twoPortB):
	''' returns twoPort representing the series combination of twoPortA
 		 and twoPortB. twoPortA and twoPortB are type twoPort.   
	'''
	s21 = twoPortA.s21 * twoPortB.s21 / (1-twoPortA.s22*twoPortB.s11)
	s12 = twoPortA.s12*twoPortB.s12 / (1-twoPortA.s22*twoPortB.s11);
	s11 = twoPortA.s11 + twoPortA.s21*twoPortB.s11*twoPortA.s12 / (1-twoPortA.s22*twoPortB.s11);
	s22 = twoPortB.s22 + twoPortB.s12*twoPortA.s22*twoPortB.s21 / (1-twoPortA.s22*twoPortB.s11);
	
	return TwoPort([],s11,s21,s12,s22)
	


##----- conversion utilities ----
# TODO: explicitly call j from numpy
def magPhase2ReIm( mag, phase):
	re = n.real(mag*n.exp(1j*(phase)))
	im = n.imag(mag*n.exp(1j*(phase)))
	return re, im
def magDeg2ReIm( mag, deg):
	re = n.real(mag*n.exp(1j*(deg*n.pi/180)))
	im = n.imag(mag*n.exp(1j*(deg*n.pi/180)))
	return re, im
def dBDeg2ReIm(dB,deg):
	re = n.real(10**((dB)/20.)*n.exp(1j*(deg*n.pi/180)))
	im = n.imag(10**((dB)/20.)*n.exp(1j*(deg*n.pi/180)))
	return re, im
	
def reIm2MagPhase( re, im):
	mag = n.abs( (re) + 1j*im )
	phase = n.angle( (re) + 1j*im)
	return mag, phase
	
def reIm2dBDeg (re, im):
	dB = 20 * n.log10(n.abs( (re) + 1j*im ))
	deg = n.angle( (re) + 1j*im) * 180/n.pi 
	return dB, deg 

def mag2dB(mag):
	return  20*n.log10(mag)
	
def dB2Mag(dB):
	return 10**((dB)/20.)
	
def rad2deg(rad):
	return (rad)*180/n.pi
	
def deg2rad(deg):
	return (deg)*n.pi/180
	





##------ ploting --------
def smith(smithRadius=1, res=1000 ):
	# smith(res=1000, smithRadius=1)
	#	plots a smith chart with radius given by smithRadius and 
	#	point density resolution given by res. 
	#TODO: this could be plotted more efficiently if all data was ploted
	#	at once. cirlces could be computer analytically, contour density
	#	could be configurable
	def circle(offset,r, numPoints ):
		circleVector = r*n.exp(1j* n.linspace(0,2*n.pi,numPoints))+ offset
		return circleVector
				
	# generate complex pairs of [center, radius] for smith chart contours
	# TODO: generate this by logical algorithm
	heavyContour = [[0,1],[1+1j,1],[1-1j,1],[.5,.5]]
	lightContour = [[1+4j,4],[1-4j,4],[1+.5j,.5],[1-.5j,.5],[1+.25j,.25],[1-.25j,.25],[1+2j,2],[1-2j,2],[.25,.75],[.75,.25],[-1,2]]

	# verticle and horizontal axis
	p.axvline(x=1,color='k')
	p.axhline(y=0,color='k')
	
	
	# loop through countour vectors and plot the circles with appropriate 
	# clipping at smithRadius
	for contour in heavyContour:	
		currentCirle= circle(contour[0],contour[1], res)
		currentCirle[abs(currentCirle)>smithRadius] = p.nan
		p.plot(n.real(currentCirle), n.imag(currentCirle),'k', linewidth=1)
		
	for contour in lightContour:	
		currentCirle= circle(contour[0],contour[1], res)
		currentCirle[abs(currentCirle)>smithRadius] = p.nan
		p.plot(n.real(currentCirle), n.imag(currentCirle),'gray', linewidth=1)
	

##------ File import -----
def loadTouchtone(inputFileName):
	
	
	''' Takes the full pathname of a touchtone plain-text file.
	Returns a network object representing its contents (1 or 2-port).
	touchtone files usually have extension of .s1p, .s2p, .s1,.s2.
	
	example:
	myTwoPort = mwavepy.loadTouchTone('inputFile.s1p') 
	'''
	
	#TODO: use the freqUnit, and paramTypes
	#	check the header, hfss does not produce correct header
	f = file(inputFileName)

	
	# ignore comments lines up untill the header line
	line = f.readline()
	while line.split()[0] != '#':
		line = f.readline()

	headerInfo = line.split()
	data = n.loadtxt(f, comments='!')

	
	
	
	# the header file contains information regarding what data lines mean
	# the format is 
	#	# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]

	freqUnit = headerInfo[1]
	freqUnit = freqUnit[:-1]+freqUnit[-1].lower() # format string nicely
	paramType = headerInfo[2].lower()
	format = headerInfo[3].lower()
	
	# try to assign a normalizing impedance.
	if (len(headerInfo) == 6):
		port1Z0 = port2Z0 = float(headerInfo[5])
		
	else:
		# BUG: i dont know how to apply which port impedance to which parameters. 
		# this possible error is compounded in the z-parameter initialization (see twoPort)
		print ('WARNING: this file does not contain a single normalizing impedance, using HFSS inserted Zo comments, if they exist.')
		port1Z0, port2Z0 = loadPortImpedanceFromTouchtone(inputFileName)
		if len(port1Z0) == 0:
			print ('Did not find any normalizing impedance defaulting to 50 Ohms')
			port1Z0 = port2Z0 = 50.0
	
	
	if ( data.shape[1] == 9 ):
		# we have a 2-port netowork 
		s11 = s(format,data[:,0],freqUnit,data[:,1],data[:,2],port1Z0)
		s21 = s(format,data[:,0],freqUnit,data[:,3],data[:,4],port1Z0)
		s12 = s(format,data[:,0],freqUnit,data[:,5],data[:,6],port2Z0)
		s22 = s(format,data[:,0],freqUnit,data[:,7],data[:,8],port2Z0)
		return twoPort(s11,s21,s12,s22)

	elif ( data.shape[1] == 3):
		# we have a 1-port
		return onePort(s(format, data[:,0], freqUnit,data[:,1],data[:,2],float(z0)))

	else:
		# TODO: handle errors correctly
		print('ERROR: file doesnt contain expected number of fields')
		return -1

	



def loadPortImpedanceFromTouchtone(inputFileName):
	'''Takes the name of a HFSS formated touchtone file, which has frequency 
	dependent port impendance, such as the case for waveguide. 
	Returns	two arrays representing the port impedances in complex format. 
	NOTE: this only supports twoPorts. needs to be generalized
	'''
	f = file(inputFileName)
	port1Z0r=[]
	port1Z0i=[]
	port2Z0r=[]
	port2Z0i=[]

	listOfLines  = f.readlines()

	for line in listOfLines:
		splitLine = line.split()
		if len(splitLine)>3 and splitLine[0] =='!' and splitLine[1] == 'Port':
			port1Z0r.append(splitLine[3])
			port1Z0i.append(splitLine[4])
			port2Z0r.append(splitLine[5])
			port2Z0i.append(splitLine[6])
		
	port1Z0 = n.array(port1Z0r,dtype=float) + 1j*n.array(port1Z0i,dtype=float)
	port2Z0 = n.array(port2Z0r,dtype=float) + 1j*n.array(port2Z0i,dtype=float)

	return port1Z0,port2Z0
	


def plotCsv(filename,rowsToSkip=1,delim=','):
	'''plots columns from csv file. plots all columns against the first
	see pylab.loadtxt for more information
	'''
	data = p.loadtxt(filename,skiprows=rowsToSkip,delimiter=delim)
	p.plot(data[:,0], data[:,1:])
	p.grid(1)
	p.title(filename)


##------ other functions ---
def psd2TimeDomain(f,y, windowType='rect'):
	'''convert a one sided complex spectrum into a real time-signal.
	takes 
		f: frequency array, 
		y: complex PSD arary 
		windowType: windowing function, defaults to rect
	
	returns in the form:
		[timeVector, signalVector]
	timeVector is in inverse units of the input variable f,
	if spectrum is not baseband then, timeSignal is modulated by 
		exp(t*2*pi*f[0])
	so keep in mind units, also due to this f must be increasing left to right'''
	
	
	# apply window function
	#TODO: make sure windowType exists in scipy.signal
	if (windowType != 'rect' ):
		exec "window = signal.%s(%i)" % (windowType,len(f))
		spectrum = spectrum * window
	
	#create other half of spectrum
	spectrum = (n.hstack([n.real(y[:0:-1]),n.real(y)])) + 1j*(n.hstack([-n.imag(y[:0:-1]),n.imag(y)]))
	
	# do the transform 
	df = abs(f[1]-f[0])
	T = 1./df
	timeVector = n.linspace(-T/2,T/2,2*len(f)-1)	
	signalVector = p.ifft(p.ifftshift(spectrum))
	
	#the imaginary part of this signal should be from fft errors only,
	signalVector= real(signalVector)
	# the response of frequency shifting is 
	# n.exp(1j*2*n.pi*timeVector*f[0])
	# but i would have to manually undo this for the inverse, which is just 
	# another  variable to require. the reason you need this is because 
	# you canttransform to a bandpass signal, only a lowpass. 
	# 
	return timeVector, signalVector


def timeDomain2Psd(t,y, windowType='rect'):
	''' returns the positive baseband PSD for a real-valued signal
	returns in the form:
		[freqVector,spectrumVector]
	freq has inverse units of t's units. also, sampling frequency is 
	fs = 1/abs(t[1]-t[0])
	the result is scaled by 1/length(n), where n is number of samples
	to attain the original spectrum you must shift the freqVector appropriatly
	'''
	# apply window function
	#TODO: make sure windowType exists in scipy.signal
	if (windowType != 'rect' ):
		exec "window = signal.%s(%i)" % (windowType,len(f))
		spectrum = spectrum * window
	
	dt = abs(t[1]-t[0])
	fs = 1./dt
	numPoints = len(t)
	f = n.linspace(-fs/2,fs/2,numPoints)

	Y = 1./len(y)* p.fftshift(p.fft(y))
	spectrumVector = Y[len(Y)/2:]
	freqVector = f[len(f)/2:]
	return [freqVector,spectrumVector]



##------- Calibrations ------
def getABC(mOpen,mShort,mMatch,aOpen,aShort,aMatch):
	'''calculates calibration coefficients for a one port OSM calibration
	 
	 returns:
		abc is a Nx3 ndarray containing the complex calibrations coefficients,
		where N is the number of frequency points in the standards that where 
		given.
	
	 takes:
		 mOpen, mShort, and mMatch are 1xN complex ndarrays representing the 
		measured reflection coefficients off the corresponding standards.
			
	 	aOpen, aShort, and aMatch are 1xN complex ndarrays representing the
	 	assumed reflection coefficients off the corresponding standards. 
	 	
	 note:
	  the standards used in OSM calibration dont actually have to be 
	  an open, short, and match. they are arbitrary but should provide
	  good seperation on teh smith chart for good accuracy 
	'''
	
	# loop through all frequencies and solve for the calibration coefficients.
	# note: abc are related to error terms with:
	# a = e10*e01-e00*e11, b=e00, c=e11  
	#TODO: check to make sure all arrays are same length
	abc= n.complex_(n.zeros([len(mOpen),3]))
	
	for k in range(len(mOpen)):
		
		Y = n.vstack( [	mShort[k],\
						mOpen[k],\
						mMatch[k]\
						] )
		
		X = n.vstack([ \
					n.hstack([aShort[k], 1, aShort[k]*mShort[k] ]),\
					n.hstack([aOpen[k],	 1, aOpen[k] *mOpen[k] ]),\
					n.hstack([aMatch[k], 1, aMatch[k]*mMatch[k] ])\
					])
		
		#matrix of correction coefficients
		abc[k,:] = n.dot(n.linalg.inv(X), Y).flatten()
		
	return abc

def applyABC( gamma, abc):
	'''
	takes a complex array of uncalibrated reflection coefficient and applies
	the one-port OSM callibration, using the coefficients abc. 

	takes:
		gamma - complex reflection coefficient
		abc - Nx3 OSM calibration coefficients. 
	'''
	# for clarity this is same as:
	# gammaCal(k)=(gammaDut(k)-b)/(a+gammaDut(k)*c); for all k 
	gammaCal = (gamma-abc[:,1]) / (abc[:,0]+ gamma*abc[:,2])
	return gammaCal
