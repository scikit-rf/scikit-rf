'''
#       __init__.py
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

#	Most of these functions have not been rigidly tested. use with caution!!

import numpy as npy
from numpy import sqrt, exp, array,tan,sin,cos,inf
import pylab as p
from scipy import constants as const
from scipy.constants import  epsilon_0, mu_0, c,pi
from scipy import signal

import os # for fileIO

# for drawing smith chart
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
	

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


############# CONSTANTS ##################


##------- objects ---------
class s:
	''' represents a s-parameter. has the following fields:
			freq, freqUnit, re, im, dB, mag, deg, complex, z0
	 	TODO: fix constructor to allow more versitile input - by using keywork arguments check for
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
	
	def plotSmith(self, radius=1):
		''' Plot the S-parameters on a smith chart.
		can be passed the smith radius and resolution of smith chart circles 
		'''
		p.hold(1)
		smith(radius)
		p.plot(self.re, self.im)
		p.axis(radius*npy.array([-1., 1., -1., 1.]))
		
	
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
		''' Plot the real and imaginary parts of Z-parameters 
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
		
		#this will be index as [i,j,f], meaning S_ij at frequency f
		self.sMat = npy.array([[self.s11.complex, self.s12.complex],\
						[self.s21.complex,self.s22.complex]])

		
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
		
		self.z11 = z('re', self.freq, npy.real(z11Complex), npy.imag(z11Complex), s11.z0)
		self.z12 = z('re', self.freq, npy.real(z12Complex), npy.imag(z12Complex), s12.z0)
		self.z21 = z('re', self.freq, npy.real(z21Complex), npy.imag(z21Complex), s21.z0)
		self.z22 = z('re', self.freq, npy.real(z22Complex), npy.imag(z22Complex), s22.z0)
		#TODO: self.y's and self.abcd's
		
		
		# these might be better as a property of the s-parameters
		# input impedance (NOT z11 of impedance matrix) of the 2 ports
		self.zin1 = s11.z0 * (1 + s11.complex) / (1 - s11.complex)
		self.zin2 = s22.z0 * (1 + s22.complex) / (1 - s22.complex)
		
		# standing wave ratio
		self.swr1 = (1 + npy.abs(s11.complex)) / (1 - npy.abs(s11.complex))
		self.swr2 = (1 + npy.abs(s22.complex)) / (1 - npy.abs(s22.complex))
				
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
		p.plot(self.freq, npy.real(self.zin1), label='Real')
		p.plot(self.freq, npy.imag(self.zin1), label='Imaginary')
		p.xlabel('Frequency (' + self.freqUnit +')')
		p.ylabel('Impedance (Ohms)')
		p.grid(1)
		p.xlim([ self.freq[0], self.freq[-1]])
		p.title('Input Impedance, Port 1')
		
	def plotZin2(self):
		p.plot(self.freq, npy.real(self.zin2),label='Real')
		p.plot(self.freq, npy.imag(self.zin2), label='Imaginary')
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
		
	def writeTouchtone(self, fileName='mytwoPort.s2p'):
		'''
		write twoPort network to a file in touchtone format. 
		
		takes 1 argument, the filename (as a string)
		
		
		the line starting with # holds import information about the data
		the lines starting with ! are just comments for the user
		this only write in mag-angle format for now. 
		'''
		#TODO:  maybe some exception handling
		#	allow for other formats instead of just mag angle (MA)
		format = 'MA'
		
		if os.access(fileName,1):
			# TODO: prompt for file removal
			os.remove(fileName)
		f=open(fileName,"w")
		

		# write header file. note: the #  line is NOT a comment it is 
		#essential and it must be exactly this format, to work
		# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
		
		#TODO check format string for acceptable values and do somthing with it
		f.write("# " + self.s11.freqUnit+ " S " + format + " R " + str(self.s11.z0) +" \n")
		
		#write comment line for users
		f.write ("!freq\t")
		for p in ['S11','S21','S12','S22']:
			f.write( p + "Mag\t" + p + "Phase\t")
		
		# loop through frequency points and write out s-parmeter info in Mag Phase
		for k in range(len(self.s11.freq)):	
			f.write("\n"+repr(self.s11.freq[k]))
			for kk in self.s11, self.s21,self.s12,self.s22:
				f.write("\t"+repr( kk.mag[k]) +"\t" + repr(kk.deg[k]))
			
		
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


	def plotReturnLoss(self):
		self.s11.plotdB()
		p.title('Return Loss')
		p.xlabel('Frequency (' + self.freqUnit +')')
	def writeTouchtonnae(fileName='myonePort.s1p'):
		'''
		write onePort network to a file in touchtone format. 
		
		takes 1 argument, the filename (as a string)
		
		
		the line starting with # holds import information about the data
		the lines starting with ! are just comments for the user
		this only write in mag-angle format for now. 
		'''
		#TODO:  maybe some exception handling
		#	allow for other formats instead of just mag angle (MA)
		if os.access(fileName,1):
			# TODO: prompt for file removal
			os.remove(fileName)
		f=open(fileName,"w")
		

		# write header file. note: the #  line is NOT a comment it is 
		#essential and it must be exactly this format, to work
		# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
		
		#TODO check format string for acceptable values and do somthing with it
		f.write("# " + self.s11.freqUnit+ " S " + format + " R " + str(self.s11.z0) +" \n")
		
		#write comment line for users
		f.write ("!freq\tS11Mag\tS11Phase\t")
		
		# loop through frequency points and write out s-parmeter info in Mag Phase
		for k in range(len(self.s11.freq)):	
			f.write("\n"+repr(self.s11.freq[k]))
			for kk in self.s11:
				f.write("\t"+repr( kk.mag[k]) +"\t" + repr(kk.deg[k]))
			
		
		f.close()


##------- networks --------
def seriesTwoPort(twoPortA ,twoPortB):
	''' returns twoPort representing the series combination of twoPortA
 		 and twoPortB. twoPortA and twoPortB are type twoPort.   
	'''
	s21 = twoPortA.s21 * twoPortB.s21 / (1-twoPortA.s22*twoPortB.s11)
	s12 = twoPortA.s12*twoPortB.s12 / (1-twoPortA.s22*twoPortB.s11);
	s11 = twoPortA.s11 + twoPortA.s21*twoPortB.s11*twoPortA.s12 / (1-twoPortA.s22*twoPortB.s11);
	s22 = twoPortB.s22 + twoPortB.s12*twoPortA.s22*twoPortB.s21 / (1-twoPortA.s22*twoPortB.s11);
	
	return twoPort(s11,s21,s12,s22)
	


##----- conversion utilities ----
# TODO: explicitly call j from numpy
def magPhase2ReIm( mag, phase):
	re = npy.real(mag*exp(1j*(phase)))
	im = npy.imag(mag*exp(1j*(phase)))
	return re, im
def magDeg2ReIm( mag, deg):
	re = npy.real(mag*exp(1j*(deg*pi/180)))
	im = npy.imag(mag*exp(1j*(deg*pi/180)))
	return re, im
def dBDeg2ReIm(dB,deg):
	re = npy.real(10**((dB)/20.)*exp(1j*(deg*pi/180)))
	im = npy.imag(10**((dB)/20.)*exp(1j*(deg*pi/180)))
	return re, im
	
def reIm2MagPhase( re, im):
	mag = npy.abs( (re) + 1j*im )
	phase = npy.angle( (re) + 1j*im)
	return mag, phase
	
def reIm2dBDeg (re, im):
	dB = 20 * npy.log10(npy.abs( (re) + 1j*im ))
	deg = npy.angle( (re) + 1j*im) * 180/pi 
	return dB, deg 

def mag2dB(mag):
	return  20*npy.log10(mag)
	
def dB2Mag(dB):
	return 10**((dB)/20.)
	
def rad2deg(rad):
	return (rad)*180/pi
	
def deg2rad(deg):
	return (deg)*pi/180
	








##------ ploting --------
def plotOnSmith(complexData,**kwargs):
	p.plot(npy.real(complexData), npy.imag(complexData), **kwargs)
	smith(1)

def smith(smithR=1):
	'''
	plots the smith chart of a given radius
	takes:
		smithR - radius of smith chart
	'''
	#TODO: fix this so that an axes object may be passed as argument
	ax = p.gca()
	# contour holds matplotlib instances of: pathes.Circle, and lines.Line2D, which 
	# are the contours on the smith chart 
	contour = []
	
	# these are hard-coded on purpose,as they should always be present
	rHeavyList = [0,1]
	xHeavyList = [1,-1]
	
	# these could be dynamically coded in the future, but work good'nuff for now 
	rLightList = p.logspace(3,-5,9,base=.5)
	xLightList = p.hstack([p.logspace(2,-5,8,base=.5), -1*p.logspace(2,-5,8,base=.5)]) 
	
	# cheap way to make a ok-looking smith chart at larger than 1 radii
	if smithR > 1:
		rMax = (1.+smithR)/(1.-smithR)
		rLightList = p.hstack([ p.linspace(0,rMax,11)  , rLightList ])
		
	
	# loops through Light and Heavy lists and draws circles using patches
	# for analysis of this see R.M. Weikles Microwave II notes (from uva)
	for r in rLightList:
		center = (r/(1.+r),0 )
		radius = 1./(1+r)
		contour.append( Circle( center, radius, ec='grey',fc = 'none'))
	for x in xLightList:
		center = (1,1./x)
		radius = 1./x
		contour.append( Circle( center, radius, ec='grey',fc = 'none'))
			
	for r in rHeavyList:
		center = (r/(1.+r),0 )
		radius = 1./(1+r)
		contour.append( Circle( center, radius, ec= 'black', fc = 'none'))	
	for x in xHeavyList:
		center = (1,1./x)
		radius = 1./x	
		contour.append( Circle( center, radius, ec='black',fc = 'none'))
	
	#draw x and y axis
	contour.append(Line2D([-smithR, smithR],[0,0],color='black'))
	contour.append(Line2D([1,1],[-smithR,smithR],color='black'))
	
	#set axis limits
	ax.axis('equal')
	ax.axis(smithR*npy.array([-1., 1., -1., 1.]))
	
	# loop though contours and draw them on the given axes
	for currentContour in contour:
		if isinstance(currentContour, Circle):
			ax.add_patch(currentContour)
		elif isinstance(currentContour, Line2D):
			ax.add_line(currentContour)


# TODO: all of these updatePlot?? utilities could be incorperated into
# 		the s-parameters functions with a try except like 
#		s.plotdB(mpl_plot):
#			try:
#				mpl_plot
#				#assert mpl_plot exists so update it 
#			except:
#				mpl_plot was not passed/doesnt exist so make a new plot
def updatePlotDb(inputS,mpl_plot):
	''' Plot the S-parameter mag in log mode. given an already existing 
	axes 'mplplot'
	'''
	mpl_plot.plot(inputS.freq, inputS.dB)
	mpl_plot.set_xlabel('Frequency (' + inputS.freqUnit +')') 
	mpl_plot.set_ylabel('Magnitude (dB)')
	mpl_plot.grid(1)
	mpl_plot.set_xlim([ inputS.freq[0], inputS.freq[-1]])

def updatePlotPhase(inputS,mpl_plot):
	''' Plot the S-parameter phase mode. given an already existing 
	axes 'mplplot'
	'''
	mpl_plot.plot(inputS.freq, inputS.deg)
	mpl_plot.set_xlabel('Frequency (' + inputS.freqUnit +')') 
	mpl_plot.set_ylabel('Phase (deg)')
	mpl_plot.grid(1)
	mpl_plot.set_xlim([ inputS.freq[0], inputS.freq[-1]])

def updatePlotSmith(inputS, mpl_plot):
	''' Plot the S-parameters on a smith chart.given an already existing 
	axes 'mplplot'
	can be passed the smith radius and resolution of smith chart circles 
	'''
	mpl_plot.plot(inputS.re, inputS.im)
	
def updateSmithChart(mpl_plot, smithRadius=1, res=1000 ):
	# smith(res=1000, smithRadius=1)
	#	plots a smith chart with radius given by smithRadius and 
	#	point density resolution given by res. 
	#TODO: this could be plotted more efficiently if all data was ploted
	#	at once. cirlces could be computer analytically, contour density
	#	could be configurable
	def circle(offset,r, numPoints ):
		circleVector = r*exp(1j* npy.linspace(0,2*pi,numPoints))+ offset
		return circleVector
				
	# generate complex pairs of [center, radius] for smith chart contours
	# TODO: generate this by logical algorithm
	heavyContour = [[0,1],[1+1j,1],[1-1j,1],[.5,.5]]
	lightContour = [[1+4j,4],[1-4j,4],[1+.5j,.5],[1-.5j,.5],[1+.25j,.25],[1-.25j,.25],[1+2j,2],[1-2j,2],[.25,.75],[.75,.25],[-1,2]]

	# verticle and horizontal axis
	mpl_plot.axvline(x=1,color='k')
	mpl_plot.axhline(y=0,color='k')
	
	
	# loop through countour vectors and plot the circles with appropriate 
	# clipping at smithRadius
	for contour in heavyContour:	
		currentCirle= circle(contour[0],contour[1], res)
		currentCirle[abs(currentCirle)>smithRadius] = npy.nan
		mpl_plot.plot(npy.real(currentCirle), npy.imag(currentCirle),'k', linewidth=1)
		
	for contour in lightContour:	
		currentCirle= circle(contour[0],contour[1], res)
		currentCirle[abs(currentCirle)>smithRadius] = npy.nan
		mpl_plot.plot(npy.real(currentCirle), npy.imag(currentCirle),'gray', linewidth=1)
	


##------ File import -----

def loadTouchtone(inputFileName):
	
	
	''' Takes the full pathname of a touchtone plain-text file.
	Returns a network object representing its contents (1 or 2-port).
	touchtone files usually have extension of .s1p, .s2p, .s1,.s2.
	
	example:
	myTwoPort = mwavepy.loadTouchTone('inputFile.s1p') 
	'''
	
	#BUG: error while reading empty header line, see line  while line.split()[0] != '#': 
	#TODO: use the freqUnit, and paramTypes
	#	check the header, hfss does not produce correct header
	f = file(inputFileName)

	
	# ignore comments lines up untill the header line
	line = f.readline()
	while line.split()[0] != '#':
		line = f.readline()

	headerInfo = line.split()
	data = npy.loadtxt(f, comments='!')

	
	
	
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
		s11 = s(format, data[:,0], freqUnit,data[:,1],data[:,2],port1Z0)
		return onePort(s11)

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
		
	port1Z0 = npy.array(port1Z0r,dtype=float) + 1j*npy.array(port1Z0i,dtype=float)
	port2Z0 = npy.array(port2Z0r,dtype=float) + 1j*npy.array(port2Z0i,dtype=float)

	return port1Z0,port2Z0
	



def loadAllTouchtonesInDir(dir = '.'):
	'''
	loads all touchtone files in a given dir 
	
	takes:
		dir  - the path to the dir, passed as a string (defalut is cwd)
	returns:
		(nameList, ntwkList)  - lists holding basenames of the files loaded, and a list of mwavepy networks. If you are using pylab you can plot these easily like so: 
	
	example usage:
		import mwavepy as m
		nameList, ntwkList = m.loadAllTouchtonesInDir()
		for n in ntwkList:
			npy.plotReturnLoss()
		legend(nameList)
	'''
	ntwkList=[]
	nameList=[]
	
	for f in os.listdir (dir):
		if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
			nameList.append(f[:-4]) #strips of extension
			ntwkList.append(loadTouchtone(f))
		
	return (nameList, ntwkList)



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
		y = y * window
	
	#create other half of spectrum
	spectrum = (npy.hstack([npy.real(y[:0:-1]),npy.real(y)])) + 1j*(npy.hstack([-npy.imag(y[:0:-1]),npy.imag(y)]))
	
	# do the transform 
	df = abs(f[1]-f[0])
	T = 1./df
	timeVector = npy.linspace(0,T,2*len(f)-1)	
	signalVector = p.ifft(p.ifftshift(spectrum))
	
	#the imaginary part of this signal should be from fft errors only,
	signalVector= npy.real(signalVector)
	# the response of frequency shifting is 
	# exp(1j*2*pi*timeVector*f[0])
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
	f = npy.linspace(-fs/2,fs/2,numPoints)

	Y = 1./len(y)* p.fftshift(p.fft(y))
	spectrumVector = Y[len(Y)/2:]
	freqVector = f[len(f)/2:]
	return [freqVector,spectrumVector]

def cutOff(a):
	'''returns the cutoff frequency (in Hz) for first  resonance of a
	waveguide with major dimension given by a. a is in meters'''
	
	return sqrt((pi/a)**2 *1/(epsilon_0*const.mu_0))/(2*pi)


def passivityTest(smat):
	'''
	check that the network represented by S matrix (smat) is passive. I-S*conj(traspose(S))
	takes:
		smat - S matrix 
	returns:
		passivity - matrix containing I-S*conj(traspose(S))
	'''
	#TODO: it probably would be better to structure this to take a 2D matrix, then have a twoPort function which itterates over all frequencies
	passivity = npy.zeros(smat.shape)
	for f in range(smat.shape[2]):
		passivity[:,:,f] = npy.eye(smat.shape[1]) - npy.dot(smat[:,:,f],smat[:,:,f].conj().transpose())
			#for tmp in  eigvals(passivity[:,:,f]):
				#if real(tmp) < 0:
					#if abs(tmp) < tol:
						## structure fails the passivity test
						#return False
			#return True
	return passivity




############### transmission lines ################
## 





#################
class transmissionLine():
	'''
	should be main class, which all transmission line sub-classes inhereit
	'''
	def __init__(self):
		raise NotImplementedError
		return None
class coax():
	def __init__(self):
		raise NotImplementedError
		return None

class microstrip():
	def __init__(self):
		raise NotImplementedError
		return None
	def eEffMicrostrip(w,h,epR):
		'''
		The above formulas are in Transmission Line Design Handbook by Brian C Wadell, Artech House 1991. The main formula is attributable to Harold A. Wheeler and was published in, "Transmission-line properties of a strip on a dielectric sheet on a plane", IEEE Tran. Microwave Theory Tech., vol. MTT-25, pp. 631-647, Aug. 1977. The effective dielectric constant formula is from: M. V. Schneider, "Microstrip lines for microwave integrated circuits," Bell Syst Tech. J., vol. 48, pp. 1422-1444, 1969.
		'''
		
		if w < h:
			return (epR+1.)/2 + (epR-1)/2 *(1/sqrt(1+12*h/w) + .04*(1-w/h)**2)
		else:
			return (epR+1.)/2 + (epR-1)/2 *(1/sqrt(1+12*h/w))
		
		
		
	
	def betaMicrostrip(w,h,epR):
		return lambda omega: omega/c * sqrt(eEffMicrostrip(w,h,epR))
		
		
	def impedanceMicrostrip(w,h,epR):
		'''
		
		
		taken from pozar
		'''
		eEff = eEffMicrostrip(w,h,epR)
		if w/h < 1:
			return 60/sqrt(eEff) * npy.ln( 8*h/w + w/(4*h))
		else:
			return 120*pi/ ( sqrt(eEff)* w/h+1.393+.667*npy.ln(w/h+1.444) )

class coplanar():
	def __init__(self):
		raise NotImplementedError
		return None
class waveguide:
	'''
	class which represents rectangular waveguide . 
	
	TODO: implement different filling materials, and wall material losses
	'''
	def __init__(self, a,b,band=None, epsilonR=1, muR=1):
		self.a = a
		self.b = b
		self.fc10 = self.fc(1,0)
		if band == None: 
			self.band = npy.array([1.25*self.fc10 , 1.9 * self.fc10]) 
		else:
			self.band= band
		self.fStart = self.band[0]
		self.fStop = self.band[1]
		#all for dominant mode
	
	def fc(self,m=1,n=0):
		return c/(2*pi)*sqrt( (m*pi/self.a)**2 +(n*pi/self.b)**2)

	def beta(self, omega,m=1,n=0):
		# TODO: should do a test below cutoff and handle imaginary sign
		# the beta here is just the space beta, which should be a method of mwavepy
		k = omega/c
		return sqrt(k**2 - (m*pi/self.a)**2- (n*pi/self.b)**2)
	
	def beta_f(self,f,m=1,n=0):
		return self.beta(2*pi*f,m,n)
		
	def lambdaG(self,omega,m=1,n=0):
		return (2*pi / self.beta(omega,m,n))
	
	def lambdaG_f(self,f,m=1,n=0):
		return self.lambdaG(2*pi *f,m,n)
	
	def vp(self, omega,m=1,n=0):
		return omega / self.beta(omega,m,n)
	
	def vp_f(self, f,m=1,n=0):
		return 2*pi*f / self.beta(2*pi*f,m,n)
		
	def zTE(self, omega,m=1,n=0):
		return eta0 * beta0(omega)/self.beta(omega,m,n)
	def zTM(self, omega,m=1,n=0):
		return eta0 * self.beta(omega,m,n)/beta0(omega)

class wr(waveguide):
	'''
	class which represents defined rectangular waveguide band. 
	
	constructor takes 
	'''
	def __init__(self, number):
		waveguide.__init__(self,number*10*const.mil ,.5 * number*10*const.mil  )
# standard waveguide bands, note that the names are not perfectly cordinated with guide dims. 
# info taken from Virginia Diodes Inc. Waveguide Band Designations
WR10 = wr(10)
WR8 = wr(8)
WR6 = wr(6.5)
WR5 = wr(5.1)
WR4 = wr(4.3)
WR3 = wr(3.4)
WR1p5 = wr(1.5)		
		
############# two-port structures ########################
# these conversions where taken from Pozar. Microwave Engineering sec 5.6

# relationships
def zl2gamma(zl,z0):
	return (zl-z0)/(zl+z0)
def zl2T(zl,z0):
	return 1 + gamma(zl,z0)

def zin(zl,z0,el):
	'''
	returns the input impedance of a transmission line of character impedance z0 and electrical length el, terminated with a load impedance zl. 
	takes:
		zl - load impedance 
		z0 - characteristic impedance of tline
		el - electrical length ( in radians)
	returns:
		input impedance ( in general complex)
	'''
	if zl == inf:
		return -1j*z0*1./(tan(el))
	elif zl == 0:
		return 1j*z0*tan(el)
	else:
		return z0 *	(zl + 1j*z0 * tan(el)) /\
					(z0 + 1j*zl * tan(el))

def zinShort (z0,el):
	'''
	returns the input impedance of a shorted transmission line of character impedance z0 and electrical length el.
	takes:
		z0 - characteristic impedance of tline
		el - electrical length ( in radians)
	returns:
		input impedance ( in general complex)
		
	note: this just calls zin()
	'''
	return zin(0,z0,el)
def zinOpen(z0,el):
	'''
	returns the input impedance of a open-circuited transmission line of character impedance z0 and electrical length el.
	takes:
		z0 - characteristic impedance of tline
		el - electrical length ( in radians)
	returns:
		input impedance ( in general complex)
		
	note: this just calls zin()
	'''
	return zin(inf,z0,el)
	



## network  representation conversions
def s2abcd(sMat,z0=50):
	'''
	converts a 2-port network represented by a  S matrix to a 2-port ABCD matrix
	takes:
		s - 2x2 complex matrix, representing a Scattering matrix for some network.
		z0 - characteristic impedance
	returns: 
		abcd - 2x2 complex matrix, representing a ABCD matrix for some network.
		
	
	there might be a matrix implementation which is more concise but i dont know it 
	'''
	a = 		( (1+sMat[0,0]) * (1-sMat[1,1]) + sMat[0,1] * sMat[1,0] )/(2*sMat[1,0])
	b = z0 * 	( (1+sMat[0,0]) * (1+sMat[1,1]) - sMat[0,1] * sMat[1,0] )/(2*sMat[1,0])
	c = 1./z0 *	( (1-sMat[0,0]) * (1-sMat[1,1]) - sMat[0,1] * sMat[1,0] )/(2*sMat[1,0])
	d = 		( (1-sMat[0,0]) * (1+sMat[1,1]) - sMat[0,1] * sMat[1,0] )/(2*sMat[1,0])
	return array([[a,b],[c,d]])

def s2z():
	raise NotImplementedError
def s2y():
	raise NotImplementedError
def z2s():
	raise NotImplementedError
def z2y():
	raise NotImplementedError
def z2abcd():
	raise NotImplementedError
def y2s():
	raise NotImplementedError
def y2z():
	raise NotImplementedError
def y2abcd():
	raise NotImplementedError
def abcd2s(abcdMat,z0=50):
	'''
	converts a 2-port network represented by a  ABCD matrix to a 2-port S matrix
	takes:
		abcdMat - 2x2 complex matrix, representing a Scattering matrix for some network
		z0 - characteristic impedance
	returns: 
		abcd - 2x2 complex matrix, representing a ABCD matrix for some network.
		
	
	there might be a matrix implementation which is more concise but i dont know it 
	'''
	a = abcdMat[0,0]
	b = abcdMat[0,1]
	c = abcdMat[1,0]
	d = abcdMat[1,1]
	
	s11 = (a + b/z0 - c*z0 - d) / ( a + b/z0 + c*z0 + d )
	s12 = 2.*( a*d-b*c) / (a + b/z0 + c*z0 + d) 
	s21 = 2./(a + b/z0 + c*z0 + d)
	s22 =  (-1*a + b/z0 - c*z0 + d)/(a + b/z0 + c*z0 + d)
	
	return array([[s11, s12],[s21,s22]])
	
def abcd2z():
	raise NotImplementedError
def abcd2y():
	raise NotImplementedError

## connections
def connectionSeries(ntwkA,ntwkB, type='s'):
	ntwkC = npy.zeros(shape=ntwkA.shape)
	if type not in 'szyabcd':
		print( type +' is not a valid Type')
		return None
	elif type == 's':
		ntwkC[1,0] = ntwkA[1,0]*ntwkB[1,0] / 1 - ntwkA[1,1]*ntwkB[0,0]
		ntwkC[0,1] = ntwkA[0,1]*ntwkB[0,1] / 1 - ntwkA[1,1]*ntwkB[0,0]
		ntwkC[0,0] = ntwkA[0,0]+ntwkA[1,0]*ntwkB[0,0]*ntwkA[0,1] / 1 - ntwkA[1,1]*ntwkB[0,0]
		ntwkC[1,1] = ntwkB[1,1]+ntwkB[0,1]*ntwkA[1,1]*ntwkB[1,0] / 1 - ntwkA[1,1]*ntwkB[0,0]
	elif type == 'abcd':
		return npy.dot(ntwkA,ntwkB)
	elif type == 'z':
		raise NotImplementedError
	elif type == 'y':
		raise NotImplementedError
		
	return ntwkC
	

## networks
class ntwk:
	def __init__(self, data,f=None, type = 's'):
		if type not in 'szyabcd':
			print( type +' is not a valid Type')
			return None
		if data.shape[0] != data.shape[1]:
			print ('ERROR: input data must be a square matrix')
		
		
		self.numPorts = data.shape[]
		
		if type == 's':
			self.s = npy.complex(npy.zeros(shape=(rank,rank, npoints))
		
		elif type == 'abcd':
			raise NotImplementedError
		elif type == 'z':
			raise NotImplementedError
		elif type == 'y':
			raise NotImplementedError
			
		self.f = 
		

def genWaveguideDelayShort(wg,l,numPoints=201):
	'''
	generate the two port S matrix for a waveguide delayed short of length l 
	
	takes:
		wg - wr type representing a waveguide band 
		l - length of thru, in meters
		numPoints - number of points to produce
	returns:
		two port S matrix for a waveguide thru section of length l 
	'''
	if isinstance(wg,waveguide):
		return genDelayShort(wg.band[0],wg.band[1],numPoints,l, lambda omega:wg.beta(omega))
	else:
		print 'ERROR: first argument must be of waveguide type'
		return None	
		
def genWaveguideThru(wg,l,numPoints=201):
	'''
	generate the two port S matrix for a waveguide thru section of length l 
	
	takes:
		wg - wr type representing a waveguide band 
		l - length of thru, in meters
		numPoints - number of points to produce
	returns:
		two port S matrix for a waveguide thru section of length l 
	'''
	if isinstance(wg,waveguide):
		return genThru(wg.band[0],wg.band[1],numPoints,l, lambda omega:wg.beta(omega))
	else:
		print 'ERROR: first argument must be of waveguide type'
		return None		
		
		
# note: these S matricies may be re-shaped if one wants the frequency index to come first, like  S = S.transpose().reshape(-1,2,2)
def genShort(numPoints):
	'''
	generates the two port S matrix for a Short. 
	
	takes:
		numPoints - number of points
	'''
	s11 = s22 = npy.complex_( -1 * npy.ones(numPoints))
	s21 = s12 = npy.complex_( npy.zeros(numPoints) )
	return npy.array([[s11, s12],\
					[s21, s22] ])

def genOpen(numPoints):
	'''
	generates the two port S matrix for a Openpy. 
	
	takes:
		numPoints - number of points
	'''
	s11 = s22 = npy.complex_( npy.ones(numPoints))
	s21 = s12 = npy.complex_( npy.zeros(numPoints) )
	return npy.array([[s11, s12],\
					[s21, s22] ])


def genMatch(numPoints):
	'''
	generates the two port S matrix for a Match. 
	
	takes:
		numPoints - number of points
	'''
	s11 = s22 = npy.complex_( npy.zeros(numPoints))
	s21 = s12 = npy.complex_( npy.ones(numPoints))
	return npy.array([[s11, s12],\
					[s21, s22] ])


	

def genThru(fStart, fStop,numPoints, l, beta = lambda omega: omega/c ):
	'''
	generates the two port S matrix for a matched Delay line of length l. 
	
	takes:
		fStart - start frequency
		fStop - stop frequency 
		numPoints - number of points
		l - length of delay in units of m 
		beta - propagation constant, which is a function of angular frequency (omega), and returns a value with units radian/m. 
		
		note: beta defaults to lossless free-space propagation constant beta = omega/c = omega*sqrt(epsilon_0*mu_0), which assumes a TEM wave	
	'''
	s11 = npy.complex_( npy.zeros(numPoints))
	s12 = npy.complex_( npy.zeros(numPoints))
	s21 = npy.complex_( npy.zeros(numPoints))
	s22 = npy.complex_( npy.zeros(numPoints))
	
	#loop through band and calculate the delay
	fband = npy.linspace(fStart,fStop,numPoints)
	for f in range(numPoints):
		s12[f] = s21[f] =  exp(-1j*electricalLength(l,fband[f],beta) )
	
	return npy.array([[s11, s12],\
					[s21, s22] ])



def genDelayShort(fStart, fStop,numPoints, l, beta = lambda omega: omega/c ):
	s11 = npy.complex_( npy.zeros(numPoints))
	s22 = npy.complex_( npy.zeros(numPoints))
	s21 = npy.complex_( npy.zeros(numPoints))
	s12 = npy.complex_( npy.zeros(numPoints))
	
	#loop through band and calculate the delay
	fband = npy.linspace(fStart,fStop,numPoints)
	for f in range(numPoints):
		s11[f] = -1*exp(-1j* 2*electricalLength(l,fband[f],beta))	
	return npy.array([[s11, s12],\
					[s21, s22] ])

############## general EM ##########################
def betaSpace(omega,epsilonR = 1, muR = 1):
	'''
	propagation constant of a material.
	takes:
		omega - radian angular frequency (rad/s)
		epsilonR - relative permativity (default = 1) 
		muR -  relative permiability (default = 1)
	returns:
		omega/c = omega*sqrt(epsilon*mu)
	'''
	return omega* sqrt((const.mu_0*muR)*(epsilonR*epsilon_0))

def beta0(omega):
	'''
	propagation constant of a free space.
	takes:
		omega - radian angular frequency (rad/s)
	returns:
		omega/c = omega*sqrt(epsilon*mu)
	'''
	return betaSpace(omega,1,1)


	
def eta(epsilonR = 1, muR = 1):
	'''
	characteristic impedance of a material.
	takes:
		epsilonR - relative permativity (default = 1) 
		muR -  relative permiability (default = 1)
	'''
	return sqrt((const.mu_0*muR)/(epsilonR*epsilon_0))
def eta0(omega):
	'''
	characteristic impedance of free space.
	'''
	return eta(omega, 1,1)
	

	
	

		


def electricalLength( l , f0, beta=lambda omega: omega/c,deg=False):
	'''
	calculates the electrical length of a section of transmission line.
	
	takes:
		l - length of line in meters
		f0 - frequency at which to calculate 
		beta - propagation constant, which is a function of angular frequency (omega), and returns a value with units radian/m. 
		
		note: defaults to lossless free-space propagation constant beta = omega/c = omega*sqrt(epsilon_0*mu_0)
	returns:
		electrical length of tline, at f0 in radians
	'''
	if deg==False:
		return  beta(2*pi*f0 ) *l 
	elif deg ==True:
		return  rad2deg(beta(2*pi*f0 ) *l )


############### calibration ##############
## one port
def getABC(mOpen,mShort,mMatch,aOpen,aShort,aMatch):
	'''
	calculates calibration coefficients for a one port OSM calibration
	 
	 returns:
		abc is a Nx3 ndarray containing the complex calibrations coefficients,
		where N is the number of frequency points in the standards that where 
		givenpy.
	
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
	abc= npy.complex_(npy.zeros([len(mOpen),3]))
	
	for k in range(len(mOpen)):
		
		Y = npy.vstack( [	mShort[k],\
						mOpen[k],\
						mMatch[k]\
						] )
		
		X = npy.vstack([ \
					npy.hstack([aShort[k], 	1, aShort[k]*mShort[k] ]),\
					npy.hstack([aOpen[k],	1, aOpen[k] *mOpen[k] ]),\
					npy.hstack([aMatch[k], 	1, aMatch[k]*mMatch[k] ])\
					])
		
		#matrix of correction coefficients
		abc[k,:] = npy.dot(npy.linalg.inv(X), Y).flatten()
		
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
## two port 

## calibrations should be classed and handled somthing like this 
#class calibration:
	#def __init__(self,kit=calKit(),measured=None):
		#self.kit = inkit
		#self.measured = measured
		#raise NotImplementedError

#class calKit:
	#def __init__(self,type = 'osm', medium = 'waveguide', fStart, fStop, nPoints, name='default cal kit',):
		#self.name = name
		#self.type = type
		#self.numberOfStandards=0
		#self.standards=[]
	#def addStandard():
		#raise NotImplementedError
	#def delStandard():
		#raise NotImplementedError
	#def editStandard():
		#raise NotImplementedError

#calGeneralOSM = calkit('OSM', 'coax', fStart,fStop,nPoints)

#calGeneralOSM.addStandard()


#class standard:
	#def __init__(self,type='waveguide',name):
		#self.type = type
		#self.name = name
		#self.sMat = insMat	
