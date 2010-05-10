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


## imports
# Builtin  libs
import os # for fileIO
from copy import copy
from time import time
# Dependencies
try:
	import numpy as npy
	from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag, interp, linspace, shape,zeros
except:
	raise ImportError ('Depedent Packages not found. Please install: numpy')
try:
	import pylab as plb
	from matplotlib.patches import Circle 	# for drawing smith chart
	from matplotlib.lines import Line2D		# for drawing smith chart
except:
	raise ImportError ('Depedent Packages not found. Please install: matplotlib')
try:
	from scipy.constants import  epsilon_0, mu_0, c,pi, mil
	from scipy import signal
	
except:
	raise ImportError ('Depedent Packages not found. Please install: scipy')
	
#Optional
try: 
	import sympy as sym
except:
	print ('Import Warning: sympy not available.')

# Internal 
from touchstone import touchstone as touch	# for loading data from touchstone files

################# TODO LIST (so i dont forget)
'''
TBD:

# HIGH PRIORITY
possible bug found in deEmbed function: when a ntwk is deEmbed 
with different ntwk from which it was cascaded with, the phase looks 
like it has a modulo error. 

# LOW PRIOTITY
use try/excepts in plotting functions to label title and legend, this 
way if the name is None we dont crash. and the default value for a ntwk's
name can be changed to None

re-write deEmbed and cascade with try/excepts instead of isinstances()

calibration.plotCoefs... need to pass a freq multiplier 


distiguish between the complex and real part of the propagation constant
ussually denoted gamma = beta + i alpha. this effecte waveguide.lambdaG,
and waveguide.vp


use get/set for states like .s and .freq (to simultaneously update 
.freqMultiplier, etc)

have plotting functions build off simpler ones to re-use code. so 
plotPhase(complexData), then ntwk.plotPhase() could call this

add a assumeReciprocity=True/False option to abc2Ntwk (to break up S12*S21)

generate other network types z,y,abcd from any given parameter
allow constructor to call using any parameter
network conversions
network connections (parrallel, series)
tranmission line classes ( what properties do all transmission lines share)
de-embeding 

check if they have usetex on and use tex formatting for legend of 
s-parameters

rewrite the getABC functions to recursivly call itself instead of just
looping over frequencies, like we do for casacde, deEmbed. (in hindsight 
im not sure i like this )

POSSIBLE CHANGES:
does ntwk need freqUnit and freqMultiplier? why not just do it all in Hz 
and have an option to the plot command for GHz, which is the only time we would want it


freqUnit and freqMultiplier are redundant. 

fix frequency unit abiquity. shhould ntwk.freq dependent on ntwk.freqMultiplier or is that just for plotting ? need to tell the user either way.



calkits?

# to make ploting faster there should be 
#	if plb.isinteractive()
#		plb.ioff()
#		do the ploting
#		plb.draw()
#		plb.ion()


'''




## constants
# electrical conductivity in S/m
conductivityDict = {'aluminium':3.8e7,'gold':4.1e7}
lengthDict ={'m':1,'mm':1e-3,'um':1e-6}


###############  mathematical conversions ############### 
def complex2dB(complx):
	dB = 20 * npy.log10(npy.abs( (npy.real(complx) + 1j*npy.imag(complx) )))
	return dB
def complex2ReIm(complx):
	return npy.real(complx), npy.imag(complx)
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

def dB2np(x):
	'''
	converts a value in nepers to dB
	'''	
	return (log(10)/20) * x
def np2dB(x):
	'''
	converts a value in dB to neper's
	'''
	return 20/log(10) * x

def rad2deg(rad):
	return (rad)*180/pi
	
def deg2rad(deg):
	return (deg)*pi/180
	








############### Ploting ############### 
def plotComplex(complexData,ax=None,**kwargs):
	# TODO: doc this and handle ax input
	plb.plot(npy.real(complexData),label='Real',**kwargs)
	plb.plot(npy.imag(complexData),label='Imag',**kwargs)
	plb.legend(loc='best')
def plotReVsIm(complexData,ax = None, **kwargs):
	plb.plot(npy.real(complexData), npy.imag(complexData),**kwargs)
	
def plotOnSmith(complexData,ax=None,**kwargs):
	if ax == None:
		ax1 = plb.gca()
	else:
		ax1 = ax
		
	ax1.plot(npy.real(complexData), npy.imag(complexData), **kwargs)
	smith(1,ax1)

def smith(smithR=1,ax=None):
	'''
	plots the smith chart of a given radius
	takes:
		smithR - radius of smith chart
	'''
	
	if ax == None:
		ax1 = plb.gca()
	else:
		ax1 = ax

	# contour holds matplotlib instances of: pathes.Circle, and lines.Line2D, which 
	# are the contours on the smith chart 
	contour = []
	
	# these are hard-coded on purpose,as they should always be present
	rHeavyList = [0,1]
	xHeavyList = [1,-1]
	
	# these could be dynamically coded in the future, but work good'nuff for now 
	rLightList = plb.logspace(3,-5,9,base=.5)
	xLightList = plb.hstack([plb.logspace(2,-5,8,base=.5), -1*plb.logspace(2,-5,8,base=.5)]) 
	
	# cheap way to make a ok-looking smith chart at larger than 1 radii
	if smithR > 1:
		rMax = (1.+smithR)/(1.-smithR)
		rLightList = plb.hstack([ plb.linspace(0,rMax,11)  , rLightList ])
		
	
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
	ax1.axis('equal')
	ax1.axis(smithR*npy.array([-1., 1., -1., 1.]))
	
	# loop though contours and draw them on the given axes
	for currentContour in contour:
		if isinstance(currentContour, Circle):
			ax1.add_patch(currentContour)
		elif isinstance(currentContour, Line2D):
			ax1.add_line(currentContour)






def saveAllFigs(dir = './', format=['eps','png']):
	for fignum in plb.get_fignums():
		plb.figure(fignum)
		fileName = plb.gca().get_title()
		if fileName == '':
				fileName = 'unamedPlot'
		for fmt in format:
			plb.savefig(dir+fileName+'.'+fmt, format=fmt)
			print (dir+fileName+'.'+fmt)
		

def plotErrorCoefsFromDictDb(errorDict,freq= None,ax = None, **kwargs):
	if not ax:
		ax = plb.gca()
	
	if freq == None:	
		for k in errorDict.keys():
			ax.plot(complex2dB(errorDict[k]),label=k,**kwargs) 
	else:
		for k in errorDict.keys():
			ax.plot(freq, complex2dB(errorDict[k]),label=k,**kwargs) 
	ax.axhline(0,color='black')		
	plb.axis('tight')
	plb.legend(loc='best')
	plb.xlabel('Frequency (GHz)')  # this shouldnt be hard coded
	plb.ylabel('Magnitude (dB)')
	plb.grid(1)
	plb.draw()
	
def plotErrorCoefsFromDictPhase(errorDict,freq= None,ax = None,unwrap=False, **kwargs):
	if not ax:
		ax = plb.gca()
	
	if freq == None:	
		for k in errorDict.keys():
			if unwrap:
				ax.plot(rad2deg(npy.unwrap(npy.angle(errorDict[k]))),label=k,**kwargs) 
			else:
				ax.plot(npy.angle(errorDict[k],deg=True),label=k,**kwargs) 
	else:
		for k in errorDict.keys():
			if unwrap:
				ax.plot(freq,rad2deg(npy.unwrap(npy.angle(errorDict[k]))),label=k,**kwargs) 
			else:
				ax.plot(freq,npy.angle(errorDict[k],deg=True),label=k,**kwargs) 
			 
			
	ax.axhline(0,color='black')		
	plb.axis('tight')
	plb.legend(loc='best')
	plb.xlabel('Frequency (GHz)') 
	plb.ylabel('Magnitude (dB)')
	plb.grid(1)
	plb.draw()
def turnLegendOff(): 
	plb.gca().legend_.set_visible(0)
	plb.draw()
############### network theory  ################
## base network class.
class ntwk(object):
	'''
	class represents a generic n-port network. 
	
	provides:
		s - a kxnxn matrix of complex values representing a n-port 
			network over a given frequency range of len k, numpy.ndarray
		freq - 1D frequency vector,  numpy.ndarray
		freqUnit - meaning frequency vector, string, ( MHz, GHz, etc)
		freqMultiplier - scale of frequency vector to 1Hz, string.
			( ie, 1e9 for freqUnit of 'GHz'
		paramType - parameter types  ( 's','z','y','abcd') , string
		z0 - characteristic impedance of network
		name - name of network, string. used in legend for plotting.
		
		smag - kxnxn array representing magnitude of s-parameters in 
			decimal, ndarray 
		sdB - kxnxn array representing magnitude of s-parameters in
			decibel (20*npy.log10(mag)) scale, ndarray
		sdeg - kxnxn array representing phase of s-parameters in deg,
			ndarray
		srad - kxnxn array representing phase of s-parameters in radians, 
			ndarray
		
		
		
		note: these matricies may be re-shaped if one wants the 
			frequency index to come last, like  
			myntwk.s.transpose().reshape(2,2,-1)
	'''
	def __set_s(self,sMatrix):
		'''
		TODO make this work so that __set_s is called when s is assinged
		update a ntwk's sparameters. ntwk.s effects other object within 
		the ntwk class, so this function is needed.
		'''

		data = npy.array(sMatrix) 
		if len(data.shape) == 1:
			# they gave us 1D array
			data = data.reshape(-1,1,1)
		if data.shape[1] != data.shape[2]:
			print ('ERROR: input data must be kxmxm, where k is frequency axis')
			raise RuntimeError
			return None
		self._s = data
		self.rank = data.shape[1]
		self.smag = abs(self._s)
		self.sdB = mag2dB( self.smag )
		self.sdeg = npy.angle(self._s, deg=True)
		self.srad = npy.angle(self._s)
	def __get_s(self):
		return self._s	
	s = property(__get_s, __set_s)
	
	def __init__(self, data=npy.zeros(shape=(1,2,2)), freq=None, freqUnit='GHz', freqMultiplier = 1e9, paramType='s',  z0=50, name = ''):
		'''
		takes:
			data - a kxnxn matrix of complex values representing a n-port network over a given frequency range, numpy.ndarray
			freq - 1D frequency vector,  numpy.ndarray
			freqUnit - meaning frequency vector, string, ( MHz, GHz, etc)
			freqMultiplier - scale of frequency vector to 1Hz, string.  ( ie, 1e9 for freqUnit of 'GHz'
			paramType - parameter types  ( 's','z','y','abcd') , string
			z0 - characteristic impedance of network
			name - name of network, string. used in legend for plotting.
		'''
		## input checking : format, shape, and existence of f
		if paramType not in 'szyabcd':
			print( paramType +' is not a valid parameter type')
			return None
		else:
			self.paramType = paramType
		
		#explicity typecasting so we can use npy.array functions
		# TOD0: do a try/except as well 	
		data = npy.array(data) 
		if len(data.shape) == 1:
			# they gave us 1D array
			data = data.reshape(-1,1,1)
		if data.shape[1] != data.shape[2]:
			raise IndexError('input data must be kxmxm, where k is frequency axis')
			return None
		if freq == None:
			self.freq = freq #None
			self.freqUnit = None
			self.freqMultiplier=None
		else:
			if len(freq) != data.shape[0]:
				raise IndexError('length of f must match data.shape[2]. There must be as many frequency points as there are	s parameter measurements.')
				return None
			else:
				#they passed a valid f vector
				self.freq = freq
				self.freqUnit = freqUnit
				#TODO:  handle this 
				self.freqMultiplier = freqMultiplier
				
		self.name = name
		self.z0 = z0
		self.rank = data.shape[1]
		self.length = data.shape[0]
		## interpret paramType
		if self.paramType == 's':
			self.s = (npy.complex_(data))
			# this is where we should calculate and assign all the other ntwk formats
			#npy.zeros(shape=(self.rank,self.rank, self.length)))
			#	
		elif self.paramType == 'abcd':
			raise NotImplementedError
		elif self.paramType == 'z':
			raise NotImplementedError
		elif self.paramType == 'y':
			raise NotImplementedError
			
			
	
		
	
	

	def __sub__(self,A):
		'''
		B = self - A
		'''
		if A.rank != self.rank:
			raise IndexError('ntwks must be of the same rank')
		else:
			B = copy(self)
			B.s =(self.s - A.s)
			B.name = self.name+'-'+A.name
			return B

	def __add__(self,A):
		'''
		B = self - A
		'''
		if A.rank != self.rank:
			raise IndexError('ntwks must be of the same rank')
		else:
			B = copy(self)
			B.s= (self.s + A.s)
			return B	
		
	
	
	def __pow__(self, A):
		'''
		the power operator is overloaded to use the cascade 
		function,.
		
		if both operands, A and B,  are 2-port ntwk's:
			A ** B = cascade(A,B)
			B ** A = cascade(B,A)
		
		if A 2-port and B is  1-port ntwk:
			A ** B = terminate(A,B), B is attached at port 2
			B ** A = terminate(B,A), B is attached at port 1
		'''

		return cascade(self, A)

	def __floordiv__(self, A):
		'''
		the divide operator is overloaded to use the deEmbed or 
		function.
				
		C//A = deEmbed(C,A)
		
		note:
			this function is not a direct mapping to the division 
			operator. specifically, the de-Embed function is implemented 
			by multiplication of the inverse of a matrix. this operation
			does not commute.
			 
			for example, 
			assume * corresponds to cascade() function, and / the 
			deEmbed() function, then 
			
			if C = A*B
				C// A = A^-1 * C = A^-1 * A*B = B
				C//B != A
			if C = B*A
				C//B = A
				C//A != B 
				
		if you wish to deEmbed from port 2, see ntwk.flip
		'''
		return deEmbed(self,A)
	#def __repr__(self):
		#'''
		#overloaded for printing of summary, 
		
		#note:
			#this is not finished
		#'''								
		#outString =	'Name:\t' + self.name +'\n' + \
				#'#OfPorts:\t' + repr(self.rank)+'\n'+\
				#'Type :\t' + self.paramType +'\n' +\
				#'Frequency Span:\t' + repr(self.freq[0])+ '\t'+repr(self.freq[-1])+'\n' +\
				#'z0:\t' + repr(self.z0) 
		#return Non
	def __mul__(self,A):
		'''
		see multiply
		'''
		return self.multiply(self,A)
	
	def __div__(self,A):
		'''
		see divide
		'''
		return self.divide(A)
	
	def divide(self,A):
		if A.rank != self.rank:
			raise IndexError('ntwks must be of the same rank')
		else:
			B = copy(self)
			B.s =(self.s/ A.s)
			B.name = self.name+'/'+A.name
			return B
	def multiply(self,A):
		if A.rank != self.rank:
			raise IndexError('ntwks must be of the same rank')
		else:
			B = copy(self)
			B.s =(self.s * A.s)
			B.name = self.name+'*'+A.name
			return B
	@classmethod
	def average(self, listOfNtwks):
		numNtwks = len(listOfNtwks)
		sumNtwks  = copy (listOfNtwks[0])
		for aNtwk in listOfNtwks[1:]:
			sumNtwks += aNtwk
		
		sumNtwks.s = sumNtwks.s/numNtwks
		return sumNtwks
		
		  
	def resizeFreq(self, numPoints):
		'''
		WARNING: DOES NOT WORK YET!!
		changes the number of points for the ntwk. this alters the 
		frequency axis, and uses linear piece-wise interpolation on the 
		real and imaginary parts of the s-parameters.
		see	numpy.interp for details.
		
		takes:
			numPoints: the number of points, desired for the ntwk
		
		returns:
			None
		
		'''
		raise NotImplimentedError
		newFreq = linspace(self.freq[0],self.freq[-1],numPoints)
		newS = zeros(shape=(numPoints, self.rank, self.rank))
		
		for m in range(self.rank):
			for n in range(self.rank):
				newS[:,m,n] = interp(newFreq, self.freq, real(self.s[:,m,n]))+\
					1j*interp(newFreq, self.freq, imag(self.s[:,m,n]))
				
		self.s = newS
		self.freq = newFreq		
	
	def flip(self):
		'''
		invert the ports of a networks s-matrix, 'flipping' it over
		
		takes:
			nothing. (self)
		returns:
			flipped version of self. ntwk type
		
		note:
			only works for 2-ports at the moment
		
		there is probably a good matrix implementation of this i dont 
		know it. 
		'''
		if self.rank == 1:
			raise IndexError('how do you flip a 1-port?')
		elif self.rank == 2: 
			flippedS = npy.array([ [self.s[:,1,1],self.s[:,0,1]],\
									[self.s[:,1,0],self.s[:,0,0]] ]).transpose().reshape(-1,2,2)
			flipNtwk = copy(self)
			flipNtwk.s=(flippedS)
			return flipNtwk
		else:
			raise NotImplementedError
			
	def plotdB(self, m=None,n=None, ax=None,**kwargs):
		'''
		plots a given parameter in log mag mode. (20*npy.log10(mag))
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if ( m==None or n==None) and (self.rank > 1):
			# if this ntwk is not a 1-port and they did not pass indecies raise error
			raise RuntimeError('Error: please specify indecies.')
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
			
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
		
		
		ax1.axhline(0,color='black')
		
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			ax1.plot(self.sdB[:,m,n],label=labelString,**kwargs)
			plb.xlabel('Index')
		else:
			ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,m,n],label=labelString,**kwargs)
			plb.xlabel('Frequency (' + self.freqUnit +')') 
			plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		
		#plb.axis('tight')
		plb.ylabel('Magnitude (dB)')
		plb.grid(1)
		plb.legend(loc='best')
		plb.draw()
	def plotMag(self, m=None,n=None, ax=None,**kwargs):
		'''
		plots a given parameter in log mag mode. (20*npy.log10(mag))
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if ( m==None or n==None) and (self.rank > 1):
			# if this ntwk is not a 1-port and they did not pass indecies raise error
			raise RuntimeError('Error: please specify indecies.')
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
			
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
		
		
		ax1.axhline(0,color='black')
		
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			ax1.plot(self.smag[:,m,n],label=labelString,**kwargs)
			plb.xlabel('Index')
		else:
			ax1.plot(self.freq/self.freqMultiplier, self.smag[:,m,n],label=labelString,**kwargs)
			plb.xlabel('Frequency (' + self.freqUnit +')') 
			plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		
		#plb.axis('tight')
		plb.ylabel('Magnitude (not dB)')
		plb.grid(1)
		plb.legend(loc='best')
		plb.draw()	
	def plotSmith(self, m=None,n=None, smithRadius = 1, ax=None,  **kwargs):
		'''
		plots a given parameter in polar mode on a smith chart.
				
		takes:
			m - first index, int
			n - second indext, int
			smithRadius - radius of smith chart
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if ( m==None or n==None) and (self.rank > 1):
			# if this ntwk is not a 1-port and they did not pass indecies raise error
			print 'Error: please specify indecies.'
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
			
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
		
		ax1.plot(npy.real(self.s[:,m,n]), npy.imag(self.s[:,m,n]) ,label=labelString,**kwargs)
		smith(smithRadius)
		plb.legend(loc='best')
		plb.draw()
		
	
	def plotPhase(self, m=None,n=None, ax=None,unwrap=False, **kwargs):
		'''
		plots a given parameter in phase (deg) mode. 
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if ( m==None or n==None) and (self.rank > 1):
			# if this ntwk is not a 1-port and they did not pass indecies raise error
			raise RuntimeError('Error: please specify indecies.')
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
			
		ax1.axhline(0,color='black')
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			if unwrap:
				ax1.plot(rad2deg(npy.unwrap(self.srad[:,m,n])),\
					label=labelString,**kwargs)
			else:
				ax1.plot(self.sdeg[:,m,n],label=labelString,**kwargs)
		else:
			if unwrap:
				ax1.plot(self.freq/self.freqMultiplier, \
					rad2deg(npy.unwrap(self.srad[:,m,n])),\
					label=labelString,**kwargs)
		
			else:
				ax1.plot(self.freq/self.freqMultiplier, self.sdeg[:,m,n],\
					label=labelString,**kwargs)
		
		
		plb.axis('tight')
		plb.xlabel('Frequency (' + self.freqUnit +')') 
		plb.ylabel('Phase (deg)')
		
		plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		plb.grid(1)
		plb.legend(loc='best')
		plb.draw()
	
	def plotReturnLossDb(self, ax= None, **kwargs):
		'''
		plots all return loss parameters in log mag mode. meaning all parameters such with matching indecies, m=n.  
		
		takes:
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
			
		for p in range(self.rank):
			try :
				labelString  = self.name+', S'+repr(p+1) + repr(p+1)
			except (TypeError):
				labelString  = 'S'+repr(p+1) + repr(p+1)
			if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
				ax1.plot(self.sdB[:,p,p],label=labelString,**kwargs)
				plb.xlabel('Index')
			else:
				ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,p,p],label=labelString,**kwargs)
				plb.xlabel('Frequency (' + self.freqUnit +')') 
				plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])

		ax1.axhline(0,color='black')
		try:
			plb.title(self.name + ', Return Loss')
		except(TypeError):
			plb.title('Return Loss')
		plb.legend(loc='best')	
		plb.axis('tight')
		plb.ylabel('Magnitude (dB)')
		plb.grid(1)
		plb.draw()
		
	
	def plotInsertionLossDb(self, ax = None, **kwargs):
		'''
		plots all Insertion loss parameters in log mag mode. meaning all parameters such with matching indecies, m!=n.  
		
		takes:
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		if self.rank == 1:
			print 'one port networks dont have insertion loss dummy.'
			return None
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
			
		for p in range(self.rank):
			for q in range(self.rank):
				if p!=q:
					labelString  = self.name+', S'+repr(p+1) + repr(q+1)
					if self.freq == None:
					# this network doesnt have a frequency axis, just plot it  
						ax1.plot(self.sdB[:,p,q],label=labelString,**kwargs)
					else:
						ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,p,q],label=labelString,**kwargs)
		ax1.axhline(0,color='black')
		try:
			plb.title(self.name + ', Insertion Loss')
		except(TypeError):
			plb.title('Insertion Loss')
		plb.legend(loc='best')				
		plb.axis('tight')
		plb.xlabel('Frequency (' + self.freqUnit +')') 
		plb.ylabel('Magnitude (dB)')
		plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		plb.grid(1)
		plb.draw()
	
	def plotAllDb(self, ax = None, twinAxis = True, **kwargs):
		'''
		plots all parameters in log mag mode. 
		
		takes:
			ax - matplotlib.axes object to plot on, used in case you want to update an existing plot. 
			twinAxis - plot on one or two y-axis, boolean. 
			**kwargs - passed to the matplotlib.plot command
		
		note: has a bug with the legend , when using twinx 
		'''
		if self.rank == 1:
			self.plotdB(0,0, ax, **kwargs)
			return None
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
			
		for p in range(self.rank):
			for q in range(self.rank):
				if p!=q:
					labelString  = self.name+', S'+repr(p+1) + repr(q+1)
					if self.freq == None:
					# this network doesnt have a frequency axis, just plot it  
						ax1.plot(self.sdB[:,p,q],label=labelString,**kwargs)
					else:
						ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,p,q],label=labelString,**kwargs)
					
		if twinAxis == True:
			plb.legend(loc='best')	
			plb.twinx()
			ax1 = plb.gca()
			
			
		for p in range(self.rank):
			labelString  = self.name+', S'+repr(p+1) + repr(p+1)
			if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
				ax1.plot(self.sdB[:,p,p],label=labelString,**kwargs)
			else:
				ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,p,p],label=labelString,**kwargs)
		
		
		plb.legend(loc='best')				
		plb.axis('tight')
		plb.xlabel('Frequency (' + self.freqUnit +')') 
		plb.ylabel('Magnitude (dB)')
		plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		plb.grid(0)
		plb.draw()
	
	def plotPassivityMetric(self,ax = None, **kwargs):
		labelString = self.name
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
			
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			ax1.plot(self.passivityMetric()[0],label=labelString+ 'at port 1',**kwargs)
			ax1.plot( self.passivityMetric()[1],label=labelString+ 'at port 2',**kwargs)
			plb.xlabel('Index')
		else:
			ax1.plot(self.freq/self.freqMultiplier, self.passivityMetric()[0],label=labelString+ ', $|S_{11}|^{2} +|S_{21}|^{2}$',**kwargs)
			ax1.plot(self.freq/self.freqMultiplier, self.passivityMetric()[1],label=labelString+ ', $|S_{22}|^{2} +|S_{12}|^{2}$',**kwargs)
			plb.xlabel('Frequency (' + self.freqUnit +')') 
			plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		
		#plb.axis('tight')
		plb.ylabel('Magnitude')
		plb.grid(1)
		plb.legend(loc='best')
		plb.title(self.name + ', Passivity Metric')
		plb.draw()
		
		
	def loadFromTouchstone(self,filename):
		'''
		loads relevent values from a touchstone file. 
		takes:
			filename - touchstone file name, string. 
		
		note: all work is tone in the touchstone class
		'''
		touchstoneFile = touch(filename)
		
		self.paramType=touchstoneFile.get_format().split()[1]
		self.rank = touchstoneFile.rank
		self.z0 = float(touchstoneFile.resistance)
		self.freqUnit = touchstoneFile.frequency_unit
		self.freqMultiplier = touchstoneFile.frequency_mult
		self.name=touchstoneFile.filename.split('/')[-1].split('.')[-2]
		self.freq, self.s = touchstoneFile.get_sparameter_arrays() # note freq in Hz
		#self.freq = self.freq /self.freqMultiplier
		self.length = len(self.freq)
		#convinience	
		self.smag = abs(self.s)
		self.sdB = mag2dB( self.smag )
		self.sdeg = npy.angle(self.s, deg=True)
		self.srad = npy.angle(self.s)
		
		
		
	def writeToTouchstone(self,filename, format = 'ri'):
		'''
		write a touchstone file representing this network. 
		
		takes: 
			filename - filename to save to , string
			format - format to save in, string. ['ri']
		note:in the future could make possible use of the touchtone class, but at the moment this would not provide any benefit as it has not set_ functions. 
		'''
		format = format.upper()
		if format not in 'RI':
			print 'ERROR:format not acceptable. only ri is supported at this time'
			raise NotImplementedError
			return None
		
		if os.access(filename,1):
			# TODO: prompt for file removal
			os.remove(filename)
		outputFile=open(filename,"w")
		
		# write header file. note: the #  line is NOT a comment it is 
		#essential and it must be exactly this format, to work
		# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
		
		#TODO check format string for acceptable values and do somthing with it
		outputFile.write("# " + self.freqUnit.upper() +' '+ self.paramType.upper() + ' '+'RI' + " R " + str(self.z0) +" \n")
		
		#write comment line for users
		outputFile.write ("!freq\t")
		for q in range(self.rank):
			for p in range(self.rank):
				outputFile.write("Re" +'S'+`p+1`+ `q+1`+  "\tIm"+'S'+`p+1`+ `q+1`+'\t')
		outputFile.write('\n')		
		
		# write out data, note: this could be done with matrix manipulations, but its more readable to me this way
		for k in range(len(self.freq)):
			outputFile.write(str(self.freq[k]/self.freqMultiplier)+'\t')
			for q in range(self.rank):
				for p in range(self.rank):
					outputFile.write( str(npy.real(self.s[k,p,q])) + '\t' + str(npy.imag(self.s[k,p,q])) +'\t')
			outputFile.write('\n')

	def passivityMetric(self):
		'''
		takes:
			na
		returns:
			port1Power, port2Power : tuple containing total Power
				associated with each port
		'''
		if self.rank == 2:
			port1Power = npy.abs(self.s[:,0,0])**2+npy.abs(self.s[:,1,0])**2
			port2Power = npy.abs(self.s[:,1,1])**2+npy.abs(self.s[:,0,1])**2
			return port1Power, port2Power
		else:
			raise IndexError('only 2-ports suported for now')
			return None
	def transform2TimeDomain(self,timeMultiplier = 1e-12, timeUnit='ps', **kwargs):
		B = copy(self)
		#TODO: should loop and do this for all s-parameters, not just s11
		timeAxis, newS = psd2TimeDomain(self.freq, self.s[:,0,0], **kwargs)
		B = ntwk(data = newS,freq=timeAxis, name= self.name, \
			freqMultiplier = timeMultiplier, freqUnit=timeUnit)
		return B
		
	def changeFreqUnit(self, newUnit='GHz'):
		freqUnitDict = {\
			'hz':'Hz',\
			'mhz':'MHz',\
			'ghz':'GHz'\
			}
		freqMultiplierDict={
			'hz':1,\
			'mhz':1e6,\
			'ghz':1e9\
			}
			
		newUnit = newUnit.lower()
		if freqUnitDict.get(newUnit,False):
			self.freqUnit = freqUnitDict.get(newUnit)
			self.freqMultiplier = freqMultiplierDict.get(newUnit)
		else:
			raise ValueError('new freq unit invalid')
	
	def attenuate(self,attenuation):
		'''
		attentuates the network by some amount given in dB
		'''	
		if self.rank != 1:
			raise NotImplementedError('you are attenuating >1port, ntwk. not sure exactly where to put attenuator.')
		else:
			attenuation = dB2Mag(attenuation)
			self.s = attenuation* self.s
		
		
		
		
def createNtwkFromTouchstone(filename):
	'''
	creates a ntwk object from a given touchstone file
	
	takes:
		filename - touchstone file to read, string.
	returns:
		mwavepy.ntwk onbject representing the network contained in the touchstone file. 
	'''
	myntwk = ntwk()
	myntwk.loadFromTouchstone(filename)
	return myntwk	
def loadAllTouchstonesInDir(dir = '.'):
	'''
	loads all touchtone files in a given dir 
	
	takes:
		dir  - the path to the dir, passed as a string (defalut is cwd)
	returns:
		ntwkDict - a Dictonary with keys equal to the file name (without
			a suffix), and values equal to the corresponding ntwk types
	
		
	'''
	ntwkDict = {}

	for f in os.listdir (dir):
		# TODO: make this s?p with reg ex
		if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
			name = f[:-4]
			ntwkDict[name]=(createNtwkFromTouchstone(dir +'/'+f))
			
		
	return ntwkDict	
def passivityTest(smat):
	'''
	check that the network represented by scattering parameter matrix (smat) is passive. returns a matrix the kxnxn, of which each k-slice is  I-smat*conj(traspose(S))
	takes:
		smat - scattering  matrix, numpy.array in shape of kxnxn.
	returns:
		passivity - matrix containing I-smat*conj(tra,spose(S))
		
		note: to be considered passive the elements allong the diagonal should be <= zero
	'''
	passivity = npy.zeros(smat.shape)
	
	for f in range(smat.shape[0]):
		passivity[f,:,:] = npy.eye(smat.shape[1]) - npy.dot(smat[f,:,:],smat[f,:,:].conj().transpose())
			#for tmp in  eigvals(passivity[:,:,f]):
				#if real(tmp) < 0:
					#if abs(tmp) < tol:
						## structure fails the passivity test
						#return False
			#return True
	return passivity










## connections
def cascade(A,B):
	'''
	cascades two  networks together, and returns resultant network
	
	takes:
		A - network at port 1. can be:
				1-port or 2-port mwavepy.ntwk
				2x2, kx2x2, 1x1, or kx1x1 numpy.ndarray 
				sympy.Matrix
			all which represent an S-matrix. 
		
		B - network at port 2. can be:
				1-port or 2-port mwavepy.ntwk
				2x2, kx2x2, 1x1, or kx1x1 numpy.ndarray 
				sympy.Matrix
			all which represent an S-matrix. 
		
	returns: 
		C - resultant network, of cascaded A, B. type is the
			same as the type passed. 
			
	note:
		A and B can not both be 1-ports, duh.
	
	
	'''
	
	try:
		if (A.rank == 2 and B.rank == 2) or \
		(A.rank == 2 or  B.rank == 2) and (A.rank == 1 or B.rank == 1):
			# this reads: both are 2-port OR  
			# 		ones a 2-port and one is a 1-port
			if A.rank > B.rank: 
				C = copy(B)
			else:
				C = copy(A)
			
			C.s=(cascade(A.s, B.s))
			C.name = A.name + ' Cascaded With ' + B.name
			return C
		else:	
			raise ValueError('Invalid rank for input ntwk\'s, see help')
		

	
	except:
		pass	
	if isinstance(A, npy.ndarray) and isinstance(B, npy.ndarray):
		# most of this code is just dumb shape checking.  the actual
		# calculation very simple, its at the end
		
		if len( A.shape ) > 2  or len( B.shape ) > 2 :
			# we have a multiple frequencies, ie  kxnxn, so we are going
			# to recursively call ourselves for each freq point
			if A.shape[0]  == B.shape[0]:
				# the returned netwok should be a 1-port if either input
				# is  a 1-port
				if A.shape[1] > B.shape[1]: 
					C = B.copy()
				else:
					#assert: A is a 1-port or A and B are 2-ports
					C = A.copy()
				
				for k in range(A.shape[0]):
					C[k] = cascade(A[k],B[k]) # C[k] = C[k,:,:]
				return C
			
			else:
				raise IndexError('A.shape[0] ! = B.shape[0], \
				meaning, they are not the same length along the \
				frequency axis')
			
		else:
			# we have only 1 frequency point	
			if A.shape == (2,2) and B.shape == (2,2):
				# A and B are 2-port s matricies
				C = A.copy()
				
				C[1,0] = (A[1,0]*B[1,0]) /\
						(1 - A[1,1]*B[0,0])
				C[0,1] = (A[0,1]*B[0,1]) /\
						(1 - A[1,1]*B[0,0])
				
				C[0,0] = A[0,0]+(A[1,0]*B[0,0]*A[0,1] )/\
						(1 - A[1,1]*B[0,0])
				C[1,1] = B[1,1]+(B[0,1]*A[1,1]*B[1,0] )/\
						(1 - A[1,1]*B[0,0])
				return C
				
			elif A.shape == (2,2) and ( B.shape == (1,1) or B.shape == (1,)):
				# A is a 2-port and B is  a 1-port s-matrix, C will be 1-port
				if B.shape == (1,1):
					C = B.copy()	
					C[0,0] = A[0,0]+(A[1,0]*B[0,0]*A[0,1] )/\
							(1 - A[1,1]*B[0,0])
					return C
				elif B.shape == (1,):
					C = B.copy()	
					C[0] = A[0,0]+(A[1,0]*B[0]*A[0,1] )/\
							(1 - A[1,1]*B[0])
					return C
				else:
					raise IndexError('bad shape. this is a coding error.')
					
			
				
			elif B.shape == (2,2) and ( A.shape == (1,1) or A.shape == (1,)):
				# B is a 2-port A is  a 1-port s-matrix, C will be 1-port
				if A.shape == (1,1):
					C = A.copy()
					C[0,0] = B[0,0]+(B[1,0]*A[0,0]*B[0,1] )/\
							(1 - B[1,1]*A[0,0])
					return C
				elif A.shape == (1,):
					C = A.copy()
					C[0,0] = B[0,0]+(B[1,0]*A[0]*B[0,1] )/\
							(1 - B[1,1]*A[0])
					return C	
					
			else:
				raise IndexError('invalid shapes of A or B,see help')
		
	
	elif isinstance(A, sym.Matrix) and isinstance(B, sym.Matrix):
		# just typecast to ndarray, calculate series network then 
		# re-cast into sym.Matrix
		A = npy.array(A)
		B = npy.array(B)
		C = cascade(A,B)
		return sym.Matrix(C)	
	else:
		raise TypeError('A and B must be mwavepy.ntwk or numpy.ndarray')
		

def deEmbed(C, A):
	'''
	calculates the de-embed network of A  embeded behind B, 
	from measurement C. ie C = A * B
	
	takes:
		C - cascade(A, B), mwavepy.ntwk type or a numpy.ndarray 
		A - the 2-port network embeding the network. a 2-port 
			mwavepy.ntwk type or a kx2x2 numpy.ndarray.
			
			can also be a tuple of the above mentioned types. In this 
			case, A[0] is deEmbeded on port 1, and A[1] is deEmbeded
			on port2.
	
	returns:
		B - the unembeded network behind B. type depends on 
			input.
	
	note:
		this function has a directionality assumption. C must be a 
		cascaded connection of A.port2 = B.port1, for B to be de-embeded
		by passing C, and A for example, 
		
		assume * corresponds to cascade function, and / the deEmbed()
		if C = A*B
			C/ A = B 
			C/B != A
		if C = B*A
			C/B = A
			C/A != B 
	
		
		if a tuple is passed for A then what you get is
			A[0].inv * C * A[1].inv
		where .inv is the inverse cascading network.  
	
	see ntwk.flip
	'''
	
	try:
		# they may  have passed us ntwk types, so lets get the relevent parameter
		#make sure its a 1-port
		if ( C.rank == 2 and A.rank == 2) or (C.rank ==1 and A.rank == 2):
			B = copy(C)
			B.s=(deEmbed(C.s, A.s))
			#B.name = A.name + ' De-Emedded From ' + B.name
			return B
		else:
			raise IndexError('C and A are or incorrect rank. see help.')

	except:
		pass
	#TODO: look into implementing this with a cascade inverse network. 
	# so C = A*B , B = A^-1*A*B 
	if isinstance(C,ntwk) and isinstance(A,tuple):
		# if they passed 1 ntwks and a tuple of ntwks, 
		# then deEmbed like A.inv*C*B.inv
		B = A[1]
		A = A[0]
		return deEmbed( deEmbed(C,A).flip(),B).flip()
	
	
	
	elif isinstance(C, npy.ndarray) and isinstance(A, npy.ndarray):
		if A.shape != (2,2):
			# A is not a 2x2 array.
			if len(A.shape) == 3 and A.shape[1:3] == (2,2):	
				# A is kx2x2
				if len(C.shape) == 3 or len(C.shape) == 2:
					# is either a kxp or a kxpxq, passing allong
					#Both A and C are  kx2x2 matrix's we can handle it 
					if C.shape[0]  == A.shape[0]:
						B = C.copy()
						for k in range(C.shape[0]):
							B[k] = deEmbed(C[k],A[k]) # C[k] = C[k,:,:]
						return B
					else:
						print 'ntwk length error'
						raise IndexError('A.shape[0] ! = C.shape[0], '+\
						'meaning, they are not the same length along the '+\
						'frequency axis')
				else:
					raise IndexError('invalid shapes of C, must be kx2x2\
					or kx1x1 or kx1, where k is arbitrary')
			else:
				raise IndexError('invalid shapes of A or B, must be 2x2\
					or kx2x2, where k is arbitrary')
	
		else:
			# A is a 2x2 smat 
			assert A.shape == (2,2)
			if C.shape == (2,2):
				# A is a 2-port and C is a 2-port 
				assert  A.shape == C.shape == (2,2)
				
				B = C.copy()
				
				B[0,0] =((C[0,0] - A[0,0]) /\
					(A[1,0]*A[0,1]-A[0,0]*A[1,1] + C[0,0]*A[1,1]))
				
				B[0,1] = (C[0,1]-C[0,1]*A[1,1]*B[0,0])/\
							A[0,1]
				B[1,0] = (C[1,0]-C[1,0]*A[1,1]*B[0,0])/\
							A[1,0]
				B[1,1] = C[1,1] - (B[1,0]*B[0,1]*A[1,1]) /\
									(1 - A[1,1]*B[0,0])
				return  B
			
			elif C.shape == (1,):
				# C is a 1-port and A is a 2-port
				B = C.copy()
				B[0] =((C[0] - A[0,0]) /\
					(A[1,0]*A[0,1]-A[0,0]*A[1,1] + C[0]*A[1,1]))
				return B					
			elif C.shape == (1,1):
				# C is a 1-port, and A is a 2-port
				B = C.copy()
				B[0,0] =((C[0,0] - A[0,0]) /\
					(A[1,0]*A[0,1]-A[0,0]*A[1,1] + C[0,0]*A[1,1]))
				return B
			else:
				raise IndexError ('Input networks have invalid shapes. see\
				help for more details. ')
	elif isinstance(C, sym.Matrix) and isinstance(A, sym.Matrix):
		# just typecast to ndarray, calculate series network then 
		# re-cast into sym.Matrix
		A = npy.array(A)
		C = npy.array(C)
		B = deEmbed(C,A)
		return sym.Matrix(B)
			
	else:
		print 'type(A)= ',type(A)
		print 'type(C)=',type(C)
		raise TypeError('A and B must be mwavepy.ntwk or numpy.ndarray')
			
	
			


## network representation conversions
# these conversions where taken from Pozar. Microwave Engineering sec 5.6
def s2r(s):
	'''
	converts a scattering parameter matrix to a wave cascading 
	matrix, aka cascading matrix, R .
		
		b1 			a2		b1		a1
		a1	=	R *	b2, 	b2 = S* a2 
	
	
	
	takes:
		s : scattering matrix, a numpy.ndarray or a sympy.Matrix of 
			shape kx2x2 or 2x2,
			. 
	returns:
		r : cascading matrix, a numpy.ndarray or a sympy.Matrix of 
			shape kx2x2 or 2x2,
			
	note:
		see, 'thru-reflect-line: an improved techniqu for calibrating
			the dual siz-port automatic network analyzer' by Engen and 
			Hoer for more info: 
	'''
	if isinstance (s, sym.Matrix):
		s = npy.array(s)
		r = s2r(s)
		return sym.Matrix(r)
	if isinstance(s,npy.ndarray):
		if len( s.shape ) > 2:
			r = s.copy()
			for k in range(s.shape[0]):
				r[k] = s2r(s[k])
			return r
		else:
			if s.shape == (2,2):
				# we have 1 frequency point of a 2-port
				try:
					r = 1./s[1,0] * \
					npy.array([	[s[1,0]*s[0,1]-s[0,0]*s[1,1], s[0,0] ],\
											[1,	-s[1,1]]])
				except ZeroDivisionError as details:
					print details, ' s21 is 0. why would you \
					cascade this? it doesnt make sense'
					return None
				
				return r
					
			else:
				raise IndexError('bad shape. must be a kx2x2 or 2x2')
		
	else:
		raise TypeError('input must be a numpy.ndarray')
		
def r2s(r):
	'''
	converts a scattering parameter matrix to a wave cascading 
	matrix, aka cascading matrix, R .
		
		b1 			a2		b1		a1
		a1	=	R *	b2, 	b2 = S* a2 
	
	
	
	takes:
		r : cascading matrix, a numpy.ndarray or a sympy.Matrix of 
			shape kx2x2 or 2x2,
			. 
	returns:
		r : scattering matrix, a numpy.ndarray or a sympy.Matrix of 
			shape kx2x2 or 2x2,
			
	note:
		see, 'thru-reflect-line: an improved techniqu for calibrating
			the dual siz-port automatic network analyzer' by Engen and 
			Hoer for more info: 
	'''
	if isinstance (r, sym.Matrix):
		r = npy.array(r)
		s = r2s(r)
		return sym.Matrix(s)
	if isinstance(r,npy.ndarray):
		if len( r.shape ) > 2:
			s = r.copy()
			for k in range(r.shape[0]):
				s[k] = r2s(r[k])
			return s
		else:
			if r.shape == (2,2):
				# we have 1 frequency point of a 2-port
				try:
					s = 1./r[1,0] * \
					npy.array([ [ r[0,1], r[1,0]*r[0,0]- r[0,1]*r[1,1]],\
								[ 1,	-r[1,1] ] ])
				except ZeroDivisionError as details:
					print details, ' s21 is 0. why would you \
					cascade this? it doesnt make sense'
					return None
				
				return s
					
			else:
				raise IndexError('bad shape. must be a kx2x2 or 2x2')
		
	else:
		raise TypeError('input must be a numpy.ndarray')
		
		
		
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


## lumped elements
def zCapacitor(capacitance,frequency):
	'''
	calculates the impedance of a capacitor.
	
	takes:
		capacitance - capacitance, in Farads. can be a single value or 
			numpy.ndarray
		frequency - frequency at which to calculate capacitance. can be 
			a single value or a numpy.ndarray()
			
	returns:
		impedance - impedance of capacitor at given frequency. type 
			depends on input
	'''
	return 1/(1j*2*npy.pi*frequency*capacitance)

def zInductor(inductance,frequency):
	'''
	calculates the impedance of an inductor.
	
	takes:
		inductance - capacitance, in Henrys. can be a single value or 
			numpy.ndarray.
		frequency - frequency at which to calculate capacitance. can be 
			a single value or a numpy.ndarray()
			
	returns:
		impedance - impedance of inductor at given frequency. type 
			depends on input
	'''
	return (1j*2*npy.pi*frequency*inductance)

def seriesZ(z1, z2):
	'''
	calculates series connection of impedances.
	
	takes: 
		z1 - impedance 1.
		z2 - impedance 2
	returns:
		z1+z2
	'''
	return z1 + z2
	
def seriesY(y1,y2):
	'''
	calculates series connection of 2 admitances.
	
	takes: 
		z1 - impedance 1.
		z2 - impedance 2
	returns:
		z1+z2
	'''
	# TODO: fix this. it wont work for arrays	
	#if y1==0:
		#return y2
	#elif y2==0:
		#return y1
	#else:
	return parallel(1./y1,1./y2)	

def parallelY(y1,y2):
	'''
	calculates parallel connection of impedances.
	
	takes: 
		z1 - impedance 1.
		z2 - impedance 2
	returns:
		1/(1/z1+1/z2)
	'''
	return y1 + y2
	
	
def parallelZ(z1,z2):
	'''
	calculates parallel connection of impedances.
	
	takes: 
		z1 - impedance 1.
		z2 - impedance 2
	returns:
		1/(1/z1+1/z2)
	'''
	if z1 == 0 or z2==0:
		return 0
	else:
		return 1/(1./z1+1./z2)

## transmission line functions
def betaPlaneWave(omega,epsilonR = 1, muR = 1):
	'''
	propagation constant of a plane wave in given  material.
	takes:
		omega - radian angular frequency (rad/s)
		epsilonR - relative permativity (default = 1) 
		muR -  relative permiability (default = 1)
	returns:
		omega/c = omega*sqrt(epsilon*mu)
	'''
	return omega* sqrt((mu_0*muR)*(epsilonR*epsilon_0))

def beta0(omega):
	'''
	propagation constant of a free space.
	takes:
		omega - radian angular frequency (rad/s)
	returns:
		omega/c = omega*sqrt(epsilon*mu)
	'''
	return betaPlaneWave(omega,1,1)


	
def eta(epsilonR = 1, muR = 1):
	'''
	characteristic impedance of a material.
	takes:
		epsilonR - relative permativity (default = 1) 
		muR -  relative permiability (default = 1)
	'''
	return sqrt((mu_0*muR)/(epsilonR*epsilon_0))
def eta0():
	'''
	characteristic impedance of free space. see eta().
	'''
	return eta( 1,1)
	
def electricalLength( l , f0, beta=beta0,deg=False):
	'''
	calculates the electrical length of a section of transmission line.
	
	takes:
		l - length of line in meters
		f0 - frequency at which to calculate 
		beta - propagation constant, which is a function of angular frequency (omega), and returns a value with units radian/m.  can pass a function on the fly, like  electricalLength(freqVector, l, beta = lambda omega: omega/c )
		
		note: beta defaults to lossless free-space propagation constant mwavepy.beta0() = omega/c = omega*sqrt(epsilon_0*mu_0)
	returns:
		electrical length of tline, at f0 in radians
	'''
	if deg==False:
		return  beta(2*pi*f0 ) *l 
	elif deg ==True:
		return  rad2deg(beta(2*pi*f0 ) *l )

def Gamma(zl,z0=50.0, theta=0):
	'''
	calculates the reflection coefficient for a given load and characteristic impedance
	takes:
		zl - load impedance (can be numpy array or float)
		z0 - characteristic impedance ( can be numpy array or float)
		theta - distance from load, given in electrical length  (rad)
	'''
	# this way of type casting allows for arrays to be passed, but does floating points arrimetic
	#TODO: if arrays test for matching lengths of zl and z0
	
	zl = 1.0*(zl)
	z0 = 1.0*(z0)
	theta = 1.0* (theta)
	
	if isinstance(zl,npy.ndarray):
		# handle the limit of open circuit. for arrays
		zl[(zl==npy.inf)]=1e100
		gammaAt0 = (zl-z0)/(zl+z0)
	else:
		if zl == inf:
			gammaAt0 = 1
		else: 
			gammaAt0 = (zl-z0)/(zl+z0)
	return gammaAt0 * npy.exp(-2j*theta)


def zin(zl,z0,theta):
	'''
	returns the input impedance of a transmission line of character impedance z0 and electrical length el, terminated with a load impedance zl. 
	takes:
		zl - load impedance 
		z0 - characteristic impedance of tline
		theta - distance from load, given in electrical length  (rad)
	returns:
		input impedance ( in general complex)
	'''
	if zl == inf:
		return -1j*z0*1./(tan(theta))
	elif zl == 0:
		return 1j*z0*tan(theta)
	else:
		return z0 *	(zl + 1j*z0 * tan(theta)) /\
					(z0 + 1j*zl * tan(theta))

def zinShort (z0,theta):
	'''
	calculates input impedance of a short.
	convinience function. see zin()
	'''
	return zin(0,z0,theta)
	
def zinOpen(z0,theta):
	'''
	calculates input impedance of a open. 
	convinience function. see zin()
	'''
	return zin(inf,z0,theta)




def surfaceImpedance(omega, conductivity, epsilon=epsilon_0, mu=mu_0):
	'''
	calculates the surface impedance of a material defined by its 
	permativity, permiablit,  and conductivity.
	
	takes:
		omega: angular frequency (rad/s)
		conductivity: conductivity, ussually denoted by sigma (S/m)
		epsilon: permitivity (F/m)
		mu: permiability (H/m)
	
	returns:
		surfaceImpedance: complex surface impedance
	'''
	return sqrt((1j*omega*mu)/(conductivity + 1j*omega*epsilon))




############### transmission line class   ################
class frequencyBand:
	'''
	represents a frequency band. 
	
	usually we are doign calcluations in a given band , so this class 
	used in other classes so user doesnt have to continually supply 
	frequency info.
	'''
	freqUnitDict = {\
		'hz':'Hz',\
		'mhz':'MHz',\
		'ghz':'GHz'\
		}
	freqMultiplierDict={
		'hz':1,\
		'mhz':1e6,\
		'ghz':1e9\
		}
	def __init__(self,start, stop, npoints, unit='hz'):
		'''
		takes:
			start: start of band.  in Hz
			stop: end of band. in Hz
			npoints: number of points in the band. 
			unit: unit you want the band in for plots. a string. can be:
				'hz', 'mhz','ghz', 
			
		note: unit sets the property freqMultiplier, which is used 
		to scale the frequncy when formatedAxis is referenced.
			
		'''
		self.start = start
		self.stop = stop
		self.npoints = npoints
		self.unit = unit
		
	
	@property
	def multiplier(self):
		return self.freqMultiplierDict[self.unit.lower()]
	@property
	def	axis(self):
		'''
		returns a frequency axis scaled to the correct units
		the unit is stored in freqDict['freqUnit']
		'''
		return linspace(self.start,self.stop,self.npoints)
	@property
	def	formatedAxis(self):
		'''
		returns a frequency axis scaled to the correct units
		the unit is stored in freqDict['freqUnit']
		'''
		return linspace(self.start,self.stop,self.npoints)\
			/self.freqMultiplier
	

class transmissionLine:
	'''
	general super-class for TEM transmission lines
	'''
	def __init__(self, \
		distributedCapacitance,	distributedInductance,\
		distributedResistance, distributedConductance, frequencyBand=None ):
		
		self.distributedCapacitance = distributedCapacitance
		self.distributedInductance = distributedInductance
		self.distributedResistance = distributedResistance
		self.distributedConductance = distributedConductance

		self.frequencyBand = frequencyBand
		
	def distributedImpedance(self,omega):
		return self.distributedResistance+1j*omega*self.distributedInductance
	
	def distributedAdmittance(self,omega):
		return self.distributedConductance+1j*omega*self.distributedCapacitance
	# could put a test for losslessness here and choose whether to make this
	# a funtion of omega or not.
	def characteristicImpedance(self,omega):
		return sqrt(self.distributedImpedance(omega)/self.distributedAdmittance(omega))
	
	def propagationConstant(self,omega):
		return sqrt(self.distributedImpedance(omega)*self.distributedAdmittance(omega))
	
	#@classmethod
	def electricalLength(self, l , f=None, gamma=None,deg=False):
		'''
		calculates the electrical length of a section of transmission line.
	
		takes:
			l - length of line in meters
			f: frequency at which to calculate, array-like or float
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.  
			
		returns:
			electricalLength: electrical length in radians or degrees, 
				if deg =True
		note:
			you can pass a function on the fly, like  
			electricalLength(freqVector, l, beta = lambda omega: omega/c )
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if f is None:
			if  self.frequencyBand is None:
				raise ValueError('please supply frequency information')
			else:
				f = self.frequencyBand.axis
				
		if deg==False:
			return  gamma(2*pi*f ) *l 
		elif deg ==True:
			return  rad2deg(gamma(2*pi*f ) *l )
	
	
	#@classmethod
	def reflectionCoefficient(self, l,f,zl,z0=None, gamma=None):
		'''
		calculates the reflection coefficient for a given load 
		takes:
			l: distance of transmission line to load, in meters (float)
			f: frequency at which to calculate, array-like or float
			zl: load impedance. may be a function of omega (2*pi*f), or 
				a number 
			z0 - characteristic impedance may be a function of omega 
				(2*pi*f), or a number 
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if z0 is None:
			z0 = self.characteristicImpedance
		
		try:
			zl = zl(2*pi*f)
		except TypeError:
			pass
		try:
			z0 = z0(2*pi*f)
		except TypeError:
			pass
		
		if len(z0) != len(zl): raise IndexError('len(zl) != len(z0)')
			
		# flexible way to typecast ints, or arrays
		zl = 1.0*(zl)
		z0 = 1.0*(z0)
		l = 1.0* (l)
		
		theta = self.electricalLength(l,f, gamma=gamma)
		
		if isinstance(zl,npy.ndarray):
			# handle the limit of open circuit. for arrays
			zl[(zl==npy.inf)]=1e100
			gammaAt0 = (zl-z0)/(zl+z0)
		else:
			if zl == inf:
				gammaAt0 = 1
			else: 
				gammaAt0 = (zl-z0)/(zl+z0)
		
		gammaAtL =gammaAt0 * npy.exp(-2j*theta)
		return gammaAtL
	
	#@classmethod
	def inputImpedance(self, l,f, zl,z0=None,gamma=None):
		'''
		returns the input impedance of a transmission line of character impedance z0 and electrical length el, terminated with a load impedance zl. 
		takes:
			l: distance from load, in meters
			f: frequency at which to calculate, array-like or float 
			zl: load impedance. may be a function of omega (2*pi*f), or 
				a number 
			z0 - characteristic impedance may be a function of omega 
				(2*pi*f), or a number
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m.
		returns:
			input impedance ( in general complex)
			
		note:
			this can also be calculated in terms of reflectionCoefficient
		'''
		if gamma is None:
			gamma = self.propagationConstant
		if z0 is None:
			z0 = self.characteristicImpedance
		
		try:
			zl = zl(2*pi*f)
		except TypeError:
			pass
		try:
			z0 = z0(2*pi*f)
		except TypeError:
			pass
		
		theta = propagationConstant(l,2*pi*f, gamma=gamma)
		
		if zl == inf:
			return -1j*z0*1./(tan(theta))
		elif zl == 0:
			return 1j*z0*tan(theta)
		else:
			return z0 *	(zl + 1j*z0 * tan(theta)) /\
						(z0 + 1j*zl * tan(theta))
	

	
	
	
	def createNtwk_delayShort(self,l,f=None, gamma=None, **kwargs ):
		'''
		generate the reflection coefficient for a  delayed short of length l 
		
		takes:
			l - length of delay, in meters
			f: frequency axis. if self.frequencyBand exists then this 
				can left as None
			gamma: propagationConstant a function of angular frequency (omega), 
				and returns a value with units radian/m. can be omited.
			kwargs: passed to ntwk constructor
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''
		
		s = -1*exp(-1j* 2*self.electricalLength(l,f,gamma))
		return ntwk(data=s,paramType='s',freq=f,**kwargs)
		

class freespace(transmissionLine):
	'''
	represents freespace, defined by [possibly complex] values of relative 
	permativity and relative permeability
	'''
	def __init__(self, relativePermativity=1, relativePermeability=1,frequencyBand=None):
		transmissionLine.__init__(self,\
			distributedCapacitance = real(epsilon_0*relativePermativity),\
			distributedResistance = imag(epsilon_0*relativePermativity),\
			distributedInductance = real(mu_0*relativePermeability),\
			distributedConductance = imag(mu_0*relativePermeability),\
			frequencyBand = frequencyBand
			)
		
class coax:
	def __init__(self):
		raise NotImplementedError
		return None

class microstrip:
	def __init__(self):
		raise NotImplementedError
		return None
	def eEffMicrostrip(w,h,epR):
		'''
		The above formulas are in Transmission Line Design Handbook by Brian C Wadell, Artech House 1991. The main formula is attributable to Harold A. Wheeler and was published in, "Transmission-line properties of a strip on a dielectric sheet on a plane", IEEE Tran. Microwave Theory Tech., vol. MTT-25, pplb. 631-647, Aug. 1977. The effective dielectric constant formula is from: M. V. Schneider, "Microstrip lines for microwave integrated circuits," Bell Syst Tech. J., vol. 48, pplb. 1422-1444, 1969.
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

class coplanar:
	def __init__(self):
		raise NotImplementedError
		return None

		
	
class waveguide:
	'''
	class which represents a rectangular waveguide . 
	
	provides:
		a - width  in meters, float
		b - height in meters, float. 
		band - tuple defining max and min frequencies. defaults to (1.25,1.9)*fc10
		fStart - start of frequency band. ( = band[0] )
		fStop - stop of frequency band ( = band[1] )
		fc10 - first mode cut on frequency ( == fc(1,0)
		epsilonR - relative permativity of filling material 
		muR - relative permiability of filling material
	TODO: implement different filling materials, and wall material losses
	'''
	

		
	
	def __init__(self, a, b, band=None, epsilonR=1, muR=1, surfaceConductivity=None, name = None, points = 201):
		'''
		takes: 
			a - width  in meters, float
			b - height in meters, float. 
			band - tuple defining max and min frequencies. defaults to (1.25,1.9)*fc10
			epsilonR - relative permativity of filling material 
			muR - relative permiability of filling material
			surfaceConductivity - the conductivity of the waveguide 
				interior surface. (S/m)
		
		note: support for dielectrically filled waveguide, ie epsilonR 
			and muR, is not supported
		'''
		if name == None:
			self.name = 'waveguide,a='+ repr(a) + 'b='+repr(b)
		else:
			self.name = name
		
		self.a = a
		self.b = b
		
		self.fc10 = self.fc(1,0)
		if band == None: 
			self.band = npy.array([1.25*self.fc10 , 1.9 * self.fc10]) 
		else:
			self.band= band
		self.fStart = self.band[0]
		self.fStop = self.band[1]
		self.fCenter = (self.band[1]-self.band[0])/2. + self.band[0]
		
		self.epsilon = epsilonR * epsilon_0
		self.mu = muR * mu_0 
		self.eta = eta(epsilonR= epsilonR, muR= muR)
		
		self.freqAxis = npy.linspace(self.fStart,self.fStop,points)
		self.surfaceConductivity= surfaceConductivity
		#all for dominant mode
	
	def fc(self,m=1,n=0):
		'''
		calculates cut-on frequency of a given mode
		takes:
			m - mode indecie for width dimension, int
			n - mode indecie for height dimension, int
		returns:
			cut of frequency in Hz, float. 
			
		'''
		return c/(2*pi)*sqrt( (m*pi/self.a)**2 +(n*pi/self.b)**2)

	
	def alphaC(self, omega, m=1,n=0):
		'''
		calculates waveguide attenuation due to conductor loss
		
		takes:
			omega: angular frequency (rad/s)
			conductivity: surface material conductivity (usually 
				written as sigma)
			m: mode number along wide dimension  
			n: mode number along height dimmension
		
		returns:
			alphaC = attenuation in np/m.  ( neper/meter)
			
			
		note: only dominant mode (TE01)  is  supported at the moment.
		This equation and a derivation can be found in:
		
		harrington: time harmonic electromagnetic fields 
		balanis: advanced engineering electromagnectics
		'''
		if m != 1 or n != 0:
			raise NotImplementedError('only dominant mode (TE01)  is  \
				supported at the moment')
		
		f = omega/(2.*pi)
		
		Rs = npy.real( surfaceImpedance(omega=omega, \
			conductivity=self.surfaceConductivity, epsilon=self.epsilon,\
			mu=self.mu))
		
		
		return Rs/(self.eta*self.b) * (1+ 2*self.b/self.a *(self.fc(m,n)/(f))**2)/\
			sqrt(1-(self.fc(m,n)/(f))**2)
	
	
	
	def beta(self, omega,m=1,n=0):
		'''
		calculates the propagation constant of given mode 
		takes:
			omega - angular frequency (rad/s)
			m - mode indecie for width dimension, int
			n - mode indecie for height dimension, int
		returns:
			propagation constant (rad/m)
		
		TODO: should do a test below cutoff and handle imaginary sign
		
		'''
		k = omega/c
		if self.surfaceConductivity == None:
			return sqrt(k**2 - (m*pi/self.a)**2- (n*pi/self.b)**2)
		else:
			# include  the conductor loss associated with the surface
			# conductivity
			return sqrt(k**2 - (m*pi/self.a)**2- (n*pi/self.b)**2) - \
				1j*self.alphaC(omega=omega,m=m,n=n)
	def beta_f(self,f,m=1,n=0):
		'''
		convinience function. see beta()
		'''
		return self.beta(2*pi*f,m,n)
		
	def lambdaG(self,omega,m=1,n=0):
		'''
		calculates the guide wavelength  of a given mode
		takes:
			omega - angular frequency (rad/s)
			m - mode indecie for width dimension, int
			n - mode indecie for height dimension, int
		returns:
			guide wavelength (m)
		'''
		return real(2*pi / self.beta(omega,m,n))
	
	def lambdaG_f(self,f,m=1,n=0):
		'''
		convinience function. see lambdaG()
		'''
		return self.lambdaG(2*pi *f,m,n)
	
	def vp(self, omega,m=1,n=0):
		'''
		calculates the phase velocity of a given mode
		takes:
			omega - angular frequency (rad/s)
			m - mode indecie for width dimension, int
			n - mode indecie for height dimension, int
		returns:
			phase velocity of mode (m/s)
		'''
		return real(omega / self.beta(omega,m,n))
	
	def vp_f(self, f,m=1,n=0):
		return 2*pi*f / self.beta(2*pi*f,m,n)
		
	
	def electricalLength(self,l, deg = False):
		'''
		calculates the electrical length in radians for a waveguide of 
		length l, at band center. 
		
		takes:
			l: length of wavegumide section in meters
			deg: return in degrees (True, False)
		returns:
			theta: electrical length in radians, or degrees
		'''
		return electricalLength(l=l, f0 = self.fCenter, beta = self.beta, deg=deg)
		
	def zTE(self, omega,m=1,n=0):
		return eta0() * beta0(omega)/self.beta(omega,m,n)
	def zTM(self, omega,m=1,n=0):
		return eta0() * self.beta(omega,m,n)/beta0(omega)

	## standard creation
	# one-port 
	def createDelayShort(self,l,numPoints, **kwargs):
		'''
		generate the reflection coefficient for a waveguide delayed short of length l 
		
		takes:
			l - length of thru, in meters
			numPoints - number of points to produce
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''

		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createDelayShort(freq,l ,self.beta)
		return ntwk(data=s,paramType='s',freq=freq,**kwargs)

			
	def createShort(self, numPoints,**kwargs):
		'''
		generate the reflection coefficient for a waveguide short.
		convinience function, see mwavepy.createShort()
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createShort(numPoints)
		return ntwk(data=s,paramType='s',freq=freq,**kwargs)
		
	def createMatch(self,numPoints,**kwargs):
		'''
		generate the reflection coefficient for a waveguide Match.
		convinience function, see mwavepy.createShort()
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createMatch(numPoints)
		return ntwk(data=s,paramType='s',freq=freq,**kwargs)
		
	# two-port 
	def createDelay(self,l,numPoints,**kwargs):
		'''
		generate the two port S matrix for a waveguide thru section of length l 
		
		takes:
			l - length of thru, in meters
			numPoints - number of points to produce
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createDelay(freqVector=freq,l=l, beta=self.beta)		
		return ntwk(data=s,paramType='s',freq=freq,**kwargs)
		
		
		
		

## subclasses
class wr(waveguide):
	'''
	class which represents a standard rectangular waveguide band.
	inherits mwavepy.waveguide type 
	'''
	def __init__(self, number, **kwargs):
		'''
		takes:
			number - waveguide designator number ( ie WR10, the number is 10)
		returns:
			mwavepy.waveguide object representing the given band
		'''
		waveguide.__init__(self,number*10*mil ,.5 * number*10*mil, \
			name='WR'+repr(number), **kwargs )

	

# standard waveguide bands, note that the names are not perfectly cordinated with guide dims. 
# info taken from Virginia Diodes Inc. Waveguide Band Designations
WR10 = wr(10,band=(75e9,110e9))
WR8 = wr(8)
WR6 = wr(6.5)
WR5 = wr(5.1)
WR4 = wr(4.3, band = (170e9,260e9))
WR3 = wr(3.4, band=(220e9,325e9))
WR1p5 = wr(1.5, band =(500e9,750e9))		

## S-parameter Network Creation
#TODO: should name these more logically. like createSMatrix_Short()
# one-port
def createShort(numPoints):
	'''
	generates the one port S matrix for a Short. 
	
	takes:
		numPoints - number of points
	'''	
	return npy.complex_(-1*npy.ones(numPoints))

def createOpen(numPoints):
	'''
	generates the one port S matrix for a Openpy. 
	
	takes:
		numPoints - number of points
	'''
	return npy.complex_(1*npy.ones(numPoints))

def createMatch(numPoints):
	'''
	generates the one port S matrix for a Match. 
	
	takes:
		numPoints - number of points
	'''
	return npy.complex_(npy.zeros(numPoints))


	

def createDelayShort(freqVector, l, beta=beta0 ):
	'''
	calculates the reflection coef of  a delayed short. 
	'''
	return  -1*exp(-1j* 2*electricalLength(l,freqVector,beta))	
	
# two-port
def createDelay(freqVector, l,beta = beta0 ):
	'''
	generates the 2-port S-matrix for a matched Delay line of length l. 
	
	takes:
		freqVector -1D  array corresponding to the frequency band over 
			which to calculate. ( in Hz)
		l - length of delay (in m )
		beta - propagation constant function. this is a function which 
			is a function of angular frequency (omega), and returns a 
			value with units radian/m. 
		
	returns:
		kx2x2 S-parameter matrix for a ideal delay.
		
	
	note: beta defaults to lossless free-space propagation constant 
		beta = omega/c = omega*sqrt(epsilon_0*mu_0), which assumes a TEM wave	
	'''
	numPoints = len(freqVector)
	s11 =  npy.zeros(numPoints,dtype=complex)
	s12 =  npy.zeros(numPoints,dtype=complex)
	s21 =  npy.zeros(numPoints,dtype=complex)
	s22 =  npy.zeros(numPoints,dtype=complex)
	
	s12 = s21=  exp(-1j*electricalLength(l,freqVector,beta) )
	
	return npy.array([[s11, s12],\
					[s21, s22] ]).transpose().reshape(-1,2,2)


def createThru(numpPoints):
	thru = npy.zeroes(shape=(numPoints,2,2))
	thru[:,0,1] = thru[:,1,0]=1
	return thru

def createImpedanceStep(z1,z2,numPoints=1):
	'''
	creates a 2-port S-matrix of an impedance mismatch where port 1 is 
	terminated 	with impedance z1, and port 2 is terminated with 
	impedance z2.
	
	takes:
		z1 - impedance at port 1. (can be a single complex value ,
			or a numpy.array)
		z2 - impedance at port 2 ((can be a single complex value ,
			or a numpy.array)
		numPoints - number of points to generate this S-matrix. This is 
			not used if z1 and z2 are arrays.
	'''
	if isinstance(z1, npy.ndarray):
		if len(z1) != len(z2):
			# z1 must be length of z2
			raise ValueError
		numPoints = len(z1)
	else:
		# assume they passed us a single value,
		numPoints = numPoints
			
	s11 = npy.ones(numPoints,dtype=complex)*gamma(zl=z2,z0=z1,theta=0)
	s21 = npy.ones(numPoints,dtype=complex)+s11
	s22 = npy.ones(numPoints,dtype=complex)*gamma(zl=z2,z0=z1,theta=0)
	s12 = npy.ones(numPoints,dtype=complex) + s22

	return npy.array([[s11, s12],\
					[s21, s22] ]).transpose().reshape(-1,2,2)




	


def createShuntAdmittance(y,z0=50,numPoints=1):
	'''
	creates a 2-port S-matrix of an impedance mismatch where port 1 is 
	terminated 	with impedance z1, and port 2 is terminated with 
	impedance z2.
	
	takes:
		z1 - impedance at port 1. (can be a single complex value ,
			or a numpy.array)
		z2 - impedance at port 2 ((can be a single complex value ,
			or a numpy.array)
		numPoints - number of points to generate this S-matrix. This is 
			not used if z1 and z2 are arrays.
	'''
	if isinstance(y, npy.ndarray):
		numPoints = len(y)
		if z0 == 50:
			z0 = npy.ones(numPoints)*z0
	else:
		# assume they passed us a single value,
		numPoints = numPoints
	
	zl =1./(y + 1./z0) 
	s11 = npy.ones(numPoints,dtype=complex)*gamma(zl,z0)
	s21 = npy.ones(numPoints,dtype=complex)+s11
	s22 = s11
	s12 = s21

	return npy.array([[s11, s12],\
					[s21, s22] ]).transpose().reshape(-1,2,2)




	




############## calibration ##############
## one port
def onePortCal(measured, ideals):
	
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		ideals - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a = abc[:,0] = e01*e10 - e00*e11
				b = abc[:,1] = e00
				c = abc[:,2] = e11
			
			residuals: a matrix of residuals from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	 
	
		
	'''
	#make  copies so list entities are not changed, when we typecast 
	mList = copy(measured)
	iList = copy(ideals)
	
	numStds = len(mList)# find number of standards given, for dimensions
	numCoefs=3
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			mList[k] = mList[k].s.reshape((-1,1))
			iList[k] = iList[k].s.reshape((-1,1))
	except:
		pass	
	
	# ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
	fLength = len(mList[0])
	
	#initialize outputs 
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residuals = npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 
	
	# loop through frequencies and form m, a vectors and 
	# the matrix M. where M = 	i1, 1, i1*m1 
	#							i2, 1, i2*m2
	#									...etc
	for f in range(fLength):
		# vectors
		one = npy.ones(shape=(numStds,1))
		m = array([ mList[k][f] for k in range(numStds)])# m-vector at f
		i = array([ iList[k][f] for k in range(numStds)])# i-vector at f			
		# construct the matrix 
		Q = npy.hstack([i, one, i*m])
		# calculate least squares
		abcTmp, residualsTmp = npy.linalg.lstsq(Q,m)[0:2]
		abc[f,:]=abcTmp.flatten()
		residuals[f,:]=residualsTmp
		
	return abc, residuals
	
def onePortCalNLS(measured, ideals):
	
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		ideals - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a = abc[:,0] = e01*e10 - e00*e11
				b = abc[:,1] = e00
				c = abc[:,2] = e11
			
			residuals: a matrix of residuals from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	 
	
		
	'''
	#make  copies so list entities are not changed, when we typecast 
	mList = copy(measured)
	iList = copy(ideals)
	
	numStds = len(mList)# find number of standards given, for dimensions
	numCoefs=3
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			mList[k] = mList[k].s.reshape((-1,1))
			iList[k] = iList[k].s.reshape((-1,1))
	except:
		pass	
	
	# ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
	fLength = len(mList[0])
	
	
	
	from scipy.optimize import leastsq 
	
	def residualFunc(e,m,i):
		e00,e01,e10,e11 = e 
		E = array([[e00,e01],[e10, e11]])
		print shape(E), shape(i)
		return m - cascade(E, i)
	
		
	abc, residuals = onePortCal(measured = measured, ideals=ideals)
	E0 = abc2Ntwk(abc).s
	output=[]
	for f in range(fLength):
		# vectors
		m = array([ mList[k][f] for k in range(numStds)])# m-vector at f
		i = array([ iList[k][f] for k in range(numStds)])# i-vector at f			
		
		
		
		E0R,E0I= real(E0[f,:,:]).flatten(), imag(E0[f,:,:]).flatten()
		
		mR,mI = real(m).flatten(),imag(m).flatten()
		iR,iI = real(i).flatten(),imag(i).flatten()
		
		#print shape(p0R)
		#print shape((mR,iR))
		
		output.append(leastsq(func=residualFunc, x0=E0R, args=(mR,iR) )[0]+\
		1j*leastsq(func=residualFuncI, x0= E0I, args=(mI,iI) )[0])
		
		
	output = array(output)
	abc = output[:,:3]
	theta1,theta2 = output[:,3],output[:,4]
	return abc, theta1,theta2

def sddl1Cal(measured, actual, ftol=1e-3):
	
	'''
	calculates calibration coefficients for a one port calibration using 
	the auto-calibration method SDDL- algorithm 1. 
	 
	NOTE:
		for both of the input lists, ORDER MATTERS. it must be *,D1,D2,*
		meaning the two standards at indecies 1, and 2, are the two which
		are optimized, so they  must be offset shorts.
		 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		actual - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		
		
	
	returns:
		(abc, residues, gammaD1, gammaD2) - a list. abc is a Nx3 ndarray
			containing the complex calibrations coefficients,where N
			is the number of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
			
			gammaD1: calculated response of delay 1, a mwavepy.ntwk type
			gammaD2: calculated response of delay 2, a mwavepy.ntwk type
	
	 
	
		
	'''
	
	
	from scipy.optimize import fmin
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 
	gammaD1 = npy.zeros(fLength,dtype=complex) 
	gammaD2 = npy.zeros(fLength,dtype=complex) 


	# loop through frequencies and form gammaM, gammaA vectors and 
	# the matrix M. where M = 	gammaA_1, 1, gammA_1*gammaM_1
	#							gammaA_2, 1, gammA_2*gammaM_2 
	#									...etc
	
	for f in range(fLength):
		#print 'f=%i'%f
		
		# intialize
		gammaM = npy.zeros(shape=(numStds,1),dtype=complex)
		gammaA = npy.zeros(shape=(numStds,1),dtype=complex)
		one = npy.ones(shape=(numStds,1),dtype=complex)
		M = npy.zeros(shape=(numStds, numCoefs),dtype=complex) 
		
		for k in range(0,numStds):
			gammaM[k] = gammaMList[k][f]
			gammaA[k] = gammaAList[k][f]
		
		def iterativeCal(thetaTuple, gammaM, gammaA):
			theta1, theta2 = thetaTuple
			gammaA[1], gammaA[2] = exp(1j*theta1),exp(1j*theta2)
			M = npy.hstack([gammaA, one  ,gammaA*gammaM ])
			residues = npy.linalg.lstsq(M, gammaM)[1]
			return sum(abs(residues))
		
		# starting point for iterative least squares loop is whatever 
		# the user has submitted
		theta1Start = npy.angle(gammaA[1])
		theta2Start = npy.angle(gammaA[2])
		
		theta1,theta2 = fmin (iterativeCal, [theta1Start,theta2Start],args=(gammaM,gammaA), disp=False,ftol=ftol)
		
		
		gammaA[1], gammaA[2] = exp(1j*theta1),exp(1j*theta2)
		
		M = npy.hstack([gammaA, one  ,gammaA*gammaM ])
			
		residues[f,:] = npy.linalg.lstsq(M, gammaM)[1]
		abc[f,:]= npy.linalg.lstsq(M, gammaM)[0].flatten()
		
		gammaD1[f],gammaD2[f] = gammaA[1,0],gammaA[2,0]
		
		
	return abc,residues,gammaD1,gammaD2 
	
	


def sddl2Cal(measured, actual, wg, d1, d2, ftol= 1e-3):
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		gammaMList - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		gammaAList - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	'''
	
	
	from scipy.optimize import fmin
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	#abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	#residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 

	dStart = [d1,d2]
	sumResidualList = []
	
	def iterativeCal(d, gammaMList, gammaAList, sumResidualList):
		d1,d2=d[0],d[1]
		gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength).s
		gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength).s
		
		abc, residues = getABCLeastSquares(gammaMList, gammaAList)
		sumResidualList.append(npy.sum(abs(residues)))
		#print sum(abs(residues))
		return sum(abs(residues))
	
	
	d,dList = fmin (iterativeCal, dStart,args=(gammaMList,gammaAList, sumResidualList),\
		disp=False,retall=True,ftol=ftol)
	d1,d2=d
	gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength, name='ideal delay').s
	gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength, name='ideal delay').s 
		
	abc, residues =  getABCLeastSquares(measured = gammaMList, actual=gammaAList)
	return abc, residues, d1,d2, sumResidualList, dList


def sdddd1Cal(measured, actual,ftol=1e-3):
	
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		gammaMList - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		gammaAList - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	'''
	
	
	from scipy.optimize import fmin
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 
	gammaD1 = npy.zeros(fLength,dtype=complex) 
	gammaD2 = npy.zeros(fLength,dtype=complex) 
	gammaD3 = npy.zeros(fLength,dtype=complex) 
	gammaD4 = npy.zeros(fLength,dtype=complex) 


	# loop through frequencies and form gammaM, gammaA vectors and 
	# the matrix M. where M = 	gammaA_1, 1, gammA_1*gammaM_1
	#							gammaA_2, 1, gammA_2*gammaM_2 
	#									...etc
	
	for f in range(fLength):
		#print 'f=%i'%f
		
		# intialize
		gammaM = npy.zeros(shape=(numStds,1),dtype=complex)
		gammaA = npy.zeros(shape=(numStds,1),dtype=complex)
		one = npy.ones(shape=(numStds,1),dtype=complex)
		M = npy.zeros(shape=(numStds, numCoefs),dtype=complex) 
		
		for k in range(0,numStds):
			gammaM[k] = gammaMList[k][f]
			gammaA[k] = gammaAList[k][f]
		
		def iterativeCal(theta, gammaM, gammaA):
			theta1, theta2,theta3,theta4 = theta
			gammaA[1], gammaA[2],gammaA[3],gammaA[4] = \
				exp(1j*theta1),exp(1j*theta2),exp(1j*theta3),exp(1j*theta4)
			M = npy.hstack([gammaA, one  ,gammaA*gammaM ])
			residues = npy.linalg.lstsq(M, gammaM)[1]
			#print npy.sum(abs(residues))
			return npy.sum(abs(residues))
		
		# starting point for iterative least squares loop is whatever 
		# the user has submitted
		thetaStart = npy.angle(gammaA[1:])
		
		theta1,theta2,theta3,theta4 = fmin (iterativeCal, thetaStart,\
			args=(gammaM,gammaA), disp=False,ftol=ftol)
		
		
		gammaA[1], gammaA[2],gammaA[3],gammaA[4] = \
			exp(1j*theta1),exp(1j*theta2),exp(1j*theta3),exp(1j*theta4)
		
		M = npy.hstack([gammaA, one  ,gammaA*gammaM ])
			
		residues[f,:] = npy.linalg.lstsq(M, gammaM)[1]
		abc[f,:]= npy.linalg.lstsq(M, gammaM)[0].flatten()
		
		gammaD1[f],gammaD2[f],gammaD3[f],gammaD4[f] = \
			gammaA[1,0],gammaA[2,0],gammaA[3,0],gammaA[4,0]
		
		
	return abc,residues,gammaD1,gammaD2,gammaD3, gammaD4

def sdddd2Cal(measured, actual, wg, d1, d2,d3,d4, ftol=1e-3):
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		gammaMList - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		gammaAList - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	'''
	
	
	from scipy.optimize import fmin
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 

	
	dStart =npy.array([d1, d2,d3,d4])
	sumResidualList = []
	
	def iterativeCal(d, gammaMList, gammaAList):
		d1,d2,d3,d4=d[0],d[1],d[2],d[3]
		gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength).s
		gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength).s
		gammaAList[3] = wg.createDelayShort(l = d3, numPoints = fLength).s
		gammaAList[4] = wg.createDelayShort(l = d4, numPoints = fLength).s
		
		
		abc, residues= getABCLeastSquares(gammaMList, gammaAList)
		sumResidualList.append(npy.sum(abs(residues)))
		#print npy.sum(abs(residues))
		print npy.sum(abs(residues)),'==>',npy.linalg.linalg.norm(d),d
		return npy.sum(abs(residues))
		
	
	d1,d2,d3,d4 = fmin (iterativeCal, dStart,args=(gammaMList,gammaAList), disp=False,ftol=ftol)
	gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength, name='ideal delay').s
	gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength, name='ideal delay').s 
	gammaAList[3] = wg.createDelayShort(l = d3, numPoints = fLength, name='ideal delay').s 
	gammaAList[4] = wg.createDelayShort(l = d4, numPoints = fLength, name='ideal delay').s 
		
	abc, residues =  getABCLeastSquares(measured = gammaMList, actual=gammaAList)
	return abc, residues, d1,d2,d3,d4, sumResidualList

def sdddd2CalUnknownLoss(measured, actual, wg, d1, d2,d3,d4, ftol=1e-3,xtol=1e-3):
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		gammaMList - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		gammaAList - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	'''
	
	
	from scipy.optimize import fmin
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	wg = copy(wg)
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 
	sumResidualList = []
	
	if not wg.surfaceConductivity:
		# they did not give us a surface conductivity to start with
		wg.surfaceConductvity =  m.conductivityDict('alumninium')
	
	conductivityStart = wg.surfaceConductivity 
	# this is a misnomer, because its got conductivity in it
	dStart = npy.array([d1, d2,d3,d4, conductivityStart])
	
	
	def iterativeCal(d, gammaMList, gammaAList):
		d1,d2,d3,d4, conductivity=d[0],d[1],d[2],d[3],d[4]
		wg.surfaceConductivity = conductivity
		gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength).s
		gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength).s
		gammaAList[3] = wg.createDelayShort(l = d3, numPoints = fLength).s
		gammaAList[4] = wg.createDelayShort(l = d4, numPoints = fLength).s
		
		
		abc, residues= getABCLeastSquares(gammaMList, gammaAList)
		sumResidualList.append(npy.sum(abs(residues)))
		print npy.sum(abs(residues)),'==>',npy.linalg.linalg.norm(d),d
		return npy.sum(abs(residues))
		
	
	d1,d2,d3,d4, conductivity = fmin (iterativeCal, dStart,\
		args=(gammaMList,gammaAList), disp=False,ftol=ftol,xtol=xtol)
	wg.surfaceConductivity = conductivity
	gammaAList[1] = wg.createDelayShort(l = d1, numPoints = fLength, name='ideal delay').s
	gammaAList[2] = wg.createDelayShort(l = d2, numPoints = fLength, name='ideal delay').s 
	gammaAList[3] = wg.createDelayShort(l = d3, numPoints = fLength, name='ideal delay').s 
	gammaAList[4] = wg.createDelayShort(l = d4, numPoints = fLength, name='ideal delay').s 
		
	abc, residues =  getABCLeastSquares(measured = gammaMList, actual=gammaAList)
	return abc, residues, d1,d2,d3,d4, wg,sumResidualList

def alexCal(measured, actual):
	'''
	alternative one-port calibration algorithm. Based off taking the 
	complex difference between the ratios of measured and actual responses.
	
	takes:	
		measured: a list of ntwk objects, representing the measured
			responses. 
		actual: a list of ntwk objects, representing the actual (aka ideal)
		 standards.
		
	returns:
		(coefsDict, residuals): a tuple containing:
			coefsDict: dictionary containing the following keys
				'directivity':e00
				'reflection tracking':e01e10
				'source match':e11
			residuals: a f x numStds-1 array of sum of the norms of the \
				residuals, where f is number of frequency points, and 
				numStds is number of Standards given
	
	
	'''
	
	numStds = len(measured)
	# convert s-parameters to arrays, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			measured[k] = measured[k].s
			actual[k] = actual[k].s
	except:
		pass	
	
	
	
	fLength = len(measured[0])
	# needed, because i use more complicated  slicing.
	measured = npy.array(measured) 
	actual = npy.array(actual)
	
	# pre-allocate output calibration coefficient vectors, and residues
	e11 = npy.zeros(shape=(fLength,1),dtype=complex) 
	e00 = npy.zeros(shape=(fLength,1),dtype=complex) 
	e10e01 = npy.zeros(shape=(fLength,1),dtype=complex) 
	residues =npy.zeros(shape=(fLength,2*numStds),dtype=complex) 
	
	# calibrate at each frequency point
	for f in range(fLength):
		m = measured[:,f] # 1xnumStds
		a = actual[:,f]
			
		P,Q,R = [],[],[]
		
		
		if numStds==4:
			permutations  = ([0,1],[0,2],[0,3],[1,2],[1,3],[2,3])
		
		elif numStds ==3 :
			permutations = ([0,1],[1,2],[2,0])
			#=[ (x,y) for x in range(numStds) for y in range(numStds) if y !=x ]#
		
		for x,y in permutations:
			P.append( m[x]/a[x] - m[y]/a[y])
			Q.append( m[x]-m[y])
			R.append(1/(a[x]) -1/a[y])
			
		
		P = npy.array(P).reshape(-1,1)
		Q = npy.array(Q).reshape(-1,1)
		R = npy.array(R).reshape(-1,1)
			
		# form matrix
		QR = npy.hstack((Q,R))
		
		
		e11[f],e00[f] = npy.linalg.lstsq(QR,P)[0]#.flatten()
		
		residues[f,:] =  (npy.linalg.lstsq(QR,P)[1])
		
		# evaluate relation to find reflection tracking term, can use 
		# any pair of standards so  0 < k < numStds, they all produce same
		# values for e10e01
		for k in [0]: 
			e10e01[f] = m[k]/a[k]-m[k]*e11[f]-e00[f]/a[k]+e00[f]*e11[f]
				
		
	
	# make dictionary from calibration coefficients
	coefsDict = {'directivity':e00, 'reflection tracking':e10e01, \
		'source match':e11}
	return coefsDict, residues
	

	

def mobiusTransform(m, a):
	'''
	returns the unique maping function between m and a planes which are
	related through	the mobius transform.
	
	takes:
		m: list containing the triplet of points in m plane m0,m1,m2
		a: list containing the triplet of points in a plane a0,a1,a2
	
	returns:
		a (m) : function of variable in m plane, which returns a value
			in the a-plane
	'''
	m0,m1,m2 = m
	a0,a1,a2 = a
	return lambda m: (a0*a1*m*m0 + a0*a1*m1*m2 + a0*a2*m*m2 + a0*a2*m0*m1 +\
	 a1*a2*m*m1 + a1*a2*m0*m2 - a0*a1*m*m1 - a0*a1*m0*m2 - a0*a2*m*m0 -\
	 a0*a2*m1*m2 - a1*a2*m*m2 - a1*a2*m0*m1)/(a0*m*m2 + a0*m0*m1 + a1*m*m0\
	  + a1*m1*m2 + a2*m*m1 + a2*m0*m2 - a0*m*m1 - a0*m0*m2 - a1*m*m2 - \
	  a1*m0*m1 - a2*m*m0 - a2*m1*m2)
	
	
def getABC(mOpen,mShort,mMatch,aOpen,aShorat,aMatch):
	'''
	calculates calibration coefficients for a one port OSM calibration
	 
	 returns:
		abc is a Nx3 ndarray containing the complex calibrations coefficients,
		where N is the number of frequency points in the standards that where 
		givenpy.
	
	 takes:
		 mOpen, mShort, and mMatch are 1xN complex ndarrays representing the 
		measured reflection coefficients off the corresponding standards OR can  be 1-port mwavepy.ntwk() types. 
			
	 	aOpen, aShort, and aMatch are 1xN complex ndarrays representing the
	 	assumed reflection coefficients off the corresponding standards. 
	 	
	 note:
	  the standards used in OSM calibration dont actually have to be 
	  an open, short, and match. they are arbitrary but should provide
	  good seperation on teh smith chart for good accuracy 
	'''
	if isinstance(mOpen,ntwk):	
		# they passed us ntwk types, so lets get the relevent parameter
		#make sure its a 1-port
		if mOpen.rank >1:
			print 'ERROR: this takes a 1-port'
			return None
		else:
			# there might be a more elegant way to do this
			mOpen = mOpen.s[:,0,0]
			mShort = mShort.s[:,0,0]
			mMatch = mMatch.s[:,0,0]
			aOpen = aOpen.s[:,0,0]
			aShort = aShort.s[:,0,0]
			aMatch = aMatch.s[:,0,0]
		

	
	# loop through all frequencies and solve for the calibration coefficients.
	# note: abc are related to error terms with:
	# a = e10*e01-e00*e11, b=e00, c=e11  
	#TODO: check to make sure all arrays are same length
	abc= npy.complex_(npy.zeros([len(mOpen),3]))
	
	for k in range(len(mOpen)):
		
		Y = npy.vstack( [\
						mShort[k],\
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
	
def getABCLeastSquaresOld(measured, actual):
	'''
	calculates calibration coefficients for a one port calibration. 
	 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		actual - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		If one makes the assumption of the error network being 
		reciprical or symmetric or both, the correction requires less 
		measurements. see mwavepy.getABLeastSquares
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for better accuracy .
	'''
	
	
	#make  copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(actual)
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	
	
	fLength = len(gammaMList[0])
	#initialize abc matrix
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 

	# loop through frequencies and form gammaM, gammaA vectors and 
	# the matrix M. where M = 	gammaA_1, 1, gammA_1*gammaM_1
	#							gammaA_2, 1, gammA_2*gammaM_2 
	#									...etc
	for f in range(fLength):
		# intialize
		gammaM = npy.zeros(shape=(numStds,1),dtype=complex)
		gammaA = npy.zeros(shape=(numStds,1),dtype=complex)
		one = npy.ones(shape=(numStds,1),dtype=complex)
		M = npy.zeros(shape=(numStds, numCoefs),dtype=complex) 
		
		for k in range(0,numStds):
			gammaM[k] = gammaMList[k][f]
			gammaA[k] = gammaAList[k][f]
			
		M = npy.hstack([gammaA, one  ,gammaA*gammaM ])
		abc[f,:]= npy.linalg.lstsq(M, gammaM)[0].flatten()
		residues[f,:] = npy.linalg.lstsq(M, gammaM)[1]
		
	return abc,residues
	
	
def getABLeastSquares(gammaMList, gammaAList):
	'''
	calculates calibration coefficients for a one port calibration with
	the assumption that the error network is reciprical and syemetric. 
	a = s21^2 - s12^2, b = s11 = s22
	 
	takes: 
		gammaMList - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or a 1-port mwavepy.ntwk types. 
		gammaAList - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or a 1-port mwavepy.ntwk types. 
	
	returns:
		
		ab - is a kx2 ndarray containing the complex calibrations 
			coefficients, where k is the number of frequency points in 
			the	standards that where givens.
		residues - kxp, where p is difference between number of 
			coefficients and number of standards.
	
	 
		
	 note:
		For calibration of general 2-port error networks, 3 standards 
		are required. 
		the standards used in OSM calibration dont actually have to be 
		an open, short, and match. they are arbitrary but should provide
		good seperation on teh smith chart for good accuracy .
	'''
	
	
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 2
	
	# test for type, 
	if isinstance(gammaMList[0], ntwk):
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
		
	fLength = len(gammaMList[0])
	#initialize abc matrix
	ab = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues =npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 

	# loop through frequencies and form gammaM, gammaA vectors and 
	# the matrix M. where M = 	gammaA_1, 1, gammA_1*gammaM_1
	#							gammaA_2, 1, gammA_2*gammaM_2 
	#									...etc
	for f in range(fLength):
		# intialize
		gammaM = npy.zeros(shape=(numStds,1),dtype=complex)
		gammaA = npy.zeros(shape=(numStds,1),dtype=complex)
		one = npy.ones(shape=(numStds,1),dtype=complex)
		M = npy.zeros(shape=(numStds, numCoefs),dtype=complex) 
		
		for k in range(0,numStds):
			gammaM[k] = gammaMList[k][f]
			gammaA[k] = gammaAList[k][f]
			
		M = npy.hstack([gammaA, one+gammaA*gammaM ])
		ab[f,:]= npy.linalg.lstsq(M, gammaM)[0].flatten()
		residues[f,:] = npy.linalg.lstsq(M, gammaM)[1]
		
	return ab,residues




def applyABC( gamma, abc):
	'''
	takes a complex array of uncalibrated reflection coefficient and applies
	the one-port OSM callibration, using the coefficients abc.  
	instead of this function, you could use abc2ntwk, and then deEmbed. 

	takes:
		gamma - complex reflection coefficient OR can  be 1-port mwavepy.ntwk() type.
		abc - Nx3 OSM calibration coefficients. 
	returns:
		either a complex reflection coefficient, or a 1-port mwavepy.ntwk() instance, depending on input
		
	note: this is a simple calculation, 
	gammaCal(k)=(gammaDut(k)-b)/(a+gammaDut(k)*c); for all k 
		
	'''
	
	#TODO: re-write this so the variables make more sense to a reader
	# type test
	if isinstance(gamma,ntwk):
		# they passed us ntwk types, so lets get the relevent parameter
		#make sure its a 1-port
		if gamma.rank >1:
			print 'ERROR: this takes a 1-port'
			raise RuntimeError
			return None
		else:
			newNtwk = copy(gamma)
			gamma = gamma.s[:,0,0]
			gammaCal= ((gamma-abc[:,1]) / (abc[:,0]+ gamma*abc[:,2]))
			newNtwk.s=(gammaCal)
			return  newNtwk
	else:
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







def abc2Ntwk(abc, isReciprocal = False, **kwargs):
	'''
	returns a 2-port ntwk for a given set of calibration coefficients
	represented by a Nx3 matrix of one port coefficients, (abc)
	
	takes:
		abc : Nx3 numpy.ndarray, which holds the complex calibration 
			coefficients. the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
		**kwargs: passed to mwavepy.ntwk constructor
	returns:
		eNtwk : 
	
	note:
		s21 = a+b*c ( which  is  e01*e10)
		s12 = 1
		s11 = b
		s22 = c
		
		This abstract ntwk should only be used for de-embeding 1-port 
		measurements, as e01*e10 cannot be seperated
	'''
	
	if len(abc.shape) == 1:
		#assert: this is abc at one frequency
		a,b,c = abc[0], abc[1],abc[2]
		if isReciprocal:
			return npy.array([	[b, npy.sqrt(a+b*c)],\
								[npy.sqrt(a+b*c), c]], dtype=complex)
		else:
			return npy.array([	[b, 1],\
								[a+b*c, c]], dtype=complex)
					
	elif len(abc.shape) == 2:
		#assert: this is an array of abc's at many frequencies
		eNtwkS = npy.zeros(shape=(abc.shape[0],2,2),dtype=complex)
		
		for k in range(abc.shape[0]):
			eNtwkS[k]=abc2Ntwk(abc[k],isReciprocal)
		
		eNtwk = ntwk(data=eNtwkS, paramType='s',**kwargs)
		return eNtwk
	else:
		raise IndexError('shape of input is incorrect')
			
	
	
	

def abc2CoefsDict(abc):
	'''
	converts an abc ndarry to a dictionarry containing the error 
	coefficients.
	
	takes: 
		abc : Nx3 numpy.ndarray, which holds the complex calibration 
			coefficients. the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
	returns:
		coefsDict: dictionary containing the following
			'directivity':e00
			'reflection tracking':e01e10
			'source match':e11
	note: 
		e00 = directivity error
		e10e01 = reflection tracking error
		e11 = source match error	
	'''
	a,b,c = abc[:,0], abc[:,1],abc[:,2]
	e01e10 = a-b*c
	e00 = b
	e11 = c
	coefsDict = {'directivity':e00, 'reflection tracking':e01e10, \
		'source match':e11}
	return coefsDict
	


class calibration(object):
	'''
	represents a calibration instance.
	
	you give it a frequency axis, and two lists of mwavepy.ntwk's: 
		ideal standards list 
		measured standards list.
	and it calculates the calbration coeffficients, and produces an 
	error ntwk for de-embeding. 	
	 
	NOTE:
		only works for one port at the moment
		
		coefficients are calculated automatically when they, or 
		their related quantities ( abc, error_ntwk, etc) are referenced,
		the FIRST time only.  once they exist, the function 
		calculateCoefs() must be called if you want to re-calculate coefs.
	'''
	
	def __init__(self,freq=[], freqMultiplier = None, ideals=[],measured =[], name = '',type='one port', d=None, wg=None, ftol=1e-3,xtol=1e-3 ):
		'''
		calibration constructor. 
		
		takes:
			freq: frequency axis over which the cal exists, a numpy.array
				[in hz]
			freqMultiplier: a multiplier to scale plot axis,  a float.
			ideals: a list of ntwk() types, which are the actual 
				response of the standards used
			measured: a list of ntwk() types, which are the measured 
				responses of the actual standards cascaded behind some 
				unkown 2-port error network.
			name: a name, which may be used in plots and such, a string
			type: the type of calibration to perform, a string. can be:
				'one port': standard one port calibration algorithm. if
					more than 3 standards are given, it uses least squares. 
				'alex one port' ; alternative to the standard one-port 
					algorithm. only inverts a Nx2 matrix, where N is the
					number of standards.
				'sddl1':  stands for Short,Delay,Delay Load, and the 
					ntwk lists must be given in that order. Asumes that 
					the magnitude of both delays is 1, and will determine
					the phase by minimized the resdiual error, iteratively.
				'sddl2':
				'sdddd1':
				'sdddd2':
			d: only used in sddl2 or sdddd2. d an array holds
				the 'guess' lengths for the delays in meters. 
				order matters. 	delay 1 length = d[0], delay 2 length = d[1]
			wg: only used in sddl2, sdddd2. a mavepy.waveguide instance 
				used to generate a predicted response of the offset short,
				which's length is adjusted  iteractively.
			
		
		returns:
			None
			
		Note: 
			only available for one-port at the moment
		'''
		
		self.name  = name
		self.ideals = copy(ideals)
		self.measured = copy(measured)
		self.freq = copy(freq)
		self.freqMultiplier = freqMultiplier
		self.type = type
		self.d = copy(d)
		self.wg = copy(wg)
		self.ftol=ftol
		self.xtol=xtol
	
	
	def calculateCoefs(self):
		'''
		calculates the calibration coefficients for the calibration instance.
		
		This is called the first time any of the calibration related 
		quantities are referenced, and thats it. so if you update the list
		of ideals, or measured  standards, then you must call this 
		explicitly for the new coefficients to be calculated.
		
		specifically, this calculates:
			coefs: a dictionary holding the calibration coefficients
			abc: an Nx3 numpy.array holdin the a, b,c values. 
			error_ntwk: a fictional 2-port network representing the 
				error network. this should only be used to de-Embed 1-port
				responses.
		
		takes:
			na
		returns:
			na
		'''
		if len(self.ideals) != len(self.measured):
			raise IndexError('you need the same number of ideals as measurements. ')
		
		# call appropriate call type
		if self.type == 'one port':
			t0 = time()
			self._abc, self._residuals = onePortCal(\
				measured = self.measured, ideals = self.ideals)
			print '%s took %i s' %(self.name, time()-t0)
		elif self.type == 'sddl1':
			t0 = time()
			
			self._abc, self._residuals, gammaD1, gammaD2 =\
				sddl1Cal(	measured = self.measured, actual = self.ideals,ftol=self.ftol)
			self.delay1 = ntwk(data = gammaD1, freq=self.freq, \
				name = self.ideals[1].name+' adjusted')
			self.delay2 = ntwk(data = gammaD2, freq=self.freq, \
				name = self.ideals[2].name+' adjusted')
			
			print '%s took %i s' %(self.name, time()-t0)	
			
		elif self.type == 'sddl2':
			t0 = time()
			self._abc, self._residuals, self.d1FromCal, self.d2FromCal,\
				 self.allResidueSums,self.dList = \
				sddl2Cal(measured = self.measured, actual = self.ideals, \
				wg = self.wg, d1 = self.d[0], d2 = self.d[1],ftol=self.ftol)
			self.delay1 = self.wg.createDelayShort(self.d1FromCal, \
				len(self.freq), name=self.ideals[1].name+' adjusted')
			self.delay2 = self.wg.createDelayShort(self.d2FromCal, \
				len(self.freq), name=self.ideals[2].name+' adjusted')
			self.d1Evolution = npy.array(self.dList)[:,0]
			self.d2Evolution = npy.array(self.dList)[:,1]
			print '%s took %i s' %(self.name, time()-t0)
		
		elif self.type == 'sdddd2':
			
			t0 = time()
			self._abc, self._residuals, self.d1FromCal, self.d2FromCal,\
			self.d3FromCal,self.d4FromCal,self.allResidueSums = \
			sdddd2Cal(measured = self.measured, actual = self.ideals, \
				wg = self.wg, d1 = self.d[0], d2 = self.d[1],d3 = self.d[2], d4 = self.d[3],ftol=self.ftol)
			
			self.delay1 = self.wg.createDelayShort(self.d1FromCal, \
				len(self.freq), name=self.ideals[1].name+' adjusted')
			self.delay2 = self.wg.createDelayShort(self.d2FromCal, \
				len(self.freq), name=self.ideals[2].name+' adjusted')
			self.delay3 = self.wg.createDelayShort(self.d3FromCal, \
				len(self.freq), name=self.ideals[3].name+' adjusted')
			self.delay4 = self.wg.createDelayShort(self.d4FromCal, \
				len(self.freq), name=self.ideals[4].name+' adjusted')
			
			
			print '%s took %i s' %(self.name, time()-t0)
		elif self.type == 'sdddd2UnkownLoss':
			
			t0 = time()
			self._abc, self._residuals, self.d1FromCal, self.d2FromCal,\
			self.d3FromCal,self.d4FromCal, self.wg, self.allResidueSums= \
			sdddd2CalUnknownLoss(measured = self.measured, actual = self.ideals, \
				wg = self.wg, d1 = self.d[0], d2 = self.d[1],d3 = self.d[2],\
				d4 = self.d[3],ftol=self.ftol,xtol=self.xtol)
			
			self.delay1 = self.wg.createDelayShort(self.d1FromCal, \
				len(self.freq), name=self.ideals[1].name+' adjusted')
			self.delay2 = self.wg.createDelayShort(self.d2FromCal, \
				len(self.freq), name=self.ideals[2].name+' adjusted')
			self.delay3 = self.wg.createDelayShort(self.d3FromCal, \
				len(self.freq), name=self.ideals[3].name+' adjusted')
			self.delay4 = self.wg.createDelayShort(self.d4FromCal, \
				len(self.freq), name=self.ideals[4].name+' adjusted')
			
			
			print '%s took %i s' %(self.name, time()-t0)
		elif self.type == 'sdddd1':
			
			t0 = time()
			self._abc, self._residuals, gammaD1, gammaD2,\
			gammaD3,gammaD4 = \
			sdddd1Cal(measured = self.measured, actual = self.ideals,ftol=self.ftol)
			
			
			self.delay1 = ntwk(data = gammaD1, freq=self.freq, \
				name = self.ideals[1].name+' adjusted')
			self.delay2 = ntwk(data = gammaD2, freq=self.freq, \
				name = self.ideals[2].name+' adjusted')
			self.delay3 = ntwk(data = gammaD3, freq=self.freq, \
				name = self.ideals[3].name+' adjusted')
			self.delay4 = ntwk(data = gammaD4, freq=self.freq, \
				name = self.ideals[4].name+' adjusted')
			
			
			
			print '%s took %i s' %(self.name, time()-t0)
		else:
			raise ValueError('Bad cal type.')
			
		self._error_ntwk = abc2Ntwk(self._abc, name = self.name,freq = self.freq, freqMultiplier= self.freqMultiplier, isReciprocal=True)
		self._coefs = abc2CoefsDict(self._abc)
		return None
		
	def __get_coefs(self):
		'''
		coefs: a dictionary holding the calibration coefficients
		'''
		
		try:
			return self._coefs
		except(AttributeError):
			self.calculateCoefs()
			return self._coefs
		
	
	def __set_coefs(self,x):
		raise TypeError('you cant set coefficients, they are calculated')
	
	coefs = property(__get_coefs, __set_coefs)
	
	def __get_error_ntwk(self):
		try:
			return self._error_ntwk
		except(AttributeError):
			self.calculateCoefs()
			return self._error_ntwk
		
	
	def __set_error_ntwk(self):
		raise TypeError('you cant set error ntwk, it is calculated')
	
	error_ntwk = property(__get_error_ntwk, __set_error_ntwk)
	
	def __get_abc(self):
		try:
			return self._abc
		except(AttributeError):
			self.calculateCoefs()
			return self._abc
	
	def __set_abc(self):
		raise TypeError('you cant set abc, they are calculated')
	
	abc = property(__get_abc, __set_abc)
		
	def __get_residuals(self):
		'''
		array of residuals. shape depends on calibration type and number
		of standards
		'''
		try:
			return self._residuals
		except(AttributeError):
			self.calculateCoefs()
			return self._residuals
	
	def __set_residuals(self):
		raise TypeError('you cant set residuals, they are calculated')
	
	residuals = property(__get_residuals, __set_residuals)	
	
	def plotCoefsDb(self, ax = None,**kwargs):
		plotErrorCoefsFromDictDb (self.coefs,freq= self.freq/1e9, ax = ax, **kwargs)
	def plotCoefsPhase(self, ax = None,**kwargs):
		plotErrorCoefsFromDictPhase (self.coefs,freq= self.freq/1e9, ax = ax, **kwargs)

	def plotD1Evolution(self,ax=None,lengthUnit='um',**kwargs):
		
		try:
			plb.plot(self.d1Evolution/lengthDict[lengthUnit], label= self.name+': d1',**kwargs)
			#plb.axhline(self.d1FromCal/lengthDict[lengthUnit], label=self.name+': d1 End Value')
			plb.legend()
			plb.xlabel('Iteration')
			plb.ylabel('Length of d1 ('+lengthUnit+')')
			plb.title(self.name + ': Evolution of Delay Line 1')
		except(AttributeError):
			raise TypeError('only available for sddl2 type, after coefs calculated')
	
	def plotD2Evolution(self,ax=None,lengthUnit='um',**kwargs):
		
		try:
			plb.plot(self.d2Evolution/lengthDict[lengthUnit], label= self.name+': d2',**kwargs)
			#plb.axhline(self.d2FromCal/lengthDict[lengthUnit], label=self.name+': d2 End Value')
			plb.legend()
			plb.xlabel('Iteration')
			plb.ylabel('Length of d2 ('+lengthUnit+')')
			plb.title(self.name + ': Evolution of Delay Line 2')
		except(AttributeError):
			raise TypeError('only available for sddl2 type, after coefs calculated')
					
	def plotResidualEvolution(self,ax=None,**kwargs):
		
		try:
			plb.semilogy(self.allResidueSums, label= self.name+': Sum(Residues)',**kwargs)
			plb.legend()
			plb.xlabel('Iteration')
			plb.ylabel('Sum(Residues)')
			plb.title(self.name + ': Evolution of Sum of Residues Accross Band')
		except(AttributeError):
			raise TypeError('only available for sddl2 type, after coefs calculated')
					
############ DEPRICATED/UNSORTED#####################
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
		raise RuntimeError
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
	








def grepLegendFromHfssCsv(hfssCsvFile,startOfString):
	'''
	produces a list which can be used as a legend info from an hfss 
	commented .csv file.
	
	takes:
		hfssCsvFile - string of filename
		startOfString - string to pull out of hfss inserted comments
	returns:
		legendList: list of strings which start with startOfString
	'''
	hfssCsvFile = open(hfssCsvFile)
	legendList=[]
	for k in hfssCsvFile.readlines()[0].split():
		if k.startswith(startOfString):
			legendList.append(k)
	return legendList
	
def getHfssLegendAndXaxis(hfssCsvFileName):
	'''
	produces the same x-axis label and legend list which is shown in 
	hfss plotter
	
	takes:
		hfssCsvFileName: string, the filename 
	returns:
		(hfssLegendList, hfssXlabel)
			hfssLegendList: list of strings which can be passed to legend 
			command
			hfssXlabel: string which is the Xlabel
		
	
	'''
	hfssCsvFile = open(hfssCsvFileName)
	commentLine = hfssCsvFile.readlines()[0]
	hfssXLabel = commentLine.split(',\"')[0]
	hfssLegendList = commentLine.split(',\"')[1:-1]
	return hfssLegendList, hfssXLabel

def plotCsv(filename,**kwargs):
	'''plots columns from csv file. plots all columns against the first
	see pylab.loadtxt for more information
	'''
	data = plb.loadtxt(filename,skiprows=1, delimiter=',',**kwargs)
	plb.plot(data[:,0], data[:,1:])
	plb.grid(1)
	plb.title(filename)

def plotHfssCsv(filename,**kwargs):
	plotCsv(filename,**kwargs)
	plb.xlabel (getHfssLegendAndXaxis(filename)[1])
	plb.legend (getHfssLegendAndXaxis(filename)[0])
	


def hfssComment2Dict(hfssCsvFileName):
	'''
	converts a hfss exported csv file's comment line into a list of 
	dictionaries corresponding to parameter
	'''
	import re
	remove_non_digits = re.compile(r'[^\d.]+')
	
	hfssFile = open(hfssCsvFileName)
	commentLine = hfssFile.readline()
	
	#convert quoted commas to quotes, so we can split on it
	commentLineList= commentLine.replace('\",\"','"').split('\"')
	xAxisLabel = commentLineList[0]
	yAxisLabel = commentLineList[1][:commentLineList[1].find(' - ')]
	# split on the double-quotes, ignore first and last entry, they are 
	# the x-axis label and a newlien character 
	
	# the parameter list actually starts after a ' - ', before that is 
	# the y-axis.
	for k in range(1,len(commentLineList)):
		commentLineList[k] = commentLineList[k][commentLineList[k].find(' - ')+3:]
		
	
	inputList = commentLineList[1:]
	
	
	#remove empty strings
	flag = True
	while flag:
	    try:
	        inputList.remove('')
	    except ValueError:
	        flag=False
	
	outputList = []
	
	for varString in inputList:
	    varStringList = varString.split()
	    varDict = {}
	    for aVar in varStringList:
	        varList = aVar.split('=')
	        varDict[varList[0]] = varList[1]
	    outputList.append(varDict)
	
	for aDict in outputList:
	    for aKey in aDict:
	        aDict[aKey] = aDict[aKey].replace("'","") #remove quote chars
	        aDict[aKey] = float(remove_non_digits.sub('', aDict[aKey]))
	
	return xAxisLabel, yAxisLabel, outputList
##------ other functions ---
def psd2TimeDomain(f,y, windowType='hamming'):
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
	signalVector = plb.ifft(plb.ifftshift(spectrum))
	
	#the imaginary part of this signal should be from fft errors only,
	signalVector= npy.real(signalVector)
	# the response of frequency shifting is 
	# exp(1j*2*pi*timeVector*f[0])
	# but i would have to manually undo this for the inverse, which is just 
	# another  variable to require. the reason you need this is because 
	# you canttransform to a bandpass signal, only a lowpass. 
	# 
	return timeVector, signalVector


def timeDomain2Psd(t,y, windowType='hamming'):
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

	Y = 1./len(y)* plb.fftshift(plb.fft(y))
	spectrumVector = Y[len(Y)/2:]
	freqVector = f[len(f)/2:]
	return [freqVector,spectrumVector]

def cutOff(a):
	'''returns the cutoff frequency (in Hz) for first  resonance of a
	waveguide with major dimension given by a. a is in meters'''
	
	return sqrt((pi/a)**2 *1/(epsilon_0*mu_0))/(2*pi)

