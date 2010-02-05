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
import pylab as plb
from numpy import sqrt, exp, array,tan,sin,cos,inf
from scipy.constants import  epsilon_0, mu_0, c,pi, mil
from scipy import signal
import os # for fileIO
from matplotlib.patches import Circle 	# for drawing smith chart
from matplotlib.lines import Line2D		# for drawing smith chart
from touchstone import touchstone as touch	# for loading data from touchstone files
from copy import copy

#TODO LIST
'''
other network types z,y,abcd
network conversions
network connections (parrallel, series)
tranmission line classes ( what properties do all transmission lines share)
de-embeding 


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



###############  mathematical conversions ############### 
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
	








############### Ploting ############### 
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





############### network theory  ################
## base network class.
class ntwk:
	'''
	class represents a generic n-port network. 
	
	provides:
		s - a kxnxn matrix of complex values representing a n-port network over a given frequency range, numpy.ndarray
		freq - 1D frequency vector,  numpy.ndarray
		freqUnit - meaning frequency vector, string, ( MHz, GHz, etc)
		freqMultiplier - scale of frequency vector to 1Hz, string.  ( ie, 1e9 for freqUnit of 'GHz'
		paramType - parameter types  ( 's','z','y','abcd') , string
		z0 - characteristic impedance of network
		name - name of network, string. used in legend for plotting.
		
		smag - kxnxn array representing magnitude of s-parameters in decimal, ndarray 
		sdB - kxnxn array representing magnitude of s-parameters in decibel (20*npy.log10(mag)) scale, ndarray
		sdeg - kxnxn array representing phase of s-parameters in deg, ndarray
		srad - kxnxn array representing phase of s-parameters in radians, ndarray
		
		
		
		note: these matricies may be re-shaped if one wants the frequency index to come last, like  myntwk.s.transpose().reshape(2,2,-1)
	'''
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
			print ('ERROR: input data must be kxmxm, where k is frequency axis')
			return None
		if freq == None:
			self.freq = freq #None
			self.freqUnit = None
			self.freqMultiplier=None
		else:
			if len(freq) != data.shape[0]:
				print 'Error: length of f must match data.shape[2]. There must be as many frequency points as there are s parameter measurements.'
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
			self.s = npy.complex_(data)
			# this is where we should calculate and assign all the other ntwk formats
			#npy.zeros(shape=(self.rank,self.rank, self.length)))
			#	
		elif self.paramType == 'abcd':
			raise NotImplementedError
		elif self.paramType == 'z':
			raise NotImplementedError
		elif self.paramType == 'y':
			raise NotImplementedError
			
			
		#convinience	
		self.smag = abs(self.s)
		self.sdB = mag2dB( self.smag )
		self.sdeg = npy.angle(self.s, deg=True)
		self.srad = npy.angle(self.s)
		
		
		

	def __sets__(self, sMatrix):
		'''
		update a ntwk's sparameters. ntwk.s effects other object within the ntwk class, so this function is needed.
		'''
		
		data = npy.array(sMatrix) 
		if len(data.shape) == 1:
			# they gave us 1D array
			data = data.reshape(-1,1,1)
		if data.shape[1] != data.shape[2]:
			print ('ERROR: input data must be kxmxm, where k is frequency axis')
			return None
		self.s = data
		self.rank = data.shape[1]
		self.smag = abs(self.s)
		self.sdB = mag2dB( self.smag )
		self.sdeg = npy.angle(self.s, deg=True)
		self.srad = npy.angle(self.s)



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
			print 'Error: please specify indecies.'
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
			
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
		
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			ax1.plot(self.sdB[:,m,n],label=labelString,**kwargs)
		else:
			ax1.plot(self.freq/self.freqMultiplier, self.sdB[:,m,n],label=labelString,**kwargs)
		
		
		plb.axis('tight')
		plb.xlabel('Frequency (' + self.freqUnit +')') 
		plb.ylabel('Magnitude (dB)')
		plb.xlim([ self.freq[0]/self.freqMultiplier, self.freq[-1]/self.freqMultiplier])
		plb.grid(1)
		plb.legend(loc='best')
		plb.draw()
		
	def plotSmith(self, m=None,n=None, smithRadius = 1, ax=None, **kwargs):
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
		
	
	def plotPhase(self, m=None,n=None, ax=None, **kwargs):
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
			print 'Error: please specify indecies.'
		elif ( m==None or n==None) and (self.rank == 1):
			m = 0
			n = 0
		labelString  = self.name+', S'+repr(m+1) + repr(n+1)
		if ax == None:
			ax1 = plb.gca()
		else:
			ax1 = ax
		
		if self.freq == None:
			# this network doesnt have a frequency axis, just plot it  
			ax1.plot(self.sdeg[:,m,n],label=labelString,**kwargs)
		else:
			ax1.plot(self.freq/self.freqMultiplier, self.sdeg[:,m,n],label=labelString,**kwargs)
		
		
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
		plb.grid(1)
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
	
	for f in range(smat.shape[2]):
		passivity[f,:,:] = npy.eye(smat.shape[1]) - npy.dot(smat[f,:,:],smat[f,:,:].conj().transpose())
			#for tmp in  eigvals(passivity[:,:,f]):
				#if real(tmp) < 0:
					#if abs(tmp) < tol:
						## structure fails the passivity test
						#return False
			#return True
	return passivity









## network representation conversions
# these conversions where taken from Pozar. Microwave Engineering sec 5.6
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



############### transmission lines   ################
class transmissionLine:
	'''
	should be main class, which all transmission line sub-classes inhereit
	'''
	def __init__(self):
		raise NotImplementedError
		return None
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
	def __init__(self, a,b,band=None, epsilonR=1, muR=1):
		'''
		takes: 
			a - width  in meters, float
			b - height in meters, float. 
			band - tuple defining max and min frequencies. defaults to (1.25,1.9)*fc10
			epsilonR - relative permativity of filling material 
			muR - relative permiability of filling material
		'''
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
		'''
		calculates cut-on frequency of a given mode
		takes:
			m - mode indecie for width dimension, int
			n - mode indecie for height dimension, int
		returns:
			cut of frequency in Hz, float. 
			
		'''
		return c/(2*pi)*sqrt( (m*pi/self.a)**2 +(n*pi/self.b)**2)

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
		the beta here is just the space beta, which should be a method of mwavepy
		'''
		k = omega/c
		return sqrt(k**2 - (m*pi/self.a)**2- (n*pi/self.b)**2)
	
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
		return (2*pi / self.beta(omega,m,n))
	
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
		return omega / self.beta(omega,m,n)
	
	def vp_f(self, f,m=1,n=0):
		return 2*pi*f / self.beta(2*pi*f,m,n)
		
	def zTE(self, omega,m=1,n=0):
		return eta0 * beta0(omega)/self.beta(omega,m,n)
	def zTM(self, omega,m=1,n=0):
		return eta0 * self.beta(omega,m,n)/beta0(omega)

	## standard creation
	# one-port 
	def createDelayShort(self,l,numPoints):
		'''
		generate the reflection coefficient for a waveguide delayed short of length l 
		
		takes:
			l - length of thru, in meters
			numPoints - number of points to produce
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''

		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s=createDelayShort(freq,l ,self.beta)
		return ntwk(data=s,paramType='s',freq=freq,freqUnit='Hz',freqMultiplier=1)

			
	def createShort(self, numPoints):
		'''
		generate the reflection coefficient for a waveguide short.
		convinience function, see mwavepy.createShort()
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createShort(numPoints)
		return ntwk(data=s,paramType='s',freq=freq,freqUnit='Hz',freqMultiplier=1)
		
	def createMatch(self,numPoints):
		'''
		generate the reflection coefficient for a waveguide Match.
		convinience function, see mwavepy.createShort()
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createMatch(numPoints)
		return ntwk(data=s,paramType='s',freq=freq,freqUnit='Hz',freqMultiplier=1)
		
	# two-port 
	def createDelay(self,l,numPoints):
		'''
		generate the two port S matrix for a waveguide thru section of length l 
		
		takes:
			l - length of thru, in meters
			numPoints - number of points to produce
		returns:
			two port S matrix for a waveguide thru section of length l 
		'''
		freq = npy.linspace(self.band[0],self.band[1],numPoints)
		s = createDelay(self.band[0],self.band[1],numPoints,l, self.beta)		
		return ntwk(data=s,paramType='s',freq=freq,freqUnit='Hz',freqMultiplier=1)
		
		
		
		
class wr(waveguide):
	'''
	class which represents a standard rectangular waveguide band.
	inherits mwavepy.waveguide type 
	'''
	def __init__(self, number):
		'''
		takes:
			number - waveguide designator number ( ie WR10, the number is 10)
		returns:
			mwavepy.waveguide object representing the given band
		'''
		waveguide.__init__(self,number*10*mil ,.5 * number*10*mil  )
# standard waveguide bands, note that the names are not perfectly cordinated with guide dims. 
# info taken from Virginia Diodes Inc. Waveguide Band Designations
WR10 = wr(10)
WR8 = wr(8)
WR6 = wr(6.5)
WR5 = wr(5.1)
WR4 = wr(4.3)
WR3 = wr(3.4)
WR1p5 = wr(1.5)		
	
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
	return omega* sqrt((const.mu_0*muR)*(epsilonR*epsilon_0))

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
	return sqrt((const.mu_0*muR)/(epsilonR*epsilon_0))
def eta0(omega):
	'''
	characteristic impedance of free space. see eta().
	'''
	return eta(omega, 1,1)
	
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

def gamma(zl,z0=50.0, theta=0):
	'''
	calculates the reflection coefficient for a given load and characteristic impedance
	takes:
		zl - load impedance
		z0 - characteristic impedance
		theta - distance from load, given in electrical length  (rad)
	'''
	# this way of type casting allows for arrays to be passed, but does floating points arrimetic
	zl = 1.0*(zl)
	z0 = 1.0*(z0)
	theta = 1.0* (theta)
	# handle the limit of open circuit
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
	## connections
def connectionSeriees(ntwkA,ntwkB, type='s'):
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
	

		
		


## Theoretically derived Networks
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
	generates the two port S matrix for a matched Delay line of length l. 
	
	takes:
		freqVector -1D  array corresponding to the frequency band over which to calculate. ( in Hz)
		l - length of delay in units of m 
		beta - propagation constant function. this is a function which is a function of angular frequency (omega), and returns a value with units radian/m. 
		
		note: beta defaults to lossless free-space propagation constant beta = omega/c = omega*sqrt(epsilon_0*mu_0), which assumes a TEM wave	
	'''
	s11 = npy.complex_( npy.zeros(numPoints))
	s12 = npy.complex_( npy.zeros(numPoints))
	s21 = npy.complex_( npy.zeros(numPoints))
	s22 = npy.complex_( npy.zeros(numPoints))
	
	s12 = s21=  exp(-1j*electricalLength(l,freqVector,beta) )
	
	return npy.array([[s11, s12],\
					[s21, s22] ])





	
############## calibration ##############
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

def applyABC( gamma, abc):
	'''
	takes a complex array of uncalibrated reflection coefficient and applies
	the one-port OSM callibration, using the coefficients abc. 

	takes:
		gamma - complex reflection coefficient OR can  be 1-port mwavepy.ntwk() type.
		abc - Nx3 OSM calibration coefficients. 
	returns:
		either a complex reflection coefficient, or a 1-port mwavepy.ntwk() instance, depending on input
		
	'''
	
	#TODO: re-write this so the variables make more sense to a reader
	# type test
	if isinstance(gamma,ntwk):
		# they passed us ntwk types, so lets get the relevent parameter
		#make sure its a 1-port
		if gamma.rank >1:
			print 'ERROR: this takes a 1-port'
			return None
		else:
			newNtwk = copy(gamma)
			gamma = gamma.s[:,0,0]
			gammaCal= ((gamma-abc[:,1]) / (abc[:,0]+ gamma*abc[:,2]))
			newNtwk.__sets__(gammaCal)
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




## Old /unsorted
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
		ntwkList  - lists holding basenames of the files loaded, and a list of mwavepy networks. If you are using pylab you can plot these easily like so: 
	
	example usage:
		import mwavepy as m
		nameList, ntwkList = m.loadAllTouchtonesInDir()
		for n in ntwkList:
			npy.plotReturnLoss()
		
	'''
	ntwkList=[]

	for f in os.listdir (dir):
		# TODO: make this s?p with reg ex
		if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
			ntwkList.append(createNtwkFromTouchstone(f))
		
	return ntwkList



def plotCsv(filename,rowsToSkip=1,delim=','):
	'''plots columns from csv file. plots all columns against the first
	see pylab.loadtxt for more information
	'''
	data = plb.loadtxt(filename,skiprows=rowsToSkip,delimiter=delim)
	plb.plot(data[:,0], data[:,1:])
	plb.grid(1)
	plb.title(filename)



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

	Y = 1./len(y)* plb.fftshift(plb.fft(y))
	spectrumVector = Y[len(Y)/2:]
	freqVector = f[len(f)/2:]
	return [freqVector,spectrumVector]

def cutOff(a):
	'''returns the cutoff frequency (in Hz) for first  resonance of a
	waveguide with major dimension given by a. a is in meters'''
	
	return sqrt((pi/a)**2 *1/(epsilon_0*mu_0))/(2*pi)

