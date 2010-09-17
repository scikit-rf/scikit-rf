'''
#       workingBand.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen 
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
#import  mwavepy1 as mv1

from frequency import Frequency
import createNetwork 

class WorkingBand(object):
	'''
	A WorkingBand is an high-level object which exists solely to make 
	 working of Networks more concise and convinient. 
	
	A WorkingBand object has two properties: 
		frequency information (Frequency object)
		transmission line information	(transmission line-like object)
		
	as stated in parenthesis both of these properties are objects 
	themselves. 
	
	
	'''
	def __init__(self, frequency, tline):
		self.frequency = frequency 
		self.tline = tline
		
	@property
	def frequency(self):
		return self._frequency
	@frequency.setter
	def frequency(self,new_frequency):
		self._frequency= new_frequency

	@property
	def f(self):
		return self.frequency.f

	@property
	def tline(self):
		return self._tline
	@tline.setter
	def tline(self,new_tline):
		self._tline = new_tline
		
	# Network creation
	def short(self,**kwargs):
		'''
		creates a delay short Network object
		'''
		ntwk = createNetwork.short(self.f,**kwargs)
		ntwk.frequency = self.frequency
		return ntwk
	def match(self,**kwargs):
		'''
		creates a perfect match Network object
		'''
		ntwk = createNetwork.match(self.f,**kwargs)
		ntwk.frequency = self.frequency
		return ntwk
	def open(self,**kwargs):
		'''
		creates a open Network object
		'''
		ntwk = createNetwork.open(self.f,**kwargs)
		ntwk.frequency = self.frequency
		return ntwk
	def line(self,d,**kwargs):
		'''
		creates a line of length 'd' Network object
		'''
		ntwk = createNetwork.delay(d=d, tline=self.tline, \
			frequency=self.f,**kwargs )
		ntwk.frequency = self.frequency
		return ntwk
	
	def delay_short(self,d,**kwargs):
		'''
		creates a delayed short of length 'd' Network object
		'''
		ntwk = createNetwork.delay_short(d=d,tline=self.tline, \
			frequency = self.f,**kwargs)
		ntwk.frequency = self.frequency
		return ntwk

	def guess_length_of_delay_short(self, aNtwk):
		raise NotImplementedError
		'''
		guess length of physical length of a Delay Short given by aNtwk
		
		takes:
			aNtwk: a mwavepy.ntwk type . (note: if this is a measurment 
				it needs to be normalized to the short plane
			tline: transmission line class of the medium. needed for the 
				calculation of propagation constant
				
		
		'''
		beta = real(self.tline.beta())
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((-2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]
	
