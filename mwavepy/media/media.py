
#       workingBand.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       
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
Contains Media class. 
'''
import warnings 

from copy import copy
import numpy as npy
from scipy import stats

from ..network import Network, connect
from ..transmissionLine import functions as tlinefunctions
from .. import mathFunctions as mf

class Media(object):
	'''
	This is the super-class for all transmission line media. It provides
	methods to produce general network components for any transmision 
	line medium, such as line, delay short, etc. These methods in turn 
	rely on the properties:
		propgation_constant
		characteristic_impedance
	Implementation of propagation_constant and characteristic_impedance 
	must be implemented in the instances of this class.
	'''
	def __init__(self, z0=None):
		'''
		
		
		'''
		
		
		if z0 is None:
			try:
				z0 = self.characteristic_impedance
			except(AttributeError):
				raise AttributeError('z0 is not inspectable. please provide one.')
		self.z0 = z0
		self.delay = self.line
		
		try:
			a=self.propagation_constant
		except(AttributeError):
			raise NotImplementedError('media instance must implement the property: propagation_constant')
		try:
			a=self.characteristic_impedance
		except(AttributeError):
			raise NotImplementedError('media instance must implement the property: characteristic_impedance')
			
	
	
	## Other Functions
	def theta_2_d(self,theta,deg=True):
		'''
		converts electrical length to physical distance
		
		takes:
			theta: electrical length, at band center (see deg for unit)[number]
			deg: is theta in degrees? [boolean]
			
		returns:
			d: physical distance in meters
			
		
		'''
		if deg == True:
			theta = mf.degree_2_radian(theta)
		
		gamma = self.propagation_constant
		return 1.0*theta/npy.imag(gamma[gamma.size/2])
	
	def electrical_length(self, d,deg=False):
		'''
		takes:
			d: distance, in meters
		
		returns:
			theta: electrical length in radians or degrees, 
				depending on  value of deg.
		'''
		gamma = self.propagation_constant
		
		if deg == False:
			return  gamma*d
		elif deg == True:
			return  mf.radian_2_degree(gamma*d )
		
	## Network creation
	def match(self,nports=1, z0=None, **kwargs):
		'''
		creates a Network for a perfect matched transmission line (Gamma0=0) 
		
		takes:
			nports: number of ports [int]
			z0: characterisitc impedance [number of array]. defaults is 
				None, in which case the WorkingBand's z0 is used. 
				Otherwise this sets the resultant network's z0. See
				Network.z0 property for more info
			**kwargs: key word arguments passed to Network Constructor
		
		returns:
			a n-port Network [mwavepy.Network]
		
		
		example:
			mymatch = wb.match(2,z0 = 50, name='Super Awesome Match')
		
		'''
		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s =  npy.zeros((self.frequency.npoints,nports, nports),\
			dtype=complex)
		if z0 is None:
			z0 = self.z0
		result.z0=z0
		return result
	
	def load(self,Gamma0,nports=1,**kwargs):
		'''
		creates a Network for a Load termianting a transmission line 
		
		takes:
			Gamma0: reflection coefficient of load (not in db)
			nports: number of ports. creates a short on all ports,
				default is 1 [int]
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network
		returns:
			a 1-port Network class, where  S = Gamma0*ones(...)
		'''
		result = self.match(nports,**kwargs)
		for f in range(self.frequency.npoints):
			result.s[f,:,:] = Gamma0*npy.eye(nports, dtype=complex)
		
		return result		
	
	def short(self,nports=1,**kwargs):
		'''
		creates a Network for a short  transmission line (Gamma0=-1) 
		
		takes:
			nports: number of ports. creates a short on all ports,
				default is 1 [int]
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network
		returns:
			a n-port Network [mwavepy.Network]
		'''
		return self.load(-1., nports, **kwargs)

	def open(self,nports=1, **kwargs):
		'''
		creates a Network for a 'open' transmission line (Gamma0=1) 
		
		takes:
			nports: number of ports. creates a short on all ports,
				default is 1 [int]
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network
		returns:
			a n-port Network [mwavepy.Network]
		'''
		
		return self.load(1., nports, **kwargs)

	def thru(self, **kwargs):
		'''
		creates a Network for a thru
		
		takes:
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network
		returns:
			a 2-port Network class, representing a thru

		note:
			this just calls self.line(0)
		'''
		return self.line(0,**kwargs)
	
	def line(self,d, unit='m',**kwargs):
		'''
		creates a Network for a section of matched transmission line
		
		takes:
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		
		returns:
			a 2-port Network class, representing a transmission line of 
			length d
	
		
		example:
			wb = WorkingBand(...) # create a working band object
			wb.line(90, 'deg', z0=50) 
		
		'''
		if unit not in ['m','deg','rad']:
			raise (ValueError('unit must be one of the following:\'m\',\'rad\',\'deg\''))
		
		result = self.match(nports=2,**kwargs)
		
		d_dict ={\
			'deg':self.theta_2_d(d,deg=True),\
			'rad':self.theta_2_d(d,deg=False),\
			'm':d\
			}
		
		theta = self.electrical_length(d_dict[unit])
		
		s11 = npy.zeros(self.frequency.npoints, dtype=complex)
		s21 = npy.exp(-1*theta)
		result.s = \
			npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
		return result

	def delay_load(self,Gamma0,d,unit='m',**kwargs):
		'''
		creates a Network for a delayed load transmission line
		
		takes:
			Gamma0: reflection coefficient of load (not in dB)
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians	
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		
		returns:
			a 1-port Network class, representing a loaded transmission
			line of length d
			
		
		note: this just calls,
		self.line(d,**kwargs) ** self.load(Gamma0, **kwargs)
		'''
		return self.line(d=d, unit=unit,**kwargs)**\
			self.load(Gamma0=Gamma0,**kwargs)	

	def delay_short(self,d,unit='m',**kwargs):
		'''
		creates a Network for a delayed short transmission line
		
		takes:
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		returns:
			a 1-port Network class, representing a shorted transmission
			line of length d
			
		
		note: this just calls,
		self.line(d,**kwargs) ** self.short(**kwargs)
		'''
		return self.delay_load(Gamma0=-1., d=d, unit=unit, **kwargs)
	
	def delay_open(self,d,unit='m',**kwargs):
		'''
		creates a Network for a delayed open transmission line
		
		takes:
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		returns:
			a 1-port Network class, representing a shorted transmission
			line of length d
			
		
		note: this just calls,
		self.line(d,**kwargs) ** self.open(**kwargs)
		'''
		return self.delay_load(Gamma0=1., d=d, unit=unit,**kwargs)
	
	def tee(self,**kwargs):
		'''
		makes a ideal, lossless tee. (aka three port splitter)
		
		takes:
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. 
		returns:
			a 3-port Network [mwavepy.Network]
		
		note:
			this just calls splitter(3)
		'''
		return self.splitter(3,**kwargs)
		
	def splitter(self, nports,**kwargs):
		'''
		returns an ideal, lossless n-way splitter.
		
		takes:
			nports: number of ports [int]
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. 
		returns:
			a n-port Network [mwavepy.Network]
		'''
		n=nports
		result = self.match(n, **kwargs)
		
		for f in range(self.frequency.npoints):
			result.s[f,:,:] =  (2*1./n-1)*npy.eye(n) + \
				npy.sqrt((1-((2.-n)/n)**2)/(n-1))*\
				(npy.ones((n,n))-npy.eye(n))
		return result
	
	def impedance_mismatch(self, z1, z2, **kwargs):
		'''
		returns a two-port network for a impedance mis-match
		
		takes:
			z1: complex impedance of port 1 [ number, list, or 1D ndarray]
			z2: complex impedance of port 2 [ number, list, or 1D ndarray]
			**kwargs: passed to mwavepy.Network constructor
		returns:
			a 2-port network [mwavepy.Network]
			
		note:
			if z1 and z2 are arrays or lists, they must be of same length
			as the frequency for this working band(WorkBand.frequency.f)
		'''	
		result = self.match(nports=2, **kwargs)
		gamma = tlinefunctions.zl_2_Gamma0(z1,z2)
		result.s[:,0,0] = gamma
		result.s[:,1,1] = -gamma
		result.s[:,1,0] = 1+gamma
		result.s[:,0,1] = 1-gamma
		return result
	
	def shunt(self,ntwk, **kwargs):
		'''
		returns a shunted ntwk. this creates a 'tee', connects 
		'ntwk' to port 1, and returns the result
		
		takes:
			ntwk: the network to be shunted. [mwavepy.Network]
			**kwargs: passed to the self.tee() function
			
		returns:
			a 2-port network [mwavepy.Network]
		'''
		return connect(self.tee(**kwargs),1,ntwk,0)
		
	def shunt_delay_load(self,*args, **kwargs):
		'''
		a shunted delayed load:
		
		takes:
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		returns:
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_load(*args, **kwargs))
		
	def shunt_delay_open(self,*args,**kwargs):	
		'''
		a shunted delayed open:
		
		takes:
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		returns:
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_open(*args, **kwargs))
	
	def shunt_delay_short(self,*args,**kwargs):	
		'''
		a shunted delayed short:
		
		takes:
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		returns:
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_short(*args, **kwargs))
	
	## Noise Networks
	def white_gaussian_polar(self,phase_dev, mag_dev,n_ports=1,**kwargs):
		'''
		creates a complex zero-mean gaussian white-noise signal of given
		standard deviations for phase and magnitude

		takes:
			phase_mag: standard deviation of magnitude
			phase_dev: standard deviation of phase
			n_ports: number of ports. defualt to 1
			**kwargs: passed to Network() initializer
		returns:
			result: Network type 
		'''
		shape = (self.frequency.npoints, n_ports,n_ports)
		phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = shape)
		mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = shape)

		result = Network(**kwargs)
		result.frequency = self.frequency
		result.s = mag_rv*npy.exp(1j*phase_rv)
		return result


	## OTHER METHODS
	def guess_length_of_delay_short(self, aNtwk):
		'''
		guess length of physical length of a Delay Short given by aNtwk
		
		takes:
			aNtwk: a mwavepy.ntwk type . (note: if this is a measurment 
				it needs to be normalized to the reference plane)
			tline: transmission line class of the medium. needed for the 
				calculation of propagation constant
				
		
		'''
		beta = npy.real(self.propagation_constant(self.frequency.f))
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]
	
