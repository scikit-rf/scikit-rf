
#       media.py
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
.. module:: mwavepy.media.media
========================================
media (:mod:`mwavepy.media.media`)
========================================

Contains Media class. 


'''
import warnings 

import numpy as npy
from scipy import stats

from ..network import Network, connect
from .. import tlineFunctions as tf
from .. import mathFunctions as mf

class Media(object):
	'''
	The base-class for all transmission line mediums.
	
	The :class:`Media` object provides generic methods to produce   :class:`~mwavepy.network.Network`'s for any transmision line medium, such  as :func:`line` and :func:`delay_short`. 
	
	The initializer for this class has flexible argument types. This
	allows for the important attributes of the :class:`Media` object 
	to be dynamic. For example, if a Media object's propagation constant
	is a function of some attribute of that object, say `conductor_width`, 
	then the propagation constant will change when that attribute 
	changes. See :func:`__init__` for details.
	
	'''
	def __init__(self, frequency,  propagation_constant,
		characteristic_impedance, z0=None):
		'''
		The Media initializer. 
		
		This initializer has flexible argument types. The parameters
		`propagation_constant`, `characterisitc_impedance` and `z0` can 
		all be either static or dynamic. This is achieved by allowing 
		those arguments to be either:
		 * functions which take no arguments or 
		 * values (numbers or arrays)
		
		In the case where the media's propagation constant may change 
		after initialization, because you adjusted a parameter of the 
		media, then passing the propagation_constant as a function 
		allows it to change when the media's parameters do.
		
		Parameters
		--------------
		frequency : :class:`~mwavepy.frequency.Frequency` object
			frequency band of this transmission line medium
		
		propagation_constant : number, array-like, or a function
			propagation constant for the medium. 
		
		characteristic_impedance : number,array-like, or a function
			characteristic impedance of transmission line medium.
		
		z0 : number, array-like, or a function
			the port impedance for media , IF its different
			from the characterisitc impedance of the transmission 
			line medium  (None) [a number].
			if z0= None then will set to characterisitc_impedance
			
		
				
		Notes
		------
		`propagation_constant` must adhere to the following convention,
		 * positive real(gamma) = attenuation
		 * positive imag(gamma) = forward propagation 
		
		the z0 parameter is needed in some cases. For example, the 
		:class:`~mwavepy.media.rectangularWaveguide.RectangularWaveguide`
		is an example  where you may need this, because the 
		characteristic impedance is frequency dependent, but the 
		touchstone's created by most VNA's have z0=1	
		'''
		self.frequency = frequency
		
		self.propagation_constant = propagation_constant
		self.characteristic_impedance = characteristic_impedance
		
		if z0 is None:
			z0 = characteristic_impedance
		self.z0 = z0
		
		# convinience names
		self.delay = self.line
	
	## Properties
	# note these are made so that a Media type can be constructed with 
	# propagation_constant, characteristic_impedance, and z0 either as:
	#	dynamic properties (if they pass a function) 
	#	static ( if they pass values)
	@property
	def propagation_constant(self):
		'''
		Propagation constant
		
		Returns
		---------
		propagation_constant : :class:`numpy.ndarray`
			complex propagation constant for this media
		
		Notes
		------
		`propagation_constant` must adhere to the following convention,
		 * positive real(propagation_constant) = attenuation
		 * positive imag(propagation_constant) = forward propagation 
		'''
		try:
			return self._propagation_constant()
		except(TypeError):
			return self._propagation_constant
	@propagation_constant.setter
	def propagation_constant(self, new_propagation_constant):
		self._propagation_constant = new_propagation_constant
	
	@property
	def characteristic_impedance(self):
		'''
		Characterisitc impedance
		
		Returns
		----------
		characteristic_impedance : :class:`numpy.ndarray`
		'''
		try:
			return self._characteristic_impedance()
		except(TypeError):
			return self._characteristic_impedance
	
	@characteristic_impedance.setter
	def characteristic_impedance(self, new_characteristic_impedance):
		self._characteristic_impedance = new_characteristic_impedance
	
	@property
	def z0(self):
		'''
		Port Impedance
		
		The port impedance  is usually equal to the 
		:attr:`characterisitc_impedance`. Therefore, if the port 
		impedance is `None` then this will return 
		:attr:`characterisitc_impedance`.
		
		However, in some cases such as rectangular waveguide, the port 
		impedance is traditionally set to 1 (normalized). In such a case
		this property may be used. 
		
		
		Returns
		----------
		port_impedance : :class:`numpy.ndarray`
			the media's port impedance
		'''
		try:
			return self._z0()
		except(TypeError):
			return self._z0
	@z0.setter
	def z0(self, new_z0):
		self._z0 = new_z0
	
	
	
	## Other Functions
	def theta_2_d(self,theta,deg=True):
		'''
		converts electrical length to physical distance. The electrical
		length is given at center frequency of self.frequency 
		
		Parameters
		----------
			theta: electrical length, at band center (see deg for unit)
				[number]
			deg: is theta in degrees? [boolean]
			
		Returns
		--------
			d: physical distance in meters
			
		
		'''
		if deg == True:
			theta = mf.degree_2_radian(theta)
		
		gamma = self.propagation_constant
		return 1.0*theta/npy.imag(gamma[gamma.size/2])
	
	def electrical_length(self, d,deg=False):
		'''
		calculates the electrical length for a given distance, at 
		the center frequency. 
		
		Parameters
		----------
			d: distance, in meters 
			deg: is d in deg?[Boolean]
		
		Returns
		--------
			theta: electrical length in radians or degrees, 
				depending on  value of deg.
		'''
		gamma = self.propagation_constant
		
		if deg == False:
			return  gamma*d
		elif deg == True:
			return  mf.radian_2_degree(gamma*d )
		
	## Network creation
	
	# lumped elements
	def match(self,nports=1, z0=None, **kwargs):
		'''
		Perfect matched load (:math:`\\Gamma_0 = 0`).
		
		Parameters
		----------
		nports : int
			number of ports
		z0 : number, or array-like
			characterisitc impedance. Default is 
			None, in which case the Media's :attr:`z0` is used. 
			This sets the resultant Network's 
			:attr:`~mwavepy.network.Network.z0`. 
		\*\*kwargs : key word arguments 
			passed to :class:`~mwavepy.network.Network` initializer
		
		Returns
		--------
		match : :class:`~mwavepy.network.Network` object 
			a n-port match  
		
		
		Examples
		------------
			>>> my_match = my_media.match(2,z0 = 50, name='Super Awesome Match')
		
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
		Load of given reflection coefficient. 
		
		Parameters
		----------
		Gamma0 : number, array-like 
			Reflection coefficient of load (linear, not in db). If its
			an array it must be of shape: kxnxn, where k is #frequency 
			points in media, and n is `nports`
		nports : int
			number of ports 
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network. 
		
		Returns
		--------
		load  :class:`~mwavepy.network.Network` object
			n-port load, where  S = Gamma0*eye(...)
		'''
		result = self.match(nports,**kwargs)
		try:
			result.s = npy.reshape(Gamma0,(-1,nports,nports)) * \
				npy.eye(nports,dtype=complex).reshape((-1,nports,nports)).\
				repeat(self.frequency.npoints,0)
		except(ValueError):
			for f in range(self.frequency.npoints):
				result.s[f,:,:] = Gamma0[f]*npy.eye(nports, dtype=complex)
		
		return result		
	
	def short(self,nports=1,**kwargs):
		'''
		Short (:math:`\\Gamma_0 = -1`) 
		
		Parameters
		----------
		nports : int
			number of ports 
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network. 
		
		Returns
		--------
		match : :class:`~mwavepy.network.Network` object 
			a n-port short circuit  
		
		See Also
		---------
		match : function called to create a 'blank' network
		'''
		return self.load(-1., nports, **kwargs)

	def open(self,nports=1, **kwargs):
		'''
		Open (:math:`\\Gamma_0 = 1`) 
		
		Parameters
		----------
		nports : int
			number of ports 
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network. 
		
		Returns
		--------
		match : :class:`~mwavepy.network.Network` object 
			a n-port open circuit  
		
		See Also
		---------
		match : function called to create a 'blank' network
		'''
		
		return self.load(1., nports, **kwargs)
	
	def capacitor(self, C, **kwargs):
		'''
		Capacitor  
		
		
		Parameters
		----------
		C : number, array 
			Capacitance, in Farads. If this is an array, must be of 
			same length as frequency vector. 
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network. 
		
		Returns
		--------
		capacitor : :class:`~mwavepy.network.Network` object 
			a n-port capacitor  
		
		See Also
		---------
		match : function called to create a 'blank' network 
		'''
		Gamma0 = tf.zl_2_Gamma0(self.z0, -1j/(self.frequency.w*C))
		return self.load(Gamma0=Gamma0, **kwargs)

	def inductor(self, L, **kwargs):
		'''
		Inductor   
		
		Parameters
		----------
		L : number, array 
			Inductance, in Henrys. If this is an array, must be of 
			same length as frequency vector. 
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network. 
		
		Returns
		--------
		inductor : :class:`~mwavepy.network.Network` object 
			a n-port inductor   
		
		See Also
		---------
		match : function called to create a 'blank' network  
		'''
		Gamma0 = tf.zl_2_Gamma0(self.z0,-1j*(self.frequency.w*L))
		return self.load(Gamma0=Gamma0, **kwargs)

	def impedance_mismatch(self, z1, z2, **kwargs):
		'''
		Two-port network for an impedance miss-match
		
		
		Parameters
		----------
		z1 : number, or array-like
			complex impedance of port 1 
		z2 : number, or array-like
			complex impedance of port 2
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network.
		
		Returns
		--------
		missmatch : :class:`~mwavepy.network.Network` object 
			a 2-port network representing the impedance missmatch
			
		Notes
		--------
		If z1 and z2 are arrays, they must be of same length
		as the :attr:`Media.frequency.npoints`
		'''	
		result = self.match(nports=2, **kwargs)
		gamma = tf.zl_2_Gamma0(z1,z2)
		result.s[:,0,0] = gamma
		result.s[:,1,1] = -gamma
		result.s[:,1,0] = 1+gamma
		result.s[:,0,1] = 1-gamma
		return result
	
		
	# splitter/couplers
	def tee(self,**kwargs):
		'''
		Ideal, lossless tee. (3-port splitter)
		
		Parameters
		----------
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network.
		
		Returns
		--------
		tee : :class:`~mwavepy.network.Network` object
			a 3-port splitter
		
		See Also
		----------
		splitter : this just calls splitter(3)
		'''
		return self.splitter(3,**kwargs)
		
	def splitter(self, nports,**kwargs):
		'''
		Ideal, lossless n-way splitter.
		
		Parameters
		----------
		nports : int
			number of ports
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network.
		
		Returns
		--------
			tee : :class:`~mwavepy.network.Network` object
			a n-port splitter
		'''
		n=nports
		result = self.match(n, **kwargs)
		
		for f in range(self.frequency.npoints):
			result.s[f,:,:] =  (2*1./n-1)*npy.eye(n) + \
				npy.sqrt((1-((2.-n)/n)**2)/(n-1))*\
				(npy.ones((n,n))-npy.eye(n))
		return result
	

	# transmission line
	def thru(self, **kwargs):
		'''
		Matched transmission line of length 0.
		
		Parameters
		----------
		\*\*kwargs : key word arguments 
			passed to :func:`match`, which is called initially to create a 
			'blank' network.
		
		Returns
		--------
		thru : :class:`~mwavepy.network.Network` object
			matched tranmission line of 0 length
		See Also
		---------
			line : this just calls line(0)
		'''
		return self.line(0,**kwargs)
	
	def line(self,d, unit='m',**kwargs):
		'''
		creates a Network for a section of matched transmission line
		
		Parameters
		----------
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		
		Returns
		--------
			a 2-port Network class, representing a transmission line of 
			length d
	
		
		example:
			my_media = mwavepy.Freespace(...)
			my_media.line(90, 'deg', z0=50) 
		
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
		
		Parameters
		----------
			Gamma0: reflection coefficient of load (not in dB)
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians	
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		
		Returns
		--------
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
		
		Parameters
		----------
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		Returns
		--------
			a 1-port Network class, representing a shorted transmission
			line of length d
			
		
		note: this just calls,
		self.line(d,**kwargs) ** self.short(**kwargs)
		'''
		return self.delay_load(Gamma0=-1., d=d, unit=unit, **kwargs)
	
	def delay_open(self,d,unit='m',**kwargs):
		'''
		creates a Network for a delayed open transmission line
		
		Parameters
		----------
			d: the length (see unit argument) [number]
			unit: string specifying the units of d. possible options are 
				'm': meters, physical length in meters (default)
				'deg':degrees, electrical length in degrees
				'rad':radians, electrical length in radians
			**kwargs: key word arguments passed to match(), which is 
				called initially to create a 'blank' network. the kwarg
				'z0' can be used to create a line of a given impedance
		Returns
		--------
			a 1-port Network class, representing a shorted transmission
			line of length d
			
		
		note: this just calls,
		self.line(d,**kwargs) ** self.open(**kwargs)
		'''
		return self.delay_load(Gamma0=1., d=d, unit=unit,**kwargs)

	def shunt(self,ntwk, **kwargs):
		'''
		returns a shunted ntwk. this creates a 'tee', connects 
		'ntwk' to port 1, and returns the result
		
		Parameters
		----------
			ntwk: the network to be shunted. [mwavepy.Network]
			**kwargs: passed to the self.tee() function
			
		Returns
		--------
			a 2-port network [mwavepy.Network]
		'''
		return connect(self.tee(**kwargs),1,ntwk,0)
		
	def shunt_delay_load(self,*args, **kwargs):
		'''
		a shunted delayed load:
		
		Parameters
		----------
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		Returns
		--------
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_load(*args, **kwargs))
		
	def shunt_delay_open(self,*args,**kwargs):	
		'''
		a shunted delayed open:
		
		Parameters
		----------
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		Returns
		--------
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_open(*args, **kwargs))
	
	def shunt_delay_short(self,*args,**kwargs):	
		'''
		a shunted delayed short:
		
		Parameters
		----------
			*args: passed to self.delay_load
			**kwargs:passed to self.delay_load
		Returns
		--------
			a 2-port network [mwavepy.Network]
		'''
		return self.shunt(self.delay_short(*args, **kwargs))
	
	def shunt_capacitor(self,C,*args,**kwargs):
		'''
		a shunt capacitor
		
		Parameters
		----------
			C: capacitance in farads
			*args: passed to self.capacitor
			**kwargs:passed to self.capacitor
		Returns
		--------
			a 2-port mwavepy.Network
		
		'''
		return self.shunt(self.capacitor(C=C,*args,**kwargs))
	
	def shunt_inductor(self,L,*args,**kwargs):
		'''
		a shunt inductor
		
		Parameters
		----------
			L: inductance in henrys
			*args: passed to self.inductor
			**kwargs:passed to self.inductor
		Returns
		--------
			a 2-port mwavepy.Network
		
		'''
		return self.shunt(self.inductor(L=L,*args,**kwargs))
		
	
	## Noise Networks
	def white_gaussian_polar(self,phase_dev, mag_dev,n_ports=1,**kwargs):
		'''
		creates a complex zero-mean gaussian white-noise signal of given
		standard deviations for phase and magnitude

		Parameters
		----------
			phase_mag: standard deviation of magnitude
			phase_dev: standard deviation of phase
			n_ports: number of ports. defualt to 1
			**kwargs: passed to Network() initializer
		Returns
		--------
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
		
		Parameters
		----------
			aNtwk: a mwavepy.ntwk type . (note: if this is a measurment 
				it needs to be normalized to the reference plane)
			tline: transmission line class of the medium. needed for the 
				calculation of propagation constant
				
		
		'''
		warnings.warn(DeprecationWarning('I have yet to update this for Media class'))
		beta = npy.real(self.propagation_constant(self.frequency.f))
		thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())
		
		A = npy.vstack((2*beta,npy.ones(len(beta)))).transpose()
		B = thetaM
		
		print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
		return npy.linalg.lstsq(A, B)[0][0]
	
