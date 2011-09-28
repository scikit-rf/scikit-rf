
#       rectangularWaveguide.py
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
Rectangular Waveguide class
'''
from scipy.constants import  epsilon_0, mu_0,pi
from numpy import sqrt
from media import Media

class RectangularWaveguide(Media):
	'''
	Rectangular Waveguide medium. 
	
	Can be used to represent any mode of a homogeneously filled 	
	rectangular waveguide of arbitrary cross-section, mode-type, and
	mode index.
	'''
	def __init__(self, frequency, a, b=None, mode_type = 'te', m=1, \
		n=0, ep_r=1, mu_r=1, *args, **kwargs):
		'''
		takes:
			frequency: mwavepy.Frequency object
			a: width of waveguide, in meters. [number]
			b: height of waveguide, in meters. defaults to a/2 [number]
			mode_type: mode type, can be either 'te' or 'tm' (to-z)
			m: mode index in 'a'-direction, (default=1) [integer]
			n: mode index in 'b'-direction, (default=0) [integer]
			ep_r: filling material relative permativity [number]
			mu_r: filling material relative permeability [number]
			*args,**kwargs: passed to Media() constructor
		returns:
			mwavepy.Media object
			
			
		example:
			most common usage is probably standard waveguide dominant 
			mode. TE10 mode of wr10 waveguide can be constructed by
			
			freq = mwavepy.Frequency(75,110,101,'ghz')
			RectangularWaveguide(freq, 100*mil)
		'''
		if b is None: 
			b = a/2.
		if mode_type.lower() not in ['te','tm']:
			raise ValueError('mode_type must be either \'te\' or \'tm\'')
			 
		self.frequency = frequency 
		self.a = a
		self.b = b
		self.mode_type = mode_type
		self.m = m
		self.n = n
		self.ep_r = ep_r
		self.mu_r = mu_r
		
		
		Media.__init__(self,\
			frequency = frequency,\
			propagation_constant = self.kz, \
			characteristic_impedance = self.Z0,\
			*args, **kwargs)
	
	def __str__(self):
		f=self.frequency
		output =  \
			'Rectangular Waveguide Media.  %i-%i %s.  %i points'%\
			(f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
			'\n a= %.2em, b= %.2em'% \
			(self.a,self.b)
		return output
		
	def __repr__(self):
		return self.__str__()
	
	@property
	def ep(self):
		'''
		the permativity of the filling material 
		'''
		return self.ep_r * epsilon_0
	
	@property
	def mu(self):
		'''
		the permeability of the filling material 
		'''
		return self.mu_r * mu_0
	
	@property
	def k0(self):
		'''
		characteristic wave number
		'''
		return 2*pi*self.frequency.f*sqrt(self.ep * self.mu)
	
	@property
	def ky(self):
		'''
		eigen-value in the 'b' direction
		'''
		return self.n*pi/self.b
	
	@property
	def kx(self):
		'''
		eigen value in the 'a' direction
		'''
		return self.m*pi/self.a
	
	@property
	def kc(self):
		'''
		cut-off wave number 
		'''
		return sqrt( self.kx**2 + self.ky**2)
	
	
	def kz(self):
		'''
		the propagation constant, which is:
			IMAGINARY for propagating modes
			REAL  for non-propagating modes, 
		'''
		k0,kc = self.k0, self.kc
		return \
			1j*sqrt(abs(k0**2 - kc**2)) * (k0>kc) +\
			sqrt(abs(kc**2- k0**2))*(k0<kc) + \
			0*(kc==k0)	
	
	
	def Z0(self):
		'''
		the characteristic impedance of a given mode
		'''
		omega = self.frequency.w
		impedance_dict = {\
			'tez':	omega*self.mu/(-1*self.kz()),\
			'te':	omega*self.mu/(-1*self.kz()),\
			'tmz':	-1*self.kz()/(omega*self.ep),\
			'tm':	-1*self.kz()/(omega*self.ep),\
			}
		
		return impedance_dict[self.mode_type]
