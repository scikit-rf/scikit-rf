'''
#       transmissionLine.py
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

from scipy.constants import  epsilon_0, mu_0, c,pi, mil,pi
import numpy as npy
from numpy import sqrt,array, cos, sin, linspace
# would there be any benefit to structuring modes as objects?
#def RectangularWaveguideMode(object,m,n):
	#self.m = m
	#self.n = n
	
	#@property 
	#def cut_off_frequency(self):
		#return 1./(2*sqrt(self.epsilon*self.mu)) * \
			#sqrt( (self.m/a)**2 + (self.n/2)**2)
import pdb		
class RectangularWaveguide(object):
	'''
	represents a homogeneously rectangular waveguide.
	
	cross-section is axb, and it is homogeneously with relative
	permativity and permiabilty, epsilon_R and mu_R respectivly
	'''
	def __init__(self, a,b=None,epsilon_R=1, mu_R=1):
		self.a = a
		if b is None: 
			self.b = a/2.
		else:
			self.b = b
			
		self.epsilon = epsilon_R * epsilon_0
		self.mu = mu_R*mu_0
		
		# link function names for convinience
		self.z0 = self.characteristic_impedance
		self.y0 = self.characteristic_admittance
		self.lambda_c = self.cutoff_wavelength
		self.f_c = self.cutoff_frequency
		self.gamma = self.kz
		
	## frequency independent functions
	def k0(self,f):
		'''
		characteristic wave number
		'''
		
		return 2*pi*f*npy.sqrt(self.epsilon * self.mu)
	
	def ky(self,n):
		'''
		eigen-value in the b direction
		'''
		return n*pi/self.b
	
	def kx(self,m):
		'''
		eigen value in teh a direction
		'''
		return m*pi/self.a
	
	
	def kc(self, m,n):
		'''
		cut-off wave number 
		'''
		return sqrt( self.kx(m)**2 + self.ky(n)**2)
	
	def cutoff_wavelength(self, m,n):
		return 2.*pi/(self.kc(m,n))
	def cutoff_frequency(self, m,n):
		return self.kc(m,n)/(2.*pi*sqrt(self.epsilon*self.mu))
	# frequency dependent functions
	def kz(self, m ,n , f):
		'''
		the propagation constant, which is:
			IMAGINARY  for propagating modes, 
			REAL for non-propagating modes
		
		takes:
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			f: frequency [Hz]
		'''
		
		k0 = array(self.k0(f),dtype=complex).reshape(-1)
		kc = array(self.kc(m,n),dtype = complex).reshape(-1)
		kz = k0.copy()
		#pdb.set_trace()
		kz =  1j*sqrt(k0**2-kc**2)*(k0>kc) + sqrt(kc**2- k0**2)*(k0<kc) \
			+ 0*(kc==k0)	
		return kz
	

	
	
	def characteristic_impedance(self, mode_type,m,n,f):
		'''
		the characteristic impedance of a given mode
		
		takes:
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			f: frequency [Hz]
			
		TODOL: write function for  'tex','tmx','tey','tmy')
					
		'''
		mode_type = mode_type.lower()
		f = array(f,dtype=complex).reshape(-1)
		omega = 2*pi *f
		impedance_dict = {\
			'tez':	1j*omega*self.mu/self.kz(m,n,f),\
			'te':	1j*omega*self.mu/self.kz(m,n,f),\
			'tmz':	self.kz(m,n,f)/(1j*omega*self.epsilon),\
			'tm':	self.kz(m,n,f)/(1j*omega*self.epsilon),\
			}
		
		return impedance_dict[mode_type]
	
	def characteristic_admittance(self, mode_type,m,n,f):
		'''
		the characteristic admittance of a given mode
		
		takes:
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			f: frequency [Hz]
			
		TODOL: write function for  'tex','tmx','tey','tmy')
					
		'''
		return 1./(self.characteristic_impedance(mode_type,m,n,f))
	
	def e_t(self,mode_type, m,n, x_points=201, y_points = 101):
		'''
		discretized transverse mode functions for the electric field. 
		
		usable for numerical evaluation of eigen-space, or field visualization
		'''
		a,b,kx,ky= self.a,self.b,self.kx(m), self.ky(n)
		#pdb.set_trace()
		x_vector= linspace(0,a,x_points)
		y_vector= linspace(0,b,y_points)
		x,y = npy.meshgrid(x_vector, y_vector)
		
		## TM
		if mode_type == 'tmz' or mode_type == 'tm':
			common_factor = 1/sqrt(m**2 *(b/a)+ n**2 *(a/b))
			e_t  = [\
				-2*m/a * common_factor * cos(kx*x)*sin(ky*y),\
				-2*n/b * common_factor * sin(kx*x)*cos(ky*y),\
				-1j/a*b * 1/common_factor * sin(kx*x)*sin(ky*y)\
				]
		## TE
		elif  mode_type == 'tez' or mode_type == 'te':
			# nuemann numbers
			ep_m = 2.-1*(m==0)
			ep_n = 2.-1*(n==0)
			common_factor = ep_m*ep_n/sqrt(m**2 *(b/a)+ n**2 *(a/b))
			
			e_t  = [\
				n/b *  common_factor * 	cos(kx*x)*sin(ky*y),\
				-m/a *  common_factor * sin(kx*x)*cos(ky*y),\
				0\
				]	
				
		return array(e_t)
