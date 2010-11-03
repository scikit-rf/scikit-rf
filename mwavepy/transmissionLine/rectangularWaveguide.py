
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
a multimoded rectangular waveguide
'''
from scipy.constants import  epsilon_0, mu_0, c,pi, mil,pi
import numpy as npy
from numpy import sqrt,array, cos, sin, linspace

from functions import electrical_length
from functions import Gamma0_2_zin
from .. import mathFunctions as mf

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
		self.zin = self.input_impedance
		self.y0 = self.characteristic_admittance
		self.yin = self.input_admittance
		self.f_c = self.cutoff_frequency
		self.gamma = self.kz
		##TODO: UNDO THIS HACK HACK
		# hack so that WorkingBand.delay, can calculate electrical length
		self.propagation_constant = lambda f: self.kz(1,0,f)

		self.lambda_c = self.cutoff_wavelength
		self.lambda_g = self.guide_wavelength
		self.lambda_0 = self.intrinsic_wavelength
		self.vp_0 = self.intrinsic_phase_velocity
		self.vp_g = self.guide_phase_velocity
		self.eta_0 = self.intrinsic_impedance
	## properties
	@property
	def intrinsic_phase_velocity(self):
		'''
		the intrinsic phase velocity of the waveguide. depends only on
		material which fills the waveguide
		'''
		return 1./sqrt(self.epsilon * self.mu)
		
	@property
	def intrinsic_impedance(self):
		'''
		the intrinsic impedance of the filling material
		'''
		return sqrt(self.mu/self.epsilon)
	##  frequency independent functions
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
		try:
			# if they gave us vectors, then return appropriate matrix
			kc = npy.zeros((len(m),len(n)),dtype=complex)
			for m_idx in range(len(m)):
				for n_idx in range(len(n)):
					kc[m_idx, n_idx] = self.kc(m[m_idx],n[n_idx])
			return kc
			
		except(TypeError):
			return sqrt( self.kx(m)**2 + self.ky(n)**2)
	
	def cutoff_wavelength(self, m,n):
		'''
		the wavelength at which mode (m,n) is cut-off
		'''
		return 2.*pi/(self.kc(m,n))
		
	def cutoff_frequency(self, m,n):
		'''
		the cutoff freqency of mode (m,n)
		'''
		return self.kc(m,n)/(2.*pi*sqrt(self.epsilon*self.mu))
	# frequency dependent functions
	
	def intrinsic_wavelength(self,f):
		'''
		the intrinisic wavelength of the waveguide at frequency f.
		(different from the guide_wavelength )
		'''	
		return self.intrinsic_phase_velocity/f

	def k0(self,f):
		'''
		characteristic wave number
		'''
		return 2*pi*f*npy.sqrt(self.epsilon * self.mu)
	def kz(self, m ,n , f):
		'''
		the propagation constant, which is:
			REAL  for propagating modes, 
			IMAGINARY for non-propagating modes
		
		takes:
			m: mode index in the 'a' direction
			n: mode index in the 'b' direction 
			f: frequency [Hz]

		output:
			kz:a complex number, and possibly a fxmxn array, depending
				on input 
		
		NOTE:
			
			a note about using arrays for input values:
			either all inputs, m,n,f can be arrays, 
			or just f
			or just m and n
			but not m or n
		'''
		## TODO: make this readable or comment more. 
		k0 = array(self.k0(f),dtype=complex).reshape(-1)
		kc = array(self.kc(m,n),dtype = complex)#.reshape(-1)

		# handle vector values for m, n , and f
		try:
			kz = npy.zeros((len(f),len(m),len(n)),dtype=complex)
			for f_idx in range(len(f)):	
				kz[f_idx] =  -sqrt(k0[f_idx]**2-kc**2) * (k0[f_idx] > kc) + \
					1j*sqrt(kc**2- k0[f_idx]**2) * (k0[f_idx]<kc) + 0*(kc==k0[f_idx])
		
		except(TypeError):
			# we have scalars for m, or n
			kz = npy.zeros(shape=k0.shape,dtype=complex)
			kz =  -sqrt(k0**2-kc**2)*(k0>kc) +1j*sqrt(kc**2- k0**2)*(k0<kc) \
				+ 0*(kc==k0)	
		return kz
		
	def guide_wavelength(self,m,n,f):
		'''
		the guide wavelength.

		'the distance that the field travels before the phase increases
		by 2*pi'. 
		'''
		fc= self.cutoff_frequency(m,n)
		lam = self.intrinsic_wavelength(f)
		return lam/sqrt(1-(fc/f)**2)

	def guide_phase_velocity(self,m,n,f):
		'''
		the guide phase velocity at which a mode propagates.  
		'''
		self.intrinsic_phase_velocity/sqrt(1-(self.cutoff_frequency(m,n)/f)**2)

	
	
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

		try:
			# needed in case m, n , and f are arrays
			omega= omega.reshape(-1,1,1).repeat(len(m),axis=1).repeat(len(n),axis=2)
		except(TypeError):
			pass
		#import pdb
		#pdb.set_trace()
		impedance_dict = {\
			'tez':	omega*self.mu/(-1*self.kz(m,n,f)),\
			'te':	omega*self.mu/(-1*self.kz(m,n,f)),\
			'tmz':	-1*self.kz(m,n,f)/(omega*self.epsilon),\
			'tm':	-1*self.kz(m,n,f)/(omega*self.epsilon),\
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

	def input_impedance(self, d, Gamma0, mode_type, m,n,f):
		'''
		calculates the input impedance for a single mode, of reflection
		coefficient Gamma0, at a specified disatnace d.

		takes:
			d: distance from load ( in meters)
			Gamma0: reflection coefficient of termination (@z=0)
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			f: frequency [Hz]

		returns:
			zin: input impedance (in ohms)

		note:
			if you want to specify load in terms of its impedance, you
			can use the function:
				transmissionLine.functions.zl_2_Gamma0().
			
			see transmissionLine.functions for more info.
		'''
		z0 = self.z0( mode_type, m,n,f)
		theta = self.electrical_length(m,n,f,d)

	
		# needed in case m, n , and f are arrays
		#Gamma0= array(Gamma0).reshape(-1,1,1).repeat(len(m),axis=1).repeat(len(n),axis=2)
		#theta = theta.reshape(-1,1,1).repeat(len(m),axis=1).repeat(len(n),axis=2)

		zin = Gamma0_2_zin(z0, Gamma0, theta).reshape(z0.shape)
		
		return zin
		
	def input_admittance(self, d, Gamma0, mode_type, m,n,f):
		'''
		calculates the input admitance for a single mode, of reflection
		coefficient Gamma0, at a specified disatnace d.

		takes:
			d: distance from load ( in meters)
			Gamma0: reflection coefficient of termination (@z=0)
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			f: frequency [Hz]

		returns:
			zin: input impedance (in 1/ohms)

		note:
			if you want to specify load in terms of its impedance, you
			can use the function:
				transmissionLine.functions.zl_2_Gamma0().
			
			see transmissionLine.functions for more info.
		'''
		return 1./self.input_impedance(d, Gamma0, mode_type, m,n,f)
	def electrical_length(self,m,n,f,d,deg=False):
		return electrical_length( \
			gamma = lambda x:self.kz(m=m,n=n, f=x),\
			f=f,d=d,deg=deg)

	## Analytical EigenFunctions
	def e_t(self,mode_type, m,n, x_points=201, y_points = 101):
		'''
		discretized transverse mode functions for the electric field. 
		
		usable for numerical evaluation of eigen-space, or field 
		visualization.
		
		takes:
		
		returns array of shape 	[3,y_points, x_points]. the three 
		components are: in order
			e_t_x: component of field in 'a' direction
			e_t_y: component of field in 'b' direction
			e_t_z:component of field in  longitudinal direction
			
			NOTE: all vectors returns are in (row, col) format which
			equates to (y,x). 
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
			ep_m = 2. - 1.*(m==0)
			ep_n = 2. - 1.*(n==0)
			common_factor = ep_m*ep_n/sqrt(m**2 *(b/a)+ n**2 *(a/b))
			
			e_t  = [\
				n/b *  common_factor * 	cos(kx*x)*sin(ky*y),\
				-m/a *  common_factor * sin(kx*x)*cos(ky*y),\
				npy.zeros((y_points, x_points))\
				]	
			#pdb.set_trace()
		return array(e_t)


	def eigenfunction_normalization(self,mode_type,m,n):
		'''
		returns the normalization for a given transverse eigen function,
		 so that the set is normalized to 1.
	
		takes:
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			
		note:
			t-to-z mode normalization can be found in marcuvitz
		'''
		a,b= self.a,self.b
		
		if m==0  and n==0:
			return 0
			
		if mode_type == 'te' or mode_type == 'tez'  :
			ep_m = mf.neuman(m)
			ep_n = mf.neuman(n)
			scaling =  -sqrt(ep_m*ep_n)*m/a * 1/sqrt(m**2 *(b/a)+ n**2 *(a/b)) 
		
		elif mode_type == 'tm' or mode_type == 'tmz' :
			if m==0 or n == 0:
				return 0 
			scaling =  -2.*n/b* 1/sqrt(m**2 *(b/a)+ n**2 *(a/b))
		
		else:
			raise(ValueError)
		
		return scaling
	
	def eigenfunction_normalization2(self,field_type,mode_type,m,n):
		'''
		returns the normalization factor for a given transverse eigenfunction,
		 so that the set is normalized to 1.
	
		takes:
			field_type: 'e' or 'h' field
			mode_type:	describes the mode type (TE,TM) and direction, 
				possible values are:
					'tez','tmz'
			m: mode index in the 'a' direction 
			n: mode index in the 'b' direction 
			
		note:
			t-to-z mode normalization can be found in marcuvitz
		'''
		
		a,b,kx,ky,kc, epsilon,mu = self.a,self.b,self.kx(m),\
			self.ky(n),self.kc(m,n),self.epsilon,self.mu
		
		if  (m==0 and n==0):
			return npy.zeros(3)
		common_factor = sqrt(mf.neuman(m)*mf.neuman(n)/ (a*b* kc**2))
		#pdb.set_trace()
		e_field_dict = {\
		'te': (m != 0  or n != 0) * common_factor * array([[ky],[-kx],[0]]),\
		'tm': (m!=0  and n!=0) * common_factor * array([[kx],[ky],[0]])\
		}
		
		h_field_dict = {\
		'te': (m!=0  or n!=0) * common_factor *	array([[-kx],[-ky],[0]]),\
		'tm': (m!=0  and n!=0) *common_factor *	array([[ky],[-kx],[0]])\
		}
		
		eigenfunction_normalization_dict = {
			'e': e_field_dict,\
			'h': h_field_dict}
			
		return eigenfunction_normalization_dict[field_type][mode_type]
	
