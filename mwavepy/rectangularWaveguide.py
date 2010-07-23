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

# would there be any benefit to structuring modes as objects?
#def RectangularWaveguideMode(object,m,n):
	#self.m = m
	#self.n = n
	
	#@property 
	#def cut_off_frequency(self):
		#return 1./(2*sqrt(self.epsilon*self.mu)) * \
			#sqrt( (self.m/a)**2 + (self.n/2)**2)
		
class RectangularWaveguide(object):
	def __init__(self, a,b=None,epsilon_R=1, mu_R=1):
		self.a = a
		if b is None: 
			self.b = a/2.
		else:
			self.b = b
		self.epsilon = epsilon_R * epsilon_0
		self.mu = mu_R*mu_0
	
	def k0(self,omega):
		'''
		characteristic wave number
		'''
		
		return omega*npy.sqrt(self.epsilon * self.mu)
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
	
	def propagation_constant(self, m ,n , f):
		'''
		the propagation constant, which is real for propagating modes, 
		imaginary for non-propagating modes
		'''k0 = self.k0(f)
		kc = self.kc(f,m,n)
		if k0 > kc:
			return 1j*sqrt(k0**2-kc**2)
		elif k0< kc: 
			return sqrt(kc**2- k0**2)
		else:
			return 0
	
	
	

	def e_t(self,xy):
		
