
#       freeSpace.py
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
contains free space class
'''
from scipy.constants import  epsilon_0, mu_0
from numpy import real, imag,pi,sqrt,log
from .media import Media	
class CPW(Media):
	'''
	Coplanar waveguide class
	'''
	def __init__(self, frequency, w , s, ep_r,*args, **kwargs):
		'''
		takes:

		returns:
			mwavepy.Media object 
		'''
		self.frequency, self.w, self.s, self.ep_r = frequency, w, s,ep_r
		
		Media.__init__(self,\
			frequency = frequency,\
			propagation_constant = self.gamma, \
			characteristic_impedance = self.Z0,\
			*args, **kwargs)
		
	def __str__(self):
		f=self.frequency
		output =  \
			'Coplanar Waveguide Media.  %i-%i %s.  %i points'%\
			(f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
			'\n W= %.2em, S= %.2em'% \
			(self.w,self.s)
		return output
		
	def __repr__(self):
		return self.__str__()
		
	@property
	def ep_re(self):
		return (self.ep_r+1)/2.
	
	@property 
	def K_ratio(self):
		
		K = self.w/(self.w +2*self.s)
		
		if (0 <= K <= 1/sqrt(2)):
			return pi/(log(2*(1+sqrt(K))/(1-sqrt(K)) ))
		elif (1/sqrt(2) < K <= 1):
			return (log(2*(1+sqrt(K))/(1-sqrt(K)) ))/pi
			
	@property
	def Z0(self):
		return 30.*pi / sqrt(self.ep_re) * self.K_ratio
	
	@property 
	def gamma(self):
		return 1j*2*pi*self.frequency.f*sqrt(self.ep_re*epsilon_0*mu_0)
