
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
contains CPW class
'''
from scipy.constants import  epsilon_0, mu_0
from scipy.special import ellipk
from numpy import real, imag,pi,sqrt,log,zeros
from .media import Media	
class CPW(Media):
	'''
	Coplanar waveguide class
	
	This class was made based off the the documentation from the
	qucs project ( qucs.sourceforge.net/ ).
	
	'''
	def __init__(self, frequency, w , s, ep_r, t=None, r_s=None, \
		*args, **kwargs):
		'''
		takes:
			w: width of center conductor, in m. 
			s: width of gap, in m. 
			ep_r: relative permativity of substrate
			r_s: surface resistivity of conductor (None)
			t: conductor thickness, in m.
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
		'''
		intermediary parameter. see qucs docs on cpw lines.
		'''
		return (self.ep_r+1)/2.
	@property
	def k1(self):
		'''
		intermediary parameter. see qucs docs on cpw lines.
		'''
		return self.w/(self.w +2*self.s)
	@property 
	def K_ratio(self):
		'''
		intermediary parameter. see qucs docs on cpw lines.
		'''
		k1 = self.k1
		
		if (0 <= k1 <= 1/sqrt(2)):
			return pi/(log(2*(1+sqrt(k1))/(1-sqrt(k1)) ))
		elif (1/sqrt(2) < k1 <= 1):
			return (log(2*(1+sqrt(k1))/(1-sqrt(k1)) ))/pi
	
	@property
	def alpha_conductor(self):
		'''
		losses due to conductor resistivity
		'''
		t, k1, ep_re, r_s = self.t, self.k1,self.ep_re,self.r_s
		a = self.w/2.
		b = self.s+self.w/2.
		K = ellipk	# complete elliptical integral of first kind
		K_p = lambda x: ellipk(sqrt(1-x**2)) # ellipk's compliment
		
		return ((r_s * sqrt(ep_re)/(480*pi*K(k1)*K_p(k1)*(1-k_1**2) ))*\
			(1./a * (pi+log((8*pi*a*(1-k1))/(t*(1+k1)))) +\
			 1./b * (pi+log((8*pi*b*(1-k1))/(t*(1+k1))))))
			


	@property
	def Z0(self):
		'''
		characterisitc impedance
		'''
		return 30.*pi / sqrt(self.ep_re) * self.K_ratio
	
	@property 
	def gamma(self):
		'''
		propagation constant
		'''
		beta = 1j*2*pi*self.frequency.f*sqrt(self.ep_re*epsilon_0*mu_0)
		alpha = zeros(len(beta))
		try:
			alpha = self.alpha_conductor 
		except(AttributeError):
			# they didnt set surface resistivity and conductor thickness
			pass
		
		return beta+alpha
