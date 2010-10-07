
#       rectangularWaveguideTE10.py
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
a dominant-mode rectangular waveguide 
'''

from rectangularWaveguide import RectangularWaveguide
from functions import electrical_length

class RectangularWaveguideTE10(RectangularWaveguide):
	def __init__(self, a,b=None,epsilon_R=1, mu_R=1):
		RectangularWaveguide.__init__(self, a,b=None,epsilon_R=1, mu_R=1)
		
	
	def cutoff_frequency(self):
		return RectangularWaveguide.cutoff_frequency(self,m=1,n=0)
	
	
	def kc(self, *args):
		'''
		cut-off wave number 
		'''
		return RectangularWaveguide.kc(self,m=1,n=0)
	
	
	def cutoff_wavelength(self):
		return RectangularWaveguide.cutoff_wavelength(self, m=1,n=0)
	
	def kz(self, f, *args):
		'''
		the propagation constant, which is:
			REAL  for propagating modes, 
			IMAGINARY for non-propagating modes
		
		takes:
			f: frequency [Hz]
		'''
		return RectangularWaveguide.kz(self,m=1,n=0,f=f)
	def characteristic_impedance(self,f,*args):
		'''
		the characteristic impedance of a given mode
		
		takes:
			f: frequency [Hz]		
		'''
		return RectangularWaveguide.characteristic_impedance(self, 'tez', m=1,n=0,f=f)
	def characteristic_admittance(self,f,*args):
		'''
		the characteristic admittance of a given mode
		
		takes:
			f: frequency [Hz]	
		'''
		return 1./(self.characteristic_impedance(f))
	
	def electrical_length(self, f,d,deg=False):
		'''
		calculate electrical length of a section fo waveguide.
		
		takes:
			f: frequency at which to calculate [Hz]
			d: length fo delay [m]
			deg: True/False
		'''
		return electrical_length( \
			gamma = self.propagation_constant,f=f,d=d,deg=deg)
