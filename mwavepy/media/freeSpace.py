
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
from numpy import real, imag
from .generic import Generic

class FreeSpace(Generic):
	'''
	Represents a plane-wave in freespace, defined by [possibly complex]
	values of relative permativity and relative permeability.
	
	The field properties of space are related to the transmission line
	model given in circuit theory by:
			
		distributed_capacitance = real(ep_0*ep_r)
		distributed_resistance = imag(ep_0*ep_r)
		distributed_inductance = real(mu_0*mu_r)
		distributed_conductance = imag(mu_0*mu_r)
	
	
	'''
	def __init__(self, frequency,  ep_r=1, mu_r=1, 	*args, **kwargs):
		'''
		takes:
			ep_r: possibly complex, relative permativity [number or array]  
			mu_r:possibly complex, relative permiability [number or array]
		returns:
			mwavepy.Media object 
		'''
		Generic.__init__(self,\
			frequency = frequency, \
			C = real(epsilon_0*ep_r),\
			G = imag(epsilon_0*ep_r),\
			I = real(mu_0*mu_r),\
			R = imag(mu_0*mu_r),\
			*args, **kwargs
			)
