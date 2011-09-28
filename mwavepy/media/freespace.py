
#       freespace.py
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
A Plane-wave in Freespace.
'''
from scipy.constants import  epsilon_0, mu_0
from numpy import real, imag
from .distributedCircuit import DistributedCircuit

class Freespace(DistributedCircuit):
	'''
	Represents a plane-wave in a homogeneous freespace, defined by
	[possibly complex] values of relative permativity and 
	relative permeability.
	
	The field properties of space are related to a disctributed 
	circuit transmission line model given in circuit theory by:
			
		distributed_capacitance = real(ep_0*ep_r)
		distributed_resistance = imag(ep_0*ep_r)
		distributed_inductance = real(mu_0*mu_r)
		distributed_conductance = imag(mu_0*mu_r)
	
	note: this class's inheritence is;
		Media->DistributedCircuit->FreeSpace
	
	'''
	def __init__(self, frequency,  ep_r=1, mu_r=1, 	*args, **kwargs):
		'''
		takes:
			ep_r: possibly complex, relative permativity [number or array]  
			mu_r:possibly complex, relative permiability [number or array]
		
		returns:
			mwavepy.Media object 
		'''
		DistributedCircuit.__init__(self,\
			frequency = frequency, \
			C = real(epsilon_0*ep_r),\
			G = imag(epsilon_0*ep_r),\
			I = real(mu_0*mu_r),\
			R = imag(mu_0*mu_r),\
			*args, **kwargs
			)
	
	def __str__(self):
		f=self.frequency
		output =  \
			'Freespace  Media.  %i-%i %s.  %i points'%\
			(f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)
		return output
		
	def __repr__(self):
		return self.__str__()
	
