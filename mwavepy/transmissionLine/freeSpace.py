
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
import numpy as npy
from numpy import pi, sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
	 interp, linspace, shape,zeros, reshape
from scipy.constants import  epsilon_0, mu_0, c,pi, mil,pi
from genericTEM import GenericTEM

class FreeSpace(GenericTEM):
	'''
	Represents a plane-wave in freespace, defined by [possibly complex]
	values of relative permativity and relative permeability.
	
	The field properties of space are related to the transmission line
	model given in circuit theory by:
			
		distributed_capacitance = real(epsilon_0*relative_permativity)
		distributed_resistance = imag(epsilon_0*relative_permativity)
		distributed_inductance = real(mu_0*relative_permeability)
		distributed_conductance = imag(mu_0*relative_permeability)
	
	
	'''
	def __init__(self, relative_permativity=1, relative_permeability=1):
		GenericTEM.__init__(self,\
			distributed_capacitance = real(epsilon_0*relative_permativity),\
			distributed_resistance = imag(epsilon_0*relative_permativity),\
			distributed_inductance = real(mu_0*relative_permeability),\
			distributed_conductance = imag(mu_0*relative_permeability),\
			)
