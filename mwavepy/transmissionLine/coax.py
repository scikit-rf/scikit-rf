
#       coax.py
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
contains coaxial cable transmission line class
'''
from genericTEM import GenericTEM
class Coax(GenericTEM):
	def __init__(self, innerRadius, outerRadius, surfaceResistance=0, relative_permativity=1, relative_permeability=1):
		raise(NotImplementedError)
		# changing variables just for readablility
		a = innerRadius
		b = outerRadius
		eR = relative_permativity
		uR = relative_permeability
		Rs = surfaceResistance
		
		GenericTEM.__init__(self,\
			distributed_capacitance = 2*pi*real(epsilon_0*eR)/log(b/a),\
			distributed_resistance = Rs/(2*pi) * (1/a + 1/b),\
			distributed_inductance = muR*mu_0/(2*pi) * log(b/a),\
			distributed_conductance = 2*pi*omega*imag(epsilon_0*eR)/log(b/a),\
			)
