
'''
#       mobius.py
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
from numpy import array
class mobius(object):
	def __init__(self, a,b,c,d):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		
		self.h = array([[a,b],[c,d]])
	
	def transform(self,w):
		return (self.a*w +self.b)/(self.c*w+self.d)
	def itransform(self,z):
		return (self.d*z-self.b)/(-self.c*z+self.a)




def mobiusTransformation(m, a):
	'''
	returns the unique maping function between m and a planes which are
	related through	the mobius transform.
	
	takes:
		m: list containing the triplet of points in m plane m0,m1,m2
		a: list containing the triplet of points in a plane a0,a1,a2
	
	returns:
		a (m) : function of variable in m plane, which returns a value
			in the a-plane
	'''
	m0,m1,m2 = m
	a0,a1,a2 = a
	return lambda m: (a0*a1*m*m0 + a0*a1*m1*m2 + a0*a2*m*m2 + a0*a2*m0*m1 +\
	 a1*a2*m*m1 + a1*a2*m0*m2 - a0*a1*m*m1 - a0*a1*m0*m2 - a0*a2*m*m0 -\
	 a0*a2*m1*m2 - a1*a2*m*m2 - a1*a2*m0*m1)/(a0*m*m2 + a0*m0*m1 + a1*m*m0\
	  + a1*m1*m2 + a2*m*m1 + a2*m0*m2 - a0*m*m1 - a0*m0*m2 - a1*m*m2 - \
	  a1*m0*m1 - a2*m*m0 - a2*m1*m2)
	
