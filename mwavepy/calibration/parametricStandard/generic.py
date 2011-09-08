#       generic.py
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
#       MA 02110-1301, USA.from media import Media
'''
Provides  generic parametric standards which dont depend on any 
specific properties of the a given media

'''

from parametricStandard import ParametricStandard



## General tline mediums
class Parameterless(ParametricStandard):
	'''
		A parameterless standard. 

		note:
		this is needed so that the calibration algorithm doesnt have to
		handle more than one class type for  standards
		'''
	def __init__(self, ideal_network):
		'''
		takes:
			ideal_network: a Network instance of the standard
		'''
		ParametricStandard.__init__(self, \
			function  = lambda: ideal_network)
		
class Line_UnknownLength(ParametricStandard):
	'''
	A matched delay line of unknown length
	
	initial guess for length should be given to constructor
	'''
	def __init__(self, media, d,**kwargs):
		'''
		takes:
			media: a Mwedia type
			d: initial guess for line length [m]
			**kwargs: passed to self.function
		'''
		ParametricStandard.__init__(self, \
			function = media.line,\
			parameters = {'d':d},\
			**kwargs\
			)
class DelayedShort_UnknownLength(ParametricStandard):
	'''
	A delay short of unknown length
	
	initial guess for length should be given to constructor
	'''
	def __init__(self, media,d,**kwargs):
		'''
		takes:
			media: a Media type
			d: initial guess for delay short physical length [m]
			**kwargs: passed to self.function
		'''
		ParametricStandard.__init__(self, \
			function = media.delay_short,\
			parameters = {'d':d},\
			**kwargs\
			)

class DelayedTermination_UnknownLength(ParametricStandard):
	'''
	A  Delayed Termination of unknown length, but known termination
	'''
	def __init__(self, media,d,Gamma0,**kwargs):
		'''
		takes:
			media: a Media type, with a RectangularWaveguide object
				for its tline property.
			d: distance to termination
			Gamma0: reflection coefficient off termination at termination
			**kwargs: passed to self.function
		'''
		kwargs.update({'Gamma0':Gamma0,})
		
		ParametricStandard.__init__(self, \
			function = media.delay_load,\
			parameters = {'d':d},\
			**kwargs\
			)

class DelayedTermination_UnknownTermination(ParametricStandard):
	'''
	A  Delayed Termination of unknown length or termination
	'''
	def __init__(self, media,d,Gamma0,**kwargs):
		'''
		takes:
			media: a Media type, with a RectangularWaveguide object
				for its tline property.
			d: distance to termination
			Gamma0: reflection coefficient off termination at termination
			**kwargs: passed to self.function
		'''
		kwargs.update({'d':d})
		ParametricStandard.__init__(self, \
			function = media.delay_load,\
			parameters = {'Gamma0':Gamma0},\
			**kwargs\
			)
class DelayedTermination_UnknownLength_UnknownTermination(ParametricStandard):
	'''
	A  Delayed Termination of unknown length or termination
	'''
	def __init__(self, media,d,Gamma0,**kwargs):
		'''
		takes:
			media: a Media type, with a RectangularWaveguide object
				for its tline property.
			d: distance to termination
			Gamma0: reflection coefficient off termination at termination
			**kwargs: passed to self.function
		'''
		
		ParametricStandard.__init__(self, \
			function = media.delay_load,\
			parameters = {'d':d,'Gamma0':Gamma0},\
			**kwargs\
			)
class DelayShort_Mulipath(ParametricStandard):
	'''
	A delay short of unknown length
	
	initial guess for length should be given to constructor
	'''
	def __init__(self, media,d1,d2,d1_to_d2_power, **kwargs):
		'''
		takes:
			media: a Media type
			d: initial guess for delay short physical length [m]
			**kwargs: passed to self.function
		'''
		def multipath(d1,d2,d1_to_d2_power):
			d2_power = 1./(d1_to_d2_power +1)
			d1_power = 1-d2_power
			ds1 = media.delay_short(d1)
			ds2 = media.delay_short(d2)
			ds1.s = ds1.s * d1_power
			ds2.s = ds2.s * d2_power
			return ds1+ds2
		kwargs.update({'d1':d1})
		ParametricStandard.__init__(self, \
			function = multipath,\
			parameters = {\
				'd2':d2,\
				'd1_to_d2_power':d1_to_d2_power\
				},\
			**kwargs\
			)
class DelayLoad_Mulipath(ParametricStandard):
	'''
	A delay short of unknown length
	
	initial guess for length should be given to constructor
	'''
	def __init__(self, media,d1,Gamma0, d2,d1_to_d2_power, **kwargs):
		'''
		takes:
			media: a Media type
			d: initial guess for delay short physical length [m]
			**kwargs: passed to self.function
		'''
		def multipath(d1,d2,Gamma0, d1_to_d2_power):
			d2_power = 1./(d1_to_d2_power +1)
			d1_power = 1-d2_power
			ds1 = media.delay_load(d1,Gamma0)
			ds2 = media.delay_short(d2)
			ds1.s = ds1.s * d1_power
			ds2.s = ds2.s * d2_power
			return ds1+ds2
		kwargs.update({'d1':d1,'Gamma0':Gamma0})
		ParametricStandard.__init__(self, \
			function = multipath,\
			parameters = {\
				'd2':d2,\
				'd1_to_d2_power':d1_to_d2_power\
				},\
			**kwargs\
			)
				
