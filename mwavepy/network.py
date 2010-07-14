'''
#       network.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen 
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

import  mwavepy1.mathFunctions as mf
import touchstone

import numpy as npy
import pylab as plb 

class Network(object):
## CONSTANTS
	f_unit_dict = {\
		'hz':'Hz',\
		'mhz':'MHz',\
		'ghz':'GHz'\
		}
	f_multiplier_dict={
		'hz':1,\
		'mhz':1e6,\
		'ghz':1e9\
		}

## CONSTRUCTOR
	def __init__(self, file = None):
		'''
		takes:
			file: if given will load information from touchstone file 
		'''
		if file is not None:
			self.load_touchstone(file)
		else:
			self.name = None
			self.s = None
			self.f = None
			self.z0 = 50 
			self.f_unit = 'hz'
	

## PRIMARY PROPERTIES
	# s-parameter matrix
	@property
	def s(self):
		'''
		The scattering parameter matrix.
		
		s-matrix has shape fxmxn, 
		where; 
			f is frequency axis and,
			m and n are port indicies
		'''
		return self._s
	
	@s.setter
	def s(self, s):
		'''
		the input s-matrix should be of shape fxmxn, 
		where f is frequency axis and m and n are port indicies
		'''
		self._s = s
	
	
	# frequency information
	@property
	def f(self):
		''' the frequency vector for the network, in Hz. '''
		return self._f
		
	@f.setter
	def f(self,f):
		self._f = f
	
	# characteristic impedance
	@property
	def z0(self):
		''' the characteristic impedance of the network.'''
		return self._z0
	
	@z0.setter
	def z0(self, z0):
		self._z0 = z0
	
## SECONDARY PROPERTIES

	# s-parameters convinience properties	
	@property
	def s_manitude(self):
		'''
		returns the magnitude of the s-parameters.
		'''
		return mf.complex_2_magnitude(self.s)
	
	@property
	def s_db(self):
		'''
		returns the magnitude of the s-parameters, in dB
		
		note:
			dB is calculated by 
				20*log10(|s|)
		'''
		return mf.complex_2_db(self.s)
		
	@property
	def s_deg(self):
		'''
		returns the phase of the s-parameters, in radians
		'''
		return mf.complex_2_degree(self.s)
		
	@property
	def s_rad(self):
		'''
		returns the phase of the s-parameters, in radians.
		'''
		return mf.complex_2_radian(self.s)
	
	@property
	def s_deg_unwrap(self):
		'''
		returns the unwrapped phase of the s-paramerts, in degrees
		'''
		return mf.rad_2_degree(self.s_rad_unwrap)
	
	@property
	def s_rad_unwrap(self):
		'''
		returns the unwrapped phase of the s-parameters, in radians.
		'''
		return npy.unwrap(mf.complex_2_radian(self.s))

	@property
	def number_of_ports(self):
		'''
		the number of ports the network has.
		'''
		return npy.shape(self.s)[1]
	# frequency formating related properties
	
	@property
	def f_unit(self):
		'''
		The unit to format the frequency axis in. see formatedAxis
		'''
		return self.f_unit_dict[self._f_unit]
	@f_unit.setter
	def f_unit(self,f_unit):
		self._f_unit = f_unit.lower()
	
	@property
	def f_multiplier(self):
		'''
		multiplier for formating axis
		'''
		return self.f_multiplier_dict[self._f_unit]
	@property
	def f_scaled(self):
		'''
		The unit to format the frequency axis in. see formatedAxis
		'''
		return self.f/self.f_multiplier
## CLASS METHODS
	# touchstone file IO
	def load_touchstone(self, filename):
		'''
		loads  values from a touchstone file. 
		
		takes:
			filename - touchstone file name, string. 
		
		note: 
			ONLY 'S' FORMAT SUPORTED AT THE MOMENT 
			all work is tone in the touchstone class. 
		'''
		touchstoneFile = touchstone.touchstone(filename)
		
		if touchstoneFile.get_format().split()[1] != 's':
			raise NotImplementedError('only s-parameters supported for now.')
		
		self.z0 = float(touchstoneFile.resistance)
		self.f, self.s = touchstoneFile.get_sparameter_arrays() # note freq in Hz
		self.f_unit = touchstoneFile.frequency_unit # for formatting plots
		self.name = touchstoneFile.filename.split('/')[-1].split('.')[-2]

	def write_touchstone(self, filename):
		'''
		write a touchstone file representing this network.  the only 
		format supported at the moment is :
			HZ S RI 
		
		takes: 
			filename - filename , duh
			
		
		note:
			in the future could make possible use of the touchtone 
			class, but at the moment this would not provide any benefit 
			as it has not set_ functions. 
		'''

		
		outputFile = open(filename,"w")
		
		# write header file. 
		# the '#'  line is NOT a comment it is essential and it must be 
		#exactly this format, to work
		# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
		outputFile.write('# ' + self.f_unit + ' S RI R ' + str(self.z0) +" \n")
		
		#write comment line for users (optional)
		outputFile.write ("!freq\t")
		for n in range(self.number_of_ports):
			for m in range(self.number_of_ports):
				outputFile.write("Re" +'S'+`m+1`+ `n+1`+  "\tIm"+\
				'S'+`m+1`+ `n+1`+'\t')
		outputFile.write('\n')		
		
		# write out data, note: this could be done with matrix 
		#manipulations, but its more readable to me this way
		for f in range(len(self.f)):
			outputFile.write(str(self.f_scaled[f])+'\t')
			
			for n in range(self.number_of_ports):
				for m in range(self.number_of_ports):
					outputFile.write( str(npy.real(self.s[f,m,n])) + '\t'\
					 + str(npy.imag(self.s[f,m,n])) +'\t')
			
			outputFile.write('\n')
		
		outputFile.close()
	# ploting 
	def plot_db(self,m=0, n=0, ax = None, **kwargs):
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_db[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Magnitude [dB]')
		plb.legend()
## FUNCTIONS
def cascade():
	raise NotImplementedError

def de_embed():
	raise NotImplementedError

def divide():
	raise NotImplementedError
