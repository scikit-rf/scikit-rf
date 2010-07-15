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
from copy import copy
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
	def __init__(self, touchstone_file = None, name = None ):
		'''
		takes:
			file: if given will load information from touchstone file 
		'''
		if touchstone_file is not None:
			self.read_touchstone(touchstone_file)
		else:
			self.name = name
			#self.s = None
			self.f = None
			self.z0 = 50 
			self.f_unit = 'ghz'
	

## OPERATORS
	def __pow__(self,other):
		'''
		 implements cascading this network with another network
		'''
		if self.number_of_ports == 2  and other.number_of_ports == 2:
			result = copy(self)
			result.s = cascade(self.s,other.s)
			return result
		elif self.number_of_ports == 2 and other.number_of_ports == 1:
			result = copy(other)
			result.s = terminate(self.s, other.s)
			return result
		elif self.number_of_ports == 1 and other.number_of_ports == 2:
			result = copy(other)
			result.s = terminate(other.s,self.s)
			return result
		else:
			raise IndexError('Incorrect number of ports.')
	def __floordiv__(self,other):
		'''
		 implements de-embeding another network[s], from this network
		'''
		try: 	
			# if they passed 1 ntwks and a tuple of ntwks, 
			# then deEmbed like A.inv*C*B.inv
			b = other[1]
			c = other[0]
			result =  copy (self)
			result.s =  flip(de_embed( flip(de_embed(c.s,self.s)),b.s))
			return result
		except TypeError:
			pass
				
		if self.number_of_ports == 2  and other.number_of_ports == 2:
			result = copy(self)
			result.s = de_embed(self.s,other.s)
			return result
		elif self.number_of_ports == 1 and other.number_of_ports == 2:
			result = copy(other)
			result.s = de_embed(self.s,other.s)
			return result
		else:
			raise IndexError('Incorrect number of ports.')
	def __mul__(self,a):
		'''
		element-wise complex multiplication  of s-matrix
		'''
		raise NotImplementedError
	def __div__(self,other):
		'''
		element-wise complex division  of s-matrix
		'''
		if other.number_of_ports != self.number_of_ports:
			raise IndexError('Networks must have same number of ports.')
		else:
			result = copy(self)
			try:
				result.name = self.name+'/'+other.name
			except TypeError:
				pass
			result.s =(self.s/ other.s)
			
			return result


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
		#s.squeeze()
	
	
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
		return mf.radian_2_degree(self.s_rad_unwrap)
	
	@property
	def s_rad_unwrap(self):
		'''
		returns the unwrapped phase of the s-parameters, in radians.
		'''
		return npy.unwrap(mf.complex_2_radian(self.s),axis=0)

	@property
	def number_of_ports(self):
		'''
		the number of ports the network has.
		'''
		if len (self.s.shape) > 1:
			return npy.shape(self.s)[1]
		else:
			return 1
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
	def read_touchstone(self, filename):
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
	def plot_s_db(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of  indecie m, n in log magnitude
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		
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
		
	def plot_s_mag(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of  indecie m, n in magnitude
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you 
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_mag[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Magnitude')
		plb.legend()
				
	def plot_s_deg(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of indecie m, n in degrees
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_deg[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Phase [deg]')
		plb.legend()
		
	def plot_s_deg_unwrapped(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of  indecie m, n in unwrapped degrees
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you 
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_deg_unwrap[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Phase [deg]')
		plb.legend()
		
	def plot_s_rad(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of  indecie m, n in radians
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you 
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_rad[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Phase [deg]')
		plb.legend()	
		
	def plot_s_rad_unwrapped(self,m=0, n=0, ax = None, **kwargs):
		'''
		plots the scattering parameter of  indecie m, n in unwrapped radians
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you 
				want to update an existing plot. 
			**kwargs - passed to the matplotlib.plot command
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		# set the legend label for this trace to the networks name if it
		# exists 
		if self.name is None:
			label_string = 'S'+repr(m+1) + repr(n+1)
		else:
			 label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
		ax.plot(self.f_scaled, self.s_rad_unwrap[:,m,n],\
			label=label_string, **kwargs)
		plb.xlabel('Frequency ['+ self.f_unit +']')
		plb.ylabel('Phase [deg]')
		plb.legend()

	
	# intrpolating  frequency axis 	
	def interpolate(self):
		raise NotImplementedError
	def flip(self):
		self.s = flip(self.s)
## FUNCTIONS
# network format conversions
def s2t(s):
	'''
	converts a scattering parameters to 'wave cascading parameters'
	
	input matrix shape should be should be 2x2, or kx2x2
	
	BUG: if s -matrix has ones for reflection, thsi will produce inf's
	you cant cascade a matrix like this anyway, but we should handle it 
	better
	'''
	t = npy.copy(s)
	if len (s.shape) > 2 :
		for f in range(s.shape[0]):
			t[f,:,:] = s2t(s[f,:,:])
	elif s.shape == (2,2):
		t = npy.array([[-1*npy.linalg.det(s),	s[0,0]],\
					[-s[1,1],1]]) / s[1,0]
	else:
		raise IndexError('matrix should be 2x2, or kx2x2')
	return t        
        

def t2s(t):
	'''
	converts a 'wave cascading parameters' to scattering parameters 
	
	input matrix shape should be should be 2x2, or kx2x2
	'''
	s = npy.copy(t)
	if len (t.shape) > 2 :
		for f in range(t.shape[0]):
			s[f,:,:] = t2s(s[f,:,:])
	
	elif t.shape== (2,2):
		s = npy.array([[t[0,1],npy.linalg.det(t)],\
			[1,-t[1,0]]])/t[1,1]
	else:
		raise IndexError('matrix should be 2x2, or kx2x2')
	return s


# network operations  s-matricies
def cascade(a,b):
	'''
	cascade two 2x2 s-matricies together.
	
	a's port 2 = b's port 1
	
	note:
		BE AWARE! this relies on s2t function which has a inf problem 
		if s11 or s22 is 1. 
	'''
	c = copy(a)
	
	if len (a.shape) > 2 :
		for f in range(a.shape[0]):
			c[f,:,:] = cascade(a[f,:,:],b[f,:,:])
	
	elif a.shape == (2,2) and b.shape == (2,2):
		c = t2s(npy.dot (s2t(a) ,s2t(b)))
	else:
		raise IndexError('matricies should be 2x2, or kx2x2')
	return c
	
def terminate(a,b):
	'''
	terminate  2x2 s-matrix with a reflectino coeffiecient .
	
	a's port 2 = b's port 1
	
	note:
		BE AWARE! this relies on s2t function which has a inf problem 
		if s11 or s22 is 1 or s12 or s21 =0
	'''
	#b= b.squeeze() # force this to 1D
	c = copy(b)
	
	if len (a.shape) > 2 :
		for f in range(a.shape[0]):
			c[f] = terminate(a[f,:,:],b[f])
	
	elif a.shape == (2,2):
		# makes b into a two-port s-matrix so taht s2t will work, but the 
		# end result is still correct
		c = t2s(npy.dot (s2t(a) ,s2t(npy.array([[b,1.0],[1.0,0.0]]))))[0,0]
	else:
		raise IndexError('the \'a\' matrix should be 2x2, or kx2x2 and b should be 1D')
	return c

def de_embed(a,b):	
	'''
	de-embed a 2x2 s-matrix from another 2x2 s-matrix
	
	c = b**-1 * a
	
	note:
		BE AWARE! this relies on s2t function which has a inf problem 
		if s11 or s22 is 1. 
	'''
	c = copy(a)
	
	if len (b.shape) > 2 :
		for f in range(b.shape[0]):
			if len(a.shape) == 1:
				# 'a' is a one-port netowrk
				c[f] = de_embed(a[f],b[f,:,:])
			else:
				c[f,:,:] = de_embed(a[f,:,:],b[f,:,:])
	
	elif b.shape == (2,2):
		c = t2s(npy.dot ( npy.linalg.inv(s2t(b)), \
			s2t(npy.array([[a.squeeze(),1.],[1.,0.]]))))[0,0]
	else:
		raise IndexError('matrix should be 2x2, or kx2x2')
	return c



def flip(a):
	'''
	invert the ports of a networks s-matrix, 'flipping' it over
	
	note:
			only works for 2-ports at the moment
	'''
	c = copy(a)
	
	if len (a.shape) > 2 :
		for f in range(a.shape[0]):
			c[f,:,:] = flip(a[f,:,:])
	elif a.shape == (2,2):
		c[0,0] = a[1,1]
		c[1,1] = a[0,0]
		c[0,1] = a[1,0]
		c[1,0] = a[0,1]
	else:
		raise IndexError('matricies should be 2x2, or kx2x2')
	return c
	
def average(list_of_networks):
	raise NotImplementedError

def psd2_2_time_domain():
	raise NotImplementedError


