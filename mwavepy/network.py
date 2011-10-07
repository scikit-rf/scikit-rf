
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
Provides the Network class and related functions. 

'''
from copy import deepcopy as copy
from copy import deepcopy
import os

import numpy as npy
import pylab as plb 
from scipy import stats		# for Network.add_noise_* 
from scipy.interpolate import interp1d # for Network.interpolate()

import  mathFunctions as mf
import touchstone
from frequency import Frequency
from plotting import smith
from tlineFunctions import zl_2_Gamma0

class Network(object):
	'''
	Represents a n-port microwave network.

	the most fundemental properties are:
		s: scattering matrix. a kxnxn complex matrix where 'n' is number
			of ports of network.
		z0: characteristic impedance
		f: frequency vector in Hz. see also frequency, which is a
			Frequency object (see help on this class for more info)
		
	
	The following operators are defined as follows:
		'+' : element-wise addition of the s-matrix
		'-' : element-wise subtraction of the s-matrix
		'*' : element-wise multiplication of the s-matrix
		'/' : element-wise division of the s-matrix
		'**': cascading of 2-port networks
		'//': de-embdeding of one network from the other.

	various other network properties are accesable as well as plotting
	routines are also defined for convenience, 
	
	most properties are derived from the specifications given for
	touchstone files. 
	'''
	global ALMOST_ZER0
	ALMOST_ZER0=1e-6 # used for testing s-parameter equivalencee
## CONSTRUCTOR
	def __init__(self, touchstone_file = None, name = None ):
		'''
		constructor.
		
		takes:
			file: if given will load information from touchstone file
			name: name of this network. 
		'''
		# although meaningless untill set with real values, this
		# needs this to exist for dependent properties
		#self.frequency = Frequency(0,0,0)
		
		if touchstone_file is not None:
			self.read_touchstone(touchstone_file)
			if name is not None:
				self.name = name
		
		else:
			self.name = name
			#self.s = None
			#self.z0 = 50 
		
		#convenience 
		#self.nports = self.number_of_ports
		

## OPERATORS
	def __pow__(self,other):
		'''
		 implements cascading this network with another network
		'''
		
		# this is the old algorithm
		'''if self.number_of_ports == 2  and other.number_of_ports == 2:
			result = copy(self)
			result.s = cascade(self.s,other.s)
			return result
		elif self.number_of_ports == 2 and other.number_of_ports == 1:
			result = copy(other)
			result.s = cascade(self.s, other.s)
			return result
		elif self.number_of_ports == 1 and other.number_of_ports == 2:
			result = copy(other)
			# this flip is to make the termination what the syntax
			# looks like  
			result.s = cascade(flip(other.s),self.s)
			return result
		else:
			raise IndexError('Incorrect number of ports.')
			'''
		return connect(self,1,other,0)

	def __floordiv__(self,other):
		'''
		 implements de-embeding another network[s], from this network

		see de_embed
		'''
		try: 	
			# if they passed 1 ntwks and a tuple of ntwks, 
			# then deEmbed like A.inv*C*B.inv
			b = other[0]
			c = other[1]
			result =  copy (self)
			result.s =  (b.inv**self**c.inv).s
			#flip(de_embed( flip(de_embed(c.s,self.s)),b.s))
			return result
		except TypeError:
			pass
				
		if other.number_of_ports == 2:
			result = copy(self)
			result.s = (other.inv**self).s
			#de_embed(self.s,other.s)
			return result
		else:
			raise IndexError('Incorrect number of ports.')

	def __mul__(self,a):
		'''
		element-wise complex multiplication  of s-matrix
		'''
		result = copy(self)
		result.s = result.s * a.s
		return result

	def __add__(self,other):
		'''
		element-wise addition of s-matrix
		'''
		result = copy(self)
		result.s = result.s + other.s
		return result
		
	def __sub__(self,other):
		'''
		element-wise subtraction of s-matrix
		'''
		result = copy(self)
		result.s = result.s - other.s
		return result

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

	def __eq__(self,other):
		if npy.mean(npy.abs(self.s - other.s)) < ALMOST_ZER0:
			return True
		else:
			return False
	def __ne__(self,other):
		return (not self.__eq__(other))
		
	def __getitem__(self,key):
		'''
		returns a Network object at a given single frequency
		'''
		a= self.z0# hack to force getter for z0 to re-shape it
		output = deepcopy(self)
		output.s = output.s[key,:,:]
		output.z0 = output.z0[key,:]
		output.frequency.f = npy.array(output.frequency.f[key]).reshape(-1)
		
		return output
	
	def __str__(self):
		'''
		'''
		f=self.frequency
		output =  \
			'%i-Port Network.  %i-%i %s.  %i points. z0='% \
			(self.number_of_ports,f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)+str(self.z0[0,:])

		return output
	def __repr__(self):
		return self.__str__()
## PRIMARY PROPERTIES
	# s-parameter matrix
	@property
	def s(self):
		'''
		The scattering parameter matrix.
		
		s-matrix has shape fxnxn, 
		where; 
			f is frequency axis and,
			n's are port indicies
		'''
		return self._s
	
	@s.setter
	def s(self, s):
		'''
		the input s-matrix should be of shape fxnxn, 
		where f is frequency axis and n is number of ports
		'''
		s_shape= npy.shape(s)
		if len(s_shape) <3:
			if len(s_shape)==2:
				# reshape to kx1x1, this simplifies indexing in function
				s = npy.reshape(s,(-1,s_shape[0],s_shape[0]))
			elif len(s_shape)==1:
				 s = npy.reshape(s,(-1,1,1))
		self._s = s
		#s.squeeze()
	@property
	def y(self):
		if self.number_of_ports == 1:
			return (1-self.s)/(1+self.s)
		else:
			raise(NotImplementedError)
	
	# t-parameters
	@property
	def t(self):
		'''
		returns the t-parameters, which are also known as wave cascading
		matrix. 
		'''
		return s2t(self.s)
	@property
	def inv(self):
		'''
		a network representing inverse s-parameters, for de-embeding
		'''
		out = copy(self)
		out.s = inv(self.s)
		return out
		
	# frequency information
	@property
	def frequency(self):
		'''
		returns a Frequency object, see  frequency.py
		'''
		try:
			return self._frequency
		except (AttributeError):
			self._frequency = Frequency(0,0,0)
			return self._frequency
	@frequency.setter
	def frequency(self, new_frequency):
		'''
		takes a Frequency object, see  frequency.py
		'''
		self._frequency= copy(new_frequency)

	
	@property
	def f(self):
		''' the frequency vector for the network, in Hz. '''
		return self.frequency.f
		
	@f.setter
	def f(self,f):
		tmpUnit = self.frequency.unit
		self._frequency  = Frequency(f[0],f[-1],len(f),'hz')
		self._frequency.unit = tmpUnit
	

	
	# characteristic impedance
	@property
	def z0(self):
		''' the characteristic impedance of the network.
		
		z0 can be may be a number, or numpy.ndarray of shape n or fxn. 
		
		'''
		# i hate this function
		# it was written this way because id like to allow the user to
		# set the z0 before the s-parameters are set. However, in this 
		# case we dont know how to re-shape the z0 to fxn. to solve this
		# i attempt to do the re-shaping when z0 is accessed, not when 
		# it is set. this is what makes this function confusing. 
		try:
			if len(npy.shape(self._z0)) ==0:
				try:
					#try and re-shape z0 to match s
					self._z0=self._z0*npy.ones(self.s.shape[:-1])
				except(AttributeError):
					print ('Warning: Network has improper \'z0\' shape.')
					#they have yet to set s .

			elif len(npy.shape(self._z0)) ==1:
				try:
					if len(self._z0) == self.frequency.npoints:
						# this z0 is frequency dependent but no port dependent
						self._z0 = \
							npy.repeat(npy.reshape(self._z0,(-1,1)),self.number_of_ports,1)

					elif len(self._z0) == self.number_of_ports:
						# this z0 is port dependent but not frequency dependent
						self._z0 = self._z0*npy.ones(\
							(self.frequency.npoints,self.number_of_ports))
						
					else:
						raise(IndexError('z0 has bad shape'))
						
				except(AttributeError):
					# there is no self.frequency, or self.number_of_ports
					raise(AttributeError('Error: i cant reshape z0 through inspection. you must provide correctly shaped z0, or s-matrix first.'))
			
			return self._z0
		
		except(AttributeError):
			print 'Warning: z0 is undefined. Defaulting to 50.'
			self.z0=50
			return self.z0 #this is not an error, its a recursive call
		
	@z0.setter
	def z0(self, z0):
		'''z0=npy.array(z0)
		if len(z0.shape) < 2:
			try:
				#try and re-shape z0 to match s
				z0=z0*npy.ones(self.s.shape[:-1])
			except(AttributeError):
				print ('Warning: you should store a Network\'s \'s\' matrix before its \'z0\'')
				#they have yet to set s .
				pass
		'''
		self._z0 = z0
	
## SECONDARY PROPERTIES

	# s-parameters convinience properties	
	@property
	def s_mag(self):
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
	def s11(self):
		result = Network()
		result.frequency = self.frequency
		result.s = self.s[:,0,0]
		return result
	@property
	def s22(self):
		if self.number_of_ports < 2:
			raise(IndexError('this network doesn have enough ports'))
		result = Network()
		result.frequency = self.frequency
		result.s = self.s[:,1,1]
		return result
	@property
	def s21(self):
		if self.number_of_ports < 2:
			raise(IndexError('this network doesn have enough ports'))
		result = Network()
		result.frequency = self.frequency
		result.s = self.s[:,1,0]
		return result
	@property
	def s12(self):
		if self.number_of_ports < 2:
			raise(IndexError('this network doesn have enough ports'))
		result = Network()
		result.frequency = self.frequency
		result.s = self.s[:,0,1]
		return result
	@property
	def number_of_ports(self):
		'''
		the number of ports the network has.
		'''
		return self.s.shape[1]
	@property
	def passivity(self):
		'''
		 passivity metric for a multi-port network. It returns
		a matrix who's diagonals are equal to the total power 
		received at all ports, normalized to the power at a single
		excitement  port.
		
		mathmatically, this is a test for unitary-ness of the 
		s-parameter matrix. 
		
		for two port this is 
			( |S11|^2 + |S21|^2, |S22|^2+|S12|^2)
		in general it is  
			S.H * S
		where H is conjugate transpose of S, and * is dot product
		
		note:
		see more at,
		http://en.wikipedia.org/wiki/Scattering_parameters#Lossless_networks
		'''
		if self.number_of_ports == 1:
			raise (ValueError('Doesnt exist for one ports'))
		
		pas_mat = copy(self.s)
		for f in range(len(self.s)):
			pas_mat[f,:,:] = npy.dot(self.s[f,:,:].conj().T, self.s[f,:,:])
		
		return pas_mat
	

	
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
		
		
		self.f, self.s = touchstoneFile.get_sparameter_arrays() # note: freq in Hz
		self.z0 = float(touchstoneFile.resistance)
		self.frequency.unit = touchstoneFile.frequency_unit # for formatting plots
		self.name = os.path.basename( os.path.splitext(filename)[0])

	def write_touchstone(self, filename=None, dir = './'):
		'''
		write a touchstone file representing this network.  the only 
		format supported at the moment is :
			HZ S RI 
		
		takes: 
			filename: a string containing filename without 
				extension[None]. if 'None', then will use the network's 
				name. if this is empty, then throws an error.
			dir: the directory to save the file in. [string]. Defaults 
				to './'
			
		
		note:
			in the future could make possible use of the touchtone 
			class, but at the moment this would not provide any benefit 
			as it has not set_ functions. 
		'''
		if filename is None:
			if self.name is not None:
				filename= self.name
			else:
				raise ValueError('No filename given. Network must have a name, or you must provide a filename')
		
		extension = '.s%ip'%self.number_of_ports
		
		outputFile = open(dir+'/'+filename+extension,"w")
		
		# write header file. 
		# the '#'  line is NOT a comment it is essential and it must be 
		#exactly this format, to work
		# [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
		outputFile.write('!Created with mwavepy.\n')
		outputFile.write('# ' + self.frequency.unit + ' S RI R ' + str(self.z0[0,0]) +" \n")
		
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
			outputFile.write(str(self.frequency.f_scaled[f])+'\t')
			
			for n in range(self.number_of_ports):
				for m in range(self.number_of_ports):
					outputFile.write( str(npy.real(self.s[f,m,n])) + '\t'\
					 + str(npy.imag(self.s[f,m,n])) +'\t')
			
			outputFile.write('\n')
			outputFile.write('! Port Impedance\t' )
			for n in range(self.number_of_ports):
				outputFile.write('%.14f\t%.14f\t'%(self.z0[f,n].real, self.z0[f,n].imag))
			outputFile.write('\n')
		
		outputFile.close()


	# self-modifications
	def interpolate(self, new_frequency,**kwargs):
		'''
		calculates an interpolated network. defualt interpolation type
		is linear. see notes about other interpolation types

		takes:
			new_frequency:
			**kwargs: passed to scipy.interpolate.interp1d initializer.
				  
		returns:
			result: an interpolated Network

		note:
			usefule keyward for  scipy.interpolate.interp1d:
			 kind : str or int
				Specifies the kind of interpolation as a string ('linear',
				'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an integer
				specifying the order of the spline interpolator to use.

			
		'''
		interpolation_s = interp1d(self.frequency.f,self.s,axis=0,**kwargs)
		interpolation_z0 = interp1d(self.frequency.f,self.z0,axis=0,**kwargs)
		result = deepcopy(self)
		result.frequency = new_frequency
		result.s = interpolation_s(new_frequency.f)
		result.z0 = interpolation_z0(new_frequency.f)
		return result

	def change_frequency(self, new_frequency, **kwargs):
		self.frequency.start = new_frequency.start
		self.frequency.stop = new_frequency.stop
		self = self.interpolate(new_frequency, **kwargs)
		
	def flip(self):
		'''
		swaps the ports of a two port 
		'''
		if self.number_of_ports == 2:
			self.s = flip(self.s)
		else:
			raise ValueError('you can only flip two-port Networks')


	# ploting

	def plot_vs_frequency_generic(self,attribute,y_label=None,\
		m=None,n=None, ax=None,show_legend=True,**kwargs):
		'''
		generic plotting function for plotting a Network's attribute
		vs frequency.
		

		takes:

		
		'''
		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
		
		
		if m is None:
			M = range(self.number_of_ports)
		else:
			M = [m]
		if n is None:
			N = range(self.number_of_ports)
		else:
			N = [n]
		
		if 'label'  not in kwargs.keys():
			generate_label=True
		else:
			generate_label=False
		
		for m in M:
			for n in N:
				# set the legend label for this trace to the networks
				# name if it exists, and they didnt pass a name key in
				# the kwargs
				if generate_label: 
					if self.name is None:
						if plb.rcParams['text.usetex']:
							label_string = '$S_{'+repr(m+1) + \
								repr(n+1)+'}$'
						else:
							label_string = 'S'+repr(m+1) + repr(n+1)
					else:
						if plb.rcParams['text.usetex']:
							label_string = self.name+', $S_{'+repr(m+1)\
								+ repr(n+1)+'}$'
						else:
							label_string = self.name+', S'+repr(m+1) +\
								repr(n+1)
					kwargs['label'] = label_string
					
				# plot the desired attribute vs frequency 
				ax.plot(self.frequency.f_scaled, getattr(self,\
					attribute)[:,m,n], **kwargs)

		# label axis
		ax.set_xlabel('Frequency ['+ self.frequency.unit +']')
		ax.set_ylabel(y_label)
		ax.axis('tight')
		#draw legend
		if show_legend:
			ax.legend()
		plb.draw()
	def plot_polar_generic (self,attribute_r, attribute_theta,	m=0,n=0,\
		ax=None,show_legend=True,**kwargs):
		'''
		generic plotting function for plotting a Network's attribute
		in polar form
		

		takes:
			
		
		'''

		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca(polar=True)
			
		# set the legend label for this trace to the networks name if it
		# exists, and they didnt pass a name key in the kwargs
		if 'label'  not in kwargs.keys(): 
			if self.name is None:
				if plb.rcParams['text.usetex']:
					label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
				else:
					label_string = 'S'+repr(m+1) + repr(n+1)
			else:
				if plb.rcParams['text.usetex']:
					label_string = self.name+', $S_{'+repr(m+1) + \
						repr(n+1)+'}$'
				else:
					label_string = self.name+', S'+repr(m+1) + repr(n+1)

			kwargs['label'] = label_string
			
		#TODO: fix this to call from ax, if possible
		plb.polar(getattr(self, attribute_theta)[:,m,n],\
			getattr(self, attribute_r)[:,m,n],**kwargs)

		#draw legend
		if show_legend:
			plb.legend()	
		
	def plot_s_db(self,m=None, n=None, ax = None, show_legend=True,*args,**kwargs):
		'''
		plots the magnitude of the scattering parameter of indecies m, n
		in log magnitude
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_db',\
			y_label='Magnitude [dB]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)

	def plot_s_mag(self,m=None, n=None, ax = None, show_legend=True,*args,**kwargs):
		'''
		plots the magnitude of a scattering parameter of indecies m, n
		not in  magnitude
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_mag',\
			y_label='Magnitude [not dB]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)

	def plot_s_deg(self,m=None, n=None, ax = None, show_legend=True,*args,**kwargs):
		'''
		plots the phase of a scattering parameter of indecies m, n in
		degrees
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_deg',\
			y_label='Phase [deg]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)
		
				
	def plot_s_deg_unwrapped(self,m=None, n=None, ax = None, show_legend=True,\
		*args,**kwargs):
		'''
		plots the phase of a scattering parameter of indecies m, n in
		unwrapped degrees
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_deg_unwrap',\
			y_label='Phase [deg]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)
	
	plot_s_deg_unwrap = plot_s_deg_unwrapped
	
	def plot_s_rad(self,m=None, n=None, ax = None, show_legend=True,*args,**kwargs):
		'''
		plots the phase of a scattering parameter of indecies m, n in
		radians
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_rad',\
			y_label='Phase [rad]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)
		
				
	def plot_s_rad_unwrapped(self,m=None, n=None, ax = None, show_legend=True,\
		*args,**kwargs):
		'''
		plots the phase of a scattering parameter of indecies m, n in
		unwrapped radians
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_vs_frequency_generic(attribute= 's_rad_unwrap',\
			y_label='Phase [rad]', m=m,n=n, ax=ax,\
			show_legend = show_legend,*args,**kwargs)	

	def plot_s_polar(self,m=0, n=0, ax = None, show_legend=True,\
		*args,**kwargs):
		'''
		plots the scattering parameter of indecies m, n in polar form
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		self.plot_polar_generic(attribute_r= 's_mag',attribute_theta='s_rad',\
			m=m,n=n, ax=ax,	show_legend = show_legend,*args,**kwargs)	

	def plot_s_smith(self,m=None, n=None,r=1,ax = None, show_legend=True,\
		chart_type='z', *args,**kwargs):
		'''
		plots the scattering parameter of indecies m, n on smith chart
		
		takes:
			m - first index, int
			n - second indext, int
			r -  radius of smith chart
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			chart_type: string determining countour type. options are:
				'z': impedance contours (default)
				'y': admittance contours
			*args,**kwargs - passed to the matplotlib.plot command	
		'''
		# TODO: prevent this from re-drawing smith chart if one alread
		# exists on current set of axes

		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
			
		
		if m is None:
			M = range(self.number_of_ports)
		else:
			M = [m]
		if n is None:
			N = range(self.number_of_ports)
		else:
			N = [n]
		
		if 'label'  not in kwargs.keys():
			generate_label=True
		else:
			generate_label=False
		
		for m in M:
			for n in N:
				# set the legend label for this trace to the networks name if it
				# exists, and they didnt pass a name key in the kwargs
				if generate_label: 
					if self.name is None:
						if plb.rcParams['text.usetex']:
							label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
						else:
							label_string = 'S'+repr(m+1) + repr(n+1)
					else:
						if plb.rcParams['text.usetex']:
							label_string = self.name+', $S_{'+repr(m+1) + \
								repr(n+1)+'}$'
						else:
							label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
					kwargs['label'] = label_string
					
				# plot the desired attribute vs frequency 
				if len (ax.patches) == 0:
					smith(ax=ax, smithR = r, chart_type=chart_type)
				ax.plot(self.s[:,m,n].real,  self.s[:,m,n].imag, *args,**kwargs)
		
		#draw legend
		if show_legend:
			ax.legend()
		ax.axis(npy.array([-1,1,-1,1])*r)
		ax.set_xlabel('Real')
		ax.set_ylabel('Imaginary')
	def plot_s_complex(self,m=None, n=None,ax = None, show_legend=True,\
		*args,**kwargs):
		'''
		plots the scattering parameter of indecies m, n on complex plane
		
		takes:
			m - first index, int
			n - second indext, int
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command	
		'''
		# TODO: prevent this from re-drawing smith chart if one alread
		# exists on current set of axes

		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()
			
		
		if m is None:
			M = range(self.number_of_ports)
		else:
			M = [m]
		if n is None:
			N = range(self.number_of_ports)
		else:
			N = [n]
		
		if 'label'  not in kwargs.keys():
			generate_label=True
		else:
			generate_label=False
		
		for m in M:
			for n in N:
				# set the legend label for this trace to the networks name if it
				# exists, and they didnt pass a name key in the kwargs
				if generate_label: 
					if self.name is None:
						if plb.rcParams['text.usetex']:
							label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
						else:
							label_string = 'S'+repr(m+1) + repr(n+1)
					else:
						if plb.rcParams['text.usetex']:
							label_string = self.name+', $S_{'+repr(m+1) + \
								repr(n+1)+'}$'
						else:
							label_string = self.name+', S'+repr(m+1) + repr(n+1)
		
					kwargs['label'] = label_string
					
				# plot the desired attribute vs frequency 
				ax.plot(self.s[:,m,n].real,  self.s[:,m,n].imag, *args,**kwargs)
		
		#draw legend
		if show_legend:
			ax.legend()
		ax.axis('equal')
		ax.set_xlabel('Real')
		ax.set_ylabel('Imaginary')
	def plot_s_all_db(self,ax = None, show_legend=True,*args,**kwargs):
		'''
		plots all s parameters in log magnitude

		takes:
			ax - matplotlib.axes object to plot on, used in case you
				want to update an existing plot.
			show_legend: boolean, to turn legend show legend of not
			*args,**kwargs - passed to the matplotlib.plot command
		'''
		for m in range(self.number_of_ports):
			for n in range(self.number_of_ports):
				self.plot_vs_frequency_generic(attribute= 's_db',\
					y_label='Magnitude [dB]', m=m,n=n, ax=ax,\
					show_legend = show_legend,*args,**kwargs)
	# noise
	def add_noise_polar(self,mag_dev, phase_dev,**kwargs):
		'''
		adds a complex zero-mean gaussian white-noise signal of given
		standard deviations for magnitude and phase

		takes:
			mag_mag: standard deviation of magnitude
			phase_dev: standard deviation of phase [in degrees]
			n_ports: number of ports. defualt to 1
		returns:
			nothing
		'''
		phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = self.s.shape)
		mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = self.s.shape)
		phase = (self.s_deg+phase_rv)
		mag = self.s_mag + mag_rv 
		self.s = mag* npy.exp(1j*npy.pi/180.*phase)
	def add_noise_polar_flatband(self,mag_dev, phase_dev,**kwargs):
		'''
		adds a flatband complex zero-mean gaussian white-noise signal of
		given standard deviations for magnitude and phase

		takes:
			mag_mag: standard deviation of magnitude
			phase_dev: standard deviation of phase [in degrees]
			n_ports: number of ports. defualt to 1
		returns:
			nothing
		'''
		phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = self.s[0].shape)
		mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = self.s[0].shape)
		
		phase = (self.s_deg+phase_rv)
		mag = self.s_mag + mag_rv 
		self.s = mag* npy.exp(1j*npy.pi/180.*phase)
	
	def multiply_noise(self,mag_dev, phase_dev, **kwargs):
		'''
		multiplys a complex bivariate gaussian white-noise signal
		of given standard deviations for magnitude and phase. 	
		magnitude mean is 1, phase mean is 0 
		
		takes:
			mag_dev: standard deviation of magnitude
			phase_dev: standard deviation of phase [in degrees]
			n_ports: number of ports. defualt to 1
		returns:
			nothing
		'''
		phase_rv = stats.norm(loc=0, scale=phase_dev).rvs(\
			size = self.s.shape)
		mag_rv = stats.norm(loc=1, scale=mag_dev).rvs(\
			size = self.s.shape)
		self.s = mag_rv*npy.exp(1j*npy.pi/180.*phase_rv)*self.s
	
	def nudge(self, amount=1e-12):
		'''
		perturb s-parameters by small amount. this is usefule to work-around
		numerical bugs.
		takes:
			amount: amount to add to s parameters
		returns:
			na
		'''
		self.s = self.s + 1e-12

## Functions operating on Network[s]
def connect(ntwkA, k, ntwkB,l):
	'''
	connect two n-port networks together. specifically, connect port 'k'
	on ntwkA to port 'l' on ntwkB. The resultant network has
	(ntwkA.nports+ntwkB.nports -2) ports. The port index's ('k','l') 
	start from 0. Port impedances are taken into account.
	
	takes:
		ntwkA: network 'A', [mwavepy.Network]
		k: port index on ntwkA [int] ( port indecies start from 0 )
		ntwkB: network 'B', [mwavepy.Network]
		l: port index on ntwkB [int]
	
	returns:
		ntwkC': new network of rank (ntwkA.nports+ntwkB.nports -2)-ports
	
	
	note:
		see functions connect_s() and innerconnect_s() for actual 
	S-parameter connection algorithm.
		
		the effect of mis-matched port impedances is handled by inserting
	a 2-port 'mismatch' network between the two connected ports. 
	'''
	ntwkC = deepcopy(ntwkA)
	# account for port impedance mis-match by inserting a two-port 
	# network at the connection. if ports are matched this becomes a 
	# thru
	ntwkC.s = connect_s(\
		ntwkA.s,k, \
		impedance_mismatch(ntwkA.z0[:,k],ntwkB.z0[:,l]),0)
			
	ntwkC.s = connect_s(ntwkC.s,k,ntwkB.s,l)
	ntwkC.z0=npy.hstack((npy.delete(ntwkA.z0,k,1),npy.delete(ntwkB.z0,l,1)))
	return ntwkC

def innerconnect(ntwkA, k, l):
	'''
	connect two ports of a single n-port network, resulting in a 
	(n-2)-port network. port indecies start from 0.
	
	takes:
		ntwk: the network. [mwavepy.Network]
		k: port index [int] (port indecies start from 0)
		l: port index [int]
	returns:
		ntwk': new network of with n-2 ports. [mwavepy.Network]
		
	note:
		see functions connect_s() and innerconnect_s() for actual 
	S-parameter connection algorithm. 
	'''
	ntwkC = deepcopy(ntwkA)
	ntwkC.s = connect_s(\
		ntwkA.s,k, \
		impedance_mismatch(ntwkA.z0[:,k],ntwkA.z0[:,l]),0)
	ntwkC.s = innerconnect_s(ntwkC.s,k,l)
	ntwkC.z0=npy.delete(ntwkC.z0,[l,k],1)
	return ntwkC

def average(list_of_networks):
	'''
	calculates the average network from a list of Networks. 
	this is complex average of the s-parameters for a  list of Networks
	
	takes:
		list_of_networks: a list of Networks
	returns:
		ntwk: the resultant averaged Network [mwavepy.Network]
		
	'''
	out_ntwk = copy(list_of_networks[0])
	
	for a_ntwk in list_of_networks[1:]:
		out_ntwk += a_ntwk

	out_ntwk.s = out_ntwk.s/(len(list_of_networks))

	return out_ntwk

def one_port_2_two_port(ntwk):
	'''
	calculates the two-port network given a  symetric, reciprocal and 
	lossless one-port network.
	
	takes:
		ntwk: a symetric, reciprocal and lossless one-port network.
	returns:
		ntwk: the resultant two-port Network
	'''
	result = copy(ntwk)
	result.s = npy.zeros((result.frequency.npoints,2,2), dtype=complex) 
	s11 = ntwk.s[:,0,0]
	result.s[:,0,0] = s11
	result.s[:,1,1] = s11
	## HACK: TODO: verify this mathematically
	result.s[:,0,1] = npy.sqrt(1- npy.abs(s11)**2)*\
		npy.exp(1j*(npy.angle(s11)+npy.pi/2.*(npy.angle(s11)<0) -npy.pi/2*(npy.angle(s11)>0)))
	result.s[:,1,0] = result.s[:,0,1]
	return result
	
def two_port_reflect(ntwk1, ntwk2, **kwargs):
	'''
	generates a two-port reflective (S21=S12=0) network, from the
	 2 one-port networks

	takes:
		ntwk1: Network on  port 1 [mwavepy.Network]
		ntwk2: Network on  port 2 [mwavepy.Network]
	
	returns:
		result: two-port reflective network, S12=S21=0 [mwavepy.Network]

	
	example:
	to use a working band to create a two-port reflective standard from
	two one-port standards
		my_media= ...
		two_port_reflect(my_media.short(), my_media.match())
	'''
	result = deepcopy(ntwk1)
	result.s = npy.zeros((ntwk1.frequency.npoints,2,2), dtype='complex')
	for f in range(ntwk1.frequency.npoints):
		result.s[f,0,0] = ntwk1.s[f,0,0]
		result.s[f,1,1] = ntwk2.s[f,0,0]
	return result	

def func_on_networks(ntwk_list, func, attribute='s',*args, **kwargs):
	'''
	Applies a function to some attribute of aa list of networks, and 
	returns the result in the form of a Network. This means information 
	that may not be s-parameters is stored in the s-matrix of the
	returned Network.
	
	takes:
		ntwk_list: list of mwavepy.Network types
		func: function to operate on ntwk_list s-matrices
		attribute: attribute of Network's  in ntwk_list for func to act on
		*args: passed to func
		**kwargs: passed to func
	
	returns:
		mwavepy.Network type, with s-matrix the result of func, 
			operating on ntwk_list's s-matrices

	
	example:
		averaging can be implemented with func_on_networks by 
			func_on_networks(ntwk_list,mean)
	'''
	data_matrix = \
		npy.array([ntwk.__getattribute__(attribute) for ntwk in ntwk_list])
	
	new_ntwk = deepcopy(ntwk_list[0])
	new_ntwk.s = func(data_matrix,axis=0,*args,**kwargs)
	return new_ntwk

def plot_uncertainty_bounds_s_mag(*args, **kwargs):
	'''
	this just calls 
		plot_uncertainty_bounds(attribute= 's_mag',*args,**kwargs)
	see plot_uncertainty_bounds for help
	
	'''
	kwargs.update({'attribute':'s_mag'})
	plot_uncertainty_bounds(*args,**kwargs)

def plot_uncertainty_bounds_s_deg(*args, **kwargs):
	'''
	this just calls 
		plot_uncertainty_bounds(attribute= 's_deg',*args,**kwargs)
	see plot_uncertainty_bounds for help
	
	'''
	kwargs.update({'attribute':'s_deg'})
	plot_uncertainty_bounds(*args,**kwargs)
	
def plot_uncertainty_bounds(ntwk_list,attribute='s_mag',m=0,n=0,\
	n_deviations=3, alpha=.3,*args,**kwargs):
	'''
	plots mean value with +- uncertainty bounds in an Network attribute,
	for a list of Networks. 
	
	takes:
		ntwk_list: list of Netmwork types [list]
		attribute: attribute of Network type to analyze [string] 
		m: first index of attribute matrix [int]
		n: second index of attribute matrix [int]
		n_deviations: number of std deviations to plot as bounds [number]
		alpha: passed to matplotlib.fill_between() command. [number, 0-1]
		*args,**kwargs: passed to Network.plot_'attribute' command
		
	returns:
		None
		
	
	Caution:
		 if your list_of_networks is for a calibrated short, then the 
		std dev of deg_unwrap might blow up, because even though each
		network is unwrapped, they may fall on either side fo the pi 
		relative to one another.
	'''
	# calculate mean response, and std dev of given attribute
	ntwk_mean = average(ntwk_list)
	ntwk_std = func_on_networks(ntwk_list,npy.std, attribute=attribute)
	
	# pull out port of interest
	ntwk_mean.s = ntwk_mean.s[:,m,n]
	ntwk_std.s = ntwk_std.s[:,m,n]
	
	# create bounds (the s_mag here is confusing but is realy in units
	# of whatever 'attribute' is. read the func_on_networks call to understand
	upper_bound =  ntwk_mean.__getattribute__(attribute) +\
		ntwk_std.s_mag*n_deviations
	lower_bound =   ntwk_mean.__getattribute__(attribute) -\
		ntwk_std.s_mag*n_deviations
	
	# find the correct ploting method
	plot_func = ntwk_mean.__getattribute__('plot_'+attribute)
	
	#plot mean response
	plot_func(*args,**kwargs)
	
	#plot bounds
	plb.fill_between(ntwk_mean.frequency.f_scaled, \
		lower_bound.squeeze(),upper_bound.squeeze(), alpha=alpha)
	plb.axis('tight')
	plb.draw()

## Functions operating on s-parameter matrices
def connect_s(S,k,T,l):
	'''
	connect two n-port networks together. specifically, connect port 'k'
	on network 'S' to port 'l' on network 'T'. The resultant network has
	(S.rank + T.rank-2)-ports
	
	takes:
		S: S-parameter matrix [numpy.ndarray].
		k: port index on S (port indecies start from 0) [int]
		T: S-parameter matrix [numpy.ndarray]
		l: port index on T [int]
	returns:
		S': new S-parameter matrix [numpy.ndarry]
		
	
	note: 
		shape of S-parameter matrices can be either nxn, or fxnxn, where
	f is the frequency axis. 
		internally, this function creates a larger composite network 
	and calls the  innerconnect() function. see that function for more 
	details about the implementation
	
	'''
	if k > S.shape[-1]-1 or l>T.shape[-1]-1:
		raise(ValueError('port indecies are out of range'))
	
	if len (S.shape) > 2:
		# assume this is a kxnxn matrix
		n = S.shape[-1]+T.shape[-1]-2 # nports of output Matrix
		Sp = npy.zeros((S.shape[0], n,n), dtype='complex')
		
		for f in range(S.shape[0]):
			Sp[f,:,:] = connect_s(S[f,:,:],k,T[f,:,:],l)
		return Sp
	else:		
		filler = npy.zeros((S.shape[0],T.shape[1]))
		#create composite matrix, appending each sub-matrix diagonally
		Sp= npy.vstack( [npy.hstack([S,filler]),npy.hstack([filler.T,T])])
		
		return innerconnect_s(Sp, k,S.shape[-1]+l)
		
def innerconnect_s(S, k, l):
	'''
	connect two ports of a single n-port network, resulting in a 
	(n-2)-port network.
	
	takes:
		S: S-parameter matrix [numpy.ndarray] 
		k: port index [int]
		l: port index [int]
	returns:
		S': new S-parameter matrix [numpy.ndarry]
		
	This function is based on the algorithm presented in the paper:
		'Perspectives in Microwave Circuit Analysis' by R. C. Compton
		 and D. B. Rutledge
	The original algorithm is given in 
		'A NEW GENERAL COMPUTER ALGORITHM FOR S-MATRIX CALCULATION OF
		INTERCONNECTED MULTIPORTS' by Gunnar Filipsson
	'''
	
	# TODO: this algorithm is a bit wasteful in that it calculates the 
	# scattering parameters for a network of rank S.rank+T.rank and 
	# then deletes the ports which are 'connected' 
	if k > S.shape[-1] -1 or l>S.shape[-1]-1:
		raise(ValueError('port indecies are out of range'))
	
	if len (S.shape) > 2:
		# assume this is a kxnxn matrix
		n = S.shape[-1]-2 # nports of output Matrix
		Sp = npy.zeros((S.shape[0], n,n), dtype='complex')
		
		for f in range(S.shape[0]):
			Sp[f,:,:] = innerconnect_s(S[f,:,:],k,l)
		return Sp
	else:
		n = S.shape[0]
		Sp = npy.zeros([n,n],dtype='complex')
		for i in range(n):
			for j in range(n):
				Sp[i,j] = S[i,j] +  \
					( S[k,j]*S[i,l]*(1-S[l,k]) + S[l,j]*S[i,k]*(1-S[k,l]) +\
					S[k,j]*S[l,l]*S[i,k] + S[l,j]*S[k,k]*S[i,l])/\
					( (1-S[k,l])*(1-S[l,k]) - S[k,k]*S[l,l] )
		Sp = npy.delete(Sp,(k,l),0)
		Sp = npy.delete(Sp,(k,l),1)
		return Sp

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
	
def inv(s):
	'''
	inverse s-parameters, used for de-embeding
	'''
	# this idea is from lihan
	i = npy.copy(s)
	if len (s.shape) > 2 :
		for f in range(len(s)):
			i[f,:,:] = inv(s[f,:,:])
	elif s.shape == (2,2):
		i = t2s(npy.linalg.inv(s2t(s)))
	else:
		raise IndexError('matrix should be 2x2, or kx2x2')
	return i

def cascade(a,b):
	'''
	DEPRECATED. see connect_s() instead.
	
	cascade two s-matricies together.
	
	a's port 2 == b's port 1
	
	if you want a different port configuration use the flip() fuction
	takes:
		a: a 2x2 or kx2x2 s-matrix
		b: a 2x2, kx2x2, 1x1, or kx1x1 s-matrix 
	note:
		BE AWARE! this relies on s2t function which has a inf problem 
		if s11 or s22 is 1. 
	'''
	c = copy(b)
	
	if len (a.shape) > 2 :
		# assume this is a kxnxn matrix
		for f in range(a.shape[0]):
			c[f,:,:] = cascade(a[f,:,:],b[f,:,:])
	
	elif a.shape == (2,2) and b.shape == (2,2):
		c = t2s(npy.dot (s2t(a) ,s2t(b)))
	elif a.shape == (2,2) and b.shape == (1,1):
		# makes b into a two-port s-matrix so that s2t will work, but  
		# only s11 of the  resultant network is returned
		c = t2s(npy.dot (s2t(a) , \
			s2t(npy.array([[b.squeeze(),1.0],[1.0,0.0]]))))[0,0]
	else:
		raise IndexError('one of the s-matricies has incorrect shape')
	return c
	
def de_embed(a,b):	
	'''
	de-embed a 2x2 s-matrix from another 2x2 s-matrix
	
	c = b**-1 * a
	
	note:
		BE AWARE! this relies on s2t function which has a inf problem 
		if s11 or s22 is 1. 
	'''
	# TODO: re-write this and make more efficient/concise
	c = copy(a)
	
	if len (b.shape) > 2 :
		for f in range(b.shape[0]):
			if len(a.shape) == 1:
				# 'a' is a one-port netowrk
				c[f] = de_embed(a[f],b[f,:,:])
			else:
				c[f,:,:] = de_embed(a[f,:,:],b[f,:,:])
	
	elif b.shape == (2,2):
		if a.shape == (2,2):
			c = t2s(npy.dot ( npy.linalg.inv(s2t(b)), s2t(a)))
		else:
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




## Other	
# dont belong here, but i needed them quickly
# this is needed for port impedance mismatches 
def impedance_mismatch(z1, z2):
		'''
		returns a two-port network for a impedance mis-match
		
		takes:
			z1: complex impedance of port 1 [ number, list, or 1D ndarray]
			z2: complex impedance of port 2 [ number, list, or 1D ndarray]
		returns:
			2-port s-matrix for the impedance mis-match
		'''	
		gamma = zl_2_Gamma0(z1,z2)
		result = npy.zeros(shape=(len(gamma),2,2), dtype='complex')
		
		result[:,0,0] = gamma
		result[:,1,1] = -gamma
		result[:,1,0] = 1+gamma
		result[:,0,1] = 1-gamma
		return result


# Touchstone manipulation	
def load_all_touchstones(dir = '.', contains=None, f_unit=None):
	'''
	loads all touchtone files in a given dir 
	
	takes:
		dir  - the path to the dir, passed as a string (defalut is cwd)
		contains - string which filename must contain to be loaded, not 
			used if None.(default None)
	returns:
		ntwkDict - a Dictonary with keys equal to the file name (without
			a suffix), and values equal to the corresponding ntwk types
	
		
	'''
	ntwkDict = {}

	for f in os.listdir (dir):
		if contains is not None and contains not in f:
			continue
			
		# TODO: make this s?p with reg ex
		if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
			name = f[:-4]
			ntwkDict[name]=(Network(dir +'/'+f))
			if f_unit is not None: ntwkDict[name].frequency.unit=f_unit
		
	return ntwkDict	

def write_dict_of_networks(ntwkDict, dir='.'):
	'''
	writes a dictionary of networks to a given directory
	'''
	for ntwkKey in ntwkDict:
		ntwkDict[ntwkKey].write_touchstone(filename = dir+'/'+ntwkKey)

def csv_2_touchstone(filename):
	'''
	converts a csv file saved from a Rhode swarz and possibly other 
	
	takes:
		filename: name of file
	returns:
		Network object
	'''
		
	ntwk = Network(name=filename[:-4])
	try: 
		data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
			usecols=range(9))
		s11 = data[:,1] +1j*data[:,2]	
		s21 = data[:,3] +1j*data[:,4]	
		s12 = data[:,5] +1j*data[:,6]	
		s22 = data[:,7] +1j*data[:,8]	
		ntwk.s = npy.array([[s11, s21],[s12,s22]]).transpose().reshape(-1,2,2)
	except(IndexError):
		data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
			usecols=range(3))		
		ntwk.s = data[:,1] +1j*data[:,2]
	
	ntwk.frequency.f = data[:,0]
	ntwk.frequency.unit='ghz'
	
	return ntwk


## convinience names
fon = func_on_networks


