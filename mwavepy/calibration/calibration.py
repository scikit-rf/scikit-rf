
#       calibration.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       
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
Contains the Calibration class, and supporting functions
'''
import numpy as npy
from numpy import mean, std
import pylab as plb
import os 
from copy import deepcopy, copy
import itertools
import warnings

from calibrationAlgorithms import *
from ..mathFunctions import complex_2_db
from ..frequency import *
from ..network import *
from ..network import func_on_networks as fon
from ..convenience import *



## main class
class Calibration(object):
	'''
	Represents a calibration instance, a class to hold sets
	of measurements, ideals, and calibration results.
	
	see init for more information on usage. 
	
	note:
	all calibration algorithms are in calibrationAlgorithms.py, and are
	referenced by the dictionary in this object called
	'calibration_algorihtm_dict'
	'''
	calibration_algorithm_dict={\
		'one port': one_port,\
		'one port nls': one_port_nls,\
		'one port parametric':parameterized_self_calibration,\
		'one port parametric bounded':parameterized_self_calibration_bounded,\
		'two port': two_port,\
		'two port parametric':parameterized_self_calibration,\
		}
	
	def __init__(self,measured, ideals, type=None, frequency=None,\
		is_reciprocal=False,switch_terms=None, name=None,**kwargs):
		'''
		Calibration initializer.
		
		takes:
			measured: a list of measured Network's (order must align 
				with 'ideals'
			ideals: a list of ideal Network's (order must align 
				with 'measured'
		
		[ optional]:		
			frequency: a Frequency object over which the calibration
				is defined [None]. if None then will take frequency
				information	from the first measurement.

			type: string representing what type of calibration is to be
				performed [None]. supported types at the moment are:

				'one port':	standard one-port cal. if more than
					2 measurement/ideal pairs are given it will
					calculate the least squares solution.

				'two port': standard two-port calibibration based on
					8-term error model. can take switch-terms into
					by using the switch_term option.
				
				if None, will inspect number of ports on first measured
				network and choose either 'one port' or 'two port'
				
			is_reciprocal: enables the reciprocity assumption on 
				the property error_network [False].

			switch_terms: tuple holding the two measured switch terms in
				the order (forward, reverse) [None]. the tuple elements
				 should	be Network types. (note: is only used in two-port
				  calibrations)

			name: string holding the name of calibration, just for your
				convenience [None].
				
			**kwargs: key-word arguments passed to the calibration
				algorithm.
		'''
		
		self.measured = copy(measured)
		self.ideals = copy(ideals)
		self.type = type
		self.frequency = frequency
		# a dictionary holding key word arguments to pass to whatever
		# calibration function we are going to call
		self.kwargs = kwargs
		self.name = name
		self.is_reciprocal = is_reciprocal
		self.switch_terms = switch_terms
		self._residual_ntwks = None
		self.has_run = False
		

	## properties
	@property 
	def frequency(self):
		return self._frequency
	
	@frequency.setter
	def frequency(self, new_frequency):
		if new_frequency is None:
			# they did not supply frequency, so i will try
			# to inspect a measured ntwk to  get it
			new_frequency = self.measured[0].frequency
		self._frequency = deepcopy(new_frequency)
		
	@property
	def type (self):
		'''
		string representing what type of calibration is to be
		performed. supported types at the moment are:

		'one port':	standard one-port cal. if more than
			2 measurement/ideal pairs are given it will
			calculate the least squares solution.

		'one port xds': self-calibration of a unknown-length
			delay-shorts.

		note:
		algorithms referenced by  calibration_algorithm_dict 
		'''
		return self._type

	@type.setter
	def type(self, new_type):
		if new_type is None:
			# they did not supply a calibration type, so i will try
			# to inspect a measured ntwk to see how many ports it has
			print ('Calibration type not supplied, inspecting type from measured Networks..'),
			if self.measured[0].number_of_ports == 1:
				new_type = 'one port'
				print (' using \'one port\' calibration')
			elif self.measured[0].number_of_ports == 2:
				new_type = 'two port'
				print (' using \'two port\' calibration')
		
		if new_type not in self.calibration_algorithm_dict.keys():
			raise ValueError('incorrect calibration type')
		
		
		self._type = new_type
		
		# set the number of ports of the calibration
		if 'one port' in new_type:
			self._nports = 1
		elif 'two port' in new_type:
			self._nports = 2
		else:
			raise NotImplementedError('only one and two ports supported right now')

	@property
	def nports(self):
		'''
		the number of ports in the calibration
		'''
		return self._nports

	@property
	def nstandards(self):
		if len(self.ideals) != len(self.measured):
			warnings.warn('number of ideals and measured dont agree')
		return len(self.ideals)
		
	@property
	def output_from_cal(self):
		'''
		a dictionary holding all of the output from the calibration
		algorithm
		'''
		return self._output_from_cal

	
	@property	
	def coefs(self):
		'''
		coefs: a dictionary holding the calibration coefficients

		for one port cal's
			'directivity':e00
			'reflection tracking':e01e10
			'source match':e11
		for 7-error term two port cal's
			TBD
		'''
		return self.output_from_cal['error coefficients']
	@property
	def residuals(self):
		'''
		from numpy.lstsq:
			residues:
			the sum of the residues; squared euclidean norm for 
			each column vector in b (given ax=b)
		
		'''
		return self.output_from_cal['residuals']
		
	@property
	def error_ntwk(self):
		'''
		a Network type which represents the error network being
		calibrated out.
		'''
		if not self.has_run:
			self.run()
			
		if self.nports ==1:
			return self._error_ntwk

		elif self.nports == 2:
			raise NotImplementedError('Not sure what to do yet')
	@property
	def Ts(self):
		'''
		T-matricies used for de-embeding. 
		'''
		
		if self.nports ==2:
			if not self.has_run:
				self.run()
			return self._Ts
		elif self.nports ==1:
			raise AttributeError('Only exists for two-port cals')
		else:
			raise NotImplementedError('Not sure what to do yet')
	
	@property
	def residual_ntwks(self):
		'''
		returns a the residuals for each calibration standard in the 
		form of a list of Network types.
		
		note:
			the residuals are only calculated if they are not existent.
		so, if you want to re-calculate the residual networks then 
		you delete the property '_residual_ntwks'. 
		'''
		if self._residual_ntwks is not None:
			return self._residual_ntwks
		else:
			ntwk_list=\
				[ ((self.error_ntwk.inv**self.measured[k])-self.ideals[k]) \
					for k in range(len(self.ideals))]
			
			for k in range(len(ntwk_list)):
				if self.ideals[k].name  is not None:
					name = self.ideals[k].name
				else:
					name='std# %i'%k
				
				ntwk_list[k].name = self.ideals[k].name 
			
			self._residual_ntwks = ntwk_list
		return ntwk_list 
	
	##  methods for manual control of internal calculations

	##  methods for manual control of internal calculations
	def run(self):
		'''
		runs the calibration algorihtm.
		
		 this is automatically called the
		first time	any dependent property is referenced (like error_ntwk)
		, but only the first time. if you change something and want to
		re-run the calibration use this.  
		'''
		# some basic checking to make sure they gave us consistent data
		if self.type == 'one port' or self.type == 'two port':
			
			#1 did they supply the same number of  ideals as measured?
			if len(self.measured) != len(self.ideals):
				raise(IndexError(' The length of measured and ideals lists are different. Number of ideals must equal the number of measured. '))
			
			#2 are all the networks' frequency's the same? 
			index_tuple = \
			list(itertools.permutations(range(len(self.measured)),2))
			
			for k in index_tuple:
				if self.measured[k[0]].frequency != \
					self.ideals[k[1]].frequency:
					raise(IndexError('Frequency information doesnt match on measured[%i], ideals[%i]. All networks must have identical frequency information'%(k[0],k[1])))
#
		
		
		
		# actually call the algorithm and run the calibration
		self._output_from_cal = \
			self.calibration_algorithm_dict[self.type](measured = self.measured, ideals = self.ideals,**self.kwargs)

		if self.nports ==1:
			self._error_ntwk = error_dict_2_network(self.coefs, \
				frequency=self.frequency, is_reciprocal=self.is_reciprocal)
		elif self.nports ==2:
			self._Ts = two_port_error_vector_2_Ts(self.coefs)
		
		#reset the residuals
		self._residual_ntwks = None
		
		self.has_run = True

	## methods 
	def apply_cal(self,input_ntwk):
		'''
		apply the current calibration to a measurement.

		takes:
			input_ntwk: the measurement to apply the calibration to, a
				Network type.
		returns:
			caled: the calibrated measurement, a Network type.
		'''
		if self.nports ==1:
			caled =  self.error_ntwk.inv**input_ntwk 
			caled.name = input_ntwk.name
			
		elif self.nports == 2:
			caled = deepcopy(input_ntwk)
			T1,T2,T3,T4 = self.Ts
			dot = npy.dot
			for f in range(len(input_ntwk.s)):
				t1,t2,t3,t4,m = T1[f,:,:],T2[f,:,:],T3[f,:,:],\
					T4[f,:,:],input_ntwk.s[f,:,:]
				caled.s[f,:,:] = dot(npy.linalg.inv(-1*dot(m,t3)+t1),(dot(m,t4)-t2))
		return caled 

	def apply_cal_to_all_in_dir(self, dir, contains=None, f_unit = 'ghz'):
		'''
		convience function to apply calibration to an entire directory
		of measurements, and return a dictionary of the calibrated
		results, optionally the user can 'grep' the direction
		by using the contains switch.

		takes:
			dir: directory of measurements (string)
			contains: will only load measurements who's filename contains
				this string.
			f_unit: frequency unit, to use for all networks. see
				frequency.Frequency.unit for info.
		returns:
			ntwkDict: a dictionary of calibrated measurements, the keys
				are the filenames.
		'''
		ntwkDict = load_all_touchstones(dir=dir, contains=contains,\
			f_unit=f_unit)

		for ntwkKey in ntwkDict:
			ntwkDict[ntwkKey] = self.apply_cal(ntwkDict[ntwkKey])
		
		return ntwkDict
		
	
	#def plot_error_coefs(self):

	## ploting
	def plot_coefs_db(self,ax=None,show_legend=True,**kwargs):
		'''
		plot magnitude of the error coeficient dictionary
		'''

		# get current axis if user doesnt supply and axis 
		if ax is None:
			ax = plb.gca()

				
		# plot the desired attribute vs frequency
		for error_term in self.coefs:
			error_term_db = complex_2_db(self.coefs[error_term])
			if plb.rcParams['text.usetex'] and '_' in error_term:
				error_term = '$'+error_term+'$'
			ax.plot(self.frequency.f_scaled, error_term_db , label=error_term,**kwargs)

		# label axis
		plb.xlabel('Frequency ['+ self.frequency.unit +']')
		plb.ylabel('Magnitude [dB]')
		plb.axis('tight')
		#draw legend
		if show_legend:
			plb.legend()
	
	#def plot_residuals_db(self,ax=None,show_legend=True,**kwargs):
		#'''
		#plot magnitude of the resdiues, if calibration is
		 #overdetermined 
		#'''

		## get current axis if user doesnt supply and axis 
		#if ax is None:
			#ax = plb.gca()

		#if self.name is None:
			#label_string = ''	
		#else:
			#label_string = self.name	
		#ax.semilogy(self.frequency.f_scaled, \
			#abs(self.residuals), label=label_string,\
			#**kwargs)

		## label axis
		#plb.xlabel('Frequency ['+ self.frequency.unit +']')
		#plb.ylabel('Residual Magnitude')
		#plb.axis('tight')
		##draw legend
		#if show_legend:
			#plb.legend()
	
	
	
	def plot_residuals(self,attribute,*args,**kwargs):
		'''
		plots a component of the residual errors on the  Calibration-plane.
		
		takes:
			attribute: name of ploting method of Network class to call
				possible options are:
					'mag', 'db', 'smith', 'deg', etc
			*args,**kwargs: passed to plot_s_'atttribute'()
			
		
		note:
		the residuals are calculated by:
			(self.error_ntwk.inv**self.measured[k])-self.ideals[k])
			
		'''
		for ntwk in self.residual_ntwks:
			ntwk.__getattribute__('plot_s_'+attribute)(*args,**kwargs)
				
	def plot_residuals_smith(self,*args,**kwargs):
		'''
		see plot_residuals
		'''
		self.plot_residuals(self,attribute='smith',*args,**kwargs)	
	
	def plot_residuals_mag(self,*args,**kwargs):
		'''
		see plot_residuals
		'''
		self.plot_residuals(self,attribute='mag',*args,**kwargs)	
	
	def plot_residuals_db(self,*args,**kwargs):
		'''
		see plot_residuals
		'''
		self.plot_residuals(self,attribute='db',*args,**kwargs)	

	
	def mean_residuals(self):
		ntwk_list=\
			[ ((self.error_ntwk.inv**self.measured[k])-self.ideals[k]) \
				for k in range(len(self.ideals))]
				
		return func_on_networks(ntwk_list, mean, 's_mag')
	
	def uncertainty_per_standard(self, std_names, attribute='s'):
		'''
		given that you have repeat-connections of single standards, 
		this calculates the complex standard deviation (distance) 
		for each standard in the calibration across connection #.
		
		takes:
			std_names: list of strings to uniquely identify each
				standard.* 
			attribute: string passed to func_on_networks to calculate 
				std deviation on a component if desired. ['s']
		
		returns:
			list of mwavepy.Networks, whose magnitude of s-parameters is
			proportional to the standard deviation for that standard
		
		
		*example:
			if your calibration had ideals named like:
				'short 1', 'short 2', 'open 1', 'open 2', etc. 
			you would pass this
				mycal.uncertainty_per_standard(['short','open','match'])
		
		'''
		return [fon([r for r in self.residual_ntwks \
			if std_name in r.name],std,attribute) \
			for std_name in std_names]
	
	def biased_error(self, std_names):
		'''
		estimate of biased error for overdetermined calibration with
		multiple connections of each standard
		
		takes:
			std_names: list of strings to uniquely identify each
				standard.* 
		returns:
			systematic error: mwavepy.Network type who's .s_mag is 
				proportional to the systematic error metric
		
		note:
			mathematically, this is 
				mean_s(|mean_c(r)|)
			where:
				r: complex residual errors
				mean_c: complex mean taken accross connection
				mean_s: complex mean taken accross standard
		'''
		biased_error= \
			fon([fon( [ntwk for ntwk in self.residual_ntwks \
				if ntwk.name==std_name],mean) \
				for std_name in std_names],mean, 's_mag')
		biased_error.name='biased error'
		return biased_error
	
	def unbiased_error(self, std_names):
		'''
		estimate of unbiased error for overdetermined calibration with
		multiple connections of each standard
		
		takes:
			std_names: list of strings to uniquely identify each
				standard.* 
		returns:
			stochastic error: mwavepy.Network type who's .s_mag is 
				proportional to the stochastic error metric
		
		see also:
			uncertainty_per_standard, for this a measure of unbiased 
			errors for each standard
			
		note:
			mathematically, this is 
				mean_s(std_c(r))
			where:
				r: complex residual errors
				std_c: standard deviation taken accross  connections
				mean_s: complex mean taken accross  standards
		'''
		unbiased_error= \
			fon([fon( [ntwk for ntwk in self.residual_ntwks \
				if ntwk.name==std_name],std) \
				for std_name in std_names],mean)
		unbiased_error.name = 'unbiased error'
		return unbiased_error
		
	def total_error(self, std_names):
		'''
		estimate of total error for overdetermined calibration with
		multiple connections of each standard. This is the combined 
		effects of both biased and un-biased errors
		
		takes:
			std_names: list of strings to uniquely identify each
				standard.* 
		returns:
			composit error: mwavepy.Network type who's .s_mag is 
				proportional to the composit error metric
		
		note:
			mathematically, this is 
				std_cs(r)
			where:
				r: complex residual errors
				std_cs: standard deviation taken accross connections
					and standards
		'''	
		total_error= \
			fon([ntwk for ntwk in self.residual_ntwks],mean,'s_mag') 
		total_error.name='total error'
		return total_error

## Functions	
def two_port_error_vector_2_Ts(error_coefficients):
	ec = error_coefficients
	npoints = len(ec['k'])
	one = npy.ones(npoints,dtype=complex)
	zero = npy.zeros(npoints,dtype=complex)
	#T_1 = npy.zeros((npoints, 2,2),dtype=complex)
	#T_1[:,0,0],T_1[:,1,1] = -1*ec['det_X'], -1*ec['k']*ec['det_Y']
	#T_1[:,1,1] = -1*ec['k']*ec['det_Y']


	T1 = npy.array([\
		[	-1*ec['det_X'], zero	],\
		[	zero,		-1*ec['k']*ec['det_Y']]]).transpose().reshape(-1,2,2)
	T2 = npy.array([\
		[	ec['e00'], zero	],\
		[	zero,			ec['k']*ec['e33']]]).transpose().reshape(-1,2,2)
	T3 = npy.array([\
		[	-1*ec['e11'], zero	],\
		[	zero,			-1*ec['k']*ec['e22']]]).transpose().reshape(-1,2,2)
	T4 = npy.array([\
		[	one, zero	],\
		[	zero,			ec['k']]]).transpose().reshape(-1,2,2)
	return T1,T2,T3,T4
	
def error_dict_2_network(coefs, frequency=None, is_reciprocal=False, **kwargs):
		'''
		convert a dictionary holding standard error terms to a Network
		object. 
		
		takes:
		
		returns:
		

		'''
		
		if len (coefs.keys()) == 3:
			# ASSERT: we have one port data
			ntwk = Network(**kwargs)
			
			if frequency is not None:
				ntwk.frequency = frequency
				
			if is_reciprocal:
				#TODO: make this better and maybe have a phase continuity
				# functionality
				tracking  = coefs['reflection tracking'] 
				s12 = npy.sqrt(tracking)
				s21 = npy.sqrt(tracking)
				
			else:
				s21 = coefs['reflection tracking'] 
				s12 = npy.ones(len(s21), dtype=complex)
			
			s11 = coefs['directivity'] 
			s22 = coefs['source match']
			ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
			return ntwk
		else:
			raise NotImplementedError('sorry')

