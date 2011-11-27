
#       networkSet.py
#       
#       
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
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
Provides the Network Set class, used for statistics and ploting for 
network sets.
'''


from network import average as network_average

from copy import deepcopy
import numpy as npy


class NetworkSet(object):
	'''
	A set of Networks.
	 
	This class is used to consolidate frequently called functions 
	on Network sets, such as mean or std. 
	'''
	
	def __init__(self, ntwk_set):
		'''
		input:
			ntwk_set: a list of Network's.
		returns:
			a NetworkSet type
		'''
		self.ntwk_set = ntwk_set
		
		for network_property_name in ['s','s_re','s_im','s_mag','s_deg']:
			for func in [npy.mean, npy.std]:
				self.add_a_property(network_property_name, func)
		
	def __str__(self):
		'''
		'''
		output =  \
			'A NetworkSet of length %i'%len(self.ntwk_set)

		return output
	def __repr__(self):
		return self.__str__()
	
	def add_a_property(self,network_property_name,func):
		'''
		dynamically adds a property to this class (NetworkSet)
		
		this is mostly used internally to genrate all of the classes 
		properties
		'''
		fget = lambda self: fon(self.ntwk_set,func,network_property_name)
		setattr(self.__class__,func.__name__+'_'+network_property_name,\
			property(fget))
	
	
	
	def func_on(self, func, a_property, *args, **kwargs):
		'''
		calls a function on a specific property of the networks in 
		this NetworkSet.
		'''
		return fon(self.ntwk_set, func, a_property, *args, **kwargs)
	
	# plotting functions
	def plot_uncertainty_bounds(self,attribute='s_mag',m=0,n=0,\
		n_deviations=3, alpha=.3,fill_color ='b',std_attribute=None,*args,**kwargs):
		'''
		plots mean value with +- uncertainty bounds in an Network attribute,
		for a list of Networks. 
		
		takes:
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
		ntwk_mean = average(self.ntwk_set)
		if std_attribute is None:
			# they want to calculate teh std deviation on a different attribute
			std_attribute = attribute
		ntwk_std = func_on_networks(self.ntwk_set,npy.std, attribute=std_attribute)
		
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
			lower_bound.squeeze(),upper_bound.squeeze(), alpha=alpha, color=fill_color)
		plb.axis('tight')
		plb.draw()

	def plot_uncertainty_bounds_s_re(self,*args, **kwargs):
		'''
		this just calls 
			plot_uncertainty_bounds(attribute= 's_re',*args,**kwargs)
		see plot_uncertainty_bounds for help
		
		'''
		kwargs.update({'attribute':'s_re'})
		plot_uncertainty_bounds(*args,**kwargs)
	
	def plot_uncertainty_bounds_s_im(self,*args, **kwargs):
		'''
		this just calls 
			plot_uncertainty_bounds(attribute= 's_im',*args,**kwargs)
		see plot_uncertainty_bounds for help
		
		'''
		kwargs.update({'attribute':'s_im'})
		plot_uncertainty_bounds(*args,**kwargs)
	
	def plot_uncertainty_bounds_s_mag(self,*args, **kwargs):
		'''
		this just calls 
			plot_uncertainty_bounds(attribute= 's_mag',*args,**kwargs)
		see plot_uncertainty_bounds for help
		
		'''
		kwargs.update({'attribute':'s_mag'})
		plot_uncertainty_bounds(*args,**kwargs)
		
	def plot_uncertainty_bounds_s_deg(self,*args, **kwargs):
		'''
		this just calls 
			plot_uncertainty_bounds(attribute= 's_deg_unwrap',*args,**kwargs)
		see plot_uncertainty_bounds for help
		
		note; the attribute 's_deg_unwrap' is called on purpose to alleviate
		the phase wraping effects on std dev. if you DO want to look at 
		's_deg' and not 's_deg_unwrap' then use plot_uncertainty_bounds
		
		'''
		kwargs.update({'attribute':'s_deg_unwrap'})
		plot_uncertainty_bounds(*args,**kwargs)
	
	def plot_uncertainty_bounds_s_db(self,attribute='s_mag',m=0,n=0,\
		n_deviations=3, alpha=.3,fill_color ='b',*args,**kwargs):
		'''
		plots mean value with +- uncertainty bounds in an Network's attribute
		for a list of Networks.
	
		This is plotted on a log scale (db), but uncertainty is calculated 
		in the linear domain
	
		takes:
			self.ntwk_set: list of Netmwork types [list]
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
		ntwk_mean = average(self.ntwk_set)
		ntwk_std = func_on_networks(self.ntwk_set,npy.std, attribute=attribute)
		
		# pull out port of interest
		ntwk_mean.s = ntwk_mean.s[:,m,n]
		ntwk_std.s = ntwk_std.s[:,m,n]
		
		# create bounds (the s_mag here is confusing but is realy in units
		# of whatever 'attribute' is. read the func_on_networks call to understand
		upper_bound =  ntwk_mean.__getattribute__(attribute) +\
			ntwk_std.s_mag*n_deviations
		lower_bound =   ntwk_mean.__getattribute__(attribute) -\
			ntwk_std.s_mag*n_deviations
		
		#convert to dB
		upper_bound_db, lower_bound_db = \
			mf.magnitude_2_db(upper_bound),mf.magnitude_2_db(lower_bound)
		
		# find the correct ploting method
		plot_func = ntwk_mean.plot_s_db
		
		#plot mean response
		plot_func(*args,**kwargs)
		
		#plot bounds
		plb.fill_between(ntwk_mean.frequency.f_scaled, \
			lower_bound_db.squeeze(),upper_bound_db.squeeze(), alpha=alpha, color=fill_color)
		plb.axis('tight')
		plb.draw()
		
	

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

# short hand name for convinnce
fon = func_on_networks
