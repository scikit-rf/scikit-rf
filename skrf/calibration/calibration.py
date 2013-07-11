
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
.. module:: skrf.calibration.calibration
================================================================
calibration (:mod:`skrf.calibration.calibration`)
================================================================


Contains the Calibration class, and supporting functions

Calibration Class
==================

.. autosummary::
   :toctree: generated/

   Calibration

'''
import numpy as npy
from numpy import linalg
from numpy import mean, std
import pylab as plb
import os
from copy import deepcopy, copy
import itertools
from warnings import warn
import cPickle as pickle

from calibrationAlgorithms import *
from ..mathFunctions import complex_2_db, sqrt_phase_unwrap, find_correct_sign
from ..frequency import *
from ..network import *
from ..networkSet import func_on_networks as fon
from ..networkSet import NetworkSet, s_dict_to_ns


## later imports. delayed to solve circular dependencies
#from io.general import write
#from io.general import read_all_networks


## main class
class Calibration(object):
    '''
    An object to represent a VNA calibration instance.

    A Calibration object is used to perform a calibration given a
    set meaurements and ideals responses. It can run a calibration,
    store results, and apply the results to calculate corrected
    measurements.

    '''
    calibration_algorithm_dict={\
            'one port': one_port,\
            'one port nls': one_port_nls,\
            'one port parametric':parameterized_self_calibration,\
            'one port parametric bounded':parameterized_self_calibration_bounded,\
            'two port': two_port,\
            '8-term': two_port,\
            'two port parametric':parameterized_self_calibration,\
            }
    '''
    dictionary holding calibration algorithms.

    See Also
    ---------
            :mod:`skrf.calibration.calibrationAlgorithms`
    '''

    def __init__(self,measured, ideals, type=None, \
            is_reciprocal=False,name=None, sloppy_input=False,switch_terms=None, 
            **kwargs):
        '''
        Calibration initializer.

        Parameters
        ----------
        measured : list of :class:`~skrf.network.Network` objects
                Raw measurements of the calibration standards. The order
                must align with the `ideals` parameter ( or use sloppy_input)

        ideals : list of :class:`~skrf.network.Network` objects
                Predicted ideal response of the calibration standards.
                The order must align with `ideals` list ( or use sloppy_input
        
        
        Other Parameters
        -----------------
        type : string
                the calibration algorithm. If `None`, the class will inspect
                number of ports on first `measured` Network and choose either
                `'one port'` or `'two port'`. See Notes_ section for more
                infor


        is_reciprocal : Boolean
                enables the reciprocity assumption on the calculation of the
                error_network, which is only relevant for one-port
                calibrations.

        switch_terms : tuple of :class:`~skrf.network.Network` objects
                The two measured switch terms in the order
                (forward, reverse).  This is only applicable in two-port
                calibrations. See Roger Mark's paper on switch terms [#]_
                for explanation of what they are, and [#]_ for description
                of measuring them on the Agilent PNA.

        name: string
                the name of calibration, just for your
                        convenience [None].

        sloppy_input :  Boolean.
                Allows ideals and measured lists to be 'aligned' based on
                the network names.

        \*\*kwargs : key-word arguments
                passed to the calibration algorithm, defined by `type`

        Notes
        -------
        All calibration algorithms are in stored in
        :mod:`skrf.calibration.calibrationAlgorithms` , refer to that
        file for documentation on the algorithms themselves. The
        Calibration class accesses those functions through the attribute
        `'calibration_algorihtm_dict'`.
        
        It is not required that the  measured Networks have the same 
        frequency objects as the ideals, but the measured frequency 
        must be a subset, so that the ideals can be interpolated.
         

        Examples
        ----------
        See the :doc:`../../../tutorials/calibration` tutorial, or the
        examples sections for :doc:`../../../examples/oneport_calibration`
        and :doc:`../../../examples/twoport_calibration`

        References
        -------------
        .. [#] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931
        
        .. [#] http://tech.groups.yahoo.com/group/mtt-11-on-wafer/message/283
        '''
        # allow them to pass di
        if hasattr(measured, 'keys'):
            measured = measured.values()
            sloppy_input = True
            warn('dictionary passed, sloppy_input automatically activated')
        if hasattr(ideals, 'keys'):
            ideals = ideals.values()
            sloppy_input = True
            warn('dictionary passed, sloppy_input automatically activated')
        
        # fill measured and ideals with copied lists of input     
        self.measured = [ntwk.copy() for ntwk in measured]
        self.ideals = [ntwk.copy() for ntwk in ideals]
        
        if sloppy_input:
            self.measured, self.ideals = \
                align_measured_ideals(self.measured, self.ideals)
        
        if len(self.measured) != len(self.ideals):
            raise(IndexError('The length of measured and ideals lists are different. Number of ideals must equal the number of measured.'))
        
        
        # ensure all the measured Networks' frequency's are the same
        for measure in self.measured:
            if self.measured[0].frequency != measure.frequency:
                raise(ValueError('measured Networks dont have matching frequencies.'))
        # ensure that all ideals have same frequency of the measured
        # if not, then attempt to interpolate
        for k in range(len(self.ideals)):
            if self.ideals[k].frequency != self.measured[0]:
                print('Warning: Frequency information doesnt match on ideals[%i], attempting to interpolate the ideal[%i] Network ..'%(k,k)),
                try:
                    # try to resample our ideals network to match
                    # the meaurement frequency
                    self.ideals[k].interpolate_self(\
                        self.measured[0].frequency)
                    print ('Success')
                    
                except:
                    raise(IndexError('Failed to interpolate. Check frequency of ideals[%i].'%k))
        
        
        self.frequency = measured[0].frequency.copy()
        self.type = type

        # passed to calibration algorithm in run()
        self.kwargs = kwargs 
        self.name = name
        self.is_reciprocal = is_reciprocal
        self.switch_terms = switch_terms
        
        
        # initialized internal properties to None
        self._residual_ntwks = None
        self._caled_ntwks =None
        self._caled_ntwk_sets = None
    
    def __str__(self):
        if self.name is None:
            name = ''
        else:
            name = self.name
            
        output = '%s Calibration: \'%s\', %s, %i-ideals/%i-measured'\
            %(self.type,name,str(self.measured[0].frequency),\
            len(self.ideals), len(self.measured))
            
        return output
        
    def __repr__(self):
        return self.__str__()    
        
    ## properties
    @property
    def type (self):
        '''
        string representing what type of calibration is to be
        performed. supported types at the moment are:

        'one port':     standard one-port cal. if more than
                2 measurement/ideal pairs are given it will
                calculate the least squares solution.

        'two port': two port calibration based on the error-box model


        note:
        algorithms referenced by  calibration_algorithm_dict, are stored
        in calibrationAlgorithms.py
        '''
        return self._type

    @type.setter
    def type(self, new_type):
        if new_type is None:
            # they did not supply a calibration type, so i will try
            # to inspect a measured ntwk to see how many ports it has
            if self.measured[0].number_of_ports == 1:
                new_type = 'one port'
            elif self.measured[0].number_of_ports == 2:
                new_type = 'two port'

        if new_type not in self.calibration_algorithm_dict.keys():
            raise ValueError('incorrect calibration type. Should be in:\n '+', '.join(self.calibration_algorithm_dict.keys()))


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
        '''
        number of ideal/measurement pairs in calibration
        '''
        if len(self.ideals) != len(self.measured):
            warn('number of ideals and measured dont agree')
        return len(self.ideals)

    @property
    def output_from_cal(self):
        '''
        a dictionary holding all of the output from the calibration
        algorithm
        '''
        try: 
            return( self._output_from_cal)
        except(AttributeError):
            self.run()
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
                TODO:
        '''
        return self.output_from_cal['error coefficients']
    @property
    def coefs_ntwks(self):
        '''
        
        for one port cal's
                'directivity':e00
                'reflection tracking':e01e10
                'source match':e11
        for 7-error term two port cal's
                TODO
        '''
        return s_dict_to_ns(self.output_from_cal['error coefficients'], self.frequency).ntwk_set
        
    @property
    def coefs_ntwks_2p(self):
        '''
        coefs: a dictionary holding the calibration coefficients

        for one port cal's
                'directivity':e00
                'reflection tracking':e01e10
                'source match':e11
        for 7-error term two port cal's
                TODO
        '''
        if self.nports !=2:
            raise ValueError('Only defined for 2-ports')
            
        return (s_dict_to_ns(eight_term_2_one_port_coefs(self.coefs)[0], self.frequency).ntwk_set,
        s_dict_to_ns(eight_term_2_one_port_coefs(self.coefs)[1], self.frequency).ntwk_set)
        
        
    @property
    def residuals(self):
        '''
        if calibration is overdeteremined, this holds the residuals
        in the form of a vector.

        also available are the complex residuals in the form
        of skrf.Network's, see the property 'residual_ntwks'

        from numpy.lstsq:
                residues:
                the sum of the residues; squared euclidean norm for
                each column vector in b (given ax=b)

        '''
        return self.output_from_cal['residuals']

    @property
    def error_ntwk(self):
        '''
        A Network object which represents the error network being
        calibrated out.
        '''
        if self.nports == 1 or self.nports ==2:
            try: 
                return( self._error_ntwk)
            except(AttributeError):
                self.run()
                return self._error_ntwk
                

        else:
            raise NotImplementedError()
            
    @property
    def Ts(self):
        '''
        T-matricies used for de-embeding, a two-port calibration.
        '''

        if self.nports == 2:
            try:
                return self._Ts
            except(AttributeError):
                self.run()
                return self._Ts
        else:
            raise AttributeError('Only defined for 2-port cals')
        

    @property
    def residual_ntwks(self):
        '''
        returns a the residuals for each calibration standard in the
        form of a list of Network types.

        these residuals are calculated in the 'calibrated domain',
        meaning they are
                r = (E.inv ** m - i)

        where,
                r: residual network,
                E: embedding network,
                m: measured network
                i: ideal network

        This way the units of the residual networks are meaningful


        note:
                the residuals are only calculated if they are not existent.
        so, if you want to re-calculate the residual networks then
        you delete the property '_residual_ntwks'.
        '''
        if self._residual_ntwks is None:
            ntwk_list=\
                    [ ((self.apply_cal(self.measured[k]))-self.ideals[k]) \
                            for k in range(len(self.ideals))]

            for k in range(len(ntwk_list)):
                if self.ideals[k].name  is not None:
                    name = self.ideals[k].name
                else:
                    name='std# %i'%k

                ntwk_list[k].name = self.ideals[k].name

            self._residual_ntwks = ntwk_list
        return self._residual_ntwks

    @property
    def caled_ntwks(self):
        '''
        list of the calibrated, calibration standards.


        '''
        if self._caled_ntwks is None:
            ntwk_list=\
                    [ self.apply_cal(self.measured[k]) \
                            for k in range(len(self.ideals))]
            
            for k in range(len(ntwk_list)):
                if self.ideals[k].name  is not None:
                    name = self.ideals[k].name
                else:
                    name='std# %i'%k

                ntwk_list[k].name = self.ideals[k].name

            self._caled_ntwks = ntwk_list
        
        return self._caled_ntwks
    
    @property
    def caled_ntwk_sets(self):
        '''
        returns a NetworkSet for each caled_ntwk, based on their names
        '''
        if self._caled_ntwk_sets is  None:
            caled_sets={}
            std_names = list(set([k.name  for k in self.caled_ntwks ]))
            for std_name in std_names:
                caled_sets[std_name] = NetworkSet(
                    [k for k in self.caled_ntwks if k.name is std_name])
            self._caled_ntwk_sets = caled_sets
        
        return self._caled_ntwk_sets

    
    ##  methods for manual control of internal calculations
    def run(self):
        '''
        runs the calibration algorihtm.

        this is automatically called the first time     any dependent
        property is referenced (like error_ntwk), but only the first
        time. if you change something and want to re-run the calibration
         use this.
        '''
        # actually call the algorithm and run the calibration
        if self.switch_terms is not None:
            self.kwargs.update({'switch_terms':self.switch_terms})
        self._output_from_cal = \
                self.calibration_algorithm_dict[self.type](
                    measured = self.measured, 
                    ideals = self.ideals,
                    **self.kwargs)

        if self.nports ==1:
            self._error_ntwk = error_dict_2_network(self.coefs, \
                    frequency=self.frequency, is_reciprocal=self.is_reciprocal)
            self._error_ntwk.name= self.name
        elif self.nports ==2:
            self._Ts = two_port_error_vector_2_Ts(self.coefs)
            self._error_ntwk = error_dict_2_network(self.coefs, \
                    frequency=self.frequency, is_reciprocal=self.is_reciprocal)
            for k in self._error_ntwk:
                k.name = self.name
        
        
        #reset the residuals
        self._residual_ntwks = None

    def remove_std(self,prefix):
        for ideal, measured in zip(self.ideals, self.measured):
            if prefix in ideal.name:
                self.ideals.remove(ideal)
                self.measured.remove(measured)
    
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
        if isinstance(input_ntwk,list):
            # if they pass a list of networks, look through them and 
            # calibrate all
            return  [self.apply_cal(ntwk) for  ntwk in input_ntwk]

        else:
            if self.nports ==1:
                caled =  self.error_ntwk.inv**input_ntwk
                caled.name = input_ntwk.name

            elif self.nports == 2:
                caled = input_ntwk.copy()
                if self.switch_terms is not None:
                    
                    input_ntwk = input_ntwk.copy()
                    intput_ntwk=unterminate_switch_terms(input_ntwk, 
                        self.switch_terms[0], self.switch_terms[1])
                    
                
                T1,T2,T3,T4 = self.Ts
                dot = npy.dot
                for f in range(len(input_ntwk.s)):
                    t1,t2,t3,t4,m = T1[f,:,:],T2[f,:,:],T3[f,:,:],\
                            T4[f,:,:],input_ntwk.s[f,:,:]
                    caled.s[f,:,:] = dot(npy.linalg.inv(-1*dot(m,t3)+t1),(dot(m,t4)-t2))
            return caled

    def apply_cal_to_all_in_dir(self, dir='.', contains=None, f_unit = 'ghz'):
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
        from ..io.general import read_all_networks
        ntwkDict = read_all_networks(dir=dir, contains=contains,\
                f_unit=f_unit)

        for ntwkKey in ntwkDict:
            ntwkDict[ntwkKey] = self.apply_cal(ntwkDict[ntwkKey])

        return ntwkDict

    ## error metrics and related functions
    def mean_residuals(self):
        '''
        '''
        return func_on_networks(self.residual_ntwks, mean, 's_mag')

    def uncertainty_per_standard(self, std_names=None, attribute='s'):
        '''
        given that you have repeat-connections of single standard,
        this calculates the complex standard deviation (distance)
        for each standard in the calibration across connection #.

        takes:
                std_names: list of strings to uniquely identify each
                        standard.*
                attribute: string passed to func_on_networks to calculate
                        std deviation on a component if desired. ['s']

        returns:
                list of skrf.Networks, whose magnitude of s-parameters is
                proportional to the standard deviation for that standard


        *example:
                if your calibration had ideals named like:
                        'short 1', 'short 2', 'open 1', 'open 2', etc.
                you would pass this
                        mycal.uncertainty_per_standard(['short','open','match'])

        '''
        if std_names is None:
            std_names = set([ntwk.name for ntwk in self.ideals])
        return [fon([r for r in self.residual_ntwks \
                if std_name in r.name],std,attribute) \
                for std_name in std_names]

    def func_per_standard(self, func,attribute='s',std_names=None):
        if std_names is None:
            std_names = set([ntwk.name for ntwk in self.ideals])
        return [fon([r for r in self.residual_ntwks \
                if std_name in r.name],func,attribute) \
                for std_name in std_names]

    def biased_error(self, std_names=None):
        '''
        estimate of biased error for overdetermined calibration with
        multiple connections of each standard

        takes:
                std_names: list of strings to uniquely identify each
                        standard.*
        returns:
                systematic error: skrf.Network type who's .s_mag is
                        proportional to the systematic error metric

        note:
                mathematically, this is
                        mean_s(|mean_c(r)|)
                where:
                        r: complex residual errors
                        mean_c: complex mean taken accross connection
                        mean_s: complex mean taken accross standard
        '''
        if std_names is None:
            std_names = set([ntwk.name for ntwk in self.ideals])
        biased_error= \
                fon([fon( [ntwk for ntwk in self.residual_ntwks \
                        if std_name in ntwk.name],mean) \
                        for std_name in std_names],mean, 's_mag')
        biased_error.name='biased error'
        return biased_error

    def unbiased_error(self, std_names=None):
        '''
        estimate of unbiased error for overdetermined calibration with
        multiple connections of each standard

        takes:
                std_names: list of strings to uniquely identify each
                        standard.*
        returns:
                stochastic error: skrf.Network type who's .s_mag is
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
        if std_names is None:
            std_names = set([ntwk.name for ntwk in self.ideals])
        unbiased_error= \
                fon([fon( [ntwk for ntwk in self.residual_ntwks \
                        if std_name in ntwk.name],std) \
                        for std_name in std_names],mean)
        unbiased_error.name = 'unbiased error'
        return unbiased_error

    def total_error(self, std_names=None):
        '''
        estimate of total error for overdetermined calibration with
        multiple connections of each standard. This is the combined
        effects of both biased and un-biased errors

        takes:
                std_names: list of strings to uniquely identify each
                        standard.*
        returns:
                composit error: skrf.Network type who's .s_mag is
                        proportional to the composit error metric

        note:
                mathematically, this is
                        std_cs(r)
                where:
                        r: complex residual errors
                        std_cs: standard deviation taken accross connections
                                and standards
        '''
        if std_names is None:
            std_names = set([ntwk.name for ntwk in self.ideals])
        total_error= \
                fon([ntwk for ntwk in self.residual_ntwks],mean,'s_mag')
        total_error.name='total error'
        return total_error

    ## ploting
    def plot_coefs(self,attr='s_db',port=None, *args, **kwargs):
        '''
        plot magnitude of the error coeficient dictionary
        '''

        # plot the desired attribute vs frequency
        if self.nports == 1:
            ns = NetworkSet(self.coefs_ntwks)
        elif self.nports == 2:
            if port is  None:
                ns = NetworkSet(self.coefs_ntwks)
            elif port == 1:
                ns = NetworkSet(self.coefs_ntwks_2p[0])
            elif port == 2:
                ns = NetworkSet(self.coefs_ntwks_2p[1])
            else:
                raise(ValueError())
        else:
            raise NotImplementedError()
        
        return ns.__getattribute__('plot_'+attr)(*args, **kwargs)            
               
        
    def plot_coefs_db(self, *args, **kwargs):
        return self.plot_coefs(attr='s_db',*args, **kwargs)
            
                
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
                (self.apply_cal(self.measured[k])-self.ideals[k])

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

    def plot_errors(self,std_names =None, scale='db', *args, **kwargs):
        '''
        plot calibration error metrics for an over-determined calibration.

        see biased_error, unbiased_error, and total_error for more info

        '''
        if scale == 'lin':
            self.biased_error(std_names).plot_s_mag(*args, **kwargs)
            self.unbiased_error(std_names).plot_s_mag(*args, **kwargs)
            self.total_error(std_names).plot_s_mag(*args, **kwargs)
            plb.ylabel('Mean Distance (linear)')
        elif scale == 'db':
            self.biased_error(std_names).plot_s_db(*args, **kwargs)
            self.unbiased_error(std_names).plot_s_db(*args, **kwargs)
            self.total_error(std_names).plot_s_db(*args, **kwargs)
            plb.ylabel('Mean Distance (dB)')
        plb.title('Error Metrics')

    def plot_uncertainty_per_standard(self, scale='db',*args, **kwargs):
        '''
        Plots uncertainty associated with each calibration standard.
        
        This requires that each calibration standard is measured 
        multiple times. The uncertainty associated with each 
        standard is calculated by the complex standard deviation. 
       
        
        
        Parameters
        ------------
        scale : 'db', 'lin'
            plot uncertainties on linear or log scale
        \\*args, \\*\\*kwargs : passed to :func:`uncertainty_per_standard`
        
        See Also
        ----------
        :func:`uncertainty_per_standard`
        '''
        plb.title('Uncertainty Per Standard')
        if scale=='lin':
            [ntwk.plot_s_mag() for ntwk in \
                self.uncertainty_per_standard(*args, **kwargs)]
            plb.ylabel('Standard Deviation (linear)')
        elif scale=='db':

            [ntwk.plot_s_db() for ntwk in \
                self.uncertainty_per_standard(*args, **kwargs)]
            plb.ylabel('Standard Deviation (dB)')
    
    def plot_caled_ntwks(self,attr='s_smith',*args, **kwargs):
        '''
        Plot specified parameters the :`caled_ntwks`.
        
        Parameters
        -----------
        attr : str
            plotable attribute of a Network object. ie 's_db', 's_smith'
        
        \*args, \*\*kwargs : 
            passed to the plotting method
        
        '''
        [k.__getattribute__('plot_%s'%attr)(*args, **kwargs) \
            for k in self.caled_ntwks]
    
    def plot_caled_ntwk_sets(self, attr='s_db', *args, **kwargs):
        '''
        plots calibrated network sets with uncertainty bounds.
        
        For use with redundantly measured calibration standards.  
        
        Parameters
        -----------
        attr : str
            plotable uncertainty bounds attribute of a NetworkSet object.
            ie 's_db', 's_deg'
        
        \*args, \*\*kwargs : 
            passed to the plotting method
        '''
        [k.__getattribute__('plot_uncertainty_bounds_%s'%attr)(*args, **kwargs) \
            for k in self.caled_ntwk_sets.values()]
        
    
    # io
    def write(self, file=None,  *args, **kwargs):
        '''
        Write the Calibration to disk using :func:`~skrf.io.general.write`
        
        
        Parameters
        -----------
        file : str or file-object
            filename or a file-object. If left as None then the 
            filename will be set to Calibration.name, if its not None. 
            If both are None, ValueError is raised.
        \*args, \*\*kwargs : arguments and keyword arguments
            passed through to :func:`~skrf.io.general.write`
        
        Notes
        ------
        If the self.name is not None and file is  can left as None
        and the resultant file will have the `.ntwk` extension appended
        to the filename. 
        
        Examples
        ---------
        >>> cal.name = 'my_cal'
        >>> cal.write()
        
        See Also
        ---------
        skrf.io.general.write
        skrf.io.general.read
        
        '''
        # this import is delayed untill here because of a circular depency
        from ..io.general import write
        
        if file is None:
            if self.name is None:
                 raise (ValueError('No filename given. You must provide a filename, or set the name attribute'))
            file = self.name

        write(file,self, *args, **kwargs) 



    
class Calibration2(object):
    def __init__(self, measured, ideals, sloppy_input=False,
        is_reciprocal=True,name=None,*args, **kwargs):
        '''
        Calibration initializer.

        Parameters
        ----------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use sloppy_input)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input
        '''
        
        # allow them to pass di
        if hasattr(measured, 'keys'):
            measured = measured.values()
            if sloppy_input == False:
                warn('dictionary passed, sloppy_input automatically activated')
                sloppy_input = True
            
        if hasattr(ideals, 'keys'):
            ideals = ideals.values()
            if sloppy_input == False:
                warn('dictionary passed, sloppy_input automatically activated')
                sloppy_input = True
        
               
        # fill measured and ideals with copied lists of input     
        self.measured = [ntwk.copy() for ntwk in measured]
        self.ideals = [ntwk.copy() for ntwk in ideals]
        
        if sloppy_input:
            self.measured, self.ideals = \
                align_measured_ideals(self.measured, self.ideals)
        
        if len(self.measured) != len(self.ideals):
            raise(IndexError('The length of measured and ideals lists are different. Number of ideals must equal the number of measured.'))
        
        
        # ensure all the measured Networks' frequency's are the same
        for measure in self.measured:
            if self.measured[0].frequency != measure.frequency:
                raise(ValueError('measured Networks dont have matching frequencies.'))
        # ensure that all ideals have same frequency of the measured
        # if not, then attempt to interpolate
        for k in range(len(self.ideals)):
            if self.ideals[k].frequency != self.measured[0]:
                print('Warning: Frequency information doesnt match on ideals[%i], attempting to interpolate the ideal[%i] Network ..'%(k,k)),
                try:
                    # try to resample our ideals network to match
                    # the meaurement frequency
                    self.ideals[k].interpolate_self(\
                        self.measured[0].frequency)
                    print ('Success')
                    
                except:
                    raise(IndexError('Failed to interpolate. Check frequency of ideals[%i].'%k))
    
    
        # passed to calibration algorithm in run()
        self.kwargs = kwargs 
        self.name = name
        
        self.is_reciprocal = is_reciprocal
        
        # initialized internal properties to None
        self._residual_ntwks = None
        self._caled_ntwks =None
        self._caled_ntwk_sets = None
    
    def __str__(self):
        if self.name is None:
            name = ''
        else:
            name = self.name
            
        output = '%s Calibration: \'%s\', %s, %i-ideals/%i-measured'\
            %(self.type,name,str(self.measured[0].frequency),\
            len(self.ideals), len(self.measured))
            
        return output
        
    def __repr__(self):
        return self.__str__()    
        
    def run(self):
        '''
        Runs the calibration algorithm.
        '''
        raise NotImplementedError('The Subclass must implement this')
    
    def apply_cal(self,ntwk):
        '''
        Apply correction to a Network
        '''
        raise NotImplementedError('The Subclass must implement this')
    
    def apply_cal_to_list(self,ntwk_list):
        '''
        Apply correction to list of dict of Networks.
        '''
        if hasattr(ntwk_list, 'keys'):
            return dict([(k, self.apply_cal(ntwk_list[k])) for k in ntwk_list])
        else:
            return [self.apply_cal(k) for k in ntwk_list]
        
    def apply_cal_to_all_in_dir(self, *args, **kwargs):
        '''
        '''
        
        from ..io.general import read_all_networks
        ntwkDict = read_all_networks(*args, **kwargs)
        return self.apply_cal_to_list(ntwkDict)
    
    def embed(self,ntwk):
        '''
        Embed an ideal response in the estimated error network[s]
        '''
        raise NotImplementedError('The Subclass must implement this')
        
    def pop(self,index=-1):
        '''
        Remove and return tuple of (ideal, measured) at index.
        '''
        return (self.ideals.pop(index),  self.measured.pop(index))
    
        
    
    @property
    def frequency(self):
        return self.measured[0].frequency.copy()
    
    @property
    def nstandards(self):
        '''
        number of ideal/measurement pairs in calibration
        '''
        if len(self.ideals) != len(self.measured):
            warn('number of ideals and measured dont agree')
        return len(self.ideals)
        
    @property
    def coefs(self):
        '''
        '''
        try:
            return self._coefs
        except(AttributeError):
            self.run()
            return self._coefs
    
    
    @property
    def output_from_run(self):
        try:
            return self._output_from_run
        except(AttributeError):
            # maybe i havent run yet
            self.run()
            try:
                return self._output_from_run
            except(AttributeError):
                # i did run and there is no output_from_run
                return None
        
    @property
    def coefs_ntwks(self):
        '''
        '''
        return s_dict_to_ns(self.coefs, self.frequency).to_dict()
    
    @property
    def coefs_3term(self):
        '''
        '''
        
    @property
    def coefs_8term(self):
        return dict([(k, self.coefs.get(k)) for k in [\
            'forward directivity',
            'forward source match',
            'forward reflection tracking',
            
            'reverse directivity',
            'reverse load match',
            'reverse reflection tracking',
                        
            'forward switch term',
            'reverse switch term',
            'k'
            ]])
    
    @property 
    def coefs_12term(self):
        return dict([(k, self.coefs.get(k)) for k in [\
            'forward directivity',
            'forward source match',
            'forward reflection tracking',
            'forward transmission tracking',
            'forward load match',

            'reverse directivity',
            'reverse load match',
            'reverse reflection tracking',
            'reverse transmission tracking',
            'reverse source match',
            ]])
    
    
    
    @property
    def verify_12term(self):
        '''
        '''
        
        Edf = self.coefs_12term['forward directivity']
        Esf = self.coefs_12term['forward source match']
        Erf = self.coefs_12term['forward reflection tracking']
        Etf = self.coefs_12term['forward transmission tracking']
        Elf = self.coefs_12term['forward load match']
        
        Edr = self.coefs_12term['reverse directivity']
        Elr = self.coefs_12term['reverse load match']
        Err = self.coefs_12term['reverse reflection tracking']
        Etr = self.coefs_12term['reverse transmission tracking']
        Esr = self.coefs_12term['reverse source match']
        
        
        return Etf*Etr - (Err + Edr*(Elf - Esr))*(Erf  + Edf *(Elr - Esf))    
    
    @property
    def verify_12term_ntwk(self):
        return Network(s= self.verify_12term, frequency = self.frequency)
        
    @property
    def residual_ntwks(self):
        '''
        Returns a the residuals for each calibration standard

        These residuals are complex differences between the ideal 
        standards and their corresponding  corrected measurements. 
        
        '''
        return [ideal - caled for (ideal, caled) in zip(self.ideals, self.caled_ntwks)]

    @property
    def caled_ntwks(self):
        '''
        List of the corrected calibration standards
        '''
        return self.apply_cal_to_list(self.measured)
    
        
    @property
    def caled_ntwk_sets(self):
        '''
        Returns a NetworkSet for each caled_ntwk, based on their names
        '''
       
        caled_sets={}
        std_names = list(set([k.name  for k in self.caled_ntwks ]))
        for std_name in std_names:
            caled_sets[std_name] = NetworkSet(
                [k for k in self.caled_ntwks if k.name is std_name])
        return caled_sets
       
    @property
    def error_ntwk(self):
        '''
        Returns the calculated two-port error Network or Networks
        '''
        return error_dict_2_network(
            self.coefs, 
            frequency = self.frequency,
            is_reciprocal= self.is_reciprocal)        
    
    def write(self, file=None,  *args, **kwargs):
        '''
        Write the Calibration to disk using :func:`~skrf.io.general.write`
        
        
        Parameters
        -----------
        file : str or file-object
            filename or a file-object. If left as None then the 
            filename will be set to Calibration.name, if its not None. 
            If both are None, ValueError is raised.
        \*args, \*\*kwargs : arguments and keyword arguments
            passed through to :func:`~skrf.io.general.write`
        
        Notes
        ------
        If the self.name is not None and file is  can left as None
        and the resultant file will have the `.ntwk` extension appended
        to the filename. 
        
        Examples
        ---------
        >>> cal.name = 'my_cal'
        >>> cal.write()
        
        See Also
        ---------
        skrf.io.general.write
        skrf.io.general.read
        
        '''
        # this import is delayed untill here because of a circular depency
        from ..io.general import write
        
        if file is None:
            if self.name is None:
                 raise (ValueError('No filename given. You must provide a filename, or set the name attribute'))
            file = self.name

        write(file,self, *args, **kwargs) 
  
class OnePort(Calibration2):
    '''
    Standard algorithm for a one port calibration.
    
    If more than three standards are supplied then a least square
    algorithm is applied.
    '''
    def __init__(self, measured, ideals,*args, **kwargs):
        '''
        One Port initializer
        
        If more than three standards are supplied then a least square
        algorithm is applied.
        
        Parameters
        -----------
        measured : list of :class:`~....network.Network` objects or numpy.ndarray
            a list of the measured reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.
        ideals : list of :class:`~skrf.network.Network` objects or numpy.ndarray
            a list of the ideal reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.
    
        Returns
        -----------
        output : a dictionary
            output information from the calibration, the keys are
             * 'error coeffcients': dictionary containing standard error
               coefficients
             * 'residuals': a matrix of residuals from the least squared
               calculation. see numpy.linalg.lstsq() for more info
    
    
        Notes
        -----
                uses numpy.linalg.lstsq() for least squares calculation
        '''
        self.type = 'OnePort'
        Calibration2.__init__(self, measured, ideals, *args, **kwargs)
    
    def run(self):
        '''
        '''
        numStds = self.nstandards
        numCoefs=3
        
        mList = [self.measured[k].s.reshape((-1,1)) for k in range(numStds)]
        iList = [self.ideals[k].s.reshape((-1,1)) for k in range(numStds)]
        
        # ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
        fLength = len(mList[0])
    
        #initialize outputs
        abc = npy.zeros((fLength,numCoefs),dtype=complex)
        residuals =     npy.zeros((fLength,\
                npy.sign(numStds-numCoefs)),dtype=complex)
        parameter_variance = npy.zeros((fLength, 3,3),dtype=complex)
        measurement_variance = npy.zeros((fLength, 1),dtype=complex)
        # loop through frequencies and form m, a vectors and
        # the matrix M. where M = i1, 1, i1*m1
        #                         i2, 1, i2*m2
        #                                 ...etc
        for f in range(fLength):
            #create  m, i, and 1 vectors
            one = npy.ones(shape=(numStds,1))
            m = npy.array([ mList[k][f] for k in range(numStds)]).reshape(-1,1)# m-vector at f
            i = npy.array([ iList[k][f] for k in range(numStds)]).reshape(-1,1)# i-vector at f
    
            # construct the matrix
            Q = npy.hstack([i, one, i*m])
            # calculate least squares
            abcTmp, residualsTmp = npy.linalg.lstsq(Q,m)[0:2]
            if numStds > 3:
                measurement_variance[f,:]= residualsTmp/(numStds-numCoefs)
                parameter_variance[f,:] = \
                        abs(measurement_variance[f,:])*\
                        npy.linalg.inv(npy.dot(Q.T,Q))
    
            
            abc[f,:] = abcTmp.flatten()
            try:
                residuals[f,:] = residualsTmp
            except(ValueError):
                raise(ValueError('matrix has singular values. ensure standards are far enough away on smith chart'))
        
        # convert the abc vector to standard error coefficients
        a,b,c = abc[:,0], abc[:,1],abc[:,2]
        e01e10 = a+b*c
        e00 = b
        e11 = c
        self._coefs = {\
                'directivity':e00,\
                'reflection tracking':e01e10, \
                'source match':e11\
                }
        
    
        # output is a dictionary of information
        self._output_from_run = {
            'residuals':residuals, 
            'parameter variance':parameter_variance
            }
        
        return None
            
    def apply_cal(self, ntwk):
        er_ntwk = Network(frequency = self.frequency, name=ntwk.name)
        tracking  = self.coefs['reflection tracking']
        s12 = npy.sqrt(tracking)
        s21 = npy.sqrt(tracking)

        s11 = self.coefs['directivity']
        s22 = self.coefs['source match']
        er_ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
        return er_ntwk.inv**ntwk
    
    def embed(self,ntwk):
        embedded = ntwk.copy()
        embedded = self.error_ntwk**embedded
        embedded.name = ntwk.name
        return embedded
        
class SOLT(Calibration2):
    '''
    Traditional 12-term, full two-port calibration.
    
    SOLT is the traditional, fully determined, two-port calibration. 
    This implementation is based off of Doug Rytting's work in [#] .
    Although the acronym SOLT implies the use of 4 standards, skrf's 
    algorithm can accept any number of reflect standards,  If  
    more than 3 reflect standards are provided a least-squares solution 
    is implemented for the one-port stage of the calibration.
    
    Redundant thru measurements can also be used, through the `n_thrus`
    parameter. See :func:`__init__`
     
    
    
    .. [#] "Network Analyzer Error Models and Calibration Methods" 
        by Doug Rytting
    
    '''
    def __init__(self, measured, ideals, n_thrus=1, *args, **kwargs):
        '''
        SOLT initializer 
        
        Parameters
        -------------
        measured : list or dict of :class:`Network` objects
            measured Networks. must align with ideals
            
        
        '''
        self.type = 'SOLT'
        self.n_thrus = n_thrus
        Calibration2.__init__(self, measured, ideals, *args, **kwargs)
    
    def run(self):
        '''
        '''
        n_thrus = self.n_thrus
        p1_m = [k.s11 for k in self.measured[:-n_thrus]]
        p2_m = [k.s22 for k in self.measured[:-n_thrus]]
        p1_i = [k.s11 for k in self.ideals[:-n_thrus]]
        p2_i = [k.s22 for k in self.ideals[:-n_thrus]]
        thru = NetworkSet(self.measured[-n_thrus:]).mean_s
        
        # create one port calibration for reflective standards  
        port1_cal = OnePort(measured = p1_m, ideals = p1_i)
        port2_cal = OnePort(measured = p2_m, ideals = p2_i)
        
        # cal coefficient dictionaries
        p1_coefs = port1_cal.coefs
        p2_coefs = port2_cal.coefs
        
        if self.kwargs.get('isolation',None) is not None:
            raise NotImplementedError()
            p1_coefs['isolation'] = isolation.s21.s.flatten()
            p2_coefs['isolation'] = isolation.s12.s.flatten()
        
        p1_coefs['load match'] = port1_cal.apply_cal(thru.s11).s.flatten()
        p2_coefs['load match'] = port2_cal.apply_cal(thru.s22).s.flatten()
        
        p1_coefs['transmission tracking'] = \
            (thru.s21.s.flatten() - p1_coefs.get('isolation',0))*\
            (1. - p1_coefs['source match']*p1_coefs['load match'])
        p2_coefs['transmission tracking'] = \
            (thru.s12.s.flatten() - p2_coefs.get('isolation',0))*\
            (1. - p2_coefs['source match']*p2_coefs['load match'])
        coefs = {}
        #import pdb;pdb.set_trace()
        coefs.update(dict([('forward %s'%k, p1_coefs[k]) for k in p1_coefs]))
        coefs.update(dict([('reverse %s'%k, p2_coefs[k]) for k in p2_coefs]))
        eight_term_coefs = convert_12term_2_8term(coefs)
        #import pdb;pdb.set_trace()
        coefs.update(dict([(l, eight_term_coefs[l]) for l in \
            ['forward switch term','reverse switch term','k'] ]))
        self._coefs = coefs
    
    def apply_cal(self,ntwk):
        '''
        '''
        caled = ntwk.copy()
        
        s11 = ntwk.s[:,0,0]
        s12 = ntwk.s[:,0,1]
        s21 = ntwk.s[:,1,0]
        s22 = ntwk.s[:,1,1]
        
        Edf = self.coefs['forward directivity']
        Esf = self.coefs['forward source match']
        Erf = self.coefs['forward reflection tracking']
        Etf = self.coefs['forward transmission tracking']
        Elf = self.coefs['forward load match']
        Eif = self.coefs.get('forward isolation',0)
        
        Edr = self.coefs['reverse directivity']
        Elr = self.coefs['reverse load match']
        Err = self.coefs['reverse reflection tracking']
        Etr = self.coefs['reverse transmission tracking']
        Esr = self.coefs['reverse source match']
        Eir = self.coefs.get('reverse isolation',0)
        
        
        D = (1+(s11-Edf)/(Erf)*Esf)*(1+(s22-Edr)/(Err)*Esr) -\
            ((s21-Eif)/(Etf))*((s12-Eir)/(Etr))*Elf*Elr
        
        
        caled.s[:,0,0] = \
            (((s11-Edf)/(Erf))*(1+(s22-Edr)/(Err)*Esr)-\
            Elf*((s21-Eif)/(Etf))*(s12-Eir)/(Etr)) /D
            
        caled.s[:,1,1] = \
            (((s22-Edr)/(Err))*(1+(s11-Edf)/(Erf)*Esf)-\
            Elr*((s21-Eif)/(Etf))*(s12-Eir)/(Etr)) /D
            
        caled.s[:,1,0] = \
            ( ((s21 -Eif)/(Etf))*(1+((s22-Edr)/(Err))*(Esr-Elf)) )/D
        
        caled.s[:,0,1] = \
            ( ((s12 -Eir)/(Etr))*(1+((s11-Edf)/(Erf))*(Esf-Elr)) )/D    
        
        return caled
    
    def embed(self, ntwk):
        measured = ntwk.copy()
        
        s11 = ntwk.s[:,0,0]
        s12 = ntwk.s[:,0,1]
        s21 = ntwk.s[:,1,0]
        s22 = ntwk.s[:,1,1]
        det = s11*s22 - s12*s21
        
        Edf = self.coefs['forward directivity']
        Esf = self.coefs['forward source match']
        Erf = self.coefs['forward reflection tracking']
        Etf = self.coefs['forward transmission tracking']
        Elf = self.coefs['forward load match']
        Eif = self.coefs.get('forward isolation',0)
        
        Edr = self.coefs['reverse directivity']
        Elr = self.coefs['reverse load match']
        Err = self.coefs['reverse reflection tracking']
        Etr = self.coefs['reverse transmission tracking']
        Esr = self.coefs['reverse source match']
        Eir = self.coefs.get('reverse isolation',0)
        
        
        measured = ntwk.copy()
        
        D1 = (1 - Esf*s11 - Elf*s22 + Esf*Elf*det)
        D2 = (1 - Elr*s11 - Esr*s22 + Esr*Elr*det)
        
        measured.s[:,0,0] =  Edf + Erf * (s11 - Elf*det)/D1 
        measured.s[:,1,0] =  Eif + Etf * s21/D1 
        measured.s[:,1,1] =  Edr + Err * (s22 - Elr*det)/D2 
        measured.s[:,0,1] =  Eir + Etr * s12/D2 
        
        return measured
        
class EightTerm(Calibration2):
    def __init__(self, measured, ideals, switch_terms=None,*args, **kwargs):
        '''
        Parameters
        --------------
        measured : 
        ideals : 
        switch_terms : tuple of :class:`~skrf.network.Network` objects
            the pair of switch terms in the order (forward, reverse)
            
        \*args, \*\*kwargs : 
            
            
        '''
        self.type = 'EightTerm'
        self.switch_terms = switch_terms
        if switch_terms is None:
            warn('No switch terms provided')
        Calibration2.__init__(self, 
            measured = measured, 
            ideals = ideals, 
            *args, **kwargs)
        
    def unterminate(self,ntwk):
        '''
        Unterminates switch terms from a raw measurement.
        
        In order to use the 8-term error model on a VNA which employs a 
        switched source, the effects of the switch must be accounted for. 
        This is done through `switch terms` as described in  [#]_ . The 
        two switch terms are defined as, 
        
        .. math :: 
            
            \\Gamma_f = \\frac{a2}{b2} ,\\qquad\\text{sourced by port 1}
            \\Gamma_r = \\frac{a1}{b1} ,\\qquad\\text{sourced by port 2}
        
        These can be measured by four-sampler VNA's by setting up 
        user-defined traces onboard the VNA. If the VNA doesnt have  
        4-samplers, then you can measure switch terms indirectly by using a 
        two-tier two-port calibration. Firts do a SOLT, then convert 
        the 12-term error coefs to 8-term, and pull out the switch terms.  
        
        Parameters
        -------------
        two_port : 2-port Network 
            the raw measurement
        gamma_f : 1-port Network
            the measured forward switch term. 
            gamma_f = a2/b2 sourced by port1
        gamma_r : 1-port Network
            the measured reverse switch term
        
        Returns
        -----------
        ntwk :  Network object
        
        References
        ------------
        
        .. [#] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks
        '''
        if self.switch_terms is not None:
            gamma_f, gamma_r = self.switch_terms
            
            unterminated = ntwk.copy()
            
            # extract scattering matrices
            m, gamma_r, gamma_f = ntwk.s, gamma_r.s, gamma_f.s
            u = m.copy()
            
            one = npy.ones(ntwk.frequency.npoints)
            
            d = one - m[:,0,1]*m[:,1,0]*gamma_r[:,0,0]*gamma_f[:,0,0]
            u[:,0,0] = (m[:,0,0] - m[:,0,1]*m[:,1,0]*gamma_f[:,0,0])/(d)
            u[:,0,1] = (m[:,0,1] - m[:,0,0]*m[:,0,1]*gamma_r[:,0,0])/(d)
            u[:,1,0] = (m[:,1,0] - m[:,1,1]*m[:,1,0]*gamma_f[:,0,0])/(d)
            u[:,1,1] = (m[:,1,1] - m[:,0,1]*m[:,1,0]*gamma_r[:,0,0])/(d)
            
            unterminated.s = u
            return unterminated
        else:
            return ntwk
    
    def terminate(self, ntwk):
        '''
        Terminate a  network with  switch terms
        
        
        Parameters
        -------------
        two_port : 2-port Network 
            an unterminated network
        gamma_f : 1-port Network
            measured forward switch term. 
            gamma_f = a2/b2 sourced by port1
        gamma_r : 1-port Network
            measured reverse switch term
            gamma_r = a1/b1 sourced by port1
        
        Returns
        -----------
        ntwk :  Network object
        
        See Also
        --------
        unterminate_switch_terms 
        
        References
        ------------
        
        .. [#] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks
        '''
        if self.switch_terms is not None:
            gamma_f, gamma_r = self.switch_terms
            m = ntwk.copy()
            ntwk_flip = ntwk.copy()
            ntwk_flip.flip()
            
            m.s[:,0,0] = (ntwk**gamma_f).s[:,0,0]
            m.s[:,1,1] = (ntwk_flip**gamma_r).s[:,0,0]
            m.s[:,1,0] = ntwk.s[:,1,0]/(1-ntwk.s[:,1,1]*gamma_f.s[:,0,0])
            m.s[:,0,1] = ntwk.s[:,0,1]/(1-ntwk.s[:,0,0]*gamma_r.s[:,0,0])
            return m
        else:
            return ntwk
    
      
    @property
    def measured_unterminated(self):        
        return [self.unterminate(k) for k in self.measured]
        
    def run(self):
        '''
        Two port calibration based on the 8-term error model.
    
        Takes two
        ordered lists of measured and ideal responses. Optionally, switch
        terms [1]_ can be taken into account by passing a tuple containing the
        forward and reverse switch terms as 1-port Networks. This algorithm
        is based on the work in [2]_ .
    
        Parameters
        -----------
        measured : list of 2-port :class:`~skrf.network.Network` objects
                Raw measurements of the calibration standards. The order
                must align with the `ideals` parameter
    
        ideals : list of 2-port :class:`~skrf.network.Network` objects
                Predicted ideal response of the calibration standards.
                The order must align with `ideals` list
                measured: ordered list of measured networks. list elements
    
        switch_terms : tuple of :class:`~skrf.network.Network` objects
                        The two measured switch terms in the order
                        (forward, reverse).  This is only applicable in two-port
                        calibrations. See Roger Mark's paper on switch terms [1]_
                        for explanation of what they are.
    
        Returns
        ----------
        output : a dictionary
                output information, contains the following keys:
                * 'error coefficients':
                * 'error vector':
                * 'residuals':
    
        Notes
        ---------
        support for gathering switch terms on HP8510C  is in
        :mod:`skrf.vi.vna`
    
    
        References
        -------------
        .. [1] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931
        .. [2] Speciale, R.A.; , "A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors," Microwave Theory and Techniques, IEEE Transactions on , vol.25, no.12, pp. 1100- 1115, Dec 1977. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1129282&isnumber=25047
    

    '''
        numStds = self.nstandards
        numCoefs = 7

        
        mList = [self.unterminate(k).s for k in self.measured]
        iList = [k.s for k in self.ideals]
        
        fLength = len(mList[0])
        #initialize outputs
        error_vector = npy.zeros(shape=(fLength,numCoefs),dtype=complex)
        residuals = npy.zeros(shape=(fLength,4*numStds-numCoefs),dtype=complex)
        Q = npy.zeros((numStds*4, 7),dtype=complex)
        M = npy.zeros((numStds*4, 1),dtype=complex)
        # loop through frequencies and form m, a vectors and
        # the matrix M. where M =       e00 + S11i
        #                                                       i2, 1, i2*m2
        #                                                                       ...etc
        for f in range(fLength):
            # loop through standards and fill matrix
            for k in range(numStds):
                m,i  = mList[k][f,:,:],iList[k][f,:,:] # 2x2 s-matrices
                Q[k*4:k*4+4,:] = npy.array([\
                        [ 1, i[0,0]*m[0,0], -i[0,0],    0,  i[1,0]*m[0,1],        0,         0   ],\
                        [ 0, i[0,1]*m[0,0], -i[0,1],    0,  i[1,1]*m[0,1],        0,     -m[0,1] ],\
                        [ 0, i[0,0]*m[1,0],     0,      0,  i[1,0]*m[1,1],   -i[1,0],        0   ],\
                        [ 0, i[0,1]*m[1,0],     0,      1,  i[1,1]*m[1,1],   -i[1,1],    -m[1,1] ],\
                        ])
                #pdb.set_trace()
                M[k*4:k*4+4,:] = npy.array([\
                        [ m[0,0]],\
                        [       0       ],\
                        [ m[1,0]],\
                        [       0       ],\
                        ])
    
            # calculate least squares
            error_vector_at_f, residuals_at_f = npy.linalg.lstsq(Q,M)[0:2]
            #if len (residualsTmp )==0:
            #       raise ValueError( 'matrix has singular values, check standards')
            
            
            error_vector[f,:] = error_vector_at_f.flatten()
            residuals[f,:] = residuals_at_f
        
        e = error_vector
        # put the error vector into human readable dictionary
        self._coefs = {\
                'forward directivity':e[:,0],
                'forward source match':e[:,1],
                'forward reflection tracking':(e[:,0]*e[:,1])-e[:,2],
                'reverse directivity':e[:,3]/e[:,6],
                'reverse source match':e[:,4]/e[:,6],
                'reverse reflection tracking':(e[:,4]/e[:,6])*(e[:,3]/e[:,6])- (e[:,5]/e[:,6]),
                'k':e[:,6],
                }
        
        
        if self.switch_terms is not None:
            self._coefs.update({
                'forward switch term': self.switch_terms[0].s.flatten(),
                'reverse switch term': self.switch_terms[1].s.flatten(),
                })
        else:
            self._coefs.update({
                'forward switch term': npy.zeros(fLength, dtype=complex),
                'reverse switch term': npy.zeros(fLength, dtype=complex),
                })
        # output is a dictionary of information
        self._output_from_run = {
                'error vector':e, 
                'residuals':residuals
                }
    
        return None    
        
    def apply_cal(self, ntwk):
        caled = ntwk.copy()
        inv = linalg.inv
        
        T1,T2,T3,T4 = self.T_matrices
        
        ntwk = self.unterminate(ntwk)  
        
        for f in range(len(ntwk.s)):
            t1,t2,t3,t4,m = T1[f,:,:],T2[f,:,:],T3[f,:,:],\
                            T4[f,:,:],ntwk.s[f,:,:]
            caled.s[f,:,:] = inv(-1*m.dot(t3)+t1).dot(m.dot(t4)-t2)
        return caled
    
    def embed(self, ntwk):
        '''
        '''
        embedded = ntwk.copy()
        inv = linalg.inv
        
        T1,T2,T3,T4 = self.T_matrices
        
        for f in range(len(ntwk.s)):
            t1,t2,t3,t4,a = T1[f,:,:],T2[f,:,:],T3[f,:,:],\
                            T4[f,:,:],ntwk.s[f,:,:]
            embedded.s[f,:,:] = (t1.dot(a)+t2).dot(inv(t3.dot(a)+t4))
        
        embedded = self.terminate(embedded)
        
        return embedded
        
      
    @property
    def T_matrices(self):
        '''
        Intermediate matrices used for embedding and de-embedding. 
        
        Returns
        --------
        T1,T2,T3,T4 : numpy ndarray
        
        '''
        ec = self.coefs
        npoints = len(ec['k'])
        one = npy.ones(npoints,dtype=complex)
        zero = npy.zeros(npoints,dtype=complex)
        
        Edf = self.coefs['forward directivity']
        Esf = self.coefs['forward source match']
        Erf = self.coefs['forward reflection tracking']
        Edr = self.coefs['reverse directivity']
        Esr = self.coefs['reverse source match']
        Err = self.coefs['reverse reflection tracking']
        k = self.coefs['k']
        
        detX = Edf*Esf-Erf
        detY = Edr*Esr-Err
        
        
        T1 = npy.array([\
                [ -1*detX,  zero    ],\
                [ zero,     -1*k*detY]])\
                .transpose().reshape(-1,2,2)
        T2 = npy.array([\
                [ Edf,      zero ],\
                [ zero,     k*Edr]])\
                .transpose().reshape(-1,2,2)
        T3 = npy.array([\
                [ -1*Esf,   zero ],\
                [ zero,     -1*k*Esr]])\
                .transpose().reshape(-1,2,2)
        T4 = npy.array([\
                [ one,      zero ],\
                [ zero,     k ]])\
                .transpose().reshape(-1,2,2)
                
        return T1,T2,T3,T4
        
class TRL(EightTerm):
    '''
    Thru Reflect Line 
    '''
    
    def __init__(self, measured, ideals,line_approx=None,*args, **kwargs):
        '''
        Init. 
        
        Parameters
        --------------
        measured : 
        ideals : 
        switch_terms : tuple of :class:`~skrf.network.Network` objects
            the pair of switch terms in the order (forward, reverse)
            
        \*args, \*\*kwargs : 
            
            
        '''
        warn('Value of Reflect is not solved for yet.')
        self.line_approx = line_approx
        
        
        EightTerm.__init__(self, 
            measured = measured, 
            ideals = ideals,
            *args, **kwargs)
        
        
        thru_m, reflect_m, line_m = self.measured_unterminated 
        self.ideals[2] = determine_line(thru_m, line_m, line_approx) # find line 
        self.type = 'TRL'

class UnknownThru(EightTerm):
    '''
    '''
    def __init__(self, measured, ideals, thru_approx=None, n_thrus=1, *args, **kwargs):
        '''
        '''
        self.n_thrus = n_thrus
        self.thru_approx = thru_approx
        
        EightTerm.__init__(self, 
            measured = measured, 
            ideals = ideals,
            *args, **kwargs)
        self.type = 'UnknownThru'
    
    def run(self):
        n_thrus = self.n_thrus
        p1_m = [k.s11 for k in self.measured[:-n_thrus]]
        p2_m = [k.s22 for k in self.measured[:-n_thrus]]
        p1_i = [k.s11 for k in self.ideals[:-n_thrus]]
        p2_i = [k.s22 for k in self.ideals[:-n_thrus]]
        
        thru_m = NetworkSet(self.measured_unterminated[-n_thrus:]).mean_s
        
        thru_approx  =  NetworkSet(self.ideals[-n_thrus:]).mean_s
        
        # create one port calibration for all reflective standards  
        port1_cal = OnePort(measured = p1_m, ideals = p1_i)
        port2_cal = OnePort(measured = p2_m, ideals = p2_i)
        
        # cal coefficient dictionaries
        p1_coefs = port1_cal.coefs.copy()
        p2_coefs = port2_cal.coefs.copy()
        
        e_rf = port1_cal.coefs_ntwks['reflection tracking']
        e_rr = port2_cal.coefs_ntwks['reflection tracking']
        X = port1_cal.error_ntwk
        Y = port2_cal.error_ntwk
        
        # create a fully-determined 8-term cal just get estimate on k's sign
        # this is really inefficient, i need to fix the math on the 
        # closed form solution
        et = EightTerm(
            measured = self.measured, 
            ideals = self.ideals,
            switch_terms= self.switch_terms)
        k_approx = et.coefs_ntwks['k']
        
        
        e_tf_s = npy.sqrt((e_rf*e_rr*(thru_m.s21/thru_m.s12)).s.flatten())
        e_tf_s = find_correct_sign(e_tf_s, -1* e_tf_s, (k_approx*e_rr).s.flatten())
        
        
        k_ = (e_tf_s.flatten()/e_rr.s.flatten())
        
               
        # create single dictionary for all error terms
        coefs = {}
        
        coefs.update(dict([('forward %s'%k, p1_coefs[k]) for k in p1_coefs]))
        coefs.update(dict([('reverse %s'%k, p2_coefs[k]) for k in p2_coefs]))
        
        if self.switch_terms is not None:
            coefs.update({
                'forward switch term': self.switch_terms[0].s.flatten(),
                'reverse switch term': self.switch_terms[1].s.flatten(),
                })
        else:
            coefs.update({
                'forward switch term': npy.zeros(len(self.frequency), dtype=complex),
                'reverse switch term': npy.zeros(len(self.frequency), dtype=complex),
                })
        
        coefs.update({'k':k_})
        
        self._coefs = coefs



## Functions
def determine_line(thru_m, line_m, line_approx=None):
    '''
    Determine S21 of a matched line. 
    
    Given raw measurements of a `thru` and a matched `line` with unknown
    s21, this will calculate the response of the line. The `line_approx`
    is an approximation to line, that is needed to choose the correct 
    root sign. 
    
    This is possible because two measurements can be combined to 
    create a relationship of similar matrices, as shown below. Equating
    the traces between these measurements allows one to solve for S21 
    of the line.
    
    .. math::
        
        M_t = X \\cdot A_t \\cdot Y    
        M_l = X \\cdot A_l \\cdot Y
        
        M_t \\cdot M_{l}^{-1} = X \\cdot A_t \\cdot A_{l}^{-1} \\cdot X^{-1}
        
        tr(M_t \\cdot M_{l}^{-1}) = tr( A_t \\cdot A_{l}^{-1})
    
    which can be solved to form a quadratic in S21 of the line
    
    Notes
    -------
    This relies on the 8-term error model, which requires that switch
    terms are accounted for. specifically, thru and line have their 
    switch terms unterminated. 
    
    Parameters
    -----------
    thru_m : :class:`~skrf.network.Network`
        a raw measurement of a thru 
    line_m : :class:`~skrf.network.Network`
        a raw measurement of a matched transmissive standard
    line_approx : :class:`~skrf.network.Network`
        an approximate network the ideal line response. if None, then 
        the response is approximated by line_approx = line/thru. This 
        makes the assumption that the error networks have much larger 
        transmission than reflection
        
        
    References 
    --------------
    
    '''
    
    npts = len(thru_m)    
    
    if line_approx is None:
        # estimate line length, by assumeing error networks are well
        # matched
        line_approx = line_m/thru_m
    
    
    fm = [ -1* npy.trace(npy.dot(thru_m.t[f], npy.linalg.inv(line_m.t[f]))) \
        for f in range(npts)]
    one = npy.ones(npts)
    zero = npy.zeros(npts)
    
    roots_v = npy.frompyfunc( lambda x,y,z:npy.roots([x,y,z]),3,1 )
    s12 = roots_v(one, fm, one)
    s12_0 = npy.array([k[0]  for k in s12])
    s12_1 = npy.array([k[1]  for k in s12])
    
    s12 = find_correct_sign(s12_0, s12_1, line_approx.s[:,1,0])
    found_line = line_m.copy()
    found_line.s = npy.array([[zero, s12],[s12,zero]]).transpose().reshape(-1,2,2)
    return found_line
    
def convert_12term_2_8term(coefs_12term, redundant_k = False):
    '''
    Convert the 12-term and 8-term error coefficients.
    
    
    Derivation of this conversion can be found in [#]_ .
    
    References
    ------------
    
    .. [#] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931
    '''
    
    # Nomenclature taken from Roger Marks
    Edf = coefs_12term['forward directivity']
    Esf = coefs_12term['forward source match']
    Erf = coefs_12term['forward reflection tracking']
    Etf = coefs_12term['forward transmission tracking']
    Elf = coefs_12term['forward load match']
    Eif = coefs_12term.get('forward isolation',0)
    
    Edr = coefs_12term['reverse directivity']
    Esr = coefs_12term['reverse source match']
    Err = coefs_12term['reverse reflection tracking']
    Elr = coefs_12term['reverse load match']    
    Etr = coefs_12term['reverse transmission tracking']
    Eir = coefs_12term.get('reverse isolation',0)
    
    # these are given in eq (30) - (33) in Roger Mark's paper listed in 
    # the docstring
    # NOTE: k = e10/e23 = alpha/beta 
    #   the 'k' nomenclature is from Soares Speciale
    gamma_f = (Elf - Esr)/(Err + Edr*(Elf  - Esr))
    gamma_r = (Elr - Esf)/(Erf  + Edf *(Elr - Esf))
    
    k_first  =   Etf/(Err + Edr*(Elf  - Esr) )
    k_second =1/(Etr/(Erf + Edf *(Elr - Esf)))
    k = k_first #npy.sqrt(k_second*k_first)# (k_first +k_second )/2.
    coefs_8term = {}
    for l in ['forward directivity','forward source match',
        'forward reflection tracking','reverse directivity',
        'reverse reflection tracking','reverse source match']:
        coefs_8term[l] = coefs_12term[l].copy() 
    
    coefs_8term['forward switch term'] = gamma_f
    coefs_8term['reverse switch term'] = gamma_r
    coefs_8term['k'] = k
    if redundant_k:
        coefs_8term['k first'] = k_first
        coefs_8term['k second'] = k_second
    return coefs_8term
 

def convert_8term_2_12term(coefs_8term):
    '''
    '''
    Edf = coefs_8term['forward directivity']
    Esf = coefs_8term['forward source match']
    Erf = coefs_8term['forward reflection tracking']
    
    Edr = coefs_8term['reverse directivity']
    Esr = coefs_8term['reverse source match']
    Err = coefs_8term['reverse reflection tracking']
    
    
    gamma_f = coefs_8term['forward switch term']
    gamma_r = coefs_8term['reverse switch term']
    k = coefs_8term['k']
    
    # taken from eq (36)-(39) in the Roger Marks paper given in the 
    # docstring
    Elf  = Esr + (Err*gamma_f)/(1. - Edr * gamma_f)
    Elr = Esf  + (Erf *gamma_r)/(1. - Edf  * gamma_r)
    Etf  = ((Elf  - Esr)/gamma_f) * k
    Etr = ((Elr - Esf )/gamma_r) * 1./k
    
    coefs_12term = {}
    for l in ['forward directivity','forward source match',
        'forward reflection tracking','reverse directivity',
        'reverse reflection tracking','reverse source match']:
        coefs_12term[l] = coefs_8term[l].copy() 
        
    coefs_12term['forward load match'] = Elf
    coefs_12term['reverse load match'] = Elr
    coefs_12term['forward transmission tracking'] =  Etf
    coefs_12term['reverse transmission tracking'] =  Etr
    return coefs_12term





def align_measured_ideals(measured, ideals):
    '''
    Aligns two lists of networks based on the intersection of their name's.
    
    '''
    measured = [ measure for measure in measured\
        for ideal in ideals if ideal.name in measure.name]
    ideals = [ ideal for measure in measured\
        for ideal in ideals if ideal.name in measure.name]
    return measured, ideals
    
    
def two_port_error_vector_2_Ts(error_coefficients):
    ec = error_coefficients
    npoints = len(ec['k'])
    one = npy.ones(npoints,dtype=complex)
    zero = npy.zeros(npoints,dtype=complex)
    #T_1 = npy.zeros((npoints, 2,2),dtype=complex)
    #T_1[:,0,0],T_1[:,1,1] = -1*ec['det_X'], -1*ec['k']*ec['det_Y']
    #T_1[:,1,1] = -1*ec['k']*ec['det_Y']


    T1 = npy.array([\
            [       -1*ec['det_X'], zero    ],\
            [       zero,           -1*ec['k']*ec['det_Y']]]).transpose().reshape(-1,2,2)
    T2 = npy.array([\
            [       ec['e00'], zero ],\
            [       zero,                   ec['k']*ec['e33']]]).transpose().reshape(-1,2,2)
    T3 = npy.array([\
            [       -1*ec['e11'], zero      ],\
            [       zero,                   -1*ec['k']*ec['e22']]]).transpose().reshape(-1,2,2)
    T4 = npy.array([\
            [       one, zero       ],\
            [       zero,                   ec['k']]]).transpose().reshape(-1,2,2)
    return T1,T2,T3,T4

def error_dict_2_network(coefs, frequency,  is_reciprocal=False, **kwargs):
    '''
    Create a Network from a dictionary of standard error terms 


    '''

    if len (coefs.keys()) == 3:
        # ASSERT: we have one port data
        ntwk = Network(**kwargs)

        if is_reciprocal:
            #TODO: make this better and maybe have a phase continuity
            # functionality
            tracking  = coefs['reflection tracking']
            s12 = npy.sqrt(tracking)
            s21 = npy.sqrt(tracking)
            #s12 =  sqrt_phase_unwrap(tracking)
            #s21 =  sqrt_phase_unwrap(tracking)

        else:
            s21 = coefs['reflection tracking']
            s12 = npy.ones(len(s21), dtype=complex)

        s11 = coefs['directivity']
        s22 = coefs['source match']
        ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
        ntwk.frequency = frequency
        return ntwk
    
    else:
        p1,p2 = {},{}
        for k in ['source match','directivity','reflection tracking']:
            p1[k] = coefs['forward '+k]
            p2[k] = coefs['reverse '+k]
        forward = error_dict_2_network(p1, frequency = frequency, 
            name='forward', is_reciprocal = is_reciprocal,**kwargs)
        reverse = error_dict_2_network(p2, frequency = frequency, 
            name='reverse', is_reciprocal = is_reciprocal,**kwargs)
        return (forward, reverse)
