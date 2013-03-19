
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
from numpy import mean, std
import pylab as plb
import os
from copy import deepcopy, copy
import itertools
from warnings import warn
import cPickle as pickle

from calibrationAlgorithms import *
from ..mathFunctions import complex_2_db, sqrt_phase_unwrap
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
            'two port parametric':parameterized_self_calibration,\
            }
    '''
    dictionary holding calibration algorithms.

    See Also
    ---------
            :mod:`skrf.calibration.calibrationAlgorithms`
    '''

    def __init__(self,measured, ideals, type=None, \
            is_reciprocal=False,name=None, sloppy_input=False,
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

        self.frequency = measured[0].frequency.copy()
        self.type = type

        # passed to calibration algorithm in run()
        self.kwargs = kwargs 
        self.name = name
        self.is_reciprocal = is_reciprocal
        self.sloppy_input= sloppy_input

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
        # some checking to make sure they gave us consistent data
        if self.type == 'one port' or self.type == 'two port':

            if self.sloppy_input == True:
                # if they gave sloppy input try to align networks based
                # on their names. This basically takes the union of the 
                # two lists. 
                self.measured = [ measure for measure in self.measured\
                    for ideal in self.ideals if ideal.name in measure.name]
                self.ideals = [ ideal for measure in self.measured\
                    for ideal in self.ideals if ideal.name in measure.name]
                self.sloppy_input = False
            else:
                # did they supply the same number of  ideals as measured?
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
#



        # actually call the algorithm and run the calibration
        self._output_from_cal = \
                self.calibration_algorithm_dict[self.type](measured = self.measured, ideals = self.ideals,**self.kwargs)

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
                caled = deepcopy(input_ntwk)
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
    def __init__(self, measured, ideals, **kwargs):
        '''
        '''
        self.measured = measured
        self.ideals = ideals
        self.kwargs = kwargs
        
        
        def run(self):
            pass
        
        @property
        def coefs(self):
            '''
            '''
            try:
                return self._coefs
            except(AttributeError):
                self.run()
                return self._coefs
        
        
        
        def apply(self):
            '''
            '''
            pass
        # to support legacy scripts
        apply_cal = apply    

class SOLT(Calibration2):
    def __init__(self, measured, ideals, **kwargs):
        Calibration2.__init__(self, measured, ideals, **kwargs)
    
    def run(self):
        '''
        '''
        if len(self.measured) != len(self.ideals): 
            raise(IndexError('Number of ideals must equal number of measurements'))
        
        p1_m = [k.s11 for k in self.measured[:-1]]
        p2_m = [k.s22 for k in self.measured[:-1]]
        p1_i = [k.s11 for k in self.ideals[:-1]]
        p2_i = [k.s22 for k in self.ideals[:-1]]
        thru = self.measured[-1]
        
        # create one port calibration for all but last standard    
        port1_cal = Calibration(measured = p1_m, ideals = p1_i)
        port2_cal = Calibration(measured = p2_m, ideals = p2_i)
        
        # cal coefficient dictionaries
        p1_coefs = port1_cal.coefs
        p2_coefs = port2_cal.coefs
        
        if self.kwargs.get('isolation',None) is not None:
            p1_coefs['isolation'] = isolation.s21.s
            p2_coefs['isolation'] = isolation.s12.s
        
        p1_coefs['reciever match'] = port1_cal.apply_cal(thru.s11).s
        p2_coefs['reciever match'] = port2_cal.apply_cal(thru.s22).s
        
        
        p1_coefs['transmission tracking'] = \
            thru.s21.s- p1_coefs.get('isolation',0)/\
            (1. - p1_coefs['source match']*p1_coefs['directivity'])
        p2_coefs['transmission tracking'] = \
            thru.s12.s- p2_coefs.get('isolation',0)/\
            (1. - p2_coefs['source match']*p2_coefs['directivity'])
        coefs = {}
        #import pdb;pdb.set_trace()
        coefs.update({ 'port1 %s':p1_coefs[k] for k in p1_coefs})
        coefs.update({ 'port2 %s':p2_coefs[k] for k in p2_coefs})
        self._coefs = coefs
        return 0 
    
    def apply(ntwk):
        '''
        '''
        caled = ntwk.copy()
        
        s11 = ntkw.s[:,0,0]
        s12 = ntkw.s[:,0,1]
        s21 = ntkw.s[:,1,0]
        s22 = ntkw.s[:,1,1]
        
        e00 = self.coefs['port1 directivity']
        e11 = self.coefs['port1 source match']
        e10e01 = self.coefs['port1 reflection tracking']
        e10e32 = self.coefs['port1 transmission tracking']
        e22 = self.coefs['port1 reciever match']
        e30 = self.coefs.get('port1 isolation',0)
        
        e33_ = self.coefs['port2 directivity']
        e11_ = self.coefs['port2 reciever match']
        e23e32_ = self.coefs['port2 reflection tracking']
        e23e01_ = self.coefs['port2 transmission tracking']
        e22_ = self.coefs['port2 source match']
        e03_ = self.coefs.get('port2 isolation',0)
        
        
        D = (1+(s11-e00)/(e10e01)*e11)*(1+(s22-e33_)/(e23e32_)*e22_) -\
            ((s21-e30)/(e10e32))*((s12-e03_)/(e23e01_))*e22*e11_
        
        
        caled.s[:,0,0] = \
            (((s11-e00)/(e10e01))*(1+(s22-e33_)/(e23e32_)*e22_)-\
            e22*((s21-e30)/(e10e32))*(s12-e03)/(e23e01_)) /\
            D
            
        caled.s[:,1,0] = \
            (((s22-e33_)/(e23e32_))*(1+(s11-e00)/(e10e01)*e11)-\
            e11_*((s21-e30)/(e10e32))*(s12-e03_)/(e23e01_)) /\
            D
        caled.s[:,1,1] = \    
            
        
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
            #s12 =  sqrt_phase_unwrap(tracking)
            #s21 =  sqrt_phase_unwrap(tracking)

        else:
            s21 = coefs['reflection tracking']
            s12 = npy.ones(len(s21), dtype=complex)

        s11 = coefs['directivity']
        s22 = coefs['source match']
        ntwk.s = npy.array([[s11, s12],[s21,s22]]).transpose().reshape(-1,2,2)
        return ntwk
    
    elif len (coefs.keys()) == 7:
        coefs_p1, coefs_p2 = eight_term_2_one_port_coefs(coefs)
        en1 = error_dict_2_network(coefs_p1, frequency, is_reciprocal, **kwargs)
        en2 = error_dict_2_network(coefs_p2, frequency, is_reciprocal, **kwargs)
        return en1, en2
         
    else:
        raise NotImplementedError('sorry')
