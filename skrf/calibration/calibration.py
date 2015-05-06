'''
.. module:: skrf.calibration.calibration
================================================================
calibration (:mod:`skrf.calibration.calibration`)
================================================================


This module  provides objects for VNA calibration. Specific algorithms
inheret from the common base class  :class:`Calibration`.

Base Class
--------------

.. autosummary::
   :toctree: generated/

   Calibration

One-port
----------------------

.. autosummary::
   :toctree: generated/

   OnePort
   SDDL
   PHN

Two-port
---------------------

.. autosummary::
   :toctree: generated/

   TwelveTerm
   SOLT
   EightTerm
   UnknownThru
   TRL
   MultilineTRL


Three Reciever (1.5 port)
----------------------------------------------

.. autosummary::
   :toctree: generated/

   TwoPortOnePath
   EnhancedResponse


Generic Methods
----------------
.. autosummary::
   :toctree: generated/

   terminate
   unterminate
   determine_line

PNA interaction
----------------
.. autosummary::
   :toctree: generated/

   convert_skrfcoefs_2_pna
   convert_pnacoefs_2_skrf

'''
import numpy as npy
from numpy import linalg
from numpy.linalg import det
from numpy import mean, std, angle, real, imag, exp, ones, zeros, poly1d
import pylab as plb
import os
from copy import deepcopy, copy
import itertools
from warnings import warn
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from ..mathFunctions import complex_2_db, sqrt_phase_unwrap, \
    find_correct_sign, find_closest,  ALMOST_ZERO, rand_c, cross_ratio
from ..frequency import *
from ..network import *
from ..networkSet import func_on_networks as fon
from ..networkSet import NetworkSet


## later imports. delayed to solve circular dependencies
#from io.general import write
#from io.general import read_all_networks

global coefs_list_12term
coefs_list_12term =[
    'forward directivity',
    'forward source match',
    'forward reflection tracking',
    'forward transmission tracking',
    'forward load match',
    'forward isolation',
    'reverse directivity',
    'reverse load match',
    'reverse reflection tracking',
    'reverse transmission tracking',
    'reverse source match',
    'reverse isolation'
    ]



global coefs_list_8term
'''
There are various notations used for this same model. Given that all
measurements have been unterminated properly the error box model holds
and the following equalities hold:

k = e10/e23    # in s-param
k = alpha/beta # in mark's notation
beta/alpha *1/Err = 1/(e10e32)  # marks -> rytting notation
'''
coefs_list_8term = [
    'forward directivity',
    'forward source match',
    'forward reflection tracking',
    'reverse directivity',
    'reverse load match',
    'reverse reflection tracking',
    'forward switch term',
    'reverse switch term',
    'k'
    ]
global coefs_list_3term
coefs_list_3term = [
    'directivity',
    'source match',
    'reflection tracking',
    ]


class Calibration(object):
    '''
    Base class for all Calibration objects.

    This class implements the common mechanisms for all calibration
    algorithms. Specific calibration algorithms should inheret this
    class and overide the methods:
        *  :func:`Calibration.run`
        *  :func:`Calibration.apply_cal`
        *  :func:`Calibration.embed` (optional)


    The familiy of properties prefixed `coefs` and
    `coefs..ntwks`  returns error coefficients. If the property coefs
    is accessed and empty, then :func:`Calibration.run` is called.


    '''
    family = ''
    def __init__(self, measured, ideals, sloppy_input=False,
        is_reciprocal=True,name=None,*args, **kwargs):
        '''
        Calibration initializer.


        Notes
        -------
        About the order of supplied standards,

        If the measured and ideals parameters are lists of Networks and
        `sloppy_input=False`, then their elements must align. However,
        if the measured and ideals are dictionaries, or
        `sloppy_input=True`, then we will try to align them for you
        based on the names of the networks (see `func:`align_measured_ideals`).

        You do not want to use this `sloppy_input` feature if the
        calibration depends on the standard order (like TRL).


        Parameters
        ----------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use `sloppy_input`)

        sloppy_input :  Boolean.
            Allows ideals and measured lists to be 'aligned' based on
            the network names.

        is_reciprocal : Boolean
            enables the reciprocity assumption on the calculation of the
            error_network, which is only relevant for one-port
            calibrations.

        name: string
            the name of this calibration instance, like 'waveguide cal'
            this is just for convenience [None].

        \*args, \*\*kwargs : key-word arguments
            stored in self.kwargs, which may be used by sub-classes
            most likely in `run`.


        '''

        # allow them to pass di
        # lets make an ideal flush thru for them :
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

        self.sloppy_input=sloppy_input
        if sloppy_input:
            self.measured, self.ideals = \
                align_measured_ideals(self.measured, self.ideals)

        if len(self.measured) != len(self.ideals):
            raise(IndexError('The length of measured and ideals lists are different. Number of ideals must equal the number of measured. If you are using `sloppy_input` ensure the names are uniquely alignable.'))


        # ensure all the measured Networks' frequency's are the same
        for measure in self.measured:
            if self.measured[0].frequency != measure.frequency:
                raise(ValueError('measured Networks dont have matching frequencies.'))
        # ensure that all ideals have same frequency of the measured
        # if not, then attempt to interpolate
        for k in list(range(len(self.ideals))):
            if self.ideals[k].frequency != self.measured[0]:
                print('Warning: Frequency information doesn\'t match on ideals[{}], attempting to interpolate the ideal[{}] Network ..'.format(k,k))
                try:
                    # try to resample our ideals network to match
                    # the meaurement frequency
                    self.ideals[k].interpolate_self(\
                        self.measured[0].frequency)
                    print('Success')

                except:
                    raise(IndexError('Failed to interpolate. Check frequency of ideals[{}].'.format(k)))


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

        if 'fromcoefs' in self.family.lower():
            output = '%s Calibration: \'%s\', %s'\
                %(self.family,name,str(self.frequency))
        else:
            output = '%s Calibration: \'%s\', %s, %i-standards'\
                %(self.family,name,str(self.frequency),\
                len(self.measured))

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

    def pop(self,std=-1):
        '''
        Remove and return tuple of (ideal, measured) at index.

        Parameters
        -------------
        std : int or str
            the integer of calibration standard to remove, or the name
            of the ideal or measured calibration standard to remove.



        Returns
        -----------
        ideal,measured : tuple of skrf.Networks
            the ideal and measured networks which were popped out of the
            calibration

        '''

        if isinstance(std, str):
            for idx,ideal in enumerate(self.ideals):
                if std  == ideal.name:
                    std = idx

        if isinstance(std, str):
            for idx,measured in enumerate(self.measured):
                if std  == measured.name:
                    std = idx

        if isinstance(std, str):
            raise (ValueError('standard %s not found in ideals'%std))

        return (self.ideals.pop(std),  self.measured.pop(std))

    @classmethod
    def from_coefs_ntwks(cls, coefs_ntwks, **kwargs):
        '''
        Creates a calibration from its error coefficients

        Parameters
        -------------
        coefs_ntwks :  dict of Networks objects
            error coefficients for the calibration

        See Also
        ----------
        Calibration.from_coefs
        '''
        # assigning this measured network is a hack so that
        # * `calibration.frequency` property evaluates correctly
        # * TRL.__init__() will not throw an error
        if not hasattr(coefs_ntwks,'keys'):
            # maybe they passed a list? lets try and make a dict from it
            coefs_ntwks = NetworkSet(coefs_ntwks).to_dict()

        coefs = NetworkSet(coefs_ntwks).to_s_dict()

        frequency = list(coefs_ntwks.values())[0].frequency

        cal= cls.from_coefs(frequency=frequency, coefs=coefs, **kwargs)
        return cal

    @classmethod
    def from_coefs(cls, frequency, coefs, **kwargs):
        '''
        Creates a calibration from its error coefficients

        Parameters
        -------------
        frequency : :class:`~skrf.frequency.Frequency`
            frequency info, (duh)
        coefs :  dict of numpy arrays
            error coefficients for the calibration

        See Also
        ----------
        Calibration.from_coefs_ntwks

        '''
        # assigning this measured network is a hack so that
        # * `calibration.frequency` property evaluates correctly
        # * TRL.__init__() will not throw an error
        n = Network(frequency = frequency,
                    s = rand_c(frequency.npoints,2,2))
        measured = [n,n,n]

        if 'forward switch term' in coefs:
            switch_terms = (Network(frequency = frequency,
                                    s=coefs['forward switch term']),
                            Network(frequency = frequency,
                                    s=coefs['reverse switch term']))
            kwargs['switch_terms'] = switch_terms


        cal = cls(measured, measured, **kwargs)
        cal.coefs = coefs
        cal.family += '(fromCoefs)'
        return  cal

    @property
    def frequency(self):
        '''
        :class:`~skrf.frequency.Frequency` object of the calibration


        '''
        return self.measured[0].frequency.copy()

    @property
    def nstandards(self):
        '''
        number of ideal/measurement pairs in calibration
        '''
        if len(self.ideals) != len(self.measured):
            warn('number of ideals and measured don\'t agree')
        return len(self.ideals)

    @property
    def coefs(self):
        '''
        Dictionary or error coefficients in form of numpy arrays

        The keys of this will be different depending on the
        Calibration Model. This dictionary should be populated
        when the `run()` function is called.

        Notes
        -------
        when setting this, property, the numpy arrays are flattened.
        this makes accessing the coefs more concise in the code.

        See Also
        ----------
        coefs_3term
        coefs_8term
        coefs_12term
        coefs_ntwks
        '''
        try:
            return self._coefs
        except(AttributeError):
            self.run()
            return self._coefs

    @coefs.setter
    def coefs(self,d):
        '''
        '''
        for k in d:
            d[k] = d[k].flatten()
        self._coefs = d

    def update_coefs(self, d):
        '''
        update currect dict of error coefficients

        '''
        for k in d:
            d[k] = d[k].flatten()

        self._coefs.update(d)

    @property
    def output_from_run(self):
        '''
        Returns any output from the :func:`run`.

        This just returns whats in  _output_from_run, and calls
        :func:`run` if that attribute is  non-existent.
        finally, returns None if run() is called, and nothing is in
        _output_from_run.
        '''
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
        Dictionary of error coefficients in form of Network objects

        See Also
        -----------
        coefs_3term_ntwks
        coefs_12term_ntwks
        coefs_8term_ntwks
        '''
        ns = NetworkSet.from_s_dict(d=self.coefs,
                                    frequency=self.frequency)
        return ns.to_dict()

    @property
    def coefs_3term(self):
        '''
        Dictionary of error coefficients for One-port Error model

        Contains the keys:
            * directivity
            * source match
            * reflection tracking'
        '''
        return dict([(k, self.coefs.get(k)) for k in [\
            'directivity',
            'source match',
            'reflection tracking',
            ]])

    @property
    def coefs_3term_ntwks(self):
        '''
        Dictionary of error coefficients in form of Network objects
        '''
        ns = NetworkSet.from_s_dict(d=self.coefs_3term,
                                    frequency=self.frequency)
        return ns.to_dict()

    @property
    def normalized_directivity(self):
        '''
        the directivity normalized to the reflection tracking
        '''
        try:
            return self.coefs_ntwks['directivity']/\
                   self.coefs_ntwks['reflection tracking']
        except:
            pass
        try:
            out = {}
            for direction in ['forward','reverse']:
                out[direction + ' normalized directvity'] =\
                    self.coefs_ntwks[direction + ' directivity']/\
                    self.coefs_ntwks[direction + ' reflection tracking']
            return out
        except:
            raise ValueError('cant find error coefs')


    @property
    def coefs_8term(self):
        '''
        Dictionary of error coefficients for 8-term (Error-box) Model


        Contains the keys:
            * forward directivity
            * forward source match
            * forward reflection tracking
            * reverse directivity
            * reverse load match
            * reverse reflection tracking
            * forward switch term
            * reverse switch term
            * k

        Notes
        --------
        If this calibration uses the 12-term model, then
        :func:`convert_12term_2_8term` is called. See [1]_

        References
        -------------

        .. [1] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks


        '''

        d = self.coefs

        for k in coefs_list_8term:
            if k not in d:
                d = convert_12term_2_8term(d)

        return d

    @property
    def coefs_8term_ntwks(self):
        '''
        Dictionary of error coefficients in form of Network objects
        '''
        ns = NetworkSet.from_s_dict(d=self.coefs_8term,
                                    frequency=self.frequency)
        return ns.to_dict()

    @property
    def coefs_12term(self):
        '''
        Dictionary of error coefficients for 12-term Model

        Contains the keys:
            * forward directivity
            * forward source match
            * forward reflection tracking
            * forward transmission tracking
            * forward load match
            * reverse directivity
            * reverse load match
            * reverse reflection tracking
            * reverse transmission tracking
            * reverse source match

        Notes
        --------
        If this calibration uses the 8-term model, then
        :func:`convert_8term_2_12term` is called. See [1]_


        References
        -------------

        .. [1] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks


        '''
        d = self.coefs

        for k in coefs_list_12term:
            if k not in d:
                d = convert_8term_2_12term(d)

        return d

    @property
    def coefs_12term_ntwks(self):
        '''
        Dictionary or error coefficients in form of Network objects
        '''
        ns = NetworkSet.from_s_dict(d=self.coefs_12term,
                                    frequency=self.frequency)
        return ns.to_dict()

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
        Dictionary of residual Networks

        These residuals are complex differences between the ideal
        standards and their corresponding  corrected measurements.

        '''
        return [ideal - caled for (ideal, caled) in zip(self.ideals, self.caled_ntwks)]

    @property
    def residual_ntwk_sets(self):
        '''
        Returns a NetworkSet for each `residual_ntwk`, grouped by their names
        '''

        residual_sets={}
        std_names = list(set([k.name  for k in self.ideals ]))
        for std_name in std_names:
            residual_sets[std_name] = NetworkSet(
                [k for k in self.residual_ntwks if k.name.startswith(std_name)])
        return residual_sets

    @property
    def caled_ntwks(self):
        '''
        List of the corrected calibration standards
        '''
        return self.apply_cal_to_list(self.measured)


    @property
    def caled_ntwk_sets(self):
        '''
        Returns a NetworkSet for each `caled_ntwk`, grouped by their names
        '''

        caled_sets={}
        std_names = list(set([k.name  for k in self.ideals ]))
        for std_name in std_names:
            caled_sets[std_name] = NetworkSet(
                [k for k in self.caled_ntwks if k.name.startswith(std_name)])
        return caled_sets

    @property
    def biased_error(self):
        '''
        Estimate of biased error for overdetermined calibration with
        multiple connections of each standard

        Returns
        ----------
        biased_error : skrf.Network
            Network with s_mag is proportional to the biased error

        Notes
        -------
        Mathematically, this is

            mean_s(|mean_c(r)|)

        Where:

        * r: complex residual errors
        * mean_c: complex mean taken accross connection
        * mean_s: complex mean taken accross standard

        See Also
        ---------
        biased_error
        unbiased_error
        total_error

        '''
        rns = self.residual_ntwk_sets
        out =  NetworkSet([rns[k].mean_s for k in rns]).mean_s_mag
        out.name = 'Biased Error'
        return out

    @property
    def unbiased_error(self):
        '''
        Estimate of unbiased error for overdetermined calibration with
        multiple connections of each standard

        Returns
        ----------
        unbiased_error : skrf.Network
            Network with s_mag is proportional to the unbiased error

        Notes
        -------
        Mathematically, this is

            mean_s(std_c(r))

        where:
        * r : complex residual errors
        * std_c : standard deviation taken accross  connections
        * mean_s : complex mean taken accross  standards

        See Also
        ---------
        biased_error
        unbiased_error
        total_error
        '''
        rns = self.residual_ntwk_sets
        out = NetworkSet([rns[k].std_s for k in rns]).mean_s_mag
        out.name = 'Unbiased Error'
        return out

    @property
    def total_error(self):
        '''
        Estimate of total error for overdetermined calibration with
        multiple connections of each standard.This is the combined
        effects of both biased and un-biased errors

        Returns
        ----------
        total_error : skrf.Network
            Network with s_mag is proportional to the total error

        Notes
        -------
        Mathematically, this is

            std_cs(r)

        where:
        * r : complex residual errors
        * std_cs : standard deviation taken accross connections
                and standards

        See Also
        ---------
        biased_error
        unbiased_error
        total_error
        '''
        out = NetworkSet(self.residual_ntwks).mean_s_mag
        out.name = 'Total Error'
        return out

    def plot_errors(self, *args, **kwargs):
        '''
        Plots biased, unbiased and total error in dB scaled

        See Also
        ---------
        biased_error
        unbiased_error
        total_error
        '''
        port_list = self.biased_error.port_tuples
        for m,n in port_list:
            plb.figure()
            plb.title('S%i%i'%(m+1,n+1))
            self.unbiased_error.plot_s_db(m,n,**kwargs)
            self.biased_error.plot_s_db(m,n,**kwargs)
            self.total_error.plot_s_db(m,n,**kwargs)
            plb.ylim(-100,0)

    @property
    def error_ntwk(self):
        '''
        The calculated error Network or Network[s]

        This will return a single two-port network for a one-port cal.
        For a 2-port calibration this will return networks
        for forward and reverse excitation. However, these are not
        sufficient to use for embedding, see the :func:`embed` function
        for that.



        '''
        return error_dict_2_network(
            self.coefs,
            frequency = self.frequency,
            is_reciprocal= self.is_reciprocal)


    def plot_caled_ntwks(self, attr='s_smith', show_legend=False,**kwargs):
        '''
        Plots corrected calibration standards

        Given that the calibration is overdetermined, this may be used
        as a heuristic verification of calibration quality.

        Parameters
        ------------------
        attr : str
            Network property to plot, ie 's_db', 's_smith', etc
        show_legend : bool
            draw a legend or not
        \\*\\*kwargs : kwargs
            passed to the plot method of Network
        '''
        ns = NetworkSet(self.caled_ntwks)
        kwargs.update({'show_legend':show_legend})

        if ns[0].nports ==1:
            ns.__getattribute__('plot_'+attr)(0,0, **kwargs)
        elif ns[0].nports ==2:
            plb.figure(figsize = (8,8))
            for k,mn in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
                plb.subplot(221+k)
                plb.title('S%i%i'%(mn[0]+1,mn[1]+1))
                ns.__getattribute__('plot_'+attr)(*mn, **kwargs)
        else:
            raise NotImplementedError
        plb.tight_layout()

    def plot_residuals(self, attr='s_db', **kwargs):
        '''
        Plot residual networks.

        Given that the calibration is overdetermined, this may be used
        as a metric of the calibration's *goodness of fit*

        Parameters
        ------------------
        attr : str
            Network property to plot, ie 's_db', 's_smith', etc
        \\*\\*kwargs : kwargs
            passed to the plot method of Network

        See Also
        --------
        Calibration.residual_networks
        '''

        NetworkSet(self.caled_ntwks).__getattribute__('plot_'+attr)(**kwargs)

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

class OnePort(Calibration):
    '''
    Standard algorithm for a one port calibration.

    Solves the linear set of equations:


    .. math::
        e_{11}\mathbf{i_1m_1}-\Delta e\,\mathbf{m_1}+e_{00}=\mathbf{i_1}

        e_{11}\mathbf{i_2m_2}-\Delta e\,\mathbf{m_2}+e_{00}=\mathbf{i_2}

        e_{11}\mathbf{i_3m_3}-\Delta e\,\mathbf{m_3}+e_{00}=\mathbf{i_3}

        ...

    Where **m**'s and **i**'s are the measured and ideal reflection coefficients,
    respectively.


    If more than three standards are supplied then a least square
    algorithm is applied.

    See [1]_  and [2]_

    References
    -------------

    .. [1] http://na.tm.agilent.com/vnahelp/tip20.html

    .. [2] Bauer, R.F., Jr.; Penfield, Paul, "De-Embedding and Unterminating," Microwave Theory and Techniques, IEEE Transactions on , vol.22, no.3, pp.282,288, Mar 1974
        doi: 10.1109/TMTT.1974.1128212
        URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1128212&isnumber=25001
    '''

    family = 'OnePort'
    def __init__(self, measured, ideals,*args, **kwargs):
        '''
        One Port initializer

        If more than three standards are supplied then a least square
        algorithm is applied.

        Notes
        ------
        See func:`Calibration.__init__` for details about
        automatic standards alignment (aka `sloppy_input`)


        Parameters
        -----------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use `sloppy_input`)

        args, kwargs :
            passed to func:`Calibration.__init__`

        Notes
        -----
        This uses numpy.linalg.lstsq() for least squares calculation

        See Also
        ---------
        Calibration.__init__
        '''
        Calibration.__init__(self, measured, ideals,
                             *args, **kwargs)

    def run(self):
        '''
        '''
        numStds = self.nstandards
        numCoefs=3

        mList = [self.measured[k].s.reshape((-1,1)) for k in list(range(numStds))]
        iList = [self.ideals[k].s.reshape((-1,1)) for k in list(range(numStds))]

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
        for f in list(range(fLength)):
            #create  m, i, and 1 vectors
            one = npy.ones(shape=(numStds,1))
            m = npy.array([ mList[k][f] for k in list(range(numStds))]).reshape(-1,1)# m-vector at f
            i = npy.array([ iList[k][f] for k in list(range(numStds))]).reshape(-1,1)# i-vector at f

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

class SDDLWeikle(OnePort):
    '''
    Short Delay Delay Load (Oneport Calibration)

    One-port self-calibration, which contains a short, a load, and
    two delays shorts of unity magnitude but unknown phase. Originally
    designed to be resistant to flange misalignment, see [1]_.


    References
    -------------
    .. [1] Z. Liu and R. M. Weikle, "A reflectometer calibration method resistant to waveguide flange misalignment," Microwave Theory and Techniques, IEEE Transactions on, vol. 54, no. 6, pp. 2447-2452, Jun. 2006.
    '''
    family = 'SDDL'
    def __init__(self, measured, ideals, *args, **kwargs):
        '''
        Short Delay Delay Load initializer


        measured and ideal networks must be in the order:

        * short
        * delay short1
        * delay short2
        * load

        Parameters
        -----------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input

        args, kwargs :
            passed to func:`Calibration.__init__`

        See Also
        ---------
        Calibration.__init__

        '''

        if (len(measured) != 4) or (len(ideals)) != 4:
            raise IndexError('Incorrect number of standards.')
        Calibration.__init__(self, measured =  measured,
                             ideals =ideals, *args, **kwargs)


    def run(self):
        #meaured reflection coefficients
        w_s = self.measured[0].s.flatten() # short
        w_1 = self.measured[1].s.flatten() # delay short 1
        w_2 = self.measured[2].s.flatten() # delay short 2
        w_l = self.measured[3].s.flatten() # load

        # ideal response of reflection coefficients
        G_l = self.ideals[3].s.flatten() # gamma_load
        # handle singularities
        G_l[G_l ==0] = ALMOST_ZERO


        w_1p  = w_1 - w_s # between (9) and (10)
        w_2p  = w_2 - w_s
        w_lp  = w_l - w_s


        ## NOTE: the published equation has an incorrect sign on this argument
        ## perhaps because they assume arg to measure clockwise angle??
        alpha = exp(1j*2*angle(1./w_2p - 1./w_1p)) # (17)

        p = alpha/( 1./w_1p - alpha/w_1p.conj() - (1.+G_l)/(G_l*w_lp )) # (22)
        q = p/(alpha* G_l)   #(23) (put in terms of p)

        Bp_re = -1*((1 + (imag(p+q)/real(q-p)) * (imag(q-p)/real(p+q)))/\
                    (1 + (imag(p+q)/real(q-p))**2)) * real(p+q) # (25)

        Bp_im = imag(q+p)/real(q-p) * Bp_re #(24)
        Bp = Bp_re + Bp_im*1j

        B = Bp + w_s    #(10)
        C = Bp * (1./w_1p - alpha/w_1p.conj()) + alpha * Bp/Bp.conj() #(20)
        A = B - w_s + w_s*C #(6)

        # convert the abc vector to standard error coefficients
        e00 = B
        e11 = -C
        e01e10 = A + e00*e11

        self._coefs = {\
                'directivity':e00,\
                'reflection tracking':e01e10, \
                'source match':e11\
                }

class SDDL(OnePort):
    '''
    Short Delay Delay Load (Oneport Calibration)

    One-port self-calibration, which contains a short, a load, and
    two delays shorts of unity magnitude but unknown phase. Originally
    designed to be resistant to flange misalignment, see [1]_.


    References
    -------------
    .. [1] Z. Liu and R. M. Weikle, "A reflectometer calibration method resistant to waveguide flange misalignment," Microwave Theory and Techniques, IEEE Transactions on, vol. 54, no. 6, pp. 2447-2452, Jun. 2006.
    '''
    family = 'SDDL'
    def __init__(self, measured, ideals, *args, **kwargs):
        '''
        Short Delay Delay Load initializer


        Measured and ideal networks must be in the order:

        [ Short, Delay short1, Delay short2, Load]

        The ideal delay shorts can be set to `None`, as they are
        determined during the calibration.

        Parameters
        -----------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input

        args, kwargs :
            passed to func:`Calibration.__init__`

        See Also
        ---------
        Calibration.__init__

        '''
        # if they pass None for the ideal responses for delay shorts
        # then we will copy the short standard in their place. this is
        # only to avoid throwing an error when initializing the cal, the
        # values are not used.
        if ideals[1] is None:
            ideals[1] = ideals[0].copy()
        if ideals[2] is None:
            ideals[2] = ideals[0].copy()

        if (len(measured) != 4) or (len(ideals)) != 4:
            raise IndexError('Incorrect number of standards.')
        Calibration.__init__(self, measured =  measured,
                             ideals =ideals, *args, **kwargs)


    def run(self):

        #meaured impedances
        d = s2z(self.measured[0].s,1) # short
        a = s2z(self.measured[1].s,1) # delay short 1
        b = s2z(self.measured[2].s,1) # delay short 2
        c = s2z(self.measured[3].s,1) # load
        l = s2z(self.ideals[-1].s,1) # ideal def of load
        cr_alpha = cross_ratio(b,a,c,d)
        cr_beta = cross_ratio(a,b,c,d)

        alpha = imag(cr_alpha)/real(cr_alpha/l)
        beta = imag(cr_beta)/real(cr_beta/l)

        self.ideals[1].s = z2s(alpha*1j,1)
        self.ideals[2].s = z2s(beta*1j,1)

        OnePort.run(self)

class PHN(OnePort):
    '''
    Pair of Half Knowns (One Port self-calibration)
    '''
    family = 'PHN'
    def __init__(self, measured, ideals, *args, **kwargs):
        '''


        '''
        if (len(measured) != 4) or (len(ideals)) != 4:
            raise IndexError('Incorrect number of standards.')

        Calibration.__init__(self, measured =  measured,
                             ideals =ideals, *args, **kwargs)


    def run(self):

        # ideals (in impedance)
        a = s2z(self.ideals[0].s,1).flatten() # half known
        b = s2z(self.ideals[1].s,1).flatten() # half known
        c = s2z(self.ideals[2].s,1).flatten() # fully known
        d = s2z(self.ideals[3].s,1).flatten() # fully known

        # meaured (in impedances)
        a_ = s2z(self.measured[0].s,1).flatten() # half known
        b_ = s2z(self.measured[1].s,1).flatten() # half known
        c_ = s2z(self.measured[2].s,1).flatten() # fully known
        d_ = s2z(self.measured[3].s,1).flatten() # fully known

        z = cross_ratio(a_,b_,c_,d_)

        # intermediate variables
        e = c-d-c*z
        f = d-c-d*z
        g = c*d*z

        A = -real(f*z.conj())
        B = 1j*imag( f*e.conj() + g.conj()*z)
        C = real( g*e.conj())

        npts = len(A)
        b1,b2 = zeros(npts, dtype=complex), zeros(npts, dtype=complex)

        for k in range(npts):
            p =  poly1d([A[k],B[k],C[k]])
            b1[k],b2[k] = p.r

        # temporarily translate into s-parameters so make the root-choice
        #  choosing a root in impedance doesnt generally work for typical
        # calibration standards
        b1_s = z2s(b1.reshape(-1,1,1),1)
        b2_s = z2s(b2.reshape(-1,1,1),1)
        b_guess = z2s(b.reshape(-1,1,1),1)
        b_found_s = find_closest(b1_s,b2_s,b_guess)

        b_found = s2z(b_found_s.reshape(-1,1,1),1).flatten()
        a_found = -(f*b_found + g)/(z*b_found + e)

        self.ideals[0].s = z2s(a_found.reshape(-1,1,1),1)
        self.ideals[1].s = z2s(b_found.reshape(-1,1,1),1)

        OnePort.run(self)


## Two Ports

class TwelveTerm(Calibration):
    '''
    12-term, full two-port calibration.

    `TwelveTerm` is the traditional, fully determined, two-port calibration
    originally developed in [1]_.

    `TwelveTerm` can accept any number of reflect and transmissive standards,
    as well as arbitrary (non-flush) transmissive standards.

    * If more than 3 reflect standards are provided, a least-squares
        solution  is implemented for the one-port stage of the calibration.
    * If more than 1 transmissive standard is given the `load match`,
        and `transmission tracking` terms are calculated multiple times
        and averaged.

    References
    ------------
    .. [1] "Calibration Process of Automatic Network Analyzer Systems"  by Stig Rehnmark


    '''
    family = 'TwelveTerm'
    def __init__(self, measured, ideals, n_thrus=None, trans_thres=-40,
                 *args, **kwargs):
        '''
        TwelveTerm initializer

        Use the  `n_thrus` argument to explicity define the number of
        transmissive standards. Otherwise, if `n_thrus=None`, then we
        will try and guess which are transmissive, by comparing the mean
        |s21| and |s12| responses (in dB) to `trans_thres`.

        Notes
        ------
        See func:`Calibration.__init__` for details about  automatic
        standards alignment (aka `sloppy_input`).

        Parameters
        -------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input

        n_thrus : int
            Number of transmissve standards. If None, we will try and
            guess for you by comparing measure transmission to trans_thres,

        trans_thres: float
            The  minimum transmission magnitude (in dB) that is
            the threshold for categorizing a transmissive standard.
            Compared to the measured s21,s12  meaned over frequency
            Only use if n_thrus=None.

        See Also
        -----------
        Calibration.__init__


        '''

        kwargs.update({'measured':measured,
                       'ideals':ideals})

        # note: this will enable sloppy_input and align stds if neccesary
        Calibration.__init__(self, *args, **kwargs)

        # if they didnt tell us the number of thrus, then lets
        # hueristcally determine it

        if n_thrus is None:
            warn('n_thrus is None, guessing which stds are transmissive')
            n_thrus=0
            for k in self.ideals:
                mean_trans = NetworkSet([k.s21, k.s12]).mean_s_mag
                trans_db = npy.mean(mean_trans.s_db.flatten())

                # this number is arbitrary but reasonable
                if trans_db > trans_thres and not npy.isneginf(trans_db).all():
                    n_thrus +=1


            if n_thrus ==0:
                raise ValueError('couldnt find a transimssive standard. check your data, or explicitly use `n_thrus` argument')
        self.n_thrus = n_thrus

        # if they didntly give explicit order, lets try and put the
        # more transmissive standards last, by sorted measured/ideals
        # based on mean s21
        if self.sloppy_input is True:
            trans = [npy.mean(k.s21.s_mag) for k in self.ideals]
            # see http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
            # get order of indecies of sorted means s21
            order = [x for (y,x) in sorted(zip(trans, range(len(trans))),\
                                           key=lambda pair: pair[0])]
            self.measured = [self.measured[k] for k in order]
            self.ideals = [self.ideals[k] for k in order]

    def run(self):
        '''
        '''
        n_thrus = self.n_thrus
        p1_m = [k.s11 for k in self.measured[:-n_thrus]]
        p2_m = [k.s22 for k in self.measured[:-n_thrus]]
        p1_i = [k.s11 for k in self.ideals[:-n_thrus]]
        p2_i = [k.s22 for k in self.ideals[:-n_thrus]]
        thrus = self.measured[-n_thrus:]
        ideal_thrus = self.ideals[-n_thrus:]

        # create one port calibration for reflective standards
        port1_cal = OnePort(measured = p1_m, ideals = p1_i)
        port2_cal = OnePort(measured = p2_m, ideals = p2_i)

        # cal coefficient dictionaries
        p1_coefs = dict(port1_cal.coefs)
        p2_coefs = dict(port2_cal.coefs)

        if self.kwargs.get('isolation',None) is not None:
            raise NotImplementedError()
            p1_coefs['isolation'] = isolation.s21.s.flatten()
            p2_coefs['isolation'] = isolation.s12.s.flatten()
        else:
            p1_coefs['isolation'] = npy.zeros(len(self.frequency), dtype=complex)
            p2_coefs['isolation'] = npy.zeros(len(self.frequency), dtype=complex)


        # loop thru thrus, and calculate error terms for each one
        # load match and transmission tracking for ports 1 and 2
        lm1, lm2,tt1, tt2 = [],[],[],[]
        for thru, thru_i in zip(thrus, ideal_thrus):
            lm1.append(thru_i.inv**port1_cal.apply_cal(thru.s11))
            lm2.append(thru_i.flipped().inv**port2_cal.apply_cal(thru.s22))

            # forward transmission tracking
            g = lm1[-1].s
            d = p1_coefs['source match'].reshape(-1,1,1)
            e,f,b,h = thru_i.s11.s, thru_i.s22.s,thru_i.s21.s,thru_i.s12.s
            m = thru.s21.s

            ac = m*1./b * (1 - (d*e + f*g + b*g*h*d) + (d*e*f*g) )
            tt1.append(ac[:])

            # reverse transmission tracking
            thru.flip(),thru_i.flip() # flip thrus to keep same ports as above
            g = lm2[-1].s
            d = p2_coefs['source match'].reshape(-1,1,1)

            e,f,b,h = thru_i.s11.s, thru_i.s22.s,thru_i.s21.s,thru_i.s12.s
            m = thru.s21.s

            ac = m*1./b * (1 - (d*e+f*g+b*g*h*d) + d*e*f*g)
            tt2.append(ac[:])

            thru.flip(), thru_i.flip() # flip em back

        p1_coefs['transmission tracking'] = npy.mean(npy.array(tt1),axis=0).flatten()
        p2_coefs['transmission tracking'] = npy.mean(npy.array(tt2),axis=0).flatten()
        p1_coefs['load match'] = NetworkSet(lm1).mean_s.s.flatten()
        p2_coefs['load match'] = NetworkSet(lm2).mean_s.s.flatten()


        # update coefs
        coefs = {}

        coefs.update(dict([('forward %s'%k, p1_coefs[k]) for k in p1_coefs]))
        coefs.update(dict([('reverse %s'%k, p2_coefs[k]) for k in p2_coefs]))
        eight_term_coefs = convert_12term_2_8term(coefs)

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


class SOLT(TwelveTerm):
    '''
    Short Open Load Thru, Full two-port calibration.

    SOLT is the traditional, fully determined, two-port calibration
    originally developed in [1]_.
    Although the acronym SOLT implies the use of 4 standards, skrf's
    algorithm can accept any number of reflect standards,  If
    more than 3 reflect standards are provided a least-squares solution
    is implemented for the one-port stage of the calibration.

    If your `thru` is not flush you need to use `TwelveTerm` instead of
    SOLT.

    Redundant flush thru measurements can also be used, through the `n_thrus`
    parameter. See :func:`__init__`

    References
    ------------
    .. [1] W. Kruppa and K. F. Sodomsky, "An Explicit Solution for the Scattering Parameters of a Linear Two-Port Measured with an Imperfect Test Set (Correspondence)," IEEE Transactions on Microwave Theory and Techniques, vol. 19, no. 1, pp. 122-123, Jan. 1971.


    See Also
    ---------
    TwelveTerm

    '''
    family = 'SOLT'
    def __init__(self, measured, ideals, n_thrus=1, *args, **kwargs):
        '''
        SOLT initializer

        If you arent using `sloppy_input`, then the order of the
        standards must align.

        If `n_thrus!=None`, then the thru standard[s] must be last in
        the list. The `n_thrus` argument can be used to allow  multiple
        measurements of the flush thru standard.

        If the ideal element for the thru is set to None, a flush thru
        is assumed. If your `thru` is not flush you need
        to use `TwelveTerm` instead of SOLT. Use

        Notes
        ------
        See func:`Calibration.__init__` for details about  automatic
        standards alignment (aka `sloppy_input`).

        Parameters
        -------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use `sloppy_input`)
            The thru standard can be None

        n_thrus : int
            number of thru measurments

        See Also
        ------------
        TwelveTerm.__init__
        '''

        # see if they passed a None for the thru, and if so lets
        # make an ideal flush thru for them
        for k in range(-n_thrus,len(ideals)):
            if ideals[k] is None:
                if (n_thrus is None) or (hasattr(ideals, 'keys')) or \
                   (hasattr(measured, 'keys')):
                    raise ValueError('Cant use sloppy_input and have the ideal thru be None. measured and ideals must be lists, or dont use None for the thru ideal.')

                ideal_thru = measured[0].copy()
                ideal_thru.s[:,0,0] = 0
                ideal_thru.s[:,1,1] = 0
                ideal_thru.s[:,1,0] = 1
                ideal_thru.s[:,0,1] = 1
                ideals[k] = ideal_thru

        kwargs.update({'measured':measured,
                       'ideals':ideals,
                       'n_thrus':n_thrus})

        TwelveTerm.__init__(self,*args, **kwargs)



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
        else:
            p1_coefs['isolation'] = npy.zeros(len(thru), dtype=complex)
            p2_coefs['isolation'] = npy.zeros(len(thru), dtype=complex)

        p1_coefs['load match'] = port1_cal.apply_cal(thru.s11).s.flatten()
        p2_coefs['load match'] = port2_cal.apply_cal(thru.s22).s.flatten()

        p1_coefs['transmission tracking'] = \
            (thru.s21.s.flatten() - p1_coefs.get('isolation',0))*\
            (1. - p1_coefs['source match']*p1_coefs['load match'])
        p2_coefs['transmission tracking'] = \
            (thru.s12.s.flatten() - p2_coefs.get('isolation',0))*\
            (1. - p2_coefs['source match']*p2_coefs['load match'])
        coefs = {}

        coefs.update(dict([('forward %s'%k, p1_coefs[k]) for k in p1_coefs]))
        coefs.update(dict([('reverse %s'%k, p2_coefs[k]) for k in p2_coefs]))
        eight_term_coefs = convert_12term_2_8term(coefs)

        coefs.update(dict([(l, eight_term_coefs[l]) for l in \
            ['forward switch term','reverse switch term','k'] ]))
        self._coefs = coefs

class TwoPortOnePath(TwelveTerm):
    '''
    Two Port One Path Calibration (aka poor man's TwelveTerm)

    Provides full errror correction  on a switchless three reciever 
    system, ie you can only measure the waves a1,b1,and b2. 
    Given this architecture, the DUT must be flipped and measured 
    twice to be fully corrected.

    To allow for this, the `apply_cal` method takes a tuple of
    measurements in the order  (forward,reverse), and creates a composite
    measurement that is correctable.

    '''
    family = 'TwoPortOnePath'

    def __init__(self, measured, ideals,n_thrus=None,  source_port=1,
                 *args, **kwargs):
        '''
        initializer

        Use the  `n_thrus` argument to explicity define the number of
        transmissive standards. Otherwise, if `n_thrus=None`, then we
        will try and guess which are transmissive, by comparing the mean
        |s21| and |s12| responses (in dB) to `trans_thres`.

        Notes
        ------
        See func:`Calibration.__init__` for details about  automatic
        standards alignment (aka `sloppy_input`).

        Parameters
        -------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input

        n_thrus : int
            number of thru measurments

        source_port : [1,2]
            The port on which the source is active. should be 1 or 2



        See Also
        ------------
        TwelveTerm.__init__
        '''



        self.sp = source_port-1
        self.rp = 1 if self.sp == 0 else 0
        kwargs.update({'measured':measured,
                       'ideals':ideals,
                       'n_thrus':n_thrus})
        TwelveTerm.__init__(self,*args, **kwargs)


    def run(self):
        '''

        if self.sp !=0:
            raise NotImplementedError('not implemented yet. you can just flip() all your data though. ')
        n_thrus = self.n_thrus
        p1_m = [k.s11 for k in self.measured[:-n_thrus]]
        p1_i = [k.s11 for k in self.ideals[:-n_thrus]]
        thru = NetworkSet(self.measured[-n_thrus:]).mean_s

        # create one port calibration for reflective standards
        port1_cal = OnePort(measured = p1_m, ideals = p1_i)

        # cal coefficient pdictionaries
        p1_coefs = port1_cal.coefs

        if self.kwargs.get('isolation',None) is not None:
            raise NotImplementedError()
            p1_coefs['isolation'] = isolation.s21.s.flatten()
        else:
            p1_coefs['isolation'] = npy.zeros(len(thru), dtype=complex)

        p1_coefs['load match'] = port1_cal.apply_cal(thru.s11).s.flatten()
        p1_coefs['transmission tracking'] = \
            (thru.s21.s.flatten() - p1_coefs.get('isolation',0))*\
            (1. - p1_coefs['source match']*p1_coefs['load match'])

        coefs = {}'''

        # run a full twelve term then just copy all forward error terms
        # over reverse error terms
        TwelveTerm.run(self)


        out_coefs = self.coefs.copy()

        if self.sp ==0:
            forward = 'forward'
            reverse = 'reverse'
        elif self.sp ==1:
            forward = 'reverse'
            reverse = 'forward'
        else:
            raise('source_port is out of range. should be 1 or 2.')
        for k in self.coefs:
            if k.startswith(forward):
                k_out = k.replace(forward,reverse)
                out_coefs[k_out] = self.coefs[k]

        eight_term_coefs = convert_12term_2_8term(out_coefs)
        out_coefs.update(dict([(l, eight_term_coefs[l]) for l in \
            ['forward switch term','reverse switch term','k'] ]))
        self._coefs = out_coefs

    def apply_cal(self, ntwk_tuple):
        '''
        apply the calibration to a measuremnt

        Notes
        -------
        Full correction is possible given you have measured your DUT 
        in both orientations. Meaning, you have measured the device, 
        then physically flipped the device and made a second measurement. 
        
        This tuple of 2-port Networks is what is meant by
        (forward,reverse), in the docstring below
        
        If you pass a single 2-port Network, then the measurement will 
        only be partially corrected using what is known as the 
        `EnhancedResponse` calibration. 

        Parameters
        -----------
        network_tuple: tuple, or Network
            tuple of 2-port Networks in order (forward, reverse) OR 
            a single 2-port Network. 



        '''
        if isinstance(ntwk_tuple,tuple) or isinstance(ntwk_tuple,list):
            f,r = ntwk_tuple[0].copy(), ntwk_tuple[1].copy()
            sp,rp = self.sp,self.rp
            ntwk = f.copy()
            ntwk.s[:,sp,sp] = f.s[:,sp,sp]
            ntwk.s[:,rp,sp] = f.s[:,rp,sp]
            ntwk.s[:,rp,rp] = r.s[:,sp,sp]
            ntwk.s[:,sp,rp] = r.s[:,rp,sp]

            out = TwelveTerm.apply_cal(self, ntwk)
            return out

        else:
            warnings.warn('only gave a single measurement orientation, error correction is partial without a tuple')
            ntwk = ntwk_tuple.copy()
            sp,rp = self.sp,self.rp

            ntwk.s[:,rp,rp] = 0
            ntwk.s[:,sp,rp] = 0
            out = TwelveTerm.apply_cal(self, ntwk)
            out.s[:,rp,rp] = 0
            out.s[:,sp,rp] = 0

            return out

class EnhancedResponse(TwoPortOnePath):
    '''
    Enhanced Response Partial Calibration

    Why are you using this?
    For full error you correction, you can measure  the DUT in both
    orientations and instead use TwoPortOnePath

    Accuracy of correct measurements will rely on having a good match
    at the passive side of the DUT.

    For code-structuring reasons, this is a dummy placeholder class.
    Its just TwoPortOnePath, which defaults to enhancedresponse correction
    when you apply the calirbation to a single network, and not a tuple 
    of networks.
    '''
    family = 'EnhancedResponse'

class EightTerm(Calibration):
    '''
    General EightTerm (aka Error-box) Two-port calibration

    This is basically an extension of the one-port algorithm to two-port
    measurements, A least squares estimator is used to determine the
    error coefficients. No self-calibration takes place.
    The concept is presented in [1]_ , but implementation follows that
    of  [2]_ .

    See :func:`__init__`

    Notes
    -------
    An important detail of implementing the error-box
    model is that the internal switch must be correctly accounted for.
    This is done through the measurement of :term:`switch terms`.



    References
    ------------

    .. [1] Speciale, R.A.; , "A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors," Microwave Theory and Techniques, IEEE Transactions on , vol.25, no.12, pp. 1100- 1115, Dec 1977. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1129282&isnumber=25047

    .. [2] Rytting, D. (1996) Network Analyzer Error Models and Calibration Methods. RF 8: Microwave Measurements for Wireless Applications (ARFTG/NIST Short Course Notes)



    '''
    family = 'EightTerm'
    def __init__(self, measured, ideals, switch_terms=None,
                 *args, **kwargs):
        '''
        EightTerm Initializer

        Notes
        ------
        See func:`Calibration.__init__` for details about  automatic
        standards alignment (aka `sloppy_input`).

        Parameters
        --------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list

        switch_terms : tuple of :class:`~skrf.network.Network` objects
            the pair of switch terms in the order (forward, reverse)


        '''

        self.switch_terms = switch_terms
        if switch_terms is None:
            warn('No switch terms provided')
        Calibration.__init__(self,
            measured = measured,
            ideals = ideals,
            *args, **kwargs)


    def unterminate(self,ntwk):
        '''
        Unterminates switch terms from a raw measurement.

        See Also
        ---------
        calibration.unterminate
        '''
        if self.switch_terms is not None:
            gamma_f, gamma_r = self.switch_terms
            return unterminate(ntwk, gamma_f, gamma_r)

        else:
            return ntwk

    def terminate(self, ntwk):
        '''
        Terminate a  network with  switch terms

        See Also
        --------
        calibration.terminate
        '''
        if self.switch_terms is not None:
            gamma_f, gamma_r = self.switch_terms
            return terminate(ntwk, gamma_f, gamma_r)
        else:
            return ntwk


    @property
    def measured_unterminated(self):
        return [self.unterminate(k) for k in self.measured]

    def run(self):
        numStds = self.nstandards
        numCoefs = 7


        mList = [k.s  for k in self.measured_unterminated]
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
        for f in list(range(fLength)):
            # loop through standards and fill matrix
            for k in list(range(numStds)):
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

        for f in list(range(len(ntwk.s))):
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

        for f in list(range(len(ntwk.s))):
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

    A Similar self-calibration algorithm as developed by Engen and
    Hoer [1]_, more closely following into a more matrix form in [2]_.
    
    
    .. warning::
        This version of TRL does not solve for the Reflect standard yet


    See Also
    ------------
    determine_line function which actually determines the line s-parameters



    References
    ------------
    .. [1] G. F. Engen and C. A. Hoer, "Thru-Reflect-Line: An Improved Technique for Calibrating the Dual Six-Port Automatic Network Analyzer," IEEE Transactions on Microwave Theory and Techniques, vol. 27, no. 12, pp. 987-993, 1979.

    .. [2] H.-J. Eul and B. Schiek, "A generalized theory and new calibration procedures for network analyzer self-calibration," IEEE Transactions on Microwave Theory and Techniques, vol. 39, no. 4, pp. 724-731, 1991.


    '''
    family = 'TRL'
    def __init__(self, measured, ideals=None, estimate_line=False, 
                n_reflects=1,*args,**kwargs):
        '''
        Initialize a TRL calibration

        Note that the order of `measured` and `ideals` is strict.
        It must be [Thru, Reflect, Line]. A multiline algorithms is 
        used if more than one line is passed. Multiple reflects can
        also be used, see `n_reflects` argument.

        All of the `ideals` can be indivdually set to None, or the entire
        list set to None (`ideals=None`). For each ideal set to None 
        the following assumptions are made: 
        
        * thru : flush thru 
        * reflect : flush shorts 
        * line : and approximaitly  90deg  matched line (can be lossy)
        
        Note you can also use the `estimate_line` option  to 
        automatically  estimate the initial guess for the line length 
        from measurements . This is sensible
        if you have no idea what the line length is, but your **error 
        networks** are well macthed (E_ij >>E_ii).


        .. warning::
            This version of TRL does not solve for the Reflect standard yet


        Notes
        -------
        This implementation inherits from :class:`EightTerm`. dont
        forget to pass switch_terms.


        Parameters
        --------------
        measured : list of :class:`~skrf.network.Network`
             must be in order [Thru, Reflect, Line]

        ideals : list of :class:`~skrf.network.Network`, None
            must be in order [Thru, Reflect, Line]. Each element in the 
            list may be None, or equivalently, the list may be None

        estimate_line : bool
            Estimates the length of the line standard from raw measurements.
            
        n_reflects :  1
            number of reflective standards 

        \*args, \*\*kwargs :  passed to EightTerm.__init__
            dont forget the `switch_terms` argument is important


        '''
        warn('Value of Reflect is not solved for yet.')

        n_stds = len(measured)
            
        if ideals is None:
            ideals = [None]*len(measured)
            
        if ideals[0] is None:
            # lets make an ideal flush thru for them
            ideal_thru = measured[0].copy()
            ideal_thru.s[:,0,0] = 0
            ideal_thru.s[:,1,1] = 0
            ideal_thru.s[:,1,0] = 1
            ideal_thru.s[:,0,1] = 1
            ideals[0] = ideal_thru
        
        for k in range(1,n_reflects+1):
            if ideals[k] is None:
                # assume they are using flushshorts
                ideal_reflect = measured[k].copy()
                ideal_reflect.s[:,0,0] = -1
                ideal_reflect.s[:,1,1] = -1
                ideal_reflect.s[:,1,0] = 0
                ideal_reflect.s[:,0,1] = 0
                ideals[k] = ideal_reflect
        
        for k in range(n_reflects+1,n_stds):
            if ideals[k] is None:
                # lets make an 90deg line for them
                ideal_line = measured[k].copy()
                ideal_line.s[:,0,0] = 0
                ideal_line.s[:,1,1] = 0
                ideal_line.s[:,1,0] = -1j
                ideal_line.s[:,0,1] = -1j
                ideals[k] = ideal_line



        EightTerm.__init__(self,
            measured = measured,
            ideals = ideals,
            *args, **kwargs)


        m_ut = self.measured_unterminated
        for k in range(n_reflects+1,n_stds):
            if estimate_line:
                # setting line_approx  to None causes determine_line() to 
                # estimate the line length from raw measurements
                line_approx = None
            else:
                line_approx = ideals[k]
            
            self.ideals[k] = determine_line(m_ut[0], m_ut[k], line_approx) # find line

MultilineTRL = TRL

    
    

class UnknownThru(EightTerm):
    '''
    Two-Port Self-Calibration allowing the *thru* standard to be unknown.

    This algorithm was originally developed in  [1]_, and
    is based on the 8-term error model (:class:`EightTerm`). It allows
    the *thru* to be unknown, other than it must be reciprocal. This
    is useful when when a well-known thru is not realizable.


    References
    -----------
    .. [1] A. Ferrero and U. Pisani, "Two-port network analyzer calibration using an unknown `thru,`" IEEE Microwave and Guided Wave Letters, vol. 2, no. 12, pp. 505-507, 1992.

    '''
    family = 'UnknownThru'
    def __init__(self, measured, ideals,  *args, **kwargs):
        '''
        Initializer

        Note that the *thru* standard must be last in both measured, and
        ideal lists. The ideal for the *thru* is only used to choose
        the sign of a square root. Thus, it only has to be have s21, s12
        known within :math:`\pi` phase.



        Parameters
        --------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter ( or use `sloppy_input`)

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list ( or use sloppy_input

        switch_terms : tuple of :class:`~skrf.network.Network` objects
            the pair of switch terms in the order (forward, reverse)
        '''

        EightTerm.__init__(self, measured = measured, ideals = ideals,
                           *args, **kwargs)


    def run(self):
        p1_m = [k.s11 for k in self.measured_unterminated[:-1]]
        p2_m = [k.s22 for k in self.measured_unterminated[:-1]]
        p1_i = [k.s11 for k in self.ideals[:-1]]
        p2_i = [k.s22 for k in self.ideals[:-1]]

        thru_m = self.measured_unterminated[-1]

        thru_approx  =  self.ideals[-1]

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
        # this is really inefficient, i need to work out the math on the
        # closed form solution
        et = EightTerm(
            measured = self.measured,
            ideals = self.ideals,
            switch_terms= self.switch_terms)
        k_approx = et.coefs['k'].flatten()

        # this is equivalent to sqrt(detX*detY/detM)
        e10e32 = npy.sqrt((e_rf*e_rr*thru_m.s21/thru_m.s12).s.flatten())

        k_ = e10e32/e_rr.s.flatten()
        k_ = find_closest(k_, -1*k_, k_approx)

        #import pylab as plb
        #plot(abs(k_-k_approx))
        #plb.show()
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
            warn('No switch terms provided')
            coefs.update({
                'forward switch term': npy.zeros(len(self.frequency), dtype=complex),
                'reverse switch term': npy.zeros(len(self.frequency), dtype=complex),
                })

        coefs.update({'k':k_})

        self.coefs = coefs

class MRC(UnknownThru):
    '''
    Misalignment Resistance Calibration

    This is an error-box based calibration that is a combination of the
    SDDL[1]_ and the UnknownThru[2]_, algorithms.
    The self-calibration aspects of these two algorithms alleviate the
    need to know the phase of the delay shorts, as well as the exact
    response of the thru. Thus the calibration is resistant to
    waveguide flangemisalignment.


    References
    -----------
    .. [1] Z. Liu and R. M. Weikle, "A reflectometer calibration method resistant to waveguide flange misalignment," Microwave Theory and Techniques, IEEE Transactions on, vol. 54, no. 6, pp. 2447-2452, Jun. 2006.

    .. [2] A. Ferrero and U. Pisani, "Two-port network analyzer calibration using an unknown `thru,`" IEEE Microwave and Guided Wave Letters, vol. 2, no. 12, pp. 505-507, 1992.


    '''
    family = 'MRC'
    def __init__(self, measured, ideals,  *args, **kwargs):
        '''
        Initializer

        This calibration takes exactly 5 standards, which must be in the
        order:

            [Short, DelayShort1, DelayShort1, Load, Thru]

        The ideals for the delay shorts are not used and the ideal for
        the *thru* is only used to choose
        the sign of a square root. Thus, it only has to be have s21, s12
        known within :math:`\pi` phase.


        Parameters
        --------------
        measured : list/dict  of :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter

        ideals : list/dict of :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list

        switch_terms : tuple of :class:`~skrf.network.Network` objects
            the pair of switch terms in the order (forward, reverse)
        '''

        UnknownThru.__init__(self, measured = measured, ideals = ideals,
                           *args, **kwargs)


    def run(self):
        p1_m = [k.s11 for k in self.measured_unterminated[:-1]]
        p2_m = [k.s22 for k in self.measured_unterminated[:-1]]
        p1_i = [k.s11 for k in self.ideals[:-1]]
        p2_i = [k.s22 for k in self.ideals[:-1]]

        thru_m = self.measured_unterminated[-1]

        thru_approx  =  self.ideals[-1]

        # create one port calibration for all reflective standards
        port1_cal = SDDL(measured = p1_m, ideals = p1_i)
        port2_cal = SDDL(measured = p2_m, ideals = p2_i)

        # cal coefficient dictionaries
        p1_coefs = port1_cal.coefs.copy()
        p2_coefs = port2_cal.coefs.copy()

        e_rf = port1_cal.coefs_ntwks['reflection tracking']
        e_rr = port2_cal.coefs_ntwks['reflection tracking']
        X = port1_cal.error_ntwk
        Y = port2_cal.error_ntwk

        # create a fully-determined 8-term cal just get estimate on k's sign
        # this is really inefficient, i need to work out the math on the
        # closed form solution
        et = EightTerm(
            measured = self.measured,
            ideals = self.ideals,
            switch_terms= self.switch_terms)
        k_approx = et.coefs['k'].flatten()

        # this is equivalent to sqrt(detX*detY/detM)
        e10e32 = npy.sqrt((e_rf*e_rr*thru_m.s21/thru_m.s12).s.flatten())

        k_ = e10e32/e_rr.s.flatten()
        k_ = find_closest(k_, -1*k_, k_approx)

        #import pylab as plb
        #plot(abs(k_-k_approx))
        #plb.show()
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
            warn('No switch terms provided')
            coefs.update({
                'forward switch term': npy.zeros(len(self.frequency), dtype=complex),
                'reverse switch term': npy.zeros(len(self.frequency), dtype=complex),
                })

        coefs.update({'k':k_})

        self.coefs = coefs

class Normalization(Calibration):
    '''
    Simple Thru Normalization
    '''
    def run(self):
        pass
    def apply_cal(self, input_ntwk):
        return input_ntwk/average(self.measured)


## Functions



    



def ideal_coefs_12term(frequency):
    '''
    An ideal set of 12term calibration coefficients

    Produces a set of error coefficients, that would result if the
    error networks were matched thrus
    '''

    zero = zeros(len(frequency), dtype='complex')
    one = ones(len(frequency), dtype='complex')
    ideal_coefs = {}
    ideal_coefs.update({k:zero for k in [\
        'forward directivity',
        'forward source match',
        'forward load match',
        'reverse directivity',
        'reverse load match',
        'reverse source match',
        ]})

    ideal_coefs.update({k:one for k in [\
        'forward reflection tracking',
        'forward transmission tracking',
        'reverse reflection tracking',
        'reverse transmission tracking',
        ]})

    return ideal_coefs

def unterminate(ntwk, gamma_f, gamma_r):
        '''
        Unterminates switch terms from a raw measurement.

        In order to use the 8-term error model on a VNA which employs a
        switched source, the effects of the switch must be accounted for.
        This is done through `switch terms` as described in  [1]_ . The
        two switch terms are defined as,

        .. math ::

            \\Gamma_f = \\frac{a2}{b2} ,\\qquad\\text{sourced by port 1}\\
            \\Gamma_r = \\frac{a1}{b1} ,\\qquad\\text{sourced by port 2}

        These can be measured by four-sampler VNA's by setting up
        user-defined traces onboard the VNA. If the VNA doesnt have
        4-samplers, then you can measure switch terms indirectly by using a
        two-tier two-port calibration. First do a SOLT, then convert
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
            gamma_r = a1/b1 sourced by port2

        Returns
        -----------
        ntwk :  Network object

        References
        ------------

        .. [1] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks
        '''


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

def terminate(ntwk, gamma_f, gamma_r):
        '''
        Terminate a  network with  switch terms

        see [1]_


        Parameters
        -------------
        two_port : 2-port Network
            an unterminated network
        gamma_f : 1-port Network
            measured forward switch term.
            gamma_f = a2/b2 sourced by port1
        gamma_r : 1-port Network
            measured reverse switch term
            gamma_r = a1/b1 sourced by port2

        Returns
        -----------
        ntwk :  Network object

        See Also
        --------
        unterminate_switch_terms

        References
        ------------

        .. [1] "Formulations of the Basic Vector Network Analyzer Error
                Model including Switch Terms" by Roger B. Marks
        '''

        m = ntwk.copy()
        ntwk_flip = ntwk.copy()
        ntwk_flip.flip()

        m.s[:,0,0] = (ntwk**gamma_f).s[:,0,0]
        m.s[:,1,1] = (ntwk_flip**gamma_r).s[:,0,0]
        m.s[:,1,0] = ntwk.s[:,1,0]/(1-ntwk.s[:,1,1]*gamma_f.s[:,0,0])
        m.s[:,0,1] = ntwk.s[:,0,1]/(1-ntwk.s[:,0,0]*gamma_r.s[:,0,0])
        return m

def determine_line(thru_m, line_m, line_approx=None):
    '''
    Determine S21 of a matched line.

    Given raw measurements of a `thru` and a matched `line` with unknown
    s21, this will calculate the response of the line. This works for 
    lossy lines, and attentuators. The `line_approx`
    is an approximation to line, this used  to choose the correct
    root sign. If left as None, it will be estimated from raw measurements, 
    which requires your error networks to be well matched  (S_ij >>S_ii). 
    

    This is possible because two measurements can be combined to
    create a relationship of similar matrices, as shown below. Equating
    the eigenvalues between these measurements allows one to solve for S21
    of the line.

    .. math::

        M_t = X \\cdot A_t \\cdot Y    \\\\
        M_l = X \\cdot A_l \\cdot Y\\\\

        M_t \\cdot M_{l}^{-1} = X \\cdot A_t \\cdot A_{l}^{-1} \\cdot X^{-1}\\\\

        eig(M_t \\cdot M_{l}^{-1}) = eig( A_t \\cdot A_{l}^{-1})\\\\

    which can be solved to yield S21 of the line

    Notes
    -------
    This relies on the 8-term error model, which requires that switch
    terms are accounted for. specifically, thru and line have their
    switch terms unterminated.

    Parameters
    -----------
    thru_m : :class:`~skrf.network.Network`
        raw measurement of a thru
    line_m : :class:`~skrf.network.Network`
        raw measurement of a matched transmissive standard
    line_approx : :class:`~skrf.network.Network`
        approximate network the ideal line response. if None, then
        the response is approximated by line_approx = line/thru. This
        makes the assumption that the error networks have much larger
        transmission than reflection


    References
    --------------

    '''

    npts = len(thru_m)
    one = npy.ones(npts)
    zero = npy.zeros(npts)

    if line_approx is None:
        # estimate line length, by assumeing error networks are well
        # matched
        line_approx = line_m/thru_m


    C = thru_m.inv**line_m
    # the eigen values of the matrix C, are equal to s12,s12^-1)
    # we need to choose the correct one
    w,v = linalg.eig(C.t)

    s12_0, s12_1 = w[:,0], w[:,1]

    s12 = find_correct_sign(s12_0, s12_1, line_approx.s[:,1,0])
    
    
    fm = [ -1* npy.trace(npy.dot(thru_m.t[f], npy.linalg.inv(line_m.t[f]))) \
        for f in list(range(npts))]
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
    gamma_f = (Elf - Esr)/(Err + Edr*(Elf - Esr))
    gamma_r = (Elr - Esf)/(Erf + Edf*(Elr - Esf))

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

def convert_pnacoefs_2_skrf(coefs):
    '''
    Convert  PNA error coefficients to skrf error coefficients

    Parameters
    ------------
    coefs : dict
        coefficients as retrieved from PNA
    ports : tuple
        port indices. in order (forward, reverse)

    Returns
    ----------
    skrf_coefs : dict
        same error coefficients but with keys matching skrf's convention

    '''

    coefs_map ={'Directivity':'directivity',
                'SourceMatch':'source match',
                'ReflectionTracking':'reflection tracking',
                'LoadMatch':'load match',
                'TransmissionTracking':'transmission tracking',
                'CrossTalk':'isolation'}

    skrf_coefs = {}

    if len(coefs) ==3:
        for k in coefs:
            coef= k[:-5]
            coef_key = coefs_map[coef]
            skrf_coefs[coef_key] = coefs[k]

    else:
        ports = list(set([k[-2] for k in coefs]))
        ports.sort(key=int)
        port_map ={ports[0]: 'forward',
                   ports[1]: 'reverse'}

        for k in coefs:
            coef,p1,p2 = k[:-5],k[-4],k[-2]
            # the source port has a different position for reflective
            # and transmissive standards
            if coef in ['Directivity','SourceMatch','ReflectionTracking']:
                coef_key = port_map[p1]+' '+coefs_map[coef]
            elif coef in ['LoadMatch','TransmissionTracking','CrossTalk']:
                coef_key = port_map[p2]+' '+coefs_map[coef]
            skrf_coefs[coef_key] = coefs[k]



    return skrf_coefs

def convert_skrfcoefs_2_pna(coefs, ports = (1,2)):
    '''
    Convert  skrf error coefficients to pna error coefficients

    Notes
    --------
    The skrf calibration terms can be found in variables
        * skrf.calibration.coefs_list_3term
        * skrf.calibration.coefs_list_12term


    Parameters
    ------------
    coefs : dict
        complex ndarrays for the cal coefficients as defined  by skrf
    ports : tuple
        port indices. in order (forward, reverse)

    Returns
    ----------
    pna_coefs : dict
        same error coefficients but with keys matching skrf's convention


    '''
    if not hasattr(ports, '__len__'):
        ports = ports,

    coefs_map ={'directivity':'Directivity',
                'source match':'SourceMatch',
                'reflection tracking':'ReflectionTracking',
                'load match':'LoadMatch',
                'transmission tracking':'TransmissionTracking',
                'isolation':'CrossTalk'}

    pna_coefs = {}

    if len(coefs)==3:
        for k in coefs:
            coef_key = coefs_map[k] + '(%i,%i)'%(ports[0],ports[0])
            pna_coefs[coef_key] = coefs[k]


    else:
        port_map_trans ={'forward':ports[1],
                         'reverse':ports[0]}
        port_map_refl  ={'forward':ports[0],
                         'reverse':ports[1]}

        for k in coefs:
            fr = k.split(' ')[0] # remove 'forward|reverse-ness'
            eterm = coefs_map[k.lstrip(fr)[1:] ]
            # the source port has a different position for reflective
            # and transmissive standards
            if eterm  in ['Directivity','SourceMatch','ReflectionTracking']:
                coef_key= eterm+'(%i,%i)'%(port_map_refl[fr],
                                           port_map_refl[fr])


            elif eterm in ['LoadMatch','TransmissionTracking','CrossTalk']:
                receiver_port = port_map_trans[fr]
                source_port = port_map_refl[fr]
                coef_key= eterm+'(%i,%i)'%(receiver_port,source_port)
            pna_coefs[coef_key] = coefs[k]


    return pna_coefs

def align_measured_ideals(measured, ideals):
    '''
    Aligns two lists of networks based on the intersection of their names.

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
            #TODO: make this better and maybe have phase continuity
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
