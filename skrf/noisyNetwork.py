# -*- coding: utf-8 -*-
"""
.. module:: skrf.noisyNetwork
========================================
noisyNetwork (:mod:`skrf.noisyNetwork`)
========================================


Provides a n-port network class and associated functions.

Much of the functionality in this module is provided as methods and
properties of the :class:`NoisyNetwork` Class.


NoisyNetwork Class
==================

.. autosummary::
    :toctree: generated/

    NoisyNetwork

Building NoisyNetwork
---------------------

.. autosummary::
    :toctree: generated/

    Network.from_z
    Network.from_y

Network Representations
============================

.. autosummary::
    :toctree: generated/

    Network.s
    Network.z
    Network.y
    Network.a
    Network.t

Connecting Networks
===============================

.. autosummary::
    :toctree: generated/

    connect
    innerconnect
    cascade
    cascade_list
    de_embed
    flip

Connecting Networks with Noise Analysis
=======================================

.. autosummary::
    :toctree: generated/

    cascade_2port
    parallel_parallel_2port
    series_series_2port


Interpolation and Concatenation Along Frequency Axis
=====================================================

.. autosummary::
    :toctree: generated/

    stitch
    overlap
    Network.resample
    Network.interpolate
    Network.interpolate_self

Combining Networks
===================================

.. autosummary::
    :toctree: generated/

    n_oneports_2_nport
    four_oneports_2_twoport
    three_twoports_2_threeport
    n_twoports_2_nport
    concat_ports

IO
====

.. autosummary::

    skrf.io.general.read
    skrf.io.general.write
    skrf.io.general.ntwk_2_spreadsheet
    Network.write
    Network.write_touchstone
    Network.read
    Network.write_spreadsheet

Noise
============
.. autosummary::
    :toctree: generated/

    Network.add_noise_polar
    Network.add_noise_polar_flatband
    Network.multiply_noise
    Network.noise_source

Network Noise Covariance Representations
========================================
.. autosummary::
    :toctree: generated/

    Network.cs
    Network.ct
    Network.cz
    Network.cy
    Network.ca


Supporting Functions
======================

.. autosummary::
    :toctree: generated/

    inv
    connect_s
    innerconnect_s
    s2z
    s2y
    s2t
    s2a
    z2s
    z2y
    z2t
    z2a
    y2s
    y2z
    y2t
    t2s
    t2z
    t2y
    fix_z0_shape
    renormalize_s
    passivity
    reciprocity

Misc Functions
=====================
.. autosummary::
    :toctree: generated/

    average
    two_port_reflect
    chopinhalf
    Network.nudge
    Network.renormalize

"""

from six.moves import xrange
from functools import reduce

import os
import warnings

import six.moves.cPickle as pickle
from six.moves.cPickle import UnpicklingError
from six import string_types

import sys
import re
import zipfile
from copy import deepcopy as copy
from numbers import Number
from itertools import product

import numpy as npy
from numpy.linalg import inv as npy_inv
from numpy import fft, gradient, reshape, shape, ones
from scipy import stats, signal  # for Network.add_noise_*, and Network.windowed
from scipy.interpolate import interp1d  # for Network.interpolate()
from scipy.ndimage.filters import convolve1d
import unittest  # fotr unitest.skip

from . import mathFunctions as mf
from .frequency import Frequency
from .network import Network
from .networkNoiseCov import NetworkNoiseCov
from .tlineFunctions import zl_2_Gamma0
from .util import get_fid, get_extn, find_nearest_index, slice_domain, network_array
from .time import time_gate
# later imports. delayed to solve circular dependencies
# from .io.general import read, write
# from .io import touchstone
# from .io.general import network_2_spreadsheet
# from media import Freespace

from .constants import ZERO, K_BOLTZMANN, T0
from .constants import S_DEFINITIONS, S_DEF_DEFAULT

from . import is_alt_ops


#from matplotlib import cm
#import matplotlib.pyplot as plt
#import matplotlib.tri as tri
#from scipy.interpolate import interp1d


class NoiseyNetwork(Network):
    """
    A n-port electrical network [#]_.

    For instructions on how to create Network see  :func:`__init__`.

    A n-port network may be defined by three quantities,
     * network parameter matrix (s, z, or y-matrix)
     * port characteristic impedance matrix
     * frequency information

    The :class:`Network` class stores these data structures internally
    in the form of complex :class:`numpy.ndarray`'s. These arrays are not
    interfaced directly but instead through the use of the properties:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s`              scattering parameter matrix
    :attr:`z0`             characteristic impedance matrix
    :attr:`f`              frequency vector
    =====================  =============================================

    Although these docs focus on s-parameters, other equivalent network
    representations such as :attr:`z` and  :attr:`y` are
    available. Scalar projections of the complex network parameters
    are accessible through properties as well. These also return
    :class:`numpy.ndarray`'s.

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s_re`           real part of the s-matrix
    :attr:`s_im`           imaginary part of the s-matrix
    :attr:`s_mag`          magnitude of the s-matrix
    :attr:`s_db`           magnitude in log scale of the s-matrix
    :attr:`s_deg`          phase of the s-matrix in degrees
    =====================  =============================================

    The following operations act on the networks s-matrix.

    =====================  =============================================
    Operator               Function
    =====================  =============================================
    \+                     element-wise addition of the s-matrix
    \-                     element-wise difference of the s-matrix
    \*                     element-wise multiplication of the s-matrix
    \/                     element-wise division of the s-matrix
    \*\*                   cascading (only for 2-ports)
    \//                    de-embedding (for 2-ports, see :attr:`inv`)
    =====================  =============================================

    Different components of the :class:`Network` can be visualized
    through various plotting methods. These methods can be used to plot
    individual elements of the s-matrix or all at once. For more info
    about plotting see the :doc:`../../tutorials/plotting` tutorial.

    =========================  =============================================
    Method                     Meaning
    =========================  =============================================
    :func:`plot_s_smith`       plot complex s-parameters on smith chart
    :func:`plot_s_re`          plot real part of s-parameters vs frequency
    :func:`plot_s_im`          plot imaginary part of s-parameters vs frequency
    :func:`plot_s_mag`         plot magnitude of s-parameters vs frequency
    :func:`plot_s_db`          plot magnitude (in dB) of s-parameters vs frequency
    :func:`plot_s_deg`         plot phase of s-parameters (in degrees) vs frequency
    :func:`plot_s_deg_unwrap`  plot phase of s-parameters (in unwrapped degrees) vs frequency

    =========================  =============================================

    :class:`Network`  objects can be  created from a touchstone or pickle
    file  (see :func:`__init__`), by a
    :class:`~skrf.media.media.Media` object, or manually by assigning the
    network properties directly. :class:`Network`  objects
    can be saved to disk in the form of touchstone files with the
    :func:`write_touchstone` method.

    An exhaustive list of :class:`Network` Methods and Properties
    (Attributes) are given below

    References
    ------------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
    """

    global PRIMARY_PROPERTIES
    PRIMARY_PROPERTIES = ['s', 'z', 'y', 'a', 'h']

    global COMPONENT_FUNC_DICT
    COMPONENT_FUNC_DICT = {
        're': npy.real,
        'im': npy.imag,
        'mag': npy.abs,
        'db': mf.complex_2_db,
        'db10': mf.complex_2_db10,
        'rad': npy.angle,
        'deg': lambda x: npy.angle(x, deg=True),
        'arcl': lambda x: npy.angle(x) * npy.abs(x),
        'rad_unwrap': lambda x: mf.unwrap_rad(npy.angle(x)),
        'deg_unwrap': lambda x: mf.radian_2_degree(mf.unwrap_rad( \
            npy.angle(x))),
        'arcl_unwrap': lambda x: mf.unwrap_rad(npy.angle(x)) * \
                                 npy.abs(x),
        # 'gd' : lambda x: -1 * npy.gradient(mf.unwrap_rad(npy.angle(x)))[0], # removed because it depends on `f` as well as `s`
        'vswr': lambda x: (1 + abs(x)) / (1 - abs(x)),
        'time': mf.ifft,
        'time_db': lambda x: mf.complex_2_db(mf.ifft(x)),
        'time_mag': lambda x: mf.complex_2_magnitude(mf.ifft(x)),
        'time_impulse': None,
        'time_step': None,
    }
    # provides y-axis labels to the plotting functions
    global Y_LABEL_DICT
    Y_LABEL_DICT = {
        're': 'Real Part',
        'im': 'Imag Part',
        'mag': 'Magnitude',
        'abs': 'Magnitude',
        'db': 'Magnitude (dB)',
        'db10': 'Magnitude (dB)',
        'deg': 'Phase (deg)',
        'deg_unwrap': 'Phase (deg)',
        'rad': 'Phase (rad)',
        'rad_unwrap': 'Phase (rad)',
        'arcl': 'Arc Length',
        'arcl_unwrap': 'Arc Length',
        'gd': 'Group Delay (s)',
        'vswr': 'VSWR',
        'passivity': 'Passivity',
        'reciprocity': 'Reciprocity',
        'time': 'Time (real)',
        'time_db': 'Magnitude (dB)',
        'time_mag': 'Magnitude',
        'time_impulse': 'Magnitude',
        'time_step': 'Magnitude',
    }

    noise_interp_kind = 'linear'

    # CONSTRUCTOR
    def __init__(self, file=None, name=None, comments=None, f_unit=None, s_def=S_DEF_DEFAULT, **kwargs):
        '''
        Network constructor.

        Creates an n-port microwave network from a `file` or directly
        from data. If no file or data is given, then an empty Network
        is created.

        Parameters
        ------------

        file : str or file-object
            file to load information from. supported formats are:
             * touchstone file (.s?p)
             * pickled Network (.ntwk, .p) see :func:`write`
        name : str
            Name of this Network. if None will try to use file, if
            its a str
        comments : str
            Comments associated with the Network
        s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
            Scattering parameter definition : 'power' for power-waves definition, 
            'pseudo' for pseudo-waves definition. 
            'traveling' corresponds to the initial implementation. 
            Default is 'power'.
            NB: results are the same for real-valued characteristic impedances.
        \*\*kwargs :
            key word arguments can be used to assign properties of the
            Network, such as `s`, `f` and `z0`.

        Examples
        ------------
        From a touchstone

        >>> n = rf.NoisyNetwork('ntwk1.s2p')

        From a pickle file

        >>> n = rf.NoisyNetwork('ntwk1.ntwk')

        Create a blank network, then fill in values

        >>> n = rf.Network()
        >>> freq = rf.Frequency(1,3,3,'ghz')
        >>> n.frequency, n.s, n.z0 = freq,[1,2,3], [1,2,3]

        Directly from values

        >>> n = rf.Network(f=[1,2,3],s=[1,2,3],z0=[1,2,3])

        See Also
        -----------
        from_z : init from impedance values
        read : read a network from a file
        write : write a network to a file, using pickle
        write_touchstone : write a network to a touchstone file
        '''

        super().__init__(self, file=None, name=None, comments=None, f_unit=None, s_def=S_DEF_DEFAULT, **kwargs)

        
        self.noise_cov = None # This is the NetworkNoiseCov object, some of this will be duplicate with noise for now
        self.noise_freq = None
        self.T0 = T0 # Temperature at measurement

    def noise_source(self, source='passive', T0=None):
        '''
        Set the :class:`.NetworkNoiseCov` within :class:`Network` to model noise.

        To model noise, use this method to set the noise covariance matrix within the network.

        Parameters
        -----------
        source : :class:`.NetworkNoiseCov` or string
            Sets the noise covariance matrix for the network. The noise covariance matrix is stored within
            a :class:`.NetworkNoiseCov` object. The matrix can be used to model all kinds of noise (e.g., thermal,
            shot, flicker, etc.). However, if the network is passive, `source` may be set to source='passive'. Doing so
            will use the matrix `s` within :class:`Network` to calculate the covariance matrix for thermal noise.
            If you don't want the network to produce any noise, and yet use the network in noise calculations, you can
            pass source='none'.

        T0 : 
            The physical temperature of the network. Leave unset for room temperature.

        Example
        --------
        Create a network and set its noise source manually: 

        >>> frequency = rf.Frequency(start=1000, stop=2000, npoints=10, unit='MHz')
        >>> ovec = npy.ones(len(frequency))
        >>> zvec = npy.zeros(len(frequency))
        >>> R = 200*ovec
        >>> R_shunt_z = rf.network_array([[R, R], [R, R]])
        >>> R_shunt_cov = 4*K_BOLTZMANN*T0*npy.real(R_shunt_z)
        >>> ntwk = rf.Network.from_z(R_shunt_z)
        >>> ntwk_cz = rf.NetworkNoiseCov(R_shunt_cov, form='z')
        >>> ntwk.noise_source(ntwk_cz)

        Create a network and set its noise source using source='passive':

        >>> frequency = rf.Frequency(start=1000, stop=2000, npoints=10, unit='MHz')
        >>> ovec = npy.ones(len(frequency))
        >>> zvec = npy.zeros(len(frequency))
        >>> R = 200*ovec
        >>> R_shunt_z = rf.network_array([[R, R], [R, R]])
        >>> ntwk = rf.Network.from_z(R_shunt_z)
        >>> ntwk.noise_source('passive')


        '''
        if isinstance(source, string_types):
            if source == 'passive':
                if T0:
                    self.T0 = T0
                self.noise_cov = NetworkNoiseCov.from_passive_s(self.s, self.f, T0=self.T0)
            elif source == 'none':
                #TODO: Clean this up
                self.noise_cov = NetworkNoiseCov.from_passive_s(npy.zeros(shape=self.s.shape), self.f, T0=self.T0) 
                self.noise_cov.mat_vec = npy.zeros(shape=self.s.shape)
        elif isinstance(source, NetworkNoiseCov):
            self._validate_covariance_setter(source.mat_vec)
            self.noise
            self.noise_cov = source
        else:
            raise ValueError("Input must be the string 'passive' or a NetworkNoiseCov object, otherwise use setters 'cs', 'ct', 'cz', etc.")

    # OPERATORS
    def __pow__(self, other):
        """
        cascade this network with another network

        See `cascade`
        """
        # if they pass a number then use power operator
        if isinstance(other, Number):
            out = self.copy()
            out.s = out.s ** other
            return out

        else:
            return cascade(self, other)

    def __mul__(self, other):
        """
        Element-wise complex multiplication of s-matrix

        if skrf.alternative_ops() has been set, this operator performs
        cascade_2port operation.
        """

        # see skrf __init__.py for is_alt_ops() usage
        if not is_alt_ops():
            result = self.copy()

            if isinstance(other, Network):
                self.__compatable_for_scalar_operation_test(other)
                result.s = self.s * other.s
            else:
                # other may be an array or a number
                result.s = self.s * npy.array(other).reshape(-1, self.nports, self.nports)

            return result

        else:
            return cascade_2port(self, other)

    def __or__(self, other):
        """parallel_parallel_2port operator

        """
        return parallel_parallel_2port(self, other)

    def __rmul__(self, other):
        """
        Element-wise complex multiplication of s-matrix
        """

        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s * other.s
        else:
            # other may be an array or a number
            result.s = self.s * npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __add__(self, other):
        """
        Element-wise complex addition of s-matrix

        if skrf.alternative_ops() has been set, this operator performs
        series_series_2port operation.

        """
        # see skrf __init__.py for is_alt_ops() usage
        if not is_alt_ops():
            
            result = self.copy()

            if isinstance(other, Network):
                self.__compatable_for_scalar_operation_test(other)
                result.s = self.s + other.s
            else:
                # other may be an array or a number
                result.s = self.s + npy.array(other).reshape(-1, self.nports, self.nports)

            return result

        else:
            return series_series_2port(self, other)

    def __radd__(self, other):
        """
        Element-wise complex addition of s-matrix
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s + other.s
        else:
            # other may be an array or a number
            result.s = self.s + npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __sub__(self, other):
        """
        Element-wise complex subtraction of s-matrix
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s - other.s
        else:
            # other may be an array or a number
            result.s = self.s - npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __rsub__(self, other):
        """
        Element-wise complex subtraction of s-matrix
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = other.s - self.s
        else:
            # other may be an array or a number
            result.s = npy.array(other).reshape(-1, self.nports, self.nports) - self.s

        return result

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        Element-wise complex multiplication of s-matrix
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s / other.s
        else:
            # other may be an array or a number
            result.s = self.s / npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __eq__(self, other):
        if other is None:
            return False
        if npy.all(npy.abs(self.s - other.s) < ZERO):
            return True
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))


    def __str__(self):
        """
        """
        f = self.frequency
        if self.name is None:
            name = ''
        else:
            name = self.name

        if len(npy.shape(self.z0)) == 0 or npy.shape(self.z0)[0] == 0:
            z0 = str(self.z0)
        else:
            z0 = str(self.z0[0, :])

        output = '%i-Port Network: \'%s\',  %s, z0=%s' % (self.number_of_ports, name, str(f), z0)

        return output

    def __repr__(self):
        return self.__str__()

    @property
    def cs(self):
        """
        Noise covariance matrix in s-form

        Returns
        ---------
        cs : complex :class:`numpy.ndarray` of shape `fxnxn`
                noise covariance matrix as a function of frequency

        See Also
        ------------
        ct
        cy
        cz
        ca

        """
        ntwkNoiseCov = self.noise_cov.get_cs(self.s)
        return ntwkNoiseCov.cc

    @cs.setter
    def cs(self, value):
       self._validate_covariance_setter(value)
       self.noise_cov = NetworkNoiseCov(value, form='s')

    @property
    def ct(self):
        """
        Noise covariance matrix in t-form

        Returns
        ---------
        ct : complex :class:`numpy.ndarray` of shape `fxnxn`
                noise covariance matrix as a function of frequency

        See Also
        ------------
        cs
        cy
        cz
        ca

        """
        ntwkNoiseCov = self.noise_cov.get_ct(self.t)
        return ntwkNoiseCov.cc

    @ct.setter
    def ct(self, value):
        self._validate_covariance_setter(value)
        self.noise_cov = NetworkNoiseCov(value, form='t')

    @property
    def cz(self):
        """
        Noise covariance matrix in z-form

        Returns
        ---------
        cz : complex :class:`numpy.ndarray` of shape `fxnxn`
                noise covariance matrix as a function of frequency

        See Also
        ------------
        cs
        ct
        cy
        ca

        """
        ntwkNoiseCov = self.noise_cov.get_cz(self.z)
        return ntwkNoiseCov.cc

    @cz.setter
    def cz(self, value):
       self._validate_covariance_setter(value)
       self.noise_cov = NetworkNoiseCov(value, form='z')

    @property
    def cy(self):
        """
        Noise covariance matrix in y-form

        Returns
        ---------
        cy : complex :class:`numpy.ndarray` of shape `fxnxn`
                noise covariance matrix as a function of frequency

        See Also
        ------------
        cs
        ct
        cz
        ca

        """
        ntwkNoiseCov = self.noise_cov.get_cy(self.y)
        return ntwkNoiseCov.cc

    @cy.setter
    def cy(self, value):
       self._validate_covariance_setter(value)
       self.noise_cov = NetworkNoiseCov(value, form='y')

    @property
    def ca(self):
        """
        Noise covariance matrix in a-form

        Returns
        ---------
        ca : complex :class:`numpy.ndarray` of shape `fxnxn`
                noise covariance matrix as a function of frequency

        See Also
        ------------
        cs
        ct
        cy
        cz

        """
        ntwkNoiseCov = self.noise_cov.get_ca(self.a)
        return ntwkNoiseCov.cc

    @ca.setter
    def ca(self, value):
       self._validate_covariance_setter(value)
       self.noise_cov = NetworkNoiseCov(value, form='a')

    def _validate_covariance_setter(self, value):
        if value.shape != self.s.shape:
            raise ValueError("Covariance data must be the same size as the network data")

  
    @property
    def y_opt(self):
      """
      the optimum source admittance to minimize noise
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.y_opt 
      else:  
        noise = self.n
        return (npy.sqrt(noise[:,1,1]/noise[:,0,0] - npy.square(npy.imag(noise[:,0,1]/noise[:,0,0])))
            + 1.j*npy.imag(noise[:,0,1]/noise[:,0,0]))

    @property
    def z_opt(self):
      """
      the optimum source impedance to minimize noise
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.z_opt
      else:  
        return 1./self.y_opt

    @property
    def g_opt(self):
      """
      the optimum source reflection coefficient to minimize noise
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return z2s(nca.z_opt.reshape((self.f.shape[0], 1, 1)), self.z0[:,0])[:,0,0]
      else:  
        return z2s(self.z_opt.reshape((self.f.shape[0], 1, 1)), self.z0[:,0])[:,0,0]

    @property
    def nfmin(self):
      """
      the minimum noise figure for the network
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.nfmin
      else:
        noise = self.n
        return npy.real(1. + (noise[:,0,1] + noise[:,0,0] * npy.conj(self.y_opt))/(2*K_BOLTZMANN*self.T0))

    @property
    def nfmin_db(self):
      """
      the minimum noise figure for the network in dB
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return mf.complex_2_db10(nca.nfmin)
      else:
        return mf.complex_2_db10(self.nfmin)

    def nf(self, z):
      """
      the noise figure for the network if the source impedance is z
      """
      z0 = self.z0
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        y_opt = nca.y_opt
        fmin = nca.nfmin
        rn = nca.rn
      else:
        y_opt = self.y_opt
        fmin = self.nfmin
        rn = self.rn

      ys = 1./z
      gs = npy.real(ys)
      return fmin + rn/gs * npy.square(npy.absolute(ys - y_opt))

    def nf_db(self, z):
        return mf.complex_2_db10(self.nf(z))
 
    def nfdb_gs(self, gs):
      """
      return dB(NF) foreach gamma_source x noise_frequency
      """
      g = self.copy().s11
      nfreq = self.noise_freq.npoints

      if isinstance(gs, (int, float, complex)) :
          g.s[:,0,0] = gs
          nfdb = 10.*npy.log10(self.nf( g.z[:,0,0]))
      elif isinstance(gs, npy.ndarray) : 
          npt =  gs.shape[0]
          z = self.z0[0,0] * (1+gs)/(1-gs)
          zf = npy.broadcast_to(z[:,None], tuple((npt, nfreq)))
          nfdb = 10.*npy.log10(self.nf( zf))
      else :
          g.s[:,0,0] = -1
          nfdb = 10.*npy.log10(self.nf( g.z[:,0,0]))
      return nfdb

    '''
    newnetw.nfdb_gs(complex(.7,-0.2))
    gs = complex(.7,-0.2)
    gs = np.arange(0,0.9,0.1)
    self = newnetw
    self.    
    
    '''

    @property
    def rn(self):
      """
      the equivalent noise resistance for the network
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.rn
      else:
        return npy.real(self.n[:,0,0]/(4.*K_BOLTZMANN*self.T0))


    ## CLASS METHODS
    def copy(self):
        '''
        Returns a copy of this Network

        Needed to allow pass-by-value for a Network instead of
        pass-by-reference
        '''
        ntwk = Network(s=self.s,
                       frequency=self.frequency.copy(),
                       z0=self.z0, s_def=self.s_def
                       )

        ntwk.name = self.name

        if self.noise is not None and self.noise_freq is not None:
          if False : 
              ntwk.noise = npy.copy(self.noise)
              ntwk.noise_freq = npy.copy(self.noise_freq)
          ntwk.noise = self.noise.copy()
          ntwk.noise_freq = self.noise_freq.copy()

        try:
            ntwk.port_names = copy(self.port_names)
        except(AttributeError):
            ntwk.port_names = None
        return ntwk


def cascade_2port(ntwkA, ntwkB, calc_noise=True):
    '''
    cascade combination of two two-ports Networks (:class:`Network`), which also combines noise covariance matrices.

    Connects two two-port Networks in cascade configuration, if noise information
    is available, use it to determine the total covariance matrix for the combined
    system.

    Notes
    ------
    For a description of the cascade two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : :class:`Network`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`Network`
            the resultant two-port network of ntwkA in cascade with ntwkB

    See Also
    ---------
    :func:`series_series_2port`
    :func:`parallel_parallel_2port`
    '''

    _noisy_two_port_verify(ntwkA, ntwkB)

    ta = ntwkA.t
    tb = ntwkB.t
    tt = npy.matmul(ta, tb)

    nwk = ntwkA.copy()
    nwk.t = tt

    if ntwkA.noise_cov and ntwkB.noise_cov and calc_noise:
        cta = ntwkA.ct
        ctb = ntwkB.ct
        ctt = cta + npy.matmul(ta, npy.matmul(ctb, npy.conjugate(ta.swapaxes(1, 2))))
        nwk.ct = ctt
    
    return nwk


def parallel_parallel_2port(ntwkA, ntwkB, calc_noise=True):
    '''
    parallel combination of two two-ports  Networks (:class:`Network`), which also combines noise covariance matrices.

    Connects two two-port Networks in parallel-parallel configuration

    Notes
    ------
    For a description of the parallel-parallel two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : :class:`Network`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`Network`
            the resultant two-port network of ntwkA parallel-parallel with ntwkB

    See Also
    ---------
    :func:`series_series_2port`
    :func:`cascade_2port`
    '''

    _noisy_two_port_verify(ntwkA, ntwkB)

    ya = ntwkA.y
    yb = ntwkB.y
    yt = ya + yb # not sure I can do this with np arrays

     # make the new resulting network
    nwk = ntwkA.copy()
    nwk.y = yt

    if ntwkA.noise_cov and ntwkB.noise_cov and calc_noise:
        cya = ntwkA.cy
        cyb = ntwkB.cy
        cyt = cya + cyb
        nwk.cy = cyt
    
    return nwk

def series_series_2port(ntwkA, ntwkB, calc_noise=True):
    '''
    series combination of two two-ports Networks (:class:`Network`), which also combines noise covariance matrices.

    Connects two two-port Networks in series-series configuration

    Notes
    ------
    For a description of the series-series two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : :class:`Network`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`Network`
            the resultant two-port network of ntwkA in series-series with ntwkB

    See Also
    ---------
    :func:`parallel_parallel_2port`
    :func:`cascade_2port`
    '''

    _noisy_two_port_verify(ntwkA, ntwkB)

    za = ntwkA.z
    zb = ntwkB.z
    zt = za + zb 

    nwk = ntwkA.copy()
    nwk.z = zt

    if ntwkA.noise_cov and ntwkB.noise_cov and calc_noise:
        cza = ntwkA.cz
        czb = ntwkB.cz
        czt = cza + czb

        nwk.cz = czt
    
    return nwk

def series_parallel_2port(ntwkA, ntwkB, calc_noise=True):
    raise NotImplemented()

def parallel_series_2port(ntwkA, ntwkB, calc_noise=True):
    raise NotImplemented()






    
    



  
