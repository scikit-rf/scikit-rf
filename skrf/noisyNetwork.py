# -*- coding: utf-8 -*-
"""
.. module:: skrf.noisyNetwork
========================================
noisyNetwork (:mod:`skrf.noisyNetwork`)
========================================

Provides an n-port network class that holds noise covariance matrices. This
is useful when you want to calculate the noise figure and noise parameters
of a network. 

:class:`NoisyNetwork` derives from :class:`Network`. Therefore, any methods
within the :class:`Network` class can be used from :class:`NoisyNetwork`. 
For example, *resample*, *interpolate*, and any of the plotting functions
within :class:`Network` and be used on this object.

NoisyNetwork Class
==================

.. autosummary::
    :toctree: generated/

    NoisyNetwork

Set Noise Source
================
*noise_source* is the primary method that sets a :class:`NoisyNetwork`
apart from a :class:`Network`. This function is used to associate a noise
covariance matrix with a network.

.. autosummary::
    :toctree: generated/

    NoisyNetwork.noise_source

Noise Parameters
================
.. autosummary::
    :toctree: generated/

    NoisyNetwork.y_opt
    NoisyNetwork.z_opt
    NoisyNetwork.g_opt
    NoisyNetwork.rn
    NoisyNetwork.nfmin
    NoisyNetwork.nfmin_db
    NoisyNetwork.nf
    NoisyNetwork.nf_db

Network Noise Covariance Representations
========================================
.. autosummary::
    :toctree: generated/

    NoisyNetwork.cs
    NoisyNetwork.ct
    NoisyNetwork.cz
    NoisyNetwork.cy
    NoisyNetwork.ca

Network Methods
---------------
These functions are part of the parent class :class:`Network` and are 
provided here for convenience.

.. autosummary::
    :toctree: generated/

    Network.from_z
    Network.from_y
    Network.from_a
    Network.s
    Network.z
    Network.y
    Network.a
    Network.t

Connecting Networks with Noise Analysis
=======================================

.. autosummary::
    :toctree: generated/

    cascade_2port
    parallel_parallel_2port
    series_series_2port


"""

from six.moves import xrange
from six import string_types


import numpy as npy
from copy import deepcopy as copy

from . import mathFunctions as mf
from .network import Network
from .network import z2a, z2s, z2t, z2y, y2a, y2s, y2t, y2z, s2a, s2h, s2t, s2y
from .network import  s2z, a2s, a2t, a2z, t2a, t2s, t2y, t2z
from .frequency import Frequency
from .networkNoiseCov import NetworkNoiseCov

from .constants import ZERO, K_BOLTZMANN, T0
from .constants import S_DEFINITIONS, S_DEF_DEFAULT


class NoisyNetwork(Network):
    """
    A :class:`Network` with associated noise covariance matrix and methods.

    For instructions on how to create a NoisyNetwork see  :func:`__init__`.

    =====================  =============================================
    Operator               Function
    =====================  =============================================
    \+                     combines noisy networks in series
    \|                     combines noisy networks in parallel
    \*\*                   cascades noisy networks
    =====================  =============================================

    Different components of the :class:`NoisyNetwork` can be visualized
    using the same methods used to visualize :class:`Network` through 
    various plotting methods. These methods can be used to plot
    individual elements of the s-matrix or all at once. For more info
    about plotting see the tutorials.

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


    An exhaustive list of :class:`NoisyNetwork` Methods and Properties
    (Attributes) are given below
    """

    # CONSTRUCTOR
    def __init__(self, *args, **kwargs):
        '''
        NoisyNetwork constructor.

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

        >>> n = rf.NoisyNetwork()
        >>> freq = rf.Frequency(1,3,3,'ghz')
        >>> n.frequency, n.s, n.z0 = freq,[1,2,3], [1,2,3]

        Directly from values

        >>> n = rf.NoisyNetwork(f=[1,2,3],s=[1,2,3],z0=[1,2,3])

        See Also
        -----------
        from_z : init from impedance values
        read : read a network from a file
        write : write a network to a file, using pickle
        write_touchstone : write a network to a touchstone file
        '''

        super(NoisyNetwork, self).__init__(*args, **kwargs)

        
        self.noise_cov = None # This is the NetworkNoiseCov object, some of this will be duplicate with noise for now
        self.noise_freq = None
        self.T0 = T0 # Temperature at measurement



    def noise_source(self, source='passive', T0=None):
        '''
        Set the :class:`.NetworkNoiseCov` within :class:`NoisyNetwork` to model noise.

        To model noise, use this method to set the noise covariance matrix within the network.

        Parameters
        -----------
        source : :class:`.NetworkNoiseCov` or string
            Sets the noise covariance matrix for the network. The noise covariance matrix is stored within
            a :class:`.NetworkNoiseCov` object. The matrix can be used to model all kinds of noise (e.g., thermal,
            shot, flicker, etc.). However, if the network is passive, `source` may be set to source='passive'. Doing so
            will use the matrix `s` within :class:`NoisyNetwork` to calculate the covariance matrix for thermal noise.
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
        >>> ntwk = rf.NoisyNetwork.from_z(R_shunt_z)
        >>> ntwk_cz = rf.NetworkNoiseCov(R_shunt_cov, form='z')
        >>> ntwk.noise_source(ntwk_cz)

        Create a network and set its noise source using source='passive':

        >>> frequency = rf.Frequency(start=1000, stop=2000, npoints=10, unit='MHz')
        >>> ovec = npy.ones(len(frequency))
        >>> zvec = npy.zeros(len(frequency))
        >>> R = 200*ovec
        >>> R_shunt_z = rf.network_array([[R, R], [R, R]])
        >>> ntwk = rf.NoisyNetwork.from_z(R_shunt_z)
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
        Cascade this network with another network

        See `cascade`
        """
        return cascade_2port(self, other)

    def __mul__(self, other):
        """
        Cascade this network with another network

        See `cascade`
        """
        return cascade_2port(self, other)

    def __or__(self, other):
        """parallel_parallel_2port operator

        """
        return parallel_parallel_2port(self, other)

    def __add__(self, other):
        """
        Add two two-port networks in series

        """
        return series_series_2port(self, other)

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

        output = '%i-Port NoisyNetwork: \'%s\',  %s, z0=%s' % (self.number_of_ports, name, str(f), z0)

        return output

    def __repr__(self):
        return self.__str__()

    @property
    def cs(self):
        """
        Returns the noise covariance matrix in s-form

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
        Returns the noise covariance matrix in t-form

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
        Returns the noise covariance matrix in z-form

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
        Returns the noise covariance matrix in y-form

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
        Returns the noise covariance matrix in a-form

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
      The optimum source admittance to minimize noise
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
      The optimum source impedance to minimize noise
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.z_opt
      else:  
        return 1./self.y_opt

    @property
    def g_opt(self):
      """
      The optimum source reflection coefficient to minimize noise
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return z2s(nca.z_opt.reshape((self.f.shape[0], 1, 1)), self.z0[:,0])[:,0,0]
      else:  
        return z2s(self.z_opt.reshape((self.f.shape[0], 1, 1)), self.z0[:,0])[:,0,0]

    @property
    def nfmin(self):
      """
      The minimum noise figure for the network
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
      The minimum noise figure for the network in dB
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return mf.complex_2_db10(nca.nfmin)
      else:
        return mf.complex_2_db10(self.nfmin)

    def nf(self, z):
      """
      The noise figure for the network if the source impedance is z
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
        """
      The noise figure for the network in dB
      """
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
      The equivalent noise resistance for the network
      """
      if self.noise_cov:
        nca = self.noise_cov.get_ca(self.a)
        return nca.rn
      else:
        return npy.real(self.n[:,0,0]/(4.*K_BOLTZMANN*self.T0))


    ## CLASS METHODS
    def copy(self):
        '''
        Returns a copy of this NoisyNetwork

        Needed to allow pass-by-value for a NoisyNetwork instead of
        pass-by-reference
        '''
        ntwk = NoisyNetwork(s=self.s,
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
    Cascade combination of two two-ports NoisyNetwork (:class:`NoisyNetwork`), which also combines noise covariance matrices.

    Connects two two-port NoisyNetwork in cascade configuration, if noise information
    is available, use it to determine the total covariance matrix for the combined
    system.

    Notes
    ------
    For a description of the cascade two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`NoisyNetwork`
            network `ntwkA`
    ntwkB : :class:`NoisyNetwork`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`NoisyNetwork`
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
    Parallel combination of two two-ports  NoisyNetwork (:class:`NoisyNetwork`), which also combines noise covariance matrices.

    Connects two two-port NoisyNetwork in parallel-parallel configuration

    Notes
    ------
    For a description of the parallel-parallel two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`NoisyNetwork`
            network `ntwkA`
    ntwkB : :class:`NoisyNetwork`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`NoisyNetwork`
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
    Series combination of two two-ports NoisyNetwork (:class:`NoisyNetwork`), which also combines noise covariance matrices.

    Connects two two-port NoisyNetwork in series-series configuration

    Notes
    ------
    For a description of the series-series two-port connection see 
    https://en.wikipedia.org/wiki/Two-port_network 

    Parameters
    -----------
    ntwkA : :class:`NoisyNetwork`
            network `ntwkA`
    ntwkB : :class:`NoisyNetwork`
            network `ntwkB`
    calc_noise : Bool
                Set to false if no noise calculations are desired

    Returns
    --------
    C : :class:`NoisyNetwork`
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

def _noisy_two_port_verify(ntwkA, ntwkB):

    if ntwkA.nports!=2 or ntwkB.nports!=2:
        raise ValueError('nports must be equal to 2 for both networks')

    if (ntwkA.frequency != ntwkB.frequency) or (ntwkA.noise_freq != ntwkB.noise_freq):
        raise ValueError('both networks must have same frequency data')

    #if ntwkA.noise_freq != ntwkA.frequency:
     #   raise ValueError('network frequency and noise frequency vectors must be the same')

    #if ntwkA.z0 != ntwkB.z0:
        #raise ValueError('currently, z0 must be the same for both networks')

    return True



    
    



  
