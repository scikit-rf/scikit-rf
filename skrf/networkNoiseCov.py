# -*- coding: utf-8 -*-
"""
.. module:: skrf.networkNoiseCov
=============================================
networkNoiseCov (:mod:`skrf.networkNoiseCov`)
=============================================

Provides a class for encapsulating the covariance matrix for a network. The covariance
matrix can be represented in a number of forms; this class provides functions for easily
transforming between those forms. 

Noise parameters (Fmin, Rn, Yopt) can be calculated directly from this object.

NetworkNoiseCov Class
=====================

.. autosummary::
    :toctree: generated/

    NetworkNoiseCov

Building NetworkNoiseCov
------------------------

.. autosummary::
    :toctree: generated/

    NetworkNoiseCov.from_passive_z
    NetworkNoiseCov.from_passive_y
    NetworkNoiseCov.from_passive_s

NetworkNoiseCov Transforms
==========================

.. autosummary::
    :toctree: generated/

    NetworkNoiseCov.get_cs
    NetworkNoiseCov.get_ct
    NetworkNoiseCov.get_cz
    NetworkNoiseCov.get_cy
    NetworkNoiseCov.get_ca

NetworkNoiseCov Noise Parameters
================================

.. autosummary::
    :toctree: generated/

    NetworkNoiseCov.nfmin
    NetworkNoiseCov.nfmin_db
    NetworkNoiseCov.rn
    NetworkNoiseCov.y_opt
    NetworkNoiseCov.z_opt
    NetworkNoiseCov.nf

NetworkNoiseCov Access Covariance Matrix
========================================

.. autosummary::
    :toctree: generated/

    NetworkNoiseCov.cc
    NetworkNoiseCov.form

"""

import numpy as npy
from copy import deepcopy as copy

from .util import network_array
from .constants import ZERO, K_BOLTZMANN, h_PLANK, S_DEF_DEFAULT
from .mathFunctions import complex_2_db10

class NetworkNoiseCov(object):
    """
    Encapsulates the Covariance Matrix [1] of a network.

    For instructions on how to instantiate :class:`NetworkNoiseCov` see :func:`__init__` as well as
    :func:`from_z`, :func:`from_y` and :func:`from_s`.

    To account for noise, a multiport network will have a covariance matrix 
    associated with it. Noise parameters are calculated directly from the 
    covariance matrix (for example, Fmin, Rn, Y_opt), and these perameters 
    enable us to characterize how signals become degraded as they pass through 
    networks.

    The :class:`NetworkNoiseCov` class contains the covariance matrix of a :class:`.Network` over 
    a set of frequencies. Properties of :class:`NetworkNoiseCov` are used to access parameters
    such as the minimum noise figure (:attr:`nfmin`) or the covariance matrix itself (:attr:`cc`).

    The greatest utility of :class:`NetworkNoiseCov` is its ability to transform covariance 
    matrices to various forms. Suppose two two-port networks are connected together in cascade; the 
    resulting combined two-port network will have its own covariance matrix that is a combination
    of the covariance matrices of the individual two-port networks. Combining covariance 
    matrices can be learned about in chapter 7 of Vendelin's book [1]. When combining networks
    in cascade, it is most convenient to have the covariance matrix in either the Ct or Ca form. When
    combining networks in parallel, it is most convenient to have the covariance matrix in Cy form. These 
    matrices can be obtained from a :class:`NetworkNoiseCov` object via :func:`get_ct`, :func:`get_ca` and
    :func:`get_cy`.

    You can set the covariance matrix of a :class:`.Network` using :meth:`.Network.noise_source`.

    The covariance matrix can be stored in any form within :class:`NetworkNoiseCov`. You can retrieve the 
    matrix as a function of frequency, and determine its form via properies: 

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`cc`             vector of covariance matrices
    :attr:`form`           the form of the covariance matrix (s, t, z, y, a)
    =====================  =============================================

    You can obtain the :class:`NetworkNoiseCov` object in various forms provided you have the associated
    network matrix used for the calculation. Note, that these functions are called automatically within
    :class:`.Network` when they are required:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :func:`get_cs`         Get the S-form of the covariance matrix
    :func:`get_ct`         Get the T-form of the covariance matrix
    :func:`get_cy`         Get the Y-form of the covariance matrix
    :func:`get_cz`         Get the Z-form of the covariance matrix
    :func:`get_ca`         Get the ABCD-form of the covariance matrix
    =====================  =============================================

    All noise parameters can be caclulated from an :class:`NetworkNoiseCov` object:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`nfmin`          Minimum noise factor
    :attr:`nfmin_db`       Minimum noise figure (only difference is in dB)
    :attr:`rn`             Rn
    :attr:`y_opt`          Optimal source admittance
    :attr:`z_opt`          Optimal source impedance
    :func:`nf`             Noise factor given source impedance
    =====================  =============================================

    Other functions

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :func:`copy`           Returns a copy of the function
    =====================  =============================================


    References
    ------------
    .. [1] George D Vendelin, Anthony M Pavio, and Ulrich L Rohde. "Noise in Linear Two-Ports" in *Microwave circuit design using linear and nonlinear techniques.* John Wiley & Sons, 2005.
    """

    COVARIANCE_FORMS = ['s', 't', 'z', 'y', 'a']

    def __init__(self, cc, form='s', z0 = 50, T0=290):
        '''
        NetworkNoiseCov constructor.

        Stores a covariance matrix vector (cc) of a particular form (s, t, y, z, a). The shape of cc must be
        the same as the shape of s within :class:`.Network`.

        Parameters
        -----------
        cc: Numpy Array
            Must be the same shape as the network matrix that :class:`NetworkNoiseCov` is attached to. 
        form: Char 
            One of 's', 't', 'z', 'y', or 'a' depending on the format of the covariance matrix
        z0: 
            The reference characteristic impedance of the ports [this is constant and the same right now for each port].
        T0: 
            The physical temperature of the device for which the covariance matrix is valid

        Examples
        --------
        A two-port network with covariance matrix in 'y' form, which has two frequencies, can be constructed as

        >>> cy = numpy.array([[[1, -1],[-1, 1]], [[1, -1],[-1, 1]]])
        >>> ntwk_cy = NetworkNoiseCov(cy, 'y')

        '''

        self._validate_mat_vec_form(cc)
        self._validate_form(form)
        self._validate_z0(z0)

        self._mat_vec = cc
        self._form = form
        self._z0 = z0
        self._T0 = T0
        self._k_norm = 1/npy.sqrt(self._z0) 

        # dictionaries of transforms. Transforming is carried out during getter and setter operations 
        self.transform_to_s = {'s': self._do_nothing, 't': self._ct2cs, 'z': self._cz2cs, 'y': self._cy2cs,'a': self._ca2cs }
        self.transform_to_t = {'s': self._cs2ct, 't': self._do_nothing, 'z': self._cz2ct, 'y': self._cy2ct,'a': self._ca2ct }
        self.transform_to_z = {'s': self._cs2cz, 't': self._ct2cz, 'z': self._do_nothing, 'y': self._cy2cz,'a': self._ca2cz }
        self.transform_to_y = {'s': self._cs2cy, 't': self._ct2cy, 'z': self._cz2cy, 'y': self._do_nothing,'a': self._ca2cy }
        self.transform_to_a = {'s': self._cs2ca, 't': self._ct2ca, 'z': self._cz2ca, 'y': self._cy2ca,'a': self._do_nothing }

    @classmethod
    def Tnoise(cls,f,T):
        '''
        This is the correct blackbody noise that accounts for the quantum limit thermal noise floor for low values of T0 as well as
        the upper noise limit for frequencies that exceed 200 GHz. 
        Insert reference here - MBG
        '''

        X = (h_PLANK*f)/(2*K_BOLTZMANN*T)
        Tn = ((h_PLANK*f)/(2*K_BOLTZMANN))*(1/npy.tanh(X))

        return Tn


    @classmethod
    def from_passive_z(cls, z, f=None, z0=50, T0=290):
        '''
        Create a :class:`NetworkNoiseCov` object from impedance matrix.

        NOTE: this function should only be used in association with passive networks. If there are any other sources
        of noise, they should be modeled with the :class:`NetworkNoiseCov` constructor.

        Parameters
        -----------
        z : Numpy Array
            Impedance matrix. Should be of shape fxnxn
            where f is the frequency axis and n is the number of ports.
        f : 
            Vector of frequencies, which is necessary when taking into consideration proper blackbody noise effects. 
            Leave this parameter if you are not interested in these effects.
        z0 : 
            Characteristic impedance reference of all ports.
        T0 : 
            Physical temperature of the passive device.

        Return
        -------
        ntw_cz : :class:`NetworkNoiseCov` 
            The object in 'z' form

        Example
        --------
        A two-port shunted resistor R=100 will have an impedance matrix of form Z = [[R R],[R R]]. To create a
        :class:`NetworkNoiseCov` object:

        >>> Z = network_array([[R, R], [R, R]])
        >>> ntwk_cz = NetworkNoiseCov.from_passive_z(Z) 

        '''

        if f is None:
            Tn_mat = T0
        else:
            Tn = cls.Tnoise(f,T0)
            Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(z)[1],npy.shape(z)[2]))

        cov = 4.*K_BOLTZMANN*Tn_mat*npy.real(z)
        return cls(cov, form='z', z0=z0, T0=Tn)

    @classmethod
    def from_passive_y(cls, y, f, z0=50, T0=290):
        '''
        Create a :class:`NetworkNoiseCov` object from admittance matrix.

        NOTE: this function should only be used in association with passive networks. If there are any other sources
        of noise, they should be modeled with the :class:`NetworkNoiseCov` constructor.

        Parameters
        -----------
        y : Numpy Array
            Admittance matrix. Should be of shape fxnxn
            where f is the frequency axis and n is the number of ports.
        f : 
            Vector of frequencies, which is necessary when taking into consideration proper blackbody noise effects. 
            Leave this parameter if you are not interested in these effects.
        z0 : 
            Characteristic impedance reference of all ports.
        T0 : 
            Physical temperature of the passive device.

        Return
        -------
        ntw_cz : :class:`NetworkNoiseCov` 
            The object in 'z' form

        Example
        --------
        A two-port series resistor R=100 will have an admittance matrix of form Y = [[1/R -1/R],[-1/R 1/R]]. To create a
        :class:`NetworkNoiseCov` object:

        >>> Y = network_array([[R, R], [R, R]])
        >>> ntwk_cy = NetworkNoiseCov.from_passive_y(Y) 

        '''

        if f is None:
            Tn_mat = T0
        else:
            Tn = cls.Tnoise(f,T0)
            Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(y)[1],npy.shape(y)[2]))

        cov = 4.*K_BOLTZMANN*Tn_mat*npy.real(y)
        return cls(cov, form='y', z0=z0, T0=T0)

    @classmethod
    def from_passive_s(cls, s, f, z0=50, T0=290):
        '''
        Create a :class:`NetworkNoiseCov` object from S-parameters.

        NOTE: this function should only be used in association with passive networks. If there are any other sources
        of noise, they should be modeled with the :class:`NetworkNoiseCov` constructor.

        Parameters
        -----------
        s : Numpy Array
            S-parameter matrix. Should be of shape fxnxn
            where f is the frequency axis and n is the number of ports.
        f : 
            Vector of frequencies, which is necessary when taking into consideration proper blackbody noise effects. 
            Leave this parameter if you are not interested in these effects.
        z0 : 
            Characteristic impedance reference of all ports.
        T0 : 
            Physical temperature of the passive device.

        Return
        -------
        ntw_cz : :class:`NetworkNoiseCov` 
            The object in 'z' form

        Example
        --------
        A two-port attenuator with attenuation attn_db = 3 will have an S matrix of S = [[0 10**(-3/10)],[10**(-3/10) 0]]. 
        To create a
        :class:`NetworkNoiseCov` object:

        >>> attn_db = 3
        >>> S = network_array([[0 10**(-attn_db/10)],[10**(-attn_db/10) 0]])
        >>> ntwk_cs = NetworkNoiseCov.from_passive_y(S) 

        '''

        if f is None:
            Tn_mat = T0
        else:
            Tn = cls.Tnoise(f,T0)
            Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(s)[1],npy.shape(s)[2]))
  
        SM =  npy.matmul(s, npy.conjugate(s.swapaxes(1, 2)))
        I_2D = npy.identity(npy.shape(s)[1])
        I = npy.repeat(I_2D[npy.newaxis,:, :], npy.shape(s)[0], axis=0)
        cov = K_BOLTZMANN*Tn_mat*(I - SM) 
        return cls(cov, form='s', z0=z0, T0=T0)

    def copy(self):
        '''
        Returns a copy of this NetworkNoiseCov

        '''
        n = NetworkNoiseCov(cc=self._mat_vec, form = self._form, z0 = self._z0)
        return n

    @property
    def form(self):
        """
        Returns the form of the covariance matrix within :class:`NetworkNoiseCov`

        Returns
        ---------
        form : Char
            One of 's', 't', 'y', 'z', 'a'

        """
        return self._form

    @form.setter
    def form(self, value):
        """
        Manually set the form of the covariance matrix within :class:`NetworkNoiseCov`

        """
        if value not in self.COVARIANCE_FORMS:
            raise ValueError("form must be one of \'s\', \'t\', \'z\', \'y\', \'a\'" )
        self._form = value

    @property
    def mat_vec(self):
        """
        Same as :attr:`cc`

        """
        return self._mat_vec

    @mat_vec.setter
    def mat_vec(self, value):
        """
        Same as :attr:`cc`

        """
        if value.shape != self._mat_vec.shape:
            raise ValueError("mat_vec " + str(value.shape) +  " to " + str(self._mat_vec.shape) + " incompatible" )
        self._mat_vec = value

    @property
    def cc(self):
        """
        Returns the vector of covariance matrices.

        Returns
        --------
        cc : Numpy Array of shape `fxnxn`
            the vector of covariance matrices

        """
        return self.mat_vec

    @cc.setter
    def cc(self, value):
        """
        the input matrix should be the same shape as s from :class:`.Network`
        """
        self.mat_vec = value

    def get_cs(self, S):
        '''
        Transform current :class:`NetworkNoiseCov` to its s-form.

        Uses the S-parameters of a network to transform the current form of :class:`NetworkNoiseCov` to 
        its s-form. 

        Parameters
        -----------
        S : Numpy Array
            S-parameter matrix. Should be the same shape as :attr:`cc`.

        Return
        -------
        ntwk_cs : :class:`NetworkNoiseCov` 
            The object in 's' form

        Example
        --------
        Transform covariance matrix in y-form to s-form

        >>> Y = network_array([[R, R], [R, R]])
        >>> ntwk_cy = NetworkNoiseCov.from_passive_y(Y) 
        >>> S = y2s(Y)
        >>> ntwk_cs = ntwk_cy.get_cs(S)

        '''
        return self.transform_to_s[self.form](self._mat_vec, S)

    def get_ct(self, T):
        '''
        Transform current :class:`NetworkNoiseCov` to its t-form.

        Uses the T-parameters of a network to transform the current form of :class:`NetworkNoiseCov` to 
        its t-form. 

        Parameters
        -----------
        T : Numpy Array
            T-parameter matrix. Should be the same shape as :attr:`cc`.

        Return
        -------
        ntwk_ct : :class:`NetworkNoiseCov` 
            The object in 't' form

        Example
        --------
        Transform covariance matrix in y-form to t-form

        >>> Y = network_array([[R, R], [R, R]])
        >>> ntwk_cy = NetworkNoiseCov.from_passive_y(Y) 
        >>> T = y2t(Y)
        >>> ntwk_ct = ntwk_cy.get_ct(T)

        '''
        return self.transform_to_t[self.form](self._mat_vec, T)

    def get_cz(self, Z):
        '''
        Transform current :class:`NetworkNoiseCov` to its z-form.

        Uses the impedance matrix of a network to transform the current form of :class:`NetworkNoiseCov` to 
        its z-form. 

        Parameters
        -----------
        Z : Numpy Array
            Impedance matrix. Should be the same shape as :attr:`cc`.

        Return
        -------
        ntwk_cz : :class:`NetworkNoiseCov` 
            The object in 'z' form

        Example
        --------
        Transform covariance matrix in s-form to z-form

        >>> S = network_array([[0, 1/2], [1/2, 0]])
        >>> ntwk_cs = NetworkNoiseCov.from_passive_s(S) 
        >>> Z = s2z(S)
        >>> ntwk_cz = ntwk_cs.get_cz(Z)

        '''
        return self.transform_to_z[self.form](self._mat_vec, Z)

    def get_cy(self, Y):
        '''
        Transform current :class:`NetworkNoiseCov` to its y-form.

        Uses the admittance matrix of a network to transform the current form of :class:`NetworkNoiseCov` to 
        its y-form. 

        Parameters
        -----------
        Y : Numpy Array
            Admittance matrix. Should be the same shape as :attr:`cc`.

        Return
        -------
        ntwk_cy : :class:`NetworkNoiseCov` 
            The object in 'y' form

        Example
        --------
        Transform covariance matrix in s-form to y-form

        >>> S = network_array([[0, 1/2], [1/2, 0]])
        >>> ntwk_cs = NetworkNoiseCov.from_passive_s(S) 
        >>> Y = s2y(S)
        >>> ntwk_cy = ntwk_cs.get_cy(Y)

        '''
        return self.transform_to_y[self.form](self._mat_vec, Y)

    def get_ca(self, A):
        '''
        Transform current :class:`NetworkNoiseCov` to its a-form.

        Uses the ABCD matrix of a network to transform the current form of :class:`NetworkNoiseCov` to 
        its a-form. 

        Parameters
        -----------
        A : Numpy Array
            ABCD matrix. Should be the same shape as :attr:`cc`.

        Return
        -------
        ntwk_ca : :class:`NetworkNoiseCov` 
            The object in 'a' form

        Example
        --------
        Transform covariance matrix in s-form to a-form

        >>> S = network_array([[0, 1/2], [1/2, 0]])
        >>> ntwk_cs = NetworkNoiseCov.from_passive_s(S) 
        >>> A = s2a(S)
        >>> ntwk_ca = ntwk_cs.get_ca(A)

        '''
        return self.transform_to_a[self.form](self._mat_vec, A)

    @property
    def y_opt(self):
        """
        Returns the optimum source admittance to minimize noise figure.

        Returns
        --------
        y_opt : Numpy Array
               a vector the same length as `frequency`

        """

        self._validate_only_if_ca()
        ca = self.mat_vec
        return (npy.sqrt(ca[:,1,1]/ca[:,0,0] - npy.square(npy.imag(ca[:,0,1]/ca[:,0,0])))
          + 1.j*npy.imag(ca[:,0,1]/ca[:,0,0]))

    @property
    def z_opt(self):
        """
        Returns the optimum source impedance to minimize noise figure.

        Returns
        --------
        z_opt : Numpy Array
               a vector the same length as `frequency`

        """
        return 1./self.y_opt

    @property
    def nfmin(self):
        """
        Returns the minimum noise factor (linear form of noise figure).

        Returns
        --------
        nfmin : Numpy Array
               a vector the same length as `frequency`

        """

        self._validate_only_if_ca()
        ca = self.mat_vec
        return npy.real(1. + (ca[:,0,1] + ca[:,0,0] * npy.conj(self.y_opt))/(2.*K_BOLTZMANN*self._T0))

    @property
    def nfmin_db(self):
        """
        Returns the minimum noise figure (logarithmic form of noise factor).

        Returns
        --------
        nfmin_db : Numpy Array
               a vector the same length as `frequency`

        """
        return complex_2_db10(self.nfmin)

    @property
    def rn(self):
        """
        Returns the equivalent noise resistance for the network

        Returns
        --------
        rn : Numpy Array
               a vector the same length as `frequency`

        """

        self._validate_only_if_ca()
        ca11 = self.mat_vec[:,0,0]
        return npy.real(ca11/(4.*K_BOLTZMANN*self._T0))

    def nf(self, z):
        """
        Returns the noise factor (linear form of noise figure) for a a given source impedance. 

        Parameters
        -----------
        z : Numpy Array
            Source impedance

        Return
        -------
        nf : Numpy Array
            a vector the same length as `frequency`

        """

     
        z0 = self.z0
        y_opt = self.y_opt
        fmin = self.nfmin
        rn = self.rn

        ys = 1./z
        gs = npy.real(ys)
        return fmin + rn/gs * npy.square(npy.absolute(ys - y_opt))

    def _validate_only_if_ca(self):
        if self.form != 'a':
            raise ValueError("Noise parameters can only be extracted from an ABCD form NetworksNoiseCov object at this time")

    def _validate_mat_vec_form(self, mat_vec):
        """make sure the shape is correct
        """
        pass

    def _validate_form(self, form):
        if form not in self.COVARIANCE_FORMS:
            raise ValueError("form must be one of 's', 't', 'z', 'y', 'a'" )

    def _validate_z0(self, z0):
        """For right now z0 needs to be a real constant
        """
        pass

    ## Covariance form conversions
    def _do_nothing(self, value, M):
        return self

    ## S to other
    def _cs2ct(self, mat, T):
        t12 = T[:, 0, 1]
        t22 = T[:, 1, 1]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -t12],[zvec, -t22]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 't'
        return n

    def _cs2cz(self, mat, Z):
        Zn = Z/self._z0
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (Zn + I)/self._k_norm
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _cs2cy(self, mat, Y):
        y0 = 1/self._z0
        Yn = Y/y0
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = y0/self._k_norm*(Yn + I)  # why is this true (fix notes)
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'y'
        return n

    def _cs2ca(self, mat, A):
        y0 = 1/self._z0

        #a11 = A[:, 0, 0]
        #a12 = A[:, 0, 1]
        #a21 = A[:, 1, 0]
        #a22 = A[:, 1, 1]

        #Tm = self._k_norm/2*network_array([[a12*y0 + 1, -a11],[a22*y0, -a21 - y0]])
        #n = self.copy()
        #n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        #n.form = 'a'

        Z = self._z2a(A)
        cz = self._cs2cz(mat, Z)
        ca = self._cz2ca(cz.mat_vec, A)

        ca.form = 'a'
        return ca

    ## T to other
    def _ct2cs(self, mat, S):
        s11 = S[:,0,0]
        s21 = S[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -s11],[zvec, -s21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _ct2cz(self, mat, Z):
        S = self._z2s(Z) 
        cs = self._ct2cs(mat, S)
        cz = self._cs2cz(cs.mat_vec, Z)
        cz.form = 'z'
        return cz

    def _ct2cy(self, mat, Y):
        S = self._y2s(Y) 
        cs = self._ct2cs(mat, S)
        cy = self._cs2cy(cs.mat_vec, Y)
        cy.form = 'y'
        return cy

    def _ct2ca(self, mat, A):
        Z = self._z2a(A) # equivalent to a2z
        S = self._z2s(Z) # 
        cs = self._ct2cs(mat, S)
        cz = self._cs2cz(cs.mat_vec, Z)
        ca = self._cz2ca(cz.mat_vec, A)
        ca.form = 'a'
        return ca

    ## Z to other
    def _cz2cs(self, mat, S):
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (I - S)*self._k_norm/2
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _cz2ct(self, mat, T):
        S = self._t2s(T) 
        cs = self._cz2cs(mat, S)
        ct = self._cs2ct(cs.mat_vec, T)
        ct.form = 't'
        return ct

    def _cz2cy(self, mat, Y):
        Tm = Y
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'y'
        return n

    def _cz2ca(self, mat, A):
        a11 = A[:,0,0]
        a21 = A[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -a11],[zvec, -a21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'a'
        return n

    ## Y to other
    def _cy2cs(self, mat, S):
        y0 = 1/self._z0
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (I + S)*self._k_norm/(y0*2)
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _cy2ct(self, mat, T):
        S = self._t2s(T) 
        cs = self._cy2cs(mat, S)
        ct = self._cs2ct(cs.mat_vec, T)
        ct.form = 't'
        return ct

    def _cy2cz(self, mat, Z):
        Tm = Z
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _cy2ca(self, mat, A):
        a12 = A[:, 0, 1]
        a22 = A[:, 1, 1]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[zvec, a12],[ovec, a22]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'a'
        return n

    ## A to other
    def _ca2cs(self, mat, S):
        Z = self._s2z(S) 
        cs = self._ca2cz(mat, Z)
        cs = self._cz2cs(cs.mat_vec, S)
        cs.form = 's'
        return cs

    def _ca2ct(self, mat, T):
        Z = self._t2z(T) 
        c = self._ca2cz(mat, Z)
        ct = self._cz2ct(c.mat_vec, T)
        ct.form = 't'
        return ct

    def _ca2cz(self, mat, Z):
        z11 = Z[:,0,0]
        z21 = Z[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -z11],[zvec, -z21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _ca2cy(self, mat, Y):
        y11 = Y[:,0,0]
        y21 = Y[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[-y11, ovec],[-y21,zvec]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'y'
        return n

    # Need these until I fill in the rest of the transforms above
    def _z2a(self,z):
        '''
        z2a = a2z
        '''
        abcd = npy.array([
            [z[:, 0, 0] / z[:, 1, 0],
            1. / z[:, 1, 0]],
            [(z[:, 0, 0] * z[:, 1, 1] - z[:, 1, 0] * z[:, 0, 1]) / z[:, 1, 0],
            z[:, 1, 1] / z[:, 1, 0]],
        ]).transpose()
        return abcd

    
    def _z2s(self, z, z0=50, s_def='power'):
        """
        convert impedance parameters [1]_ to scattering parameters [2]_

        For power-waves, Eq.(18) from [3]:

        .. math::
            S = F (Z – G^*) (Z + G)^{-1} F^{-1}

        where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\\sqrt{|Re(Z_0)|}])`  
            
        For pseudo-waves, Eq.(73) from [4]:

        .. math::
            S = U (Z - G) (Z + G)^{-1}  U^{-1}

        where :math:`U = \\sqrt{Re(Z_0)}/|Z_0|`


        Parameters
        ------------
        z : complex array-like
            impedance parameters
        z0 : complex array-like or number
            port impedances
        s_def : str -> s_def : ['power','pseudo']
            Scattering parameter definition : 'power' for power-waves definition [3], 
            'pseudo' for pseudo-waves definition [4]. Default is 'power'.

        Returns
        ---------
        s : complex array-like
            scattering parameters



        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/impedance_parameters
        .. [2] http://en.wikipedia.org/wiki/S-parameters
        .. [3] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [4] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.
        
        """
        nfreqs, nports, nports = z.shape
        z0 = self._fix_z0_shape(z0, nfreqs, nports)

        # Add a small real part in case of pure imaginary char impedance
        # to prevent numerical errors for both pseudo and power waves definitions
        z0 = z0.astype(dtype=npy.complex)
        z0[z0.real == 0] += ZERO    

        if s_def == 'power':
            # Power-waves. Eq.(18) from [3]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs 
            F, G = npy.zeros_like(z), npy.zeros_like(z)
            npy.einsum('ijj->ij', F)[...] = 1.0/npy.sqrt(z0.real)*0.5
            npy.einsum('ijj->ij', G)[...] = z0
            # s = F @ (z - npy.conjugate(G)) @ npy.linalg.inv(z + G) @ npy.linalg.inv(F)  # Python > 3.5
            s = npy.matmul(F, 
                        npy.matmul((z - npy.conjugate(G)), 
                                    npy.matmul(npy.linalg.inv(z + G), npy.linalg.inv(F))))


        elif s_def == 'pseudo':    
            # Pseudo-waves. Eq.(73) from [4]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            ZR, U = npy.zeros_like(z), npy.zeros_like(z)
            npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
            npy.einsum('ijj->ij', ZR)[...] = z0
            # s = U @ (z - ZR) @ npy.linalg.inv(z + ZR) @ npy.linalg.inv(U)  # Python > 3.5
            s = npy.matmul(U, 
                        npy.matmul((z - ZR),
                                    npy.matmul(npy.linalg.inv(z + ZR), npy.linalg.inv(U))))

        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating Identity matrices of shape (nports,nports) for each nfreqs 
            Id = npy.zeros_like(z)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', Id)[...] = 1.0  
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrty0 = npy.zeros_like(z)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', sqrty0)[...] = npy.sqrt(1.0/z0)
            # z -> s 
            s = npy.zeros_like(z)
            # s = (sqrty0 @ z @ sqrty0 - Id) @  npy.linalg.inv(sqrty0 @ z @ sqrty0 + Id)  # Python>3.5
            s = npy.matmul((npy.matmul(npy.matmul(sqrty0, z), sqrty0) - Id), 
                            npy.linalg.inv(npy.matmul(npy.matmul(sqrty0, z), sqrty0) + Id))
    
        
        return s

    def _fix_z0_shape(self, z0, nfreqs, nports):
        '''
        Make a port impedance of correct shape for a given network's matrix

        This attempts to broadcast z0 to satisfy
            npy.shape(z0) == (nfreqs,nports)

        Parameters
        --------------
        z0 : number, array-like
            z0 can be:
            * a number (same at all ports and frequencies)
            * an array-like of length == number ports.
            * an array-like of length == number frequency points.
            * the correct shape ==(nfreqs,nports)

        nfreqs : int
            number of frequency points
        nports : int
            number of ports

        Returns
        ----------
        z0 : array of shape ==(nfreqs,nports)
            z0  with the right shape for a nport Network

        Examples
        ----------
        For a two-port network with 201 frequency points, possible uses may
        be

        >>> z0 = rf.fix_z0_shape(50 , 201,2)
        >>> z0 = rf.fix_z0_shape([50,25] , 201,2)
        >>> z0 = rf.fix_z0_shape(range(201) , 201,2)


        '''

        if npy.shape(z0) == (nfreqs, nports):
            # z0 is of correct shape. super duper.return it quick.
            return z0.copy()

        elif npy.isscalar(z0):
            # z0 is a single number
            return npy.array(nfreqs * [nports * [z0]])

        elif len(z0) == nports:
            # assume z0 is a list of impedances for each port,
            # but constant with frequency
            return npy.array(nfreqs * [z0])

        elif len(z0) == nfreqs:
            # assume z0 is a list of impedances for each frequency,
            # but constant with respect to ports
            return npy.array(nports * [z0]).T

        else:
            raise IndexError('z0 is not an acceptable shape')

    def _s2z(self, s, z0=50, s_def=S_DEF_DEFAULT):
        '''
        Convert scattering parameters [1]_ to impedance parameters [2]_


        For power-waves, Eq.(19) from [3]:

        .. math::
            Z = F^{-1} (1 - S)^{-1} (S G + G^*) F

        where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\\sqrt{|Re(Z_0)|}])`  
            
        For pseudo-waves, Eq.(74) from [4]:

        .. math::
            Z = (1 - U^{-1} S U)^{-1}  (1 + U^{-1} S U) G

        where :math:`U = \\sqrt{Re(Z_0)}/|Z_0|`
        
        Parameters
        ------------
        s : complex array-like
            scattering parameters
        z0 : complex array-like or number
            port impedances.
        s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
            Scattering parameter definition : 'power' for power-waves definition [3], 
            'pseudo' for pseudo-waves definition [4]. 
            'traveling' corresponds to the initial implementation. 
            Default is 'power'.
                
        Returns
        ---------
        z : complex array-like
            impedance parameters



        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/S-parameters
        .. [2] http://en.wikipedia.org/wiki/impedance_parameters
        .. [3] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [4] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.

        '''
        nfreqs, nports, nports = s.shape
        z0 = self._fix_z0_shape(z0, nfreqs, nports)
    
        # Add a small real part in case of pure imaginary char impedance
        # to prevent numerical errors for both pseudo and power waves definitions
        z0 = z0.astype(dtype=npy.complex)
        z0[z0.real == 0] += ZERO  

        s = s.copy()  # to prevent the original array from being altered
        s[s == -1.] = -1. + 1e-12  # solve numerical singularity
        s[s == 1.] = 1. + 1e-12  # solve numerical singularity

        # The following is a vectorized version of a for loop for all frequencies.    
        # # Creating Identity matrices of shape (nports,nports) for each nfreqs 
        Id = npy.zeros_like(s)  # (nfreqs, nports, nports)
        npy.einsum('ijj->ij', Id)[...] = 1.0     
        
        if s_def == 'power':    
            # Power-waves. Eq.(19) from [3]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            F, G = npy.zeros_like(s), npy.zeros_like(s)
            F = F.astype(dtype=npy.complex)
            G = G.astype(dtype=npy.complex)
            npy.einsum('ijj->ij', F)[...] = 1.0/npy.sqrt(z0.real)*0.5
            npy.einsum('ijj->ij', G)[...] = z0
            # z = npy.linalg.inv(F) @ npy.linalg.inv(Id - s) @ (s @ G + npy.conjugate(G)) @ F  # Python > 3.5
            z = npy.matmul(npy.linalg.inv(F), 
                        npy.matmul(npy.linalg.inv(Id - s), 
                                    npy.matmul(npy.matmul(s, G) + npy.conjugate(G), F)))
            
        elif s_def == 'pseudo':
            # Pseudo-waves. Eq.(74) from [4]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs 
            ZR, U = npy.zeros_like(s), npy.zeros_like(s)
            npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
            npy.einsum('ijj->ij', ZR)[...] = z0
            # USU = npy.linalg.inv(U) @ s @ U
            # z = npy.linalg.inv(Id - USU) @ (Id + USU) @ ZR
            USU = npy.matmul(npy.linalg.inv(U), npy.matmul(s , U))
            z = npy.matmul(npy.linalg.inv(Id - USU), npy.matmul((Id + USU), ZR))

        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrtz0 = npy.zeros_like(s)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', sqrtz0)[...] = npy.sqrt(z0)
            # s -> z 
            z = npy.zeros_like(s)
            # z = sqrtz0 @ npy.linalg.inv(Id - s) @ (Id + s) @ sqrtz0  # Python>3.5
            z = npy.matmul(npy.matmul(npy.matmul(sqrtz0, npy.linalg.inv(Id - s)), (Id + s)), sqrtz0)


        return z

    def _t2s(self, t):
        '''
        converts scattering transfer parameters [#]_ to scattering parameters [#]_

        transfer parameters are also referred to as
        'wave cascading matrix', this function only operates on 2N-ports
        networks with same number of input and output ports, also known as
        'balanced networks'.

        Parameters
        -----------
        t : :class:`numpy.ndarray` (shape fx2nx2n)
                scattering transfer parameters

        Returns
        -------
        s : :class:`numpy.ndarray`
                scattering parameter matrix.

        See Also
        ---------
        inv : calculates inverse s-parameters
        s2z
        s2y
        s2t
        z2s
        z2y
        z2t
        y2s
        y2z
        y2z
        t2s
        t2z
        t2y
        Network.s
        Network.y
        Network.z
        Network.t

        References
        -----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
        .. [#] http://en.wikipedia.org/wiki/S-parameters
        .. [#] Janusz A. Dobrowolski, "Scattering Parameter in RF and Microwave Circuit Analysis and Design",
            Artech House, 2016, pp. 65-68
        '''
        z, y, x = t.shape
        # test here for even number of ports.
        # t-parameter networks are square matrix, so x and y are equal.
        if(x % 2 != 0):
            raise IndexError('Network don\'t have an even number of ports')
        s = npy.zeros((z, y, x), dtype=complex)
        yh = int(y/2)
        xh = int(x/2)
        # T_II,II^-1
        tinv = npy.linalg.inv(t[:, yh:y, xh:x])
        # np.linalg.inv test for singularity (matrix not invertible)
        for k in range(len(s)):
            # S_I,I = T_I,II . T_II,II^-1
            s[k, 0:yh, 0:xh] = t[k, 0:yh, xh:x].dot(tinv[k])
            # S_I,II = T_I,I - T_I,I,II . T_II,II^-1 . T_II,I
            s[k, 0:yh, xh:x] = t[k, 0:yh, 0:xh]-t[k, 0:yh, xh:x].dot(tinv[k].dot(t[k, yh:y, 0:xh]))
            # S_II,I = T_II,II^-1
            s[k, yh:y, 0:xh] = tinv[k]
            # S_II,II = -T_II,II^-1 . T_II,I
            s[k, yh:y, xh:x] = -tinv[k].dot(t[k, yh:y, 0:xh])
        return s

    def _y2s(self, y, z0=50, s_def=S_DEF_DEFAULT):
        '''
        convert admittance parameters [#]_ to scattering parameters [#]_

        For power-waves, from [3]:

        .. math::        
            S = F (1 – G Y) (1 + G Y)^{-1} F^{-1}

        where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\\sqrt{|Re(Z_0)|}])`  
            
        For pseudo-waves, Eq.(73) from [4]:

        .. math::
            S = U (Y^{-1} - G) (Y^{-1} + G)^{-1}  U^{-1}        

        where :math:`U = \\sqrt{Re(Z_0)}/|Z_0|`


        Parameters
        ------------
        y : complex array-like
            admittance parameters

        z0 : complex array-like or number
            port impedances

        s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
            Scattering parameter definition : 'power' for power-waves definition [3], 
            'pseudo' for pseudo-waves definition [4]. 
            'traveling' corresponds to the initial implementation. 
            Default is 'power'.

        Returns
        ---------
        s : complex array-like or number
            scattering parameters

        See Also
        ----------
        s2z
        s2y
        s2t
        z2s
        z2y
        z2t
        y2s
        y2z
        y2z
        t2s
        t2z
        t2y
        Network.s
        Network.y
        Network.z
        Network.t


        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
        .. [#] http://en.wikipedia.org/wiki/S-parameters
        .. [3] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [4] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.    
        '''
        nfreqs, nports, nports = y.shape
        z0 = self._fix_z0_shape(z0, nfreqs, nports)

        # Add a small real part in case of pure imaginary char impedance
        # to prevent numerical errors for both pseudo and power waves definitions
        z0 = z0.astype(dtype=npy.complex)
        z0[z0.real == 0] += ZERO  

        # The following is a vectorized version of a for loop for all frequencies.        
        # Creating Identity matrices of shape (nports,nports) for each nfreqs 
        Id = npy.zeros_like(y)  # (nfreqs, nports, nports)
        npy.einsum('ijj->ij', Id)[...] = 1.0  
            
        if s_def == 'power':
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs 
            F, G = npy.zeros_like(y), npy.zeros_like(y)
            F = F.astype(dtype=npy.complex)
            G = G.astype(dtype=npy.complex)
            npy.einsum('ijj->ij', F)[...] = 1.0/npy.sqrt(z0.real)*0.5
            npy.einsum('ijj->ij', G)[...] = z0
            # s = F @ (Id - npy.conjugate(G) @ y) @ npy.linalg.inv(Id + G @ y) @ npy.linalg.inv(F)  # Python > 3.5
            s = npy.matmul(F, 
                        npy.matmul((Id - npy.matmul(npy.conjugate(G), y)), 
                                    npy.matmul(npy.linalg.inv(Id + npy.matmul(G, y)), npy.linalg.inv(F))))

        elif s_def == 'pseudo':
            # Pseudo-waves
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            ZR, U = npy.zeros_like(y), npy.zeros_like(y)
            npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
            npy.einsum('ijj->ij', ZR)[...] = z0
            # s = U @ (npy.linalg.inv(y) - ZR) @ npy.linalg.inv(npy.linalg.inv(y) + ZR) @ npy.linalg.inv(U)  # Python > 3.5
            s = npy.matmul(U, 
                        npy.matmul((npy.linalg.inv(y) - ZR), 
                                    npy.matmul(npy.linalg.inv(npy.linalg.inv(y) + ZR), npy.linalg.inv(U))))

        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrtz0 = npy.zeros_like(y)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', sqrtz0)[...] = npy.sqrt(z0)
            # y -> s 
            s = npy.zeros_like(y)
            # s = (Id - sqrtz0 @ y @ sqrtz0) @ npy.linalg.inv(Id + sqrtz0 @ y @ sqrtz0)  # Python>3.5
            s = npy.matmul( Id - npy.matmul(npy.matmul(sqrtz0, y), sqrtz0),
                        npy.linalg.inv(Id + npy.matmul(npy.matmul(sqrtz0, y), sqrtz0)))

            
        return s

    def _t2z(self, t, z0=50, s_def=S_DEF_DEFAULT):
        s = self._t2s(t)
        return self._s2z(s, z0, s_def)




