
#       media.py
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
.. module:: skrf.media.media
========================================
media (:mod:`skrf.media.media`)
========================================

Contains Media class.


'''
import warnings

import numpy as npy
from scipy import stats

from ..network import Network, connect
from .. import tlineFunctions as tf
from .. import mathFunctions as mf

class Media(object):
    '''
    The base-class for all transmission line mediums.

    The :class:`Media` object provides generic methods to produce   :class:`~skrf.network.Network`'s for any transmision line medium, such  as :func:`line` and :func:`delay_short`.

    The initializer for this class has flexible argument types. This
    allows for the important attributes of the :class:`Media` object
    to be dynamic. For example, if a Media object's propagation constant
    is a function of some attribute of that object, say `conductor_width`,
    then the propagation constant will change when that attribute
    changes. See :func:`__init__` for details.


    The network creation methods build off of each other. For example,
    the specicial load cases, suc as :func:`short` and :func:`open` call
    :func:`load` with given arguments for Gamma0, and the delay_ and
    shunt_ functions call :func:`line` and :func:`shunt` respectively.
    This minimizes re-implementation.

    Most methods initialize the :class:`~skrf.network.Network` by
    calling :func:`match` to create a 'blank'
    :class:`~skrf.network.Network`, and then fill in the s-matrix.
    '''
    def __init__(self, frequency,  propagation_constant,
            characteristic_impedance, z0=None):
        '''
        The Media initializer.

        This initializer has flexible argument types. The parameters
        `propagation_constant`, `characterisitc_impedance` and `z0` can
        all be either static or dynamic. This is achieved by allowing
        those arguments to be either:
         * functions which take no arguments or
         * values (numbers or arrays)

        In the case where the media's propagation constant may change
        after initialization, because you adjusted a parameter of the
        media, then passing the propagation_constant as a function
        allows it to change when the media's parameters do.

        Parameters
        --------------
        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium

        propagation_constant : number, array-like, or a function
                propagation constant for the medium.

        characteristic_impedance : number,array-like, or a function
                characteristic impedance of transmission line medium.

        z0 : number, array-like, or a function
                the port impedance for media , IF its different
                from the characterisitc impedance of the transmission
                line medium  (None) [a number].
                if z0= None then will set to characterisitc_impedance



        Notes
        ------
        `propagation_constant` must adhere to the following convention,
         * positive real(gamma) = attenuation
         * positive imag(gamma) = forward propagation

        the z0 parameter is needed in some cases. For example, the
        :class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`
        is an example  where you may need this, because the
        characteristic impedance is frequency dependent, but the
        touchstone's created by most VNA's have z0=1
        '''
        self.frequency = frequency

        self.propagation_constant = propagation_constant
        self.characteristic_impedance = characteristic_impedance

        if z0 is None:
            z0 = characteristic_impedance
        self.z0 = z0

        # convinience names
        self.delay = self.line

    ## Properties
    # note these are made so that a Media type can be constructed with
    # propagation_constant, characteristic_impedance, and z0 either as:
    #       dynamic properties (if they pass a function)
    #       static ( if they pass values)
    @property
    def propagation_constant(self):
        '''
        Propagation constant

        The propagation constant can be either a number, array-like, or
        a function. If it is a function is must take no arguments. The
        reason to make it a function is if you want the propagation
        constant to be dynamic, meaning changing with some attribute
        of the media. See :func:`__init__` for more explanation.

        Returns
        ---------
        propagation_constant : :class:`numpy.ndarray`
                complex propagation constant for this media

        Notes
        ------
        `propagation_constant` must adhere to the following convention,
         * positive real(propagation_constant) = attenuation
         * positive imag(propagation_constant) = forward propagation
        '''
        try:
            return self._propagation_constant()
        except(TypeError):
            return self._propagation_constant
    @propagation_constant.setter
    def propagation_constant(self, new_propagation_constant):
        self._propagation_constant = new_propagation_constant

    @property
    def characteristic_impedance(self):
        '''
        Characterisitc impedance

        The characteristic_impedance can be either a number, array-like, or
        a function. If it is a function is must take no arguments. The
        reason to make it a function is if you want the characterisitc
        impedance to be dynamic, meaning changing with some attribute
        of the media. See :func:`__init__` for more explanation.

        Returns
        ----------
        characteristic_impedance : :class:`numpy.ndarray`
        '''
        try:
            return self._characteristic_impedance()
        except(TypeError):
            return self._characteristic_impedance

    @characteristic_impedance.setter
    def characteristic_impedance(self, new_characteristic_impedance):
        self._characteristic_impedance = new_characteristic_impedance

    @property
    def z0(self):
        '''
        Port Impedance

        The port impedance  is usually equal to the
        :attr:`characteristic_impedance`. Therefore, if the port
        impedance is `None` then this will return
        :attr:`characteristic_impedance`.

        However, in some cases such as rectangular waveguide, the port
        impedance is traditionally set to 1 (normalized). In such a case
        this property may be used.

        The Port Impedance can be either a number, array-like, or
        a function. If it is a function is must take no arguments. The
        reason to make it a function is if you want the Port Impedance
        to be dynamic, meaning changing with some attribute
        of the media. See :func:`__init__` for more explanation.


        Returns
        ----------
        port_impedance : :class:`numpy.ndarray`
                the media's port impedance
        '''
        try:
            return self._z0()
        except(TypeError):
            return self._z0
    @z0.setter
    def z0(self, new_z0):
        self._z0 = new_z0



    ## Other Functions
    def theta_2_d(self,theta,deg=True):
        '''
        Converts electrical length to physical distance.

        The given electrical length is to be  at the center frequency.

        Parameters
        ----------
        theta : number
                electrical length, at band center (see deg for unit)
        deg : Boolean
                is theta in degrees?

        Returns
        --------
        d : number
                physical distance in meters


        '''
        if deg == True:
            theta = mf.degree_2_radian(theta)

        gamma = self.propagation_constant
        return 1.0*theta/npy.imag(gamma[gamma.size/2])

    def electrical_length(self, d,deg=False):
        '''
        calculates the electrical length for a given distance, at
        the center frequency.

        Parameters
        ----------
                d: distance, in meters
                deg: is d in deg?[Boolean]

        Returns
        --------
                theta: electrical length in radians or degrees,
                        depending on  value of deg.
        '''
        gamma = self.propagation_constant

        if deg == False:
            return  gamma*d
        elif deg == True:
            return  mf.radian_2_degree(gamma*d )

    ## Network creation

    # lumped elements
    def match(self,nports=1, z0=None, **kwargs):
        '''
        Perfect matched load (:math:`\\Gamma_0 = 0`).

        Parameters
        ----------
        nports : int
                number of ports
        z0 : number, or array-like
                characterisitc impedance. Default is
                None, in which case the Media's :attr:`z0` is used.
                This sets the resultant Network's
                :attr:`~skrf.network.Network.z0`.
        \*\*kwargs : key word arguments
                passed to :class:`~skrf.network.Network` initializer

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port match


        Examples
        ------------
                >>> my_match = my_media.match(2,z0 = 50, name='Super Awesome Match')

        '''
        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s =  npy.zeros((self.frequency.npoints,nports, nports),\
                dtype=complex)
        if z0 is None:
            z0 = self.z0
        result.z0=z0
        return result

    def load(self,Gamma0,nports=1,**kwargs):
        '''
        Load of given reflection coefficient.

        Parameters
        ----------
        Gamma0 : number, array-like
                Reflection coefficient of load (linear, not in db). If its
                an array it must be of shape: kxnxn, where k is #frequency
                points in media, and n is `nports`
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        load  :class:`~skrf.network.Network` object
                n-port load, where  S = Gamma0*eye(...)
        '''
        result = self.match(nports,**kwargs)
        try:
            result.s =  Gamma0* \
                    npy.eye(nports,dtype=complex).reshape((-1,nports,nports)).\
                    repeat(self.frequency.npoints,0)
        except(ValueError):
            for f in range(self.frequency.npoints):
                result.s[f,:,:] = Gamma0[f]*npy.eye(nports, dtype=complex)

        return result

    def short(self,nports=1,**kwargs):
        '''
        Short (:math:`\\Gamma_0 = -1`)

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port short circuit

        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        return self.load(-1., nports, **kwargs)

    def open(self,nports=1, **kwargs):
        '''
        Open (:math:`\\Gamma_0 = 1`)

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port open circuit

        See Also
        ---------
        match : function called to create a 'blank' network
        '''

        return self.load(1., nports, **kwargs)

    def capacitor(self, C, **kwargs):
        '''
        Capacitor


        Parameters
        ----------
        C : number, array
                Capacitance, in Farads. If this is an array, must be of
                same length as frequency vector.
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        capacitor : :class:`~skrf.network.Network` object
                a n-port capacitor

        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        Gamma0 = tf.zl_2_Gamma0(self.z0, -1j/(self.frequency.w*C))
        return self.load(Gamma0=Gamma0, **kwargs)

    def inductor(self, L, **kwargs):
        '''
        Inductor

        Parameters
        ----------
        L : number, array
                Inductance, in Henrys. If this is an array, must be of
                same length as frequency vector.
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        inductor : :class:`~skrf.network.Network` object
                a n-port inductor

        See Also
        ---------
        match : called to create a 'blank' network
        '''
        Gamma0 = tf.zl_2_Gamma0(self.z0,-1j*(self.frequency.w*L))
        return self.load(Gamma0=Gamma0, **kwargs)

    def impedance_mismatch(self, z1, z2, **kwargs):
        '''
        Two-port network for an impedance miss-match


        Parameters
        ----------
        z1 : number, or array-like
                complex impedance of port 1
        z2 : number, or array-like
                complex impedance of port 2
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        missmatch : :class:`~skrf.network.Network` object
                a 2-port network representing the impedance missmatch

        Notes
        --------
        If z1 and z2 are arrays, they must be of same length
        as the :attr:`Media.frequency.npoints`

        See Also
        ---------
        match : called to create a 'blank' network
        '''
        result = self.match(nports=2, **kwargs)
        gamma = tf.zl_2_Gamma0(z1,z2)
        result.s[:,0,0] = gamma
        result.s[:,1,1] = -gamma
        result.s[:,1,0] = 1+gamma
        result.s[:,0,1] = 1-gamma
        return result


    # splitter/couplers
    def tee(self,**kwargs):
        '''
        Ideal, lossless tee. (3-port splitter)

        Parameters
        ----------
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        tee : :class:`~skrf.network.Network` object
                a 3-port splitter

        See Also
        ----------
        splitter : this just calls splitter(3)
        match : called to create a 'blank' network
        '''
        return self.splitter(3,**kwargs)

    def splitter(self, nports,**kwargs):
        '''
        Ideal, lossless n-way splitter.

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        tee : :class:`~skrf.network.Network` object
                a n-port splitter

        See Also
        ---------
        match : called to create a 'blank' network
        '''
        n=nports
        result = self.match(n, **kwargs)

        for f in range(self.frequency.npoints):
            result.s[f,:,:] =  (2*1./n-1)*npy.eye(n) + \
                    npy.sqrt((1-((2.-n)/n)**2)/(n-1))*\
                    (npy.ones((n,n))-npy.eye(n))
        return result


    # transmission line
    def thru(self, **kwargs):
        '''
        Matched transmission line of length 0.

        Parameters
        ----------
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        thru : :class:`~skrf.network.Network` object
                matched tranmission line of 0 length

        See Also
        ---------
        line : this just calls line(0)
        '''
        return self.line(0,**kwargs)

    def line(self,d, unit='m',**kwargs):
        '''
        Matched transmission line of given length

        The units of `length` are interpreted according to the value
        of `unit`.

        Parameters
        ----------
        d : number
                the length of transmissin line (see unit argument)
        unit : ['m','deg','rad']
                the units of d. possible options are:
                 * *m* : meters, physical length in meters (default)
                 * *deg* :degrees, electrical length in degrees
                 * *rad* :radians, electrical length in radians
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        line : :class:`~skrf.network.Network` object
                matched tranmission line of given length


        Examples
        ----------
        >>> my_media.line(90, 'deg', z0=50)

        '''
        if unit not in ['m','deg','rad']:
            raise (ValueError('unit must be one of the following:\'m\',\'rad\',\'deg\''))

        result = self.match(nports=2,**kwargs)

        d_dict ={\
                'deg':self.theta_2_d(d,deg=True),\
                'rad':self.theta_2_d(d,deg=False),\
                'm':d\
                }

        theta = self.electrical_length(d_dict[unit])

        s11 = npy.zeros(self.frequency.npoints, dtype=complex)
        s21 = npy.exp(-1*theta)
        result.s = \
                npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)
        return result

    def delay_load(self,Gamma0,d,unit='m',**kwargs):
        '''
        Delayed load

        A load with reflection coefficient `Gamma0` at the end of a
        matched line of length `d`.

        Parameters
        ----------
        Gamma0 : number, array-like
                reflection coefficient of load (not in dB)
        d : number
                the length of transmissin line (see unit argument)
        unit : ['m','deg','rad']
                the units of d. possible options are:
                 * *m* : meters, physical length in meters (default)
                 * *deg* :degrees, electrical length in degrees
                 * *rad* :radians, electrical length in radians
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_load : :class:`~skrf.network.Network` object
                a delayed load


        Examples
        ----------
        >>> my_media.delay_load(-.5, 90, 'deg', z0=50)


        Notes
        ------
        This calls ::

                line(d,unit, **kwargs) ** load(Gamma0, **kwargs)

        See Also
        ---------
        line : creates the network for line
        load : creates the network for the load


        '''
        return self.line(d=d, unit=unit,**kwargs)**\
                self.load(Gamma0=Gamma0,**kwargs)

    def delay_short(self,d,unit='m',**kwargs):
        '''
        Delayed Short

        A transmission line of given length terminated with a short.

        Parameters
        ----------
        d : number
                the length of transmissin line (see unit argument)
        unit : ['m','deg','rad']
                the units of d. possible options are:
                 * *m* : meters, physical length in meters (default)
                 * *deg* :degrees, electrical length in degrees
                 * *rad* :radians, electrical length in radians
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_short : :class:`~skrf.network.Network` object
                a delayed short


        See Also
        --------
        delay_load : delay_short just calls this function

        '''
        return self.delay_load(Gamma0=-1., d=d, unit=unit, **kwargs)

    def delay_open(self,d,unit='m',**kwargs):
        '''
        Delayed open transmission line

        Parameters
        ----------
        d : number
                the length of transmissin line (see unit argument)
        unit : ['m','deg','rad']
                the units of d. possible options are:
                 * *m* : meters, physical length in meters (default)
                 * *deg* :degrees, electrical length in degrees
                 * *rad* :radians, electrical length in radians
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_open : :class:`~skrf.network.Network` object
                a delayed open


        See Also
        ---------
        delay_load : delay_short just calls this function
        '''
        return self.delay_load(Gamma0=1., d=d, unit=unit,**kwargs)

    def shunt(self,ntwk, **kwargs):
        '''
        Shunts a :class:`~skrf.network.Network`

        This creates a :func:`tee` and connects connects
        `ntwk` to port 1, and returns the result

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
        \*\*kwargs : keyword arguments
                passed to :func:`tee`

        Returns
        --------
        shunted_ntwk : :class:`~skrf.network.Network` object
                a shunted a ntwk. The resultant shunted_ntwk will have
                (2 + ntwk.number_of_ports -1) ports.

        '''
        return connect(self.tee(**kwargs),1,ntwk,0)

    def shunt_delay_load(self,*args, **kwargs):
        '''
        Shunted delayed load

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_load`

        Returns
        --------
        shunt_delay_load : :class:`~skrf.network.Network` object
                a shunted delayed load (2-port)

        Notes
        --------
        This calls::

                shunt(delay_load(*args, **kwargs))

        '''
        return self.shunt(self.delay_load(*args, **kwargs))

    def shunt_delay_open(self,*args,**kwargs):
        '''
        Shunted delayed open

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_delay_open : :class:`~skrf.network.Network` object
                shunted delayed open (2-port)

        Notes
        --------
        This calls::

                shunt(delay_open(*args, **kwargs))
        '''
        return self.shunt(self.delay_open(*args, **kwargs))

    def shunt_delay_short(self,*args,**kwargs):
        '''
        Shunted delayed short

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_delay_load : :class:`~skrf.network.Network` object
                shunted delayed open (2-port)

        Notes
        --------
        This calls::

                shunt(delay_short(*args, **kwargs))
        '''
        return self.shunt(self.delay_short(*args, **kwargs))

    def shunt_capacitor(self,C,*args,**kwargs):
        '''
        Shunted capacitor

        Parameters
        ----------
        C : number, array-like
                Capacitance in Farads.
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_capacitor : :class:`~skrf.network.Network` object
                shunted capcitor(2-port)

        Notes
        --------
        This calls::

                shunt(capacitor(C,*args, **kwargs))

        '''
        return self.shunt(self.capacitor(C=C,*args,**kwargs))

    def shunt_inductor(self,L,*args,**kwargs):
        '''
        Shunted inductor

        Parameters
        ----------
        L : number, array-like
                Inductance in Farads.
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_inductor : :class:`~skrf.network.Network` object
                shunted inductor(2-port)

        Notes
        --------
        This calls::

                shunt(inductor(C,*args, **kwargs))

        '''
        return self.shunt(self.inductor(L=L,*args,**kwargs))


    ## Noise Networks
    def white_gaussian_polar(self,phase_dev, mag_dev,n_ports=1,**kwargs):
        '''
        Complex zero-mean gaussian white-noise network.

        Creates a network whose s-matrix is complex zero-mean gaussian
        white-noise, of given standard deviations for phase and
        magnitude components.
        This 'noise' network can be added to networks to simulate
        additive noise.

        Parameters
        ----------
        phase_mag : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase
        n_ports : int
                number of ports.
        \*\*kwargs : passed to :class:`~skrf.network.Network`
                initializer

        Returns
        --------
        result : :class:`~skrf.network.Network` object
                a noise network
        '''
        shape = (self.frequency.npoints, n_ports,n_ports)
        phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = shape)

        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s = mag_rv*npy.exp(1j*phase_rv)
        return result


    ## OTHER METHODS
    def guess_length_of_delay_short(self, aNtwk):
        '''
        Guess physical length of a delay short.

        Unwraps the phase and determines the slope, which is then used
        in conjunction with :attr:`propagation_constant` to estimate the
        physical distance to the short.

        Parameters
        ----------
        aNtwk : :class:`~skrf.network.Network` object
                (note: if this is a measurment
                it needs to be normalized to the reference plane)


        '''
        warnings.warn(DeprecationWarning('I have yet to update this for Media class'))
        beta = npy.real(self.propagation_constant(self.frequency.f))
        thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())

        A = npy.vstack((2*beta,npy.ones(len(beta)))).transpose()
        B = thetaM

        print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
        return npy.linalg.lstsq(A, B)[0][0]
