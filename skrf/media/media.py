"""
media (:mod:`skrf.media.media`)
========================================

Media class.

.. autosummary::
    :toctree: generated/

    Media
    DefinedGammaZ0

"""
from numbers import Number
from pathlib import Path
import warnings

import numpy as npy
from numpy import real, imag, ones, any, gradient, array
from scipy import stats
from scipy.constants import  c, inch, mil

from ..frequency import Frequency
from ..network import Network, connect, impedance_mismatch

from .. import tlineFunctions as tf
from .. import mathFunctions as mf

from ..constants import NumberLike, to_meters, ZERO
from typing import Union

from abc import ABC, abstractmethod
import re
from copy import deepcopy as copy
from ..constants import S_DEF_DEFAULT

class Media(ABC):
    """
    Abstract Base Class for a single mode on a transmission line media.


    This class init's with `frequency` and `z0` (the port impedance);
    attributes shared by all media. Methods defined here make use of the
    properties :

    * `gamma` - (complex) media propagation constant
    * `Z0` - (complex) media characteristic impedance

    Which define the properties of a specific media. Any sub-class of Media
    must implement these properties. `gamma` and `Z0` should return
    complex arrays of the same length as `frequency`. `gamma` must
    follow the convention:

    * positive real(gamma) = attenuation
    * positive imag(gamma) = forward propagation

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object or None
        frequency band of this transmission line medium.
        Defaults to None, which produces a 1-10ghz band with 101 points.
    z0 : number, array-like, or None
        the port impedance for media. Only needed if its different
        from the characteristic impedance of the media.
        If z0 is None then will default to Z0.
        Default is None.


    Notes
    -----
    The `z0` parameter (port impedance) is needed in some cases.
    :class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`
    is an example where you may need this, because the
    characteristic impedance is frequency dependent, but the
    touchstone's created by most VNA's have z0=1 or 50. So to
    prevent accidental impedance mis-match, you may want to manually
    set the `z0`.
    """

    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None):
        if frequency is None:
            frequency = Frequency(1,10,101,'ghz')

        self.frequency = frequency.copy()
        self.z0 = z0

    def mode(self,  **kw) -> 'Media':
        r"""
        Create another mode in this medium.

        Convenient way to return a copy this Media object
        with eventually different properties.

        Parameters
        ----------
        \*\*kwargs : keyword arguments passed to the copy

        Returns
        -------
        copy : :class:`Media`
            A copy of this Media object with \*\*kwargs attribute
        """
        out = copy(self)
        for k in kw:
            setattr(self, k, kw[k])
        return out

    def copy(self) -> 'Media':
        """
        Copy of this Media object.

        Returns
        -------
        copy : :class:`Media`
            A copy of this Media object
        """
        return copy(self)

    def __eq__(self,other):
        """
        Test for numerical equality (up to :data:`~skrf.constants.ZERO`).
        """

        if self.frequency != other.frequency:
            return False

        if max(abs(self.Z0 - other.Z0)) > ZERO:
            return False

        if max(abs(self.gamma - other.gamma)) > ZERO:
            return False

        if max(abs(self.z0 - other.z0)) > ZERO:
            return False

        return True

    def __len__(self) -> int:
        """
        Length of frequency axis.
        """
        return len(self.frequency)

    @property
    def npoints(self) -> int:
        """
        Number of points of the frequency axis.

        Returns
        -------
        npoints : int
            Number of points of the frequency axis.
        """
        return self.frequency.npoints

    @npoints.setter
    def npoints(self, val):
        self.frequency.npoints = val

    @property
    def z0(self) -> npy.ndarray:
        """
        Characteristic Impedance.

        Returns
        -------
        z0 : :class:`numpy.ndarray`
        """
        if self._z0 is None:
            return self.Z0
        return self._z0*ones(len(self))

    @z0.setter
    def z0(self, val):
        self._z0 = val

    @property
    @abstractmethod
    def gamma(self):
        r"""
        Propagation constant.

        In skrf, defined as :math:`\gamma = \alpha + j \beta`.

        Returns
        -------
        gamma : :class:`numpy.ndarray`
            complex propagation constant for this media

        Notes
        -----
        `gamma` must adhere to the following convention:

         * positive real(gamma) = attenuation
         * positive imag(gamma) = forward propagation
        """
        return None


    @property
    def alpha(self) -> npy.ndarray:
        """
        Real (attenuation) component of gamma.

        Returns
        -------
        alpha : :class:`numpy.ndarray`
        """
        return real(self.gamma)

    @property
    def beta(self) -> npy.ndarray:
        """
        Imaginary (propagating) component of gamma.

        Returns
        -------
        beta : :class:`numpy.ndarray`
        """
        return imag(self.gamma)

    @property
    @abstractmethod
    def Z0(self):
        return None


    @property
    def v_p(self) -> npy.ndarray:
        r"""
        Complex phase velocity (in m/s).

        .. math::
            j \cdot \omega / \gamma

        Notes
        -----
        The `j` is used so that real phase velocity corresponds to
        propagation

        where:

        * :math:`\omega` is angular frequency (rad/s),
        * :math:`\gamma` is complex propagation constant (rad/m)

        Returns
        -------
        v_p : :class:`numpy.ndarray`

        See Also
        --------
        propagation_constant
        gamma

        """
        return 1j*(self.frequency.w/self.gamma)


    @property
    def v_g(self):
        r"""
        Complex group velocity (in m/s).

        .. math::
            j \cdot d \omega / d \gamma


        where:

        * :math:`\omega` is angular frequency (rad/s),
        * :math:`\gamma` is complex propagation constant (rad/m)

        Notes
        -----
        the `j` is used to make propagation real, this is needed because
        skrf defined the gamma as :math:`\gamma= \alpha +j\beta`.

        Returns
        -------
        v_g : :class:`numpy.ndarray`

        References
        ----------
        https://en.wikipedia.org/wiki/Group_velocity

        See Also
        --------
        propagation_constant
        v_p
        gamma
        """
        dw = self.frequency.dw
        dk = gradient(self.gamma)

        return dw/dk


    def get_array_of(self, x):
        try:
            if len(x)!= len(self):
                # we have to make a decision
                pass
        except(TypeError):
            y = x* ones(len(self))

        return y

    ## Other Functions
    def theta_2_d(self, theta: NumberLike, deg:bool = True, bc: bool = True) -> NumberLike:
        r"""
        Convert electrical length to physical distance.

        The electrical length is given by :math:`d=\theta/\beta`.

        The given electrical length can be given either at the center frequency
        or on the entire band depending of the parameter `bc`.

        Parameters
        ----------
        theta : number
            electrical length, at band center (see deg for unit)
        deg : Boolean, optional
            is theta in degrees?
            Default is True (theta is assumed in degrees)
        bc : bool, optional.
            evaluate only at band center, or across the entire band?
            Default is True (evaluation assumed at band center)

        Returns
        --------
        d : number, array-like
            physical distance in meters

        """
        if deg == True:
            theta = mf.degree_2_radian(theta)

        gamma = self.gamma
        if bc:
                return 1.0*theta/npy.imag(gamma[int(gamma.size/2)])
        else:
                return 1.0*theta/npy.imag(gamma)

    def electrical_length(self, d: NumberLike, deg: bool = False) -> NumberLike:
        r"""
        Calculate the complex electrical length for a given distance.

        Electrical length is given by :math:`\theta=\gamma d`.

        Parameters
        ----------
        d: number or array-like
            delay distance, in meters
        deg: Boolean, optional
            return electrical length in deg?
            Default is False (returns electrical length in radians)

        Returns
        -------
        theta: number or array-like
            complex electrical length in radians or degrees, depending on
            value of deg.
        """
        gamma = self.gamma

        if deg == False:
            return gamma*d
        elif deg == True:
            return  mf.radian_2_degree(gamma*d)

    ## Network creation

    # lumped elements
    def match(self, nports: int = 1, z0: Union[NumberLike, None] = None,
              z0_norm: bool = False, **kwargs) -> Network:
        r"""
        Perfect matched load (:math:`\Gamma_0 = 0`).

        Parameters
        ----------
        nports : int
            number of ports
        z0 : number, or array-like or None
            port impedance. Default is
            None, in which case the Media's :attr:`z0` is used.
            This sets the resultant Network's
            :attr:`~skrf.network.Network.z0`.
        z0_norm : bool
            is z0 normalized to this media's `z0`?
        \*\*kwargs : key word arguments
            passed to :class:`~skrf.network.Network` initializer

        Returns
        -------
        match : :class:`~skrf.network.Network` object
            a n-port match

        Examples
        --------
        >>> my_match = my_media.match(2,z0 = 50, name='Super Awesome Match')

        """
        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s = npy.zeros((self.frequency.npoints, nports, nports), dtype=complex)
        if z0 is None:
            z0 = self.z0
        elif isinstance(z0, str):
            z0 = npy.ones(result.s.shape[:2]) * parse_z0(z0)

        if z0_norm:
            z0 = z0*self.z0

        result.z0 = z0
        return result

    def load(self, Gamma0: NumberLike, nports: int = 1, **kwargs) -> Network:
        r"""
        Load of given reflection coefficient.

        Parameters
        ----------
        Gamma0 : number, array-like
            Reflection coefficient of load (linear, not in db). If its
            an array it must be of shape: `kxnxn`, where k is number of frequency
            points in media, and n is `nports`
        nports : int
            number of ports
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        load : :class:`~skrf.network.Network` object
            n-port load, where  S = Gamma0*eye(...)

        See Also
        --------
        match
        open
        short
        """
        result = self.match(nports, **kwargs)
        result.s = npy.array(Gamma0).reshape(-1, 1, 1) * \
            npy.eye(nports, dtype=complex).reshape((-1, nports, nports)).\
            repeat(self.frequency.npoints, 0)
        return result

    def short(self, nports: int = 1, **kwargs) -> Network:
        r"""
        Short (:math:`\Gamma_0 = -1`)

        For s_def = 'power' (:math:`-Z_{ref}^*/Z_{ref}`)

        Parameters
        ----------
        nports : int
            number of ports
        \*\*kwargs : key word arguments passed to :func:`load`.

        Returns
        -------
        match : :class:`~skrf.network.Network` object
            a n-port short circuit

        Notes
        -----
        This calls ::

            load(-1.0, nports, **kwargs)

        See Also
        --------
        match
        open
        load
        """
        s_short = -1
        # Powerwave short is not necessarily -1
        if kwargs.get('s_def', S_DEF_DEFAULT) == 'power':
            z0 = kwargs.get('z0', self.z0)
            s_short = -npy.conjugate(z0) / z0
        return self.load(s_short, nports, **kwargs)

    def open(self, nports: int = 1, **kwargs) -> Network:
        r"""
        Open (:math:`\Gamma_0 = 1`).

        Parameters
        ----------
        nports : int
            number of ports
        \*\*kwargs : key word arguments passed to :func:`load`

        Returns
        -------
        match : :class:`~skrf.network.Network` object
            a n-port open circuit

        Notes
        -----
        This calls ::

            load(1.0, nports, **kwargs)

        See Also
        --------
        match
        load
        short
        """

        return self.load(1.0, nports, **kwargs)

    def resistor(self, R: NumberLike, *args, **kwargs) -> Network:
        r"""
        Resistor.

        Parameters
        ----------
        R : number, array
            Resistance , in Ohms. If this is an array, must be of
            same length as frequency vector.
        \*args, \*\*kwargs : arguments, key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        resistor : a 2-port :class:`~skrf.network.Network`

        See Also
        --------
        match
        short
        open
        load
        capacitor
        inductor
        """
        result = self.match(nports=2, *args, **kwargs)
        y = npy.zeros(shape=result.s.shape, dtype=complex)
        R = npy.array(R)
        y[:, 0, 0] = 1.0 / R
        y[:, 1, 1] = 1.0 / R
        y[:, 0, 1] = -1.0 / R
        y[:, 1, 0] = -1.0 / R
        result.y = y
        return result

    def capacitor(self, C: NumberLike, **kwargs) -> Network:
        r"""
        Capacitor.

        Parameters
        ----------
        C : number, array
            Capacitance, in Farads. If this is an array, must be of
            same length as frequency vector.
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        capacitor : a 2-port :class:`~skrf.network.Network`

        See Also
        --------
        match
        short
        open
        load
        resistor
        inductor
        """
        result = self.match(nports=2, **kwargs)
        w = self.frequency.w
        y = npy.zeros(shape=result.s.shape, dtype=complex)
        C = npy.array(C)
        y[:, 0, 0] = 1j * w * C
        y[:, 1, 1] = 1j * w * C
        y[:, 0, 1] = -1j * w * C
        y[:, 1, 0] = -1j * w * C
        result.y = y
        return result

    def inductor(self, L: NumberLike, **kwargs) -> Network:
        r"""
        Inductor.

        Parameters
        ----------
        L : number, array
            Inductance, in Henrys. If this is an array, must be of
            same length as frequency vector.
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        inductor : a 2-port :class:`~skrf.network.Network`

        See Also
        --------
        match
        short
        open
        load
        capacitor
        resistor
        """
        result = self.match(nports=2, **kwargs)
        w = self.frequency.w
        y = npy.zeros(shape=result.s.shape, dtype=complex)
        L = npy.array(L)
        y[:, 0, 0] = 1.0 / (1j * w * L)
        y[:, 1, 1] = 1.0 / (1j * w * L)
        y[:, 0, 1] = -1.0 / (1j * w * L)
        y[:, 1, 0] = -1.0 / (1j * w * L)
        result.y = y
        return result

    def impedance_mismatch(self, z1: NumberLike, z2: NumberLike, **kwargs) -> Network:
        r"""
        Two-port network for an impedance mismatch.

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
        -------
        missmatch : :class:`~skrf.network.Network` object
            a 2-port network representing the impedance mismatch

        Notes
        -----
        If z1 and z2 are arrays, they must be of same length
        as the :attr:`Media.frequency.npoints`

        See Also
        --------
        match
        short
        open
        load
        capacitor
        inductor
        resistor
        """
        result = self.match(nports=2, **kwargs)
        s_def = kwargs.get('s_def', S_DEF_DEFAULT)
        z1 = npy.array(z1)
        z2 = npy.array(z2)
        mismatch = npy.broadcast_to(impedance_mismatch(z1, z2, s_def), result.s.shape)
        result.s = mismatch
        return result


    # splitter/couplers
    def tee(self, **kwargs) -> Network:
        r"""
        Ideal, lossless tee. (3-port splitter).

        Parameters
        ----------
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        tee : :class:`~skrf.network.Network` object
            a 3-port splitter

        See Also
        ----------
        splitter : this just calls splitter(3)
        match : called to create a 'blank' network
        """
        return self.splitter(3, **kwargs)

    def splitter(self, nports: int, **kwargs) -> Network:
        r"""
        Ideal, lossless n-way splitter.

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        -------
        tee : :class:`~skrf.network.Network` object
                a n-port splitter

        See Also
        --------
        match : called to create a 'blank' network
        """
        n=nports
        result = self.match(n, **kwargs)

        for f in range(self.frequency.npoints):
            result.s[f,:,:] =  (2*1./n-1)*npy.eye(n) + \
                    npy.sqrt((1-((2.-n)/n)**2)/(n-1))*\
                    (npy.ones((n,n))-npy.eye(n))
        return result


    # transmission line

    def to_meters(self, d: NumberLike, unit: str = 'deg') -> NumberLike:
        """
        Translate various units of distance into meters.

        This is a method of media to allow for electrical lengths as
        inputs. For dispersive media, mean group velocity is used to
        translate time-based units to distance.

        Parameters
        ----------
        d : number or array-like
            the value
        unit : str
            the unit to that x is in:
            ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']

        Returns
        -------
        d_m : number, array-like
            d in meters

        See Also
        --------
        skrf.constants.to_meters
        """
        unit = unit.lower()
        #import pdb;pdb.set_trace()

        d_dict ={'deg':self.theta_2_d(d,deg=True),
                 'rad':self.theta_2_d(d,deg=False),
                 }

        if unit in d_dict:
            return d_dict[unit]
        else:
            # mean group velocity is used to translate time-based
            # units to distance
            if 's' in unit:
                # they are specifying  a time unit so calculate
                # the group velocity. (note this fails for media of
                # too little points, as it uses gradient)
                v_g = -self.v_g.imag.mean()
            else:
                v_g = c
            return to_meters(d=d,unit=unit, v_g=v_g)

    def thru(self, **kwargs) -> Network:
        r"""
        Matched transmission line of length 0.

        Parameters
        ----------
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        thru : :class:`~skrf.network.Network` object
            matched transmission line of 0 length

        See Also
        --------
        line : this just calls line(0)
        open, short, match
        """
        return self.line(0, **kwargs)

    def line(self, d: NumberLike, unit: str = 'deg',
             z0: Union[NumberLike, str, None] = None, embed: bool = False, **kwargs) -> Network:
        r"""
        Transmission line of a given length and impedance.

        The units of `length` are interpreted according to the value
        of `unit`. If `z0` is not None, then a line specified impedance
        is produced. if `embed` is also True, then the line is embedded
        in this media's z0 environment, creating a mismatched line.

        Parameters
        ----------
        d : number
                the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        z0 : number, string, or array-like or None
            the characteristic impedance of the line, if different
            from self.z0. To set z0 in terms of normalized impedance,
            pass a string, like `z0='1+.2j'`
        embed : bool
            if `Z0` is given, should the line be embedded in z0
            environment? or left in a `z` environment. if embedded,
            there will be reflections
        \*\*kwargs : key word arguments
            passed to :func:`match`, which is called initially to create a
            'blank' network.

        Returns
        -------
        line : :class:`~skrf.network.Network` object
            matched transmission line of given length

        Examples
        --------
        >>> my_media.line(1, 'mm', z0=100)
        >>> my_media.line(90, 'deg', z0='2') # set z0 as normalized impedance

        """

        if isinstance(z0,str):
            z0 = parse_z0(z0)* self.z0

        kwargs.update({'z0':z0})
        s_def = kwargs.pop('s_def', S_DEF_DEFAULT)
        # The use of either traveling or pseudo waves s-parameters definition
        # is required here.
        # The definition of the reflection coefficient for power waves has
        # conjugation. 
        result = self.match(nports=2, s_def='traveling', **kwargs)

        theta = self.electrical_length(self.to_meters(d=d, unit=unit))

        s11 = npy.zeros(self.frequency.npoints, dtype=complex)
        s21 = npy.exp(-1*theta)
        result.s = \
                npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)

        if embed and self.z0 is not None:
            # warns of future deprecation
            warnings.warn('In a future version,`embed` will be deprecated.\n'
                          'The line and media port impedance z0 and '
                          'characteristic impedance Z0 will be used instead '
                          'to determine if the line has to be renormalized.',
              FutureWarning, stacklevel = 2)
            result.renormalize(self.z0, s_def=s_def)
        else:
            result.renormalize(result.z0, s_def=s_def)

        return result


    def delay_load(self, Gamma0: NumberLike, d: Number, unit: str = 'deg', **kwargs) -> Network:
        r"""
        Delayed load.

        A load with reflection coefficient `Gamma0` at the end of a
        matched line of length `d`.

        Parameters
        ----------
        Gamma0 : number, array-like
            reflection coefficient of load (not in dB)
        d : number
            the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
            the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments
            passed to :func:`line` and :func:`load`

        Returns
        -------
        delay_load : :class:`~skrf.network.Network` object
                a delayed load

        Examples
        ----------
        >>> my_media.delay_load(-.5, 90, 'deg', Z0=50)

        Notes
        -----
        This calls ::

            line(d, unit, **kwargs) ** load(Gamma0, **kwargs)

        See Also
        --------
        line : creates the network for line
        load : creates the network for the load
        delay_short
        delay_open
        """
        return self.line(d=d, unit=unit, **kwargs) ** self.load(Gamma0=Gamma0,
                                                                **kwargs)

    def delay_short(self, d: Number, unit: str = 'deg', **kwargs) -> Network:
        r"""
        Delayed Short.

        A transmission line of given length terminated with a short.

        Parameters
        ----------
        d : number
            the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
            the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments passed to :func:`delay_load`.

        Returns
        -------
        delay_short : :class:`~skrf.network.Network` object
                a delayed short

        Notes
        -----
        This calls::

                delay_load(Gamma0=-1.0, d=d, unit=unit, **kwargs)

        See Also
        --------
        delay_load
        delay_open
        """
        return self.delay_load(Gamma0=-1.0, d=d, unit=unit, **kwargs)

    def delay_open(self, d: Number, unit: str = 'deg', **kwargs) -> Network:
        r"""
        Delayed open transmission line.

        Parameters
        ----------
        d : number
            the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
            the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments passed to :func:`delay_load`

        Returns
        -------
        delay_open : :class:`~skrf.network.Network` object
            a delayed open

        Notes
        -----
        This calls::

                delay_load(Gamma0=1.0, d=d, unit=unit, **kwargs)

        See Also
        --------
        delay_load
        delay_short
        """
        return self.delay_load(Gamma0=1.0, d=d, unit=unit, **kwargs)

    def shunt(self, ntwk: Network, **kwargs) -> Network:
        r"""
        Shunts a :class:`~skrf.network.Network`.

        This creates a :func:`tee` and connects
        `ntwk` to port 1, and returns the result.

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
        \*\*kwargs : keyword arguments passed to :func:`tee`

        Returns
        -------
        shunted_ntwk : :class:`~skrf.network.Network` object
                a shunted a ntwk. The resultant shunted_ntwk will have
                (2 + ntwk.number_of_ports -1) ports.

        See Also
        --------
        shunt_delay_load
        shunt_delay_open
        shunt_delay_short
        shunt_capacitor
        shunt_inductor
        """
        return connect(self.tee(**kwargs), 1, ntwk, 0)

    def shunt_delay_load(self, *args, **kwargs) -> Network:
        r"""
        Shunted delayed load.

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
            passed to func:`delay_load`

        Returns
        --------
        shunt_delay_load : :class:`~skrf.network.Network` object
            a shunted delayed load (2-port)

        Notes
        -----
        This calls::

                shunt(delay_load(*args, **kwargs))

        See Also
        --------
        shunt
        shunt_delay_open
        shunt_delay_short
        shunt_capacitor
        shunt_inductor
        """
        return self.shunt(self.delay_load(*args, **kwargs), **kwargs)

    def shunt_delay_open(self,*args,**kwargs) -> Network:
        r"""
        Shunted delayed open.

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
            passed to func:`delay_open`

        Returns
        -------
        shunt_delay_open : :class:`~skrf.network.Network` object
            shunted delayed open (2-port)

        Notes
        -----
        This calls::

                shunt(delay_open(*args, **kwargs))

        See Also
        --------
        shunt
        shunt_delay_load
        shunt_delay_short
        shunt_capacitor
        shunt_inductor
        """
        return self.shunt(self.delay_open(*args, **kwargs), **kwargs)

    def shunt_delay_short(self, *args, **kwargs) -> Network:
        r"""
        Shunted delayed short.

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_short`

        Returns
        -------
        shunt_delay_load : :class:`~skrf.network.Network` object
                shunted delayed open (2-port)

        Notes
        -----
        This calls::

                shunt(delay_short(*args, **kwargs))

        See Also
        --------
        shunt
        shunt_delay_load
        shunt_delay_open
        shunt_capacitor
        shunt_inductor
        """
        return self.shunt(self.delay_short(*args, **kwargs), **kwargs)

    def shunt_capacitor(self, C: NumberLike, **kwargs) -> Network:
        r"""
        Shunted capacitor.

        Parameters
        ----------
        C : number, array-like
            Capacitance in Farads.
        \*\*kwargs : arguments, keyword arguments
            passed to func:`capacitor`

        Returns
        -------
        shunt_capacitor : :class:`~skrf.network.Network` object
            shunted capacitor (2-port)

        Notes
        -----
        This calls::

                shunt(capacitor(C, **kwargs) ** short())

        See Also
        --------
        shunt
        shunt_delay_load
        shunt_delay_open
        shunt_delay_short
        shunt_inductor
        """
        return self.shunt(self.capacitor(C=C, **kwargs) ** 
                          self.short(**kwargs), **kwargs)

    def shunt_inductor(self, L: NumberLike, **kwargs) -> Network:
        r"""
        Shunted inductor.

        Parameters
        ----------
        L : number, array-like
            Inductance in Farads.
        \*\*kwargs : arguments, keyword arguments
            passed to func:`inductor`

        Returns
        -------
        shunt_inductor : :class:`~skrf.network.Network` object
            shunted inductor(2-port)

        Notes
        -----
        This calls::

                shunt(inductor(L, **kwargs) ** short())

        See Also
        --------
        shunt
        shunt_delay_load
        shunt_delay_open
        shunt_delay_short
        shunt_capacitor
        """
        return self.shunt(self.inductor(L=L, **kwargs) **
                          self.short(**kwargs), **kwargs)

    def attenuator(self, s21: NumberLike, db: bool = True, d: Number = 0,
                   unit: str = 'deg', name: str = '', **kwargs) -> Network:
        r"""
        Ideal matched attenuator of a given length.

        Parameters
        ----------
        s21 : number, array-like
            the attenuation
        db : bool, optional
            is s21 in dB? otherwise assumes linear. Default is True (dB).
        d : number, optional
            length of attenuator. Default is 0.
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
            the units of d.  See :func:`to_meters`, for details. 
            Default is 'deg'
        name : str
            Name for the returned attenuator Network
        \*\*kwargs : arguments, keyword arguments
            passed to func:`line`

        Returns
        -------
        ntwk : :class:`~skrf.network.Network` object
            2-port attenuator

        """

        s21 = npy.array(s21)
        if db:
            s21 = mf.db_2_magnitude(s21)

        result = self.match(nports=2)
        result.s[:, 0, 1] = s21
        result.s[:, 1, 0] = s21
        result = result ** self.line(d=d, unit=unit, **kwargs)
        result.name = name
        return result

    def lossless_mismatch(self, s11: NumberLike, db: bool = True, **kwargs) -> Network:
        r"""
        Lossless, symmetric mismatch defined by its return loss.

        Parameters
        ----------
        s11 : complex number, number, or array-like
            the reflection coefficient. if db==True, then phase is ignored

        db : bool, optional
            is s11 in db? otherwise assumes linear. Default is True (dB)

        \*\*kwargs : arguments, keyword arguments
            passed to func:`match`

        Returns
        -------
        ntwk : :class:`~skrf.network.Network` object
            2-port lossless mismatch

        """

        result = self.match(nports=2, **kwargs)
        s11 = npy.array(s11)
        if db:
            s11 = mf.db_2_magnitude(s11)

        result.s[:, 0, 0] = s11
        result.s[:, 1, 1] = s11

        s21_mag = npy.sqrt(1 - npy.abs(s11) ** 2)
        s21_phase = npy.angle(s11) + npy.pi / 2 * (npy.angle(s11) <= 0) - npy.pi / 2 * (npy.angle(s11) > 0)
        result.s[:, 0, 1] = s21_mag * npy.exp(1j * s21_phase)
        result.s[:, 1, 0] = result.s[:, 0, 1]
        return result

    def isolator(self, source_port: int = 0, **kwargs) -> Network:
        r"""
        Two-port isolator.

        Parameters
        -------------
        source_port: int in [0,1], optional
            port at which power can flow from. Default is 0.
        \*\*kwargs : arguments, keyword arguments
            passed to func:`thru`

        Returns
        -------
        ntwk : :class:`~skrf.network.Network` object
            2-port isolator

        """
        result = self.thru(**kwargs)
        if source_port == 0:
            result.s[:, 0, 1] = 0
        elif source_port == 1:
            result.s[:, 1, 0] = 0
        return result



    ## Noisy Networks

    def white_gaussian_polar(self, phase_dev: Number, mag_dev: Number,
                             n_ports: int = 1, **kwargs) -> Network:
        r"""
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
        """
        shape = (self.frequency.npoints, n_ports,n_ports)
        phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = shape)

        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s = mag_rv*npy.exp(1j*phase_rv)
        return result

    def random(self, n_ports: int = 1, reciprocal: bool = False, matched: bool = False,
               symmetric: bool = False, **kwargs) -> Network:
        r"""
        Complex random network.

        Creates a n-port network whose s-matrix is filled with random
        complex numbers. Optionally, result can be matched or reciprocal.

        Parameters
        ----------
        n_ports : int
            number of ports.
        reciprocal : bool
            makes s-matrix symmetric ($S_{mn} = S_{nm}$)
        symmetric : bool
            makes s-matrix diagonal have single value ($S_{mm}=S_{nn}$)
        matched : bool
            makes diagonals of s-matrix zero

        \*\*kwargs : passed to :class:`~skrf.network.Network`
                initializer

        Returns
        -------
        result : :class:`~skrf.network.Network` object
                the network
        """
        result = self.match(nports = n_ports, **kwargs)
        result.s = mf.rand_c(self.frequency.npoints, n_ports,n_ports)
        if reciprocal and n_ports>1:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m>n:
                        result.s[:,m,n] = result.s[:,n,m]
        if symmetric:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m==n:
                        result.s[:,m,n] = result.s[:,0,0]
        if matched:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m==n:
                        result.s[:,m,n] = 0

        return result

    ## OTHER METHODS
    def extract_distance(self, ntwk: Network) -> NumberLike:
        """
        Determines physical distance from a transmission or reflection Network.

        Given a matched transmission or reflection measurement the
        physical distance is estimated at each frequency point based on
        the scattering parameter phase of the ntwk and propagation constant.

        Notes
        -----
        If the Network is a reflect measurement, the returned distance will
        be twice the physical distance.

        Parameters
        ----------
        ntwk : `Network`
            A one-port network of either the reflection or the transmission.

        Returns
        -------
        d : number or array_like
            physical distance

        Examples
        --------
        >>> air = rf.air50
        >>> l = air.line(1, 'cm')
        >>> d_found = air.extract_distance(l.s21)
        >>> d_found
        """
        if ntwk.nports ==1:
            dphi = gradient(ntwk.s_rad_unwrap.flatten())
            dgamma = gradient(self.gamma.imag)
            return  -dphi/dgamma
        else:
            raise ValueError('ntwk must be one-port. Select s21 or s12 for a two-port.')



    def plot(self, *args, **kw):
        return self.frequency.plot(*args, **kw)



    def write_csv(self, filename: str = 'f,gamma,Z0,z0.csv'):
        """
        write this media's frequency, gamma, Z0, and z0 to a csv file.

        Parameters
        ----------
        filename : string, optional
            file name to write out data to.
            Default is 'f,gamma,Z0,z0.csv', so you probably want to specify it.

        See Also
        --------
        from_csv : class method to initialize Media object from a
            csv file written from this function
        """

        header = 'f[%s], Re(Z0), Im(Z0), Re(gamma), Im(gamma), Re(port Z0), Im(port Z0)\n'%self.frequency.unit

        g,z,pz  = self.gamma, \
                self.Z0, self.z0

        data = npy.vstack(\
                [self.frequency.f_scaled, z.real, z.imag, \
                g.real, g.imag, pz.real, pz.imag]).T

        npy.savetxt(filename,data,delimiter=',',header=header)



class DefinedGammaZ0(Media):
    """
    A media directly defined by its propagation constant and characteristic impedance.

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object or None
        frequency band of this transmission line medium.
        Default is None, which produces a 1-10ghz band with 101 points.
    z0 : number, array-like, or None
        The port impedance for media. Only needed if its different
        from the characteristic impedance of the transmission
        line. if `z0` is `None` then it will default to `Z0`
    gamma : number or array-like, optional
        complex propagation constant. `gamma` must adhere to
        the following convention:

            * positive real(gamma) = attenuation
            * positive imag(gamma) = forward propagation
        Default is 1j (lossless).
    Z0 : number or array-like, optional.
        complex characteristic impedance of the media.
        Default is 50 ohm.
    """

    def __init__(self, frequency: Union[Frequency, None] = None,
                 z0: Union[NumberLike, None] = None, Z0: NumberLike = 50,
                 gamma: NumberLike = 1j):
        super().__init__(frequency=frequency,
                                             z0=z0)
        self.gamma= gamma
        self.Z0 = Z0

    @classmethod
    def from_csv(cls, filename: str, *args, **kwargs) -> Media:
        """
        Create a Media from numerical values stored in a csv file.

        The csv file format must be written by the function :func:`write_csv`,
        or similar method which produces the following format::

            f[$unit], Re(Z0), Im(Z0), Re(gamma), Im(gamma), Re(port Z0), Im(port Z0)
            1, 1, 1, 1, 1, 1, 1
            2, 1, 1, 1, 1, 1, 1
            .....

        See Also
        --------
        write_csv
        """
        try:
            fid = open(filename)
        except(TypeError):
            # they may have passed a file
            fid = filename

        header = fid.readline()
        # this is not the correct way to do this ... but whatever
        f_unit = header.split(',')[0].split('[')[1].split(']')[0]

        f,z_re,z_im,g_re,g_im,pz_re,pz_im = \
                npy.loadtxt(fid,  delimiter=',').T

        if isinstance(filename, (str, Path)):
            fid.close()

        return cls(
            frequency = Frequency.from_f(f, unit=f_unit),
            Z0 = z_re+1j*z_im,
            gamma = g_re+1j*g_im,
            z0 = pz_re+1j*pz_im,
            *args, **kwargs
            )

    @property
    def npoints(self):
        return self.frequency.npoints

    @npoints.setter
    def npoints(self,val):
        # this is done to trigger checks on vector lengths for
        # gamma/Z0/z0
        new_freq= self.frequency.copy()
        new_freq.npoints = val
        self.frequency = new_freq


    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, val):
        if hasattr(self, '_frequency') and self._frequency is not None:

            # they are updating the frequency, we may have to do something
            attrs_to_test = [self._gamma, self._Z0, self._z0]
            if any([has_len(k) for k in attrs_to_test]):
                 raise NotImplementedError('updating a Media frequency, with non-constant gamma/Z0/z0 is not worked out yet')
        self._frequency = val

    @property
    def Z0(self):
        """
        Characteristic Impedance of the media.
        """
        return self._Z0*ones(len(self))

    @Z0.setter
    def Z0(self, val):
        self._Z0 = val

    @property
    def gamma(self):
        """
        Propagation constant.

        Returns
        ---------
        gamma : :class:`numpy.ndarray`
            complex propagation constant for this media

        Notes
        ------
        `gamma` must adhere to the following convention:

         * positive real(gamma) = attenuation
         * positive imag(gamma) = forward propagation
        """
        return self._gamma*ones(len(self))

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

def has_len(x: NumberLike) -> bool:
    """
    Test of x has any length (ie is a vector).

    This is slightly non-trivial because [3] has len() but is
    doesn't really have any length.
    """
    try:
        return (len(array(x))>1)
    except TypeError:
        return False

def parse_z0(s: str) -> NumberLike:
    """
    Parse a z0 string.

    Parameters
    ----------
    s : str
        z0 string, like '50+10j'

    Returns
    -------
    z0 : npy.ndarray

    Raises
    ------
    ValueError
        If could not arse the z0 string.
    """
    # they passed a string for z0, try to parse it
    re_numbers = re.compile(r'\d+')
    numbers = re.findall(re_numbers, s)
    if len(numbers)==2:
        out = float(numbers[0]) +1j*float(numbers[1])
    elif len(numbers)==1:
        out = float(numbers[0])
    else:
        raise ValueError('couldnt parse z0 string')
    return out

