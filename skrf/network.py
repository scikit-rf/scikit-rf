"""
.. module:: skrf.network
========================================
network (:mod:`skrf.network`)
========================================


Provides a n-port network class and associated functions.

Much of the functionality in this module is provided as methods and
properties of the :class:`Network` Class.


Network Class
===============

.. autosummary::
    :toctree: generated/

    Network

Building Network
----------------

.. autosummary::
    :toctree: generated/

    Network.from_z

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



Interpolation and Concatenation Along Frequency Axis
=====================================================

.. autosummary::
    :toctree: generated/

    stitch
    overlap
    Network.resample
    Network.interpolate
    Network.interpolate_self


Combining and Splitting Networks
===================================

.. autosummary::
    :toctree: generated/

    subnetwork

    one_port_2_two_port
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
    skrf.io.general.network_2_spreadsheet
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
    s2h
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
    h2s
    h2z
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
    Network.drop_non_monotonic_increasing

"""
from typing import (Any, NoReturn, Optional, Sequence,
    Sized, Union, Tuple, Callable, TYPE_CHECKING, Dict, List, TextIO)
from numbers import Number
from functools import reduce

import os
import warnings
import io
from pathlib import Path
import pickle
from pickle import UnpicklingError

import sys
import re
import zipfile
from copy import deepcopy as copy
from numbers import Number
from itertools import product

import numpy as npy
from numpy.linalg import inv as npy_inv
from numpy import fft, gradient, ndarray, reshape, shape, ones, std
from scipy import stats, signal  # for Network.add_noise_*, and Network.windowed
from scipy.interpolate import interp1d  # for Network.interpolate()
import unittest  # fotr unitest.skip

from . import mathFunctions as mf
from .frequency import Frequency
from .util import get_fid, get_extn, find_nearest_index, slice_domain
from .time import time_gate

from .constants import NumberLike, ZERO, K_BOLTZMANN, T0
from .constants import S_DEFINITIONS, S_DEF_DEFAULT

if TYPE_CHECKING:
    import pandas as pd

#from matplotlib import cm
#import matplotlib.pyplot as plt
#import matplotlib.tri as tri
#from scipy.interpolate import interp1d

class Network:
    r"""
    A n-port electrical network.

    For instructions on how to create Network see  :func:`__init__`.
    A n-port network [#TwoPortWiki]_ may be defined by three quantities
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
    about plotting see the :doc:`../../tutorials/Plotting` tutorial.

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
    ----------
    .. [#TwoPortWiki] http://en.wikipedia.org/wiki/Two-port_network

    """

    PRIMARY_PROPERTIES = ['s', 'z', 'y', 'a', 'h', 't']
    """
    Primary Network Properties list like 's', 'z', 'y', etc.
    """

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
    """
    Component functions like 're', 'im', 'mag', 'db', etc.
    """

    # provides y-axis labels to the plotting functions
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
    """
    y-axis labels to the plotting functions
    """

    noise_interp_kind = 'linear'
    """
    Default interpolation method.
    """

    # CONSTRUCTOR
    def __init__(self, file: str = None, name: str = None, params: dict = None,
                 comments: str = None, f_unit: str = None,
                 s_def: str = S_DEF_DEFAULT, **kwargs) -> None:
        r"""
        Network constructor.

        Creates an n-port microwave network from a `file` or directly
        from data. If no file or data is given, then an empty Network
        is created.

        Parameters
        ----------

        file : str or file-object
            file to load information from. supported formats are:
             * touchstone file (.s?p) (or .ts)
             * io.StringIO object (with `.name` property which contains the file extension, such as `myfile.s4p`)
             * pickled Network (.ntwk, .p) see :func:`write`
        name : str, optional
            Name of this Network. if None will try to use file, if it is a str
        params : dict, optional
            Dictionnary of parameters associated with the Network
        comments : str, optional
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
            keyword `encoding` can be used to define the Touchstone file encoding.

        Examples
        --------
        From a touchstone

        >>> n = rf.Network('ntwk1.s2p')

        From a pickle file

        >>> n = rf.Network('ntwk1.ntwk')

        Create a blank network, then fill in values

        >>> n = rf.Network()
        >>> freq = rf.Frequency(1, 3, 3, 'ghz')
        >>> n.frequency, n.s, n.z0 = freq, [1,2,3], [1,2,3]

        Directly from values

        >>> n = rf.Network(f=[1,2,3], s=[1,2,3], z0=[1,2,3])

        Define some parameters associated with the Network

        >>> n = rf.Network('ntwk1.s2p', params={'temperature': 25, 'voltage':5})

        See Also
        --------
        from_z : init from impedance values
        read : read a network from a file
        write : write a network to a file, using pickle
        write_touchstone : write a network to a touchstone file
        """

        # allow for old kwarg for backward compatibility
        if 'touchstone_filename' in kwargs:
            file = kwargs['touchstone_filename']

        self.name = name
        self.params = params
        self.comments = comments
        self.port_names = None
        self.encoding = kwargs.pop('encoding', None)

        self.deembed = None
        self.noise = None
        self.noise_freq = None
        self._z0 = npy.array(50, dtype=complex)

        if s_def not in S_DEFINITIONS:
            raise ValueError('s_def parameter should be either:', S_DEFINITIONS)
        else:
            self.s_def = s_def

        if file is not None:
            if isinstance(file, Path):
                file = str(file.resolve())

            # allows user to pass StringIO, filename or file obj
            if isinstance(file, io.StringIO):
                self.read_touchstone(file, self.encoding)

            else:
                # open file in 'binary' mode because we are going to try and
                # unpickle it first
                fid = get_fid(file, 'rb')

                try:
                    self.read(fid)
                except UnicodeDecodeError:  # Support for pickles created in Python2 and loaded in Python3
                    self.read(fid, encoding='latin1')
                except (UnpicklingError, TypeError):
                    # if unpickling doesn't work then, close fid, reopen in
                    # non-binary mode and try to read it as touchstone
                    filename = fid.name
                    fid.close()
                    self.read_touchstone(filename, self.encoding)

                if not fid.closed:
                    fid.close()

            if name is None and isinstance(file, str):
                name = os.path.splitext(os.path.basename(file))[0]

        if self.frequency is not None and f_unit is not None:
            self.frequency.unit = f_unit

        # Check for multiple attributes
        params = [attr for attr in PRIMARY_PROPERTIES if attr in kwargs]
        if len(params) > 1:
            raise ValueError(f'Multiple input parameters provided: {params}')

        # When initializing Network from different parameters than s
        # we need to make sure that z0 has been set first because it will be
        # needed in conversion to S-parameters. s is initialized with zeros here,
        # to determine the correct z0 shape afterwards.

        if params:
            s_shape = npy.array(kwargs[params[0]]).shape
            self.s = npy.zeros(s_shape, dtype=complex)

        self.z0 = kwargs.get('z0', self._z0)


        if "f" in kwargs.keys():
            if f_unit is None:
                f_unit = "ghz"
            kwargs["frequency"] = Frequency.from_f(kwargs.pop("f"), unit=f_unit)

        for attr in PRIMARY_PROPERTIES + ['frequency', 'noise', 'noise_freq']:
            if attr in kwargs:
                self.__setattr__(attr, kwargs[attr])

    @classmethod
    def from_z(cls, z: npy.ndarray, *args, **kw) -> 'Network':
        r"""
        Create a Network from its Z-parameters

        Parameters
        ----------
        z : Numpy array
            Impedance matrix. Should be of shape fxnxn,
            where f is frequency axis and n is number of ports
        \*\*kwargs :
            key word arguments can be used to assign properties of the
            Network, `f` and `z0`.

        Returns
        -------
        ntw : :class:`Network`
            Created Network

        Examples
        --------
        >>> f = rf.Frequency(start=1, stop=2, npoints=4)  # 4 frequency points
        >>> z = np.random.rand(len(f),2,2) + np.random.rand(len(f),2,2)*1j  # 2-port z-matrix: shape=(4,2,2)
        >>> ntw = rf.Network.from_z(z, f=f)

        """
        s = npy.zeros(shape=z.shape)
        me = cls(s=s, *args, **kw)
        me.z = z
        return me

    # OPERATORS
    def __pow__(self, other: 'Network') -> 'Network':
        """
        Cascade this network with another network

        Returns
        -------
        ntw : :class:`Network`
            Cascaded Network

        See Also
        --------
        cascade

        """
        check_frequency_exist(self)

        # if they pass a number then use power operator
        if isinstance(other, Number):
            out = self.copy()
            out.s = out.s ** other
            return out

        else:
            return cascade(self, other)

    def __rshift__(self, other: 'Network') -> 'Network':
        """
        Cascade two 4-port networks with "1=>2/3=>4" port numbering.
        
        Note
        ----
        connection diagram::

              A               B
           +---------+   +---------+
          -|0        1|---|0        1|-
          -|2        3|---|1        3|-
          ...       ... ...       ...
          -|2N-4  2N-3|---|2N-4  2N-3|-
          -|2N-2  2N-1|---|2N-2  2N-1|-
           +---------+   +---------+

        Returns
        -------
        ntw : :class:`Network`
            Cascaded Network

        See Also
        --------
        cascade

        """
        check_nports_equal(self, other)
        check_frequency_exist(self)
        (n,_) = shape(self.s[0])
        if (n / 2) != (n // 2):
            raise ValueError("Operator >> requires an even number of ports.")

        ix_old = list(range(n))
        n_2    = n//2
        n_2_1  = list(range(n_2))
        ix_new = list(sum(zip(n_2_1, list(map((lambda x: x + n_2), n_2_1))), ()))

        _ntwk1 = self.copy()
        _ntwk1.renumber(ix_old,ix_new)
        _ntwk2 = other.copy()
        _ntwk2.renumber(ix_old,ix_new)
        _rslt = _ntwk1 ** _ntwk2
        _rslt.renumber(ix_new,ix_old)
        return _rslt

    def __floordiv__(self, other: Union['Network', Tuple['Network', ...]] ) -> 'Network':
        """
        de-embedding 1 or 2 network[s], from this network

        :param other: skrf.Network, list, tuple: Network(s) to de-embed
        :return: skrf.Network: De-embedded network

        Returns
        -------
        ntw : :class:`Network`

        See Also
        --------
        inv : inverse s-parameters
        """

        if isinstance(other, (list, tuple)):
            if len(other) >= 3:
                warnings.warn(
                    "Number of networks greater than 2. Truncating!",
                    RuntimeWarning
                )
                other_tpl = other[:2]
        else:
            other_tpl = (other, )

        for o in other_tpl:
            if o.number_of_ports != 2:
                raise IndexError(f'Incorrect number of ports in network {o.name}.')

        if len(other_tpl) == 1:
            # if passed 1 network (A) and another network B
            #   e.g. A // B
            #   e.g. A // (B)
            # then de-embed like B.inv * A
            b = other_tpl[0]
            result = self.copy()
            result.s = (b.inv ** self).s
            # de_embed(self.s, b.s)
            return result
        else:
            # if passed 1 network (A) and a list/tuple of 2 networks (B, C),
            #   e.g. A // (B, C)
            #   e.g. A // [B, C]
            # then de-embed like B.inv * A * C.inv
            b = other_tpl[0]
            c = other_tpl[1]
            result = self.copy()
            result.s = (b.inv ** self ** c.inv).s
            # flip(de_embed(flip(de_embed(c.s, self.s)), b.s))
            return result

    def __mul__(self, other:'Network') -> 'Network':
        """
        Element-wise complex multiplication of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """
        check_frequency_exist(self)

        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s * other.s
        else:
            # other may be an array or a number
            result.s = self.s * npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __rmul__(self, other: 'Network') -> 'Network':
        """
        Element-wise complex multiplication of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """

        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s * other.s
        else:
            # other may be an array or a number
            result.s = self.s * npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __add__(self, other:'Network') -> 'Network':
        """
        Element-wise complex addition of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s + other.s
        else:
            # other may be an array or a number
            result.s = self.s + npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __radd__(self, other:'Network') -> 'Network':
        """
        Element-wise complex addition of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s + other.s
        else:
            # other may be an array or a number
            result.s = self.s + npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __sub__(self, other:'Network') -> 'Network':
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

    def __rsub__(self, other:'Network') -> 'Network':
        """
        Element-wise complex subtraction of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = other.s - self.s
        else:
            # other may be an array or a number
            result.s = npy.array(other).reshape(-1, self.nports, self.nports) - self.s

        return result

    def __truediv__(self, other: 'Network') -> 'Network':
        return self.__div__(other)

    def __div__(self, other: 'Network') -> 'Network':
        """
        Element-wise complex multiplication of s-matrix

        Returns
        -------
        ntw : :class:`Network`
        """
        result = self.copy()

        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s / other.s
        else:
            # other may be an array or a number
            result.s = self.s / npy.array(other).reshape(-1, self.nports, self.nports)

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if npy.all(npy.abs(self.s - other.s) < ZERO):
            return True
        else:
            return False

    def __ne__(self, other:object) -> bool:
        return (not self.__eq__(other))

    def __getitem__(self, key:Union[str, int, slice, Sized]) -> 'Network':
        """
        Slices a Network object based on an index, or human readable string

        Parameters
        ----------
        key : str, or slice
            if slice; like [2-10] then it is interpreted as the index of
            the frequency.
            if str, then should be like '50.1-75.5ghz', or just '50'.
            If the frequency unit is omitted then self.frequency.unit is
            used. This will also accept a 2 or 3 dimensional index of the
            forms:
                port1, port2
                key, port1, port2
            where port1 and port2 are allowed to be string port names if
            the network has them defined (Network.port_names)
            If port1 and port2 are integers, will return the single-port
            network based on matrix notation (indices starts at 1 not 0)

        Returns
        -------
        ntwk : skrf.Network
            interpolated in frequency if single dimension provided
            OR
            1-port network if multi-dimensional index provided

        Examples
        --------
        >>> from skrf.data import ring_slot
        >>> a = ring_slot['80-90ghz']
        >>> a.plot_s_db()

        Multidimensional indexing:
        >>> import skrf as rf
        >>> b = rf.Network("sometouchstonefile.s2p")
        >>> c = b['80mhz', 'first_port_name', 'second_port_name']
        >>> d = b['first_port_name', 'second_port_name']

        Equivalently:
        >>> d = b[1,2]

        Equivalent to:
        >>> d = b.s12
        """

        a = self.z0  # HACK: to force getter for z0 to re-shape it (z0 getter has side effects...)
        # If user passes a multidimensional index, try to return that 1 port subnetwork
        if isinstance(key, tuple):
            if len(key) == 3:
                slice_like, p1_name, p2_name = key
                return self[slice_like][p1_name, p2_name]
            elif len(key) == 2:
                p1_name, p2_name = key
                if type(p1_name) == int and type(p2_name) == int:  # allow integer indexing if desired
                    if p1_name <= 0 or p2_name <= 0 or p1_name > self.nports or p2_name > self.nports:
                        raise ValueError("Port index out of bounds")
                    p1_index = p1_name - 1
                    p2_index = p2_name - 1
                else:
                    if self.port_names is None:
                        raise ValueError("Can't index without named ports")
                    try:
                        p1_index = self.port_names.index(p1_name)
                    except ValueError as e:
                        raise KeyError(f"Unknown port {p1_name}")
                    try:
                        p2_index = self.port_names.index(p2_name)
                    except ValueError as e:
                        raise KeyError(f"Unknown port {p2_name}")
                ntwk = self.copy()
                ntwk.s = self.s[:, p1_index, p2_index]
                ntwk.z0 = self.z0[:, p1_index]
                ntwk.name = f"{self.name}({p1_name}, {p2_name})"
                ntwk.port_names = None
                return ntwk
            else:
                raise ValueError(f"Don't understand index: {key}")
            sliced_frequency = self.frequency[key]
            return self.interpolate(sliced_frequency)
        if isinstance(key, str):
            sliced_frequency = self.frequency[key]
            return self.interpolate(sliced_frequency)
        if isinstance(key, Frequency):
            return self.interpolate(key)
        # The following avoids interpolation when the slice is done directly with indices
        ntwk = self.copy_subset(key)
        return ntwk

    def __str__(self) -> str:
        """
        """
        f = self.frequency
        if self.name is None:
            name = ''
        else:
            name = self.name

        _z0 = self.z0

        if _z0.ndim < 2:
            z0 = _z0
        else:
            z0 = self.z0[0, :]

        output = '%i-Port Network: \'%s\',  %s, z0=%s' % (self.number_of_ports, name, str(f), str(z0))

        return output

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        """
        length of frequency axis
        """
        return len(self.s)

    # INTERNAL CODE GENERATION METHODS
    def __compatable_for_scalar_operation_test(self, other:'Network') -> None:
        """
        tests to make sure other network's s-matrix is of same shape
        """
        if other.frequency != self.frequency:
            raise IndexError('Networks must have same frequency. See `Network.interpolate`')

        if other.s.shape != self.s.shape:
            raise IndexError('Networks must have same number of ports.')

    def __generate_secondary_properties(self) -> None:
        """
        creates numerous `secondary properties` which are various
        different scalar projects of the primary properties. the primary
        properties are s,z, and y.
        """
        for prop_name in PRIMARY_PROPERTIES:
            for func_name in COMPONENT_FUNC_DICT:
                func = COMPONENT_FUNC_DICT[func_name]
                if 'gd' in func_name:  # scaling of gradient by frequency
                    def fget(self: 'Network', f: Callable = func, p: str = prop_name) -> npy.ndarray:
                        return f(getattr(self, p)) / (2 * npy.pi * self.frequency.step)
                else:
                    def fget(self: 'Network', f: Callable = func, p: str = prop_name) -> npy.ndarray:
                        return f(getattr(self, p))
                doc = """
                The {} component of the {}-matrix


                See Also
                --------
                {}
                """.format(func_name, prop_name, prop_name)

                setattr(self.__class__, f'{prop_name}_{func_name}', \
                        property(fget, doc=doc))

    def __generate_subnetworks(self) -> None:
        """
        generates all one-port sub-networks
        """
        for m in range(self.number_of_ports):
            for n in range(self.number_of_ports):
                def fget(self: 'Network', m:int=m, n:int=n) -> 'Network':
                    ntwk = self.copy()
                    ntwk.s = self.s[:, m, n]
                    ntwk.z0 = self.z0[:, m]
                    return ntwk

                doc = """
                one-port sub-network.
                """
                setattr(self.__class__, 's%i_%i'%(m+1, n+1),
                        property(fget, doc=doc))
                if m < 9 and n < 9:
                    setattr(self.__class__, 's%i%i' % (m + 1, n + 1),
                            getattr(self.__class__, 's%i_%i'%(m+1, n+1)))


    # PRIMARY PROPERTIES
    @property
    def s(self) -> npy.ndarray:
        """
        Scattering parameter matrix.

        The s-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so s11 can be accessed by
        taking the slice s[:,0,0].


        Returns
        -------
        s : complex :class:`numpy.ndarray` of shape `fxnxn`
            The scattering parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters
        """
        return self._s

    @s.setter
    def s(self, s: npy.ndarray) -> None:
        """
        Scattering parameter matrix.

        Parameters
        ----------
        s : :class:`numpy.ndarray`
            The input s-matrix should be of shape `fxnxn`,
            where f is frequency axis and n is number of ports.

        """
        self._s = fix_param_shape(s)
        self.__generate_secondary_properties()
        self.__generate_subnetworks()
        
        if self.z0.ndim == 0:
            self.z0 = self.z0

    @property
    def s_traveling(self) -> npy.ndarray:
        """
        Scattering parameter matrix with s_def = 'traveling'.

        Returns
        -------
        s : complex :class:`numpy.ndarray` of shape `fxnxn`
            The scattering parameter matrix.

        See Also
        --------
        s
        s_power
        s_pseudo
        s_traveling

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters
        """
        return s2s(self._s, self.z0, 'traveling', self.s_def)

    @s_traveling.setter
    def s_traveling(self, s) -> npy.ndarray:
        self.s = s2s(s, self.z0, self.s_def, 'traveling')

    @property
    def s_power(self) -> npy.ndarray:
        """
        Scattering parameter matrix with s_def = 'power'.

        Returns
        -------
        s : complex :class:`numpy.ndarray` of shape `fxnxn`
            The scattering parameter matrix.

        See Also
        --------
        s
        s_power
        s_pseudo
        s_traveling

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters
        """
        return s2s(self._s, self.z0, 'power', self.s_def)

    @s_power.setter
    def s_power(self, s) -> npy.ndarray:
        self.s = s2s(s, self.z0, self.s_def, 'power')

    @property
    def s_pseudo(self) -> npy.ndarray:
        """
        Scattering parameter matrix with s_def = 'pseudo'.

        Returns
        -------
        s : complex :class:`numpy.ndarray` of shape `fxnxn`
            The scattering parameter matrix.

        See Also
        --------
        s
        s_power
        s_pseudo
        s_traveling

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters
        """
        return s2s(self._s, self.z0, 'pseudo', self.s_def)

    @s_pseudo.setter
    def s_pseudo(self, s) -> npy.ndarray:
        self.s = s2s(s, self.z0, self.s_def, 'pseudo')

    @property
    def h(self) -> npy.ndarray:
        """
        Hybrid parameter matrix.

        The h-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so h11 can be accessed by
        taking the slice `h[:,0,0]`.


        Returns
        -------
        h : complex :class:`numpy.ndarray` of shape `fxnxn`
                the hybrid parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a
        h

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Two-port_network#Hybrid_parameters_(h-parameters)
        """
        return s2h(self.s, self.z0)

    @h.setter
    def h(self, value: npy.ndarray) -> None:
        self._s = h2s(fix_param_shape(value), self.z0)

    @property
    def y(self) -> npy.ndarray:
        """
        Admittance parameter matrix.

        The y-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so y11 can be accessed by
        taking the slice `y[:,0,0]`.


        Returns
        -------
        y : complex :class:`numpy.ndarray` of shape `fxnxn`
                the admittance parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
        """
        return s2y(self._s, self.z0, s_def=self.s_def)

    @y.setter
    def y(self, value: npy.ndarray) -> None:
        self._s = y2s(fix_param_shape(value), self.z0, s_def=self.s_def)

    @property
    def z(self) -> npy.ndarray:
        """
        Impedance parameter matrix.

        The z-matrix  [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so z11 can be accessed by
        taking the slice `z[:,0,0]`.


        Returns
        -------
        z : complex :class:`numpy.ndarray` of shape `fxnxn`
                the Impedance parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/impedance_parameters
        """
        return s2z(self._s, self.z0, s_def=self.s_def)

    @z.setter
    def z(self, value: npy.ndarray) -> None:
        self._s = z2s(fix_param_shape(value), self.z0, s_def=self.s_def)

    @property
    def t(self) -> npy.ndarray:
        """
        Scattering transfer parameters

        The t-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray`
        which has shape `fx2x2`, where `f` is frequency axis.
        Note that indexing starts at 0, so t11 can be accessed by
        taking the slice `t[:,0,0]`.

        The t-matrix, also known as the wave cascading matrix, is
        only defined for a 2-port Network.

        Returns
        -------
        t : complex npy.ndarray of shape `fx2x2`
                t-parameters, aka scattering transfer parameters


        See Also
        --------
        s
        y
        z
        t
        a

        References
        -----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters
        """
        return s2t(self.s)

    @t.setter
    def t(self, value: npy.ndarray) -> None:
        self._s = t2s(fix_param_shape(value))

    @property
    def s_invert(self) -> npy.ndarray:
        """
        Inverted scattering parameter matrix.

        Inverted scattering parameters are simply inverted s-parameters,
        defined as a = 1/s. Useful in analysis of active networks.
        The a-matrix is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so a11 can be accessed by
        taking the slice a[:,0,0].


        Returns
        -------
        s_inv : complex :class:`numpy.ndarray` of shape `fxnxn`
                the inverted scattering parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a
        """
        return 1 / self.s

    @s_invert.setter
    def s_invert(self, value: npy.ndarray) -> NoReturn:
        raise NotImplementedError

    @property
    def a(self) -> npy.ndarray:
        """
        abcd parameter matrix. Used to cascade two-ports

        The abcd-matrix  [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so abcd11 can be accessed by
        taking the slice `abcd[:,0,0]`.


        Returns
        -------
        abcd : complex :class:`numpy.ndarray` of shape `fxnxn`
                the Impedance parameter matrix.

        See Also
        --------
        s
        y
        z
        t
        a
        abcd

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/impedance_parameters
        """
        return s2a(self.s, self.z0)

    @a.setter
    def a(self, value: npy.ndarray) -> None:
        self._s = a2s(fix_param_shape(value), self.z0)

    @property
    def z0(self) -> npy.ndarray:
        """
        Characteristic impedance[s] of the network ports.

        This property stores the  characteristic impedance of each port
        of the network. Because it is possible that each port has
        a different characteristic impedance each varying with
        frequency, `z0` is stored internally as a `fxn` array.

        However because  `z0` is frequently simple (like 50ohm), it can
        be set with just number as well.

        Returns
        -------
        z0 : :class:`numpy.ndarray` of shape fxn
                characteristic impedance for network

        """
        return self._z0

    @z0.setter
    def z0(self, z0: NumberLike) -> None:
        # cast any array like type (tuple, list) to a npy.array
        z0 = npy.array(z0, dtype=complex)

        # if _z0 is a vector or matrix, we check if _s is already assigned.
        # If not, we cannot proof the correct dimensions and silently accept
        # any vector or fxn array
        if not hasattr(self, '_s'):
            self._z0 = z0
            return

        # if _z0 is a scalar, we broadcast to the correct shape.
        # 
        # if _z0 is a vector, we check if the dimension matches with either
        # nports or frequency.npoints. If yes, we accept the value.
        # Note that there can be an ambiguity in theory, if nports == npoints
        #
        # if _z0 is a matrix, we check if the shape matches with _s
        # In any other case raise an Exception
        self._z0 = npy.empty(self.s.shape[:2], dtype=complex)
        if z0.ndim == 0:
            self._z0[:] = z0 
        elif z0.ndim == 1 and z0.shape[0] == self.s.shape[0]:
            self._z0[:] = z0[:, None]
        elif z0.ndim == 1 and z0.shape[0] == self.s.shape[1]:
            self._z0[:] = z0[None, :]
        elif z0.shape == self.s.shape[:2]:
            self._z0 = z0
        else:
            raise AttributeError(f'Unable to broadcast z0 shape {z0.shape} to s shape {self.s.shape}.')

    @property
    def frequency(self) -> Frequency:
        """
        frequency information for the network.

        This property is a :class:`~skrf.frequency.Frequency` object.
        It holds the frequency vector, as well frequency unit, and
        provides other properties related to frequency information, such
        as start, stop, etc.

        Returns
        -------
        frequency :  :class:`~skrf.frequency.Frequency` object
                frequency information for the network.


        See Also
        --------
        f : property holding frequency vector in Hz
        change_frequency : updates frequency property, and
            interpolates s-parameters if needed
        interpolate : interpolate function based on new frequency
            info
        """
        try:
            return self._frequency
        except (AttributeError):
            self._frequency = Frequency(0, 0, 0)
            return self._frequency

    @frequency.setter
    def frequency(self, new_frequency: Union[Frequency, int, Sequence[float], npy.ndarray]) -> None:
        """
        takes a Frequency object, see  frequency.py
        """
        if isinstance(new_frequency, Frequency):
            self._frequency = new_frequency.copy()
        else:
            try:
                self._frequency = Frequency.from_f(new_frequency)
            except (TypeError):
                raise TypeError('Could not convert argument to a frequency vector')

    @property
    def inv(self) -> 'Network':
        """
        a :class:`Network` object with 'inverse' s-parameters.

        This is used for de-embedding.
        It is defined such that the inverse of the s-matrix cascaded with itself is a unity scattering transfer parameter (T) matrix.

        Returns
        -------
        inv : a :class:`Network` object
                a :class:`Network` object with 'inverse' s-parameters.

        See Also
        --------
                inv : function which implements the inverse s-matrix
        """
        if self.number_of_ports < 2:
            raise (TypeError('One-Port Networks don\'t have inverses'))
        out = self.copy()
        out.s = inv(self.s)
        out.deembed = True
        return out

    @property
    def f(self) -> npy.ndarray:
        """
        the frequency vector for the network, in Hz.

        Returns
        -------
        f : :class:`numpy.ndarray`
                frequency vector in Hz

        See Also
        --------
                frequency : frequency property that holds all frequency
                        information
        """
        return self.frequency.f

    @f.setter
    def f(self, f: Union[NumberLike, Frequency]) -> None:
        warnings.warn('frequency.f parameter will be immutable in the next release.',
             DeprecationWarning, stacklevel=2)

        if isinstance(f, Frequency):
            self.frequency = f
        else:
            tmpUnit = self.frequency.unit
            self.frequency = Frequency.from_f(f, unit=tmpUnit)

    @property
    def noisy(self) -> bool:
      """
      whether this network has noise
      """
      try:
        return self.noise is not None and self.noise_freq is not None
      except:
        return False

    @property
    def n(self) -> npy.ndarray:
        """
        the ABCD form of the noise correlation matrix for the network
        """
        if not self.noisy:
            raise ValueError('network does not have noise')

        if self.noise_freq.f.size > 1 :
            noise_real = interp1d(self.noise_freq.f, self.noise.real, axis=0, kind=Network.noise_interp_kind)
            noise_imag = interp1d(self.noise_freq.f, self.noise.imag, axis=0, kind=Network.noise_interp_kind)
            return noise_real(self.frequency.f) + 1.0j * noise_imag(self.frequency.f)
        else :
            noise_real =  self.noise.real
            noise_imag = self.noise.imag
            return noise_real + 1.0j * noise_imag




    @property
    def f_noise(self) -> Frequency:
      """
      the frequency vector for the noise of the network, in Hz.
      """
      if not self.noisy:
        raise ValueError('network does not have noise')
      return self.noise_freq

    @property
    def y_opt(self) -> npy.ndarray:
      """
      the optimum source admittance to minimize noise
      """
      noise = self.n
      return (npy.sqrt(noise[:,1,1]/noise[:,0,0] - npy.square(npy.imag(noise[:,0,1]/noise[:,0,0])))
          + 1.j*npy.imag(noise[:,0,1]/noise[:,0,0]))

    @property
    def z_opt(self) -> npy.ndarray:
      """
      the optimum source impedance to minimize noise
      """
      return 1./self.y_opt

    @property
    def g_opt(self) -> npy.ndarray:
      """
      the optimum source reflection coefficient to minimize noise
      """
      return z2s(self.z_opt.reshape((self.f.shape[0], 1, 1)), self.z0[:,0])[:,0,0]

    @property
    def nfmin(self) -> npy.ndarray:
      """
      the minimum noise figure for the network
      """
      noise = self.n
      return npy.real(1. + (noise[:,0,1] + noise[:,0,0] * npy.conj(self.y_opt))/(2*K_BOLTZMANN*T0))

    @property
    def nfmin_db(self) -> npy.ndarray:
      """
      the minimum noise figure for the network in dB
      """
      return mf.complex_2_db10(self.nfmin)

    def nf(self, z: NumberLike) -> npy.ndarray:
      """
      the noise figure for the network if the source impedance is z
      """
      z0 = self.z0
      y_opt = self.y_opt
      fmin = self.nfmin
      rn = self.rn

      ys = 1./z
      gs = npy.real(ys)
      return fmin + rn/gs * npy.square(npy.absolute(ys - y_opt))

    def nfdb_gs(self, gs: NumberLike) -> npy.ndarray:
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

    @property
    def rn(self) -> npy.ndarray:
      """
      the equivalent noise resistance for the network
      """
      return npy.real(self.n[:,0,0]/(4.*K_BOLTZMANN*T0))

    # SECONDARY PROPERTIES
    @property
    def number_of_ports(self) -> int:
        """
        the number of ports the network has.

        Returns
        -------
        number_of_ports : number
                the number of ports the network has.

        """
        try:
            return self.s.shape[1]
        except (AttributeError):
            return 0

    @property
    def nports(self) -> int:
        """
        the number of ports the network has.

        Returns
        -------
        number_of_ports : number
                the number of ports the network has.

        """
        return self.number_of_ports

    @property
    def port_tuples(self) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples, for each port index pair

        A convenience function for the common task for iterating over
        all s-parameters index pairs

        This just calls::

            [(y,x) for x in range(self.nports) for y in range(self.nports)]


        Returns
        -------
        ports_ind : list of tuples
            list of all port index tuples.

        Examples
        --------
        >>> ntwk = skrf.data.ring_slot
        >>> for (idx_i, idx_j) in ntwk.port_tuples: print(idx_i, idx_j)

        """
        return [(y, x) for x in range(self.nports) for y in range(self.nports)]

    @property
    def passivity(self) -> ndarray:
        r"""
        passivity metric for a multi-port network.

        This returns a matrix who's diagonals are equal to the total
        power received at all ports, normalized to the power at a single
        excitement port.

        mathematically, this is a test for unitary-ness of the
        s-parameter matrix [#]_.

        for two port this is

        .. math::

                ( |S_{11}|^2 + |S_{21}|^2 \, , \, |S_{22}|^2+|S_{12}|^2)

        in general it is

        .. math::

                S^H \cdot S

        where :math:`H` is conjugate transpose of S, and :math:`\cdot`
        is dot product.

        Returns
        -------
        passivity : :class:`numpy.ndarray` of shape fxnxn

        References
        ----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Lossless_networks
        """
        return passivity(self.s)

    @property
    def reciprocity(self) -> npy.ndarray:
        """
        reciprocity metric for a multi-port network.

        This returns the difference between the s-parameter matrix
        and its transpose.

        for two port this is

        .. math::

                S - S^T

        where :math:`T` is transpose of S

        Returns
        -------
        reciprocity : :class:`numpy.ndarray` of shape `fxnxn`

        """
        return reciprocity(self.s)

    @property
    def reciprocity2(self) -> npy.ndarray:
        """
        Reciprocity metric #2

        .. math::

                abs(1 - S/S^T )

        for the two port case, this evaluates to the distance of the
        determinant of the wave-cascading matrix from unity.

        Returns
        -------
        reciprocity : :class:`numpy.ndarray` of shape `fxnxn`

        """
        return abs(1 - self.s / self.s.swapaxes(1, 2))

    @property
    def stability(self) -> npy.ndarray:
        """
        Stability factor

        .. math::

                K = ( 1 - |S_{11}|^2 - |S_{22}|^2 + |D|^2 ) / (2 * |S_{12}| * |S_{21}|)

            with

                D = S_{11} S_{22} - S_{12} S_{21}

        Returns
        -------
        K : :class:`numpy.ndarray` of shape `f`

        """
        assert self.nports == 2, "Stability factor K is only defined for two ports"

        D = self.s[:, 0, 0] * self.s[:, 1, 1] - self.s[:, 0, 1] * self.s[:, 1, 0]
        K = (1 - npy.abs(self.s[:, 0, 0]) ** 2 - npy.abs(self.s[:, 1, 1]) ** 2 + npy.abs(D) ** 2) / (
        2 * npy.abs(self.s[:, 0, 1]) * npy.abs(self.s[:, 1, 0]))
        return K

    @property
    def group_delay(self) -> npy.ndarray:
        """
        The group delay

        Usually used as a measure of dispersion (or distortion).

        Defined as the derivative of the unwrapped s-parameter phase
        (in rad) with respect to the frequency::

            -d(self.s_rad_unwrap)/d(self.frequency.w)

        Returns
        -------
        gd : :class:`numpy.ndarray` of shape `xnxn`

        References
        ----------
        https://en.wikipedia.org/wiki/Group_delay_and_phase_delay
        """
        gd = self.s * 0  # quick way to make a new array of correct shape

        phi = self.s_rad_unwrap
        dw = self.frequency.dw

        for m, n in self.port_tuples:
            dphi = gradient(phi[:, m, n])
            gd[:, m, n] = -dphi / dw

        return gd

    ## NETWORK CLASSIFIERs
    def is_reciprocal(self, tol: float = mf.ALMOST_ZERO) -> bool:
        """
        Test for reciprocity

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance. The default is :data:`skrf.mathFunctions.ALMOST_ZERO`.

        Returns
        -------
        bool : boolean

        See Also
        --------
        reciprocity

        """
        return npy.allclose(reciprocity(self.s), npy.zeros_like(self.s), atol=tol)

    def is_symmetric(self, n: int = 1, port_order: Dict[int, int] = {}, tol: float = mf.ALMOST_ZERO) -> bool:
        """
        Returns whether the 2N-port network has n-th order reflection symmetry by checking
        :math:`S_{i,i} == S_{j,j}` for appropriate pair(s) of :math:`i` and :math:`j`.

        Parameters
        ----------
        n : int
            Order of line symmetry to test for
        port_order : dict[int, int]
            Renumbering of zero-indexed ports before testing
        tol : float
            Tolerance in numeric comparisons. Default is :data:`skrf.mathFunctions.ALMOST_ZERO`.

        Returns
        -------
        bool : boolean

        Raises
        ------
        ValueError
            (1) If the network has an odd number of ports
            (2) If n is not in the range 1 to N
            (3) If n does not evenly divide 2N
            (4) If port_order is not a valid reindexing of ports
            e.g. specifying x->y but not y->z, specifying x->y twice,
            or using an index outside the range 0 to 2N-1

        References
        ----------

        https://en.wikipedia.org/wiki/Two-port_network#Scattering_parameters_(S-parameters)

        """

        nfreqs, ny, nx = self.s.shape  # nfreqs is number of frequencies, and nx, ny both are number of ports (2N)
        if nx % 2 != 0 or nx != ny:
            raise ValueError('Using is_symmetric() is only valid for a 2N-port network (N=2,4,6,8,...)')
        n_ports = nx // 2
        if n <= 0 or n > n_ports:
            raise ValueError('specified order n = ' + str(n) + ' must be ' +
                             'between 1 and N = ' + str(n_ports) + ', inclusive')
        if nx % n != 0:
            raise ValueError('specified order n = ' + str(n) + ' must evenly divide ' +
                             'N = ' + str(n_ports))

        from_ports = list(map(lambda key: int(key), port_order.keys()))
        to_ports = list(map(lambda val: int(val), port_order.values()))
        test_network = self.copy()  # TODO: consider defining renumbered()
        if len(from_ports) > 0 and len(to_ports) > 0:
            test_network.renumber(from_ports, to_ports)

        offs = npy.array(range(0, n_ports))  # port index offsets from each mirror line
        for k in range(0, n_ports, n_ports // n):  # iterate through n mirror lines
            mirror = k * npy.ones_like(offs)
            i = mirror - 1 - offs
            j = mirror + offs
            if not npy.allclose(test_network.s[:, i, i], test_network.s[:, j, j], atol=tol):
                return False
        return True

    def is_passive(self, tol: float = mf.ALMOST_ZERO) -> bool:
        """
        Test for passivity

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance. The default is :data:`skrf.mathFunctions.ALMOST_ZERO`

        Returns
        -------
        bool : boolean

        """
        try:
            M = npy.square(self.passivity)
        except ValueError:
            return False

        I = npy.identity(M.shape[-1])
        for f_idx in range(len(M)):
            D = I - M[f_idx, :, :]  # dissipation matrix
            if not mf.is_positive_definite(D) \
                    and not mf.is_positive_semidefinite(mat=D, tol=tol):
                return False
        return True

    def is_lossless(self, tol: float = mf.ALMOST_ZERO) -> bool:
        """
        Test for losslessness

        [S] is lossless if [S] is unitary, i.e. if :math:`([S][S]^* = [1])`


        Parameters
        ----------
        tol : float, optional
            Numerical tolerance. The default is :data:`skrf.mathFunctions.ALMOST_ZERO`

        Returns
        -------
        bool : boolean

        See Also
        --------
        is_passive, is_symmetric, is_reciprocal

        References
        ----------
        https://en.wikipedia.org/wiki/Unitary_matrix
        """
        for f_idx in range(len(self.s)):
            mat = self.s[f_idx, :, :]
            if not mf.is_unitary(mat, tol=tol):
                return False
        return True

    ## CLASS METHODS
    def copy(self) -> 'Network':
        """
        Returns a copy of this Network

        Needed to allow pass-by-value for a Network instead of
        pass-by-reference

        Returns
        -------
        ntwk : :class:`Network`
            Copy of the Network

        """
        ntwk = Network(s=self.s,
                       frequency=self.frequency.copy(),
                       z0=self.z0, s_def=self.s_def,
                       comments=self.comments, params=self.params
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

    def copy_from(self, other: 'Network') -> None:
        """
        Copies the contents of another Network into self

        Uses copy, so that the data is passed-by-value, not reference

        Parameters
        ----------
        other : Network
            the network to copy the contents of

        Examples
        --------
        >>> a = rf.N()
        >>> b = rf.N('my_file.s2p')
        >>> a.copy_from (b)
        """
        for attr in ['_s', 'frequency', '_z0', 'name']:
            self.__setattr__(attr, copy(other.__getattribute__(attr)))

    def copy_subset(self, key: npy.ndarray) -> 'Network':
        """
        Returns a copy of a frequency subset of this Network

        Needed to allow pass-by-value for a subset Network instead of
        pass-by-reference

        Parameters
        -----------
        key : numpy array
            the array indices of the frequencies to take

        Returns
        -------
        ntwk : :class:`Network`
            Copy of the frequency subset of the Network

        """
        ntwk = Network(s=self.s[key,:],
                       frequency=self.frequency[key].copy(),
                       z0=self.z0[key,:],
                       )

        if isinstance(self.name, str):
            ntwk.name = self.name + '_subset'
        else:
            ntwk.name = self.name

        if self.noise is not None and self.noise_freq is not None:
            ntwk.noise = npy.copy(self.noise[key,:])
            ntwk.noise_freq = npy.copy(self.noise_freq[key])

        try:
            ntwk.port_names = copy(self.port_names)
        except(AttributeError):
            ntwk.port_names = None
        return ntwk

    def drop_non_monotonic_increasing(self) -> None:
        """Drop invalid values based on duplicate and non increasing frequency values

        Example
        -------

        The following example shows how to use the :func:`drop_non_monotonic_increasing`
        automatically, if invalid frequency data is detected and an
        :class:`~skrf.frequency.InvalidFrequencyWarning` is thrown.


        >>> import warnings
        >>> import skrf as rf
        >>> from skrf.frequency import InvalidFrequencyWarning
        >>> with warnings.catch_warnings(record=True) as warns:
        >>>     net = rf.Network('corrupted_network.s2p')
        >>>     w = [w for w in warns if issubclass(w.category, InvalidFrequencyWarning)]
        >>>     if w:
        >>>         net.drop_non_monotonic_increasing()

        """
        idx = self.frequency.drop_non_monotonic_increasing()

        # z0 getter and setter depend on s.shape matching z0.shape.
        # Call z0 getter and setter only when s and z0 shapes match.
        z0_new = npy.delete(self.z0, idx, axis=0)
        self.s = npy.delete(self.s, idx, axis=0)
        self.z0 = z0_new

        if self.noisy:
            idx = self.noise_freq.drop_non_monotonic_increasing()
            self.noise = npy.delete(self.noise, idx, axis=0)


    def set_noise_a(self, noise_freq: Frequency = None, nfmin_db: float = 0,
        gamma_opt: float = 0, rn: NumberLike = 1 ) -> None:
          """
          sets the "A" (ie cascade) representation of the correlation matrix, based on the
          noise frequency and input parameters.
          """
          sh_fr = noise_freq.f.shape
          nfmin_db = npy.broadcast_to(npy.atleast_1d(nfmin_db), sh_fr)
          gamma_opt = npy.broadcast_to(npy.atleast_1d(gamma_opt), sh_fr)
          rn = npy.broadcast_to(npy.atleast_1d(rn), sh_fr)


          nf_min = npy.power(10., nfmin_db/10.)
          # TODO maybe interpolate z0 as above
          y_opt = 1./(self.z0[0, 0] * (1. + gamma_opt)/(1. - gamma_opt))
          noise = 4.*K_BOLTZMANN*T0*npy.array(
                [[rn, (nf_min-1.)/2. - rn*npy.conj(y_opt)],
                [(nf_min-1.)/2. - rn*y_opt, npy.square(npy.absolute(y_opt)) * rn]]
              )
          self.noise = noise.swapaxes(0, 2).swapaxes(1, 2)
          self.noise_freq = noise_freq




    # touchstone file IO
    def read_touchstone(self, filename: Union[str, TextIO],
                        encoding: Union[str, None] = None) -> None:
        """
        loads values from a touchstone file.

        The work of this function is done through the
        :class:`~skrf.io.touchstone` class.

        Parameters
        ----------
        filename : str or file-object
            touchstone file name.
        encoding : str, optional
            define the file encoding to use. Default value is None,
            meaning the encoding is guessed.

        Note
        ----
        Only the scattering parameters format is supported at the moment


        """
        from .io import touchstone
        touchstoneFile = touchstone.Touchstone(filename, encoding=encoding)

        if touchstoneFile.get_format().split()[1] != 's':
            raise NotImplementedError('only s-parameters supported for now.')

        self.comments = touchstoneFile.get_comments()

        try:
            self.variables = touchstoneFile.get_comment_variables()
        except:
            pass

        self.port_names = touchstoneFile.port_names

        f, self.s = touchstoneFile.get_sparameter_arrays()  # note: freq in Hz
        self.frequency = Frequency.from_f(f, unit='hz')
        self.frequency.unit = touchstoneFile.frequency_unit

        if touchstoneFile.has_hfss_port_impedances:
            self.gamma, self.z0 = touchstoneFile.get_gamma_z0()
            # This is the s_def that HFSS uses
            self.s_def = 'traveling'
        else:
            self.z0 = touchstoneFile.resistance
        

        if touchstoneFile.noise is not None:
          noise_freq = touchstoneFile.noise[:, 0] * touchstoneFile.frequency_mult
          nfmin_db = touchstoneFile.noise[:, 1]
          gamma_opt_mag = touchstoneFile.noise[:, 2]
          gamma_opt_angle = npy.deg2rad(touchstoneFile.noise[:, 3])

          # TODO maybe properly interpolate z0?
          # it probably never actually changes
          if touchstoneFile.version == '1.0':
            rn = touchstoneFile.noise[:, 4] * self.z0[0, 0]
          else:
            rn = touchstoneFile.noise[:, 4]

          gamma_opt = gamma_opt_mag * npy.exp(1j * gamma_opt_angle)

          nf_min = npy.power(10., nfmin_db/10.)
          # TODO maybe interpolate z0 as above
          y_opt = 1./(self.z0[0, 0] * (1. + gamma_opt)/(1. - gamma_opt))
          # use the voltage/current correlation matrix; this works nicely with
          # cascading networks
          self.noise_freq = Frequency.from_f(noise_freq, unit='hz')
          self.noise_freq.unit = touchstoneFile.frequency_unit
          self.set_noise_a(self.noise_freq, nfmin_db, gamma_opt, rn )

        if self.name is None:
            try:
                self.name = os.path.basename(os.path.splitext(filename)[0])
                # this may not work if filename is a file object
            except(AttributeError, TypeError):
                # in case they pass a file-object instead of file name,
                # get the name from the touchstone file
                try:
                    self.name = os.path.basename(os.path.splitext(touchstoneFile.filename)[0])
                except():
                    print('warning: couldn\'t inspect network name')
                    self.name = ''
                pass

    @classmethod
    def zipped_touchstone(cls, filename: str, archive: zipfile.ZipFile) -> 'Network':
        """
        read a Network from a Touchstone file in a ziparchive

        Parameters
        ----------
        filename : str
            the full path filename of the touchstone file
        archive : zipfile.ZipFile
            the opened zip archive

        Returns
        -------
        ntwk : :class:`Network`
            Network from the Touchstone file

        """
        # Touchstone requires file objects to be seekable (for get_gamma_z0_from_fid)
        # A ZipExtFile object is not seekable prior to Python 3.7, so use StringIO
        # and manually add a name attribute
        fileobj = io.StringIO(archive.open(filename).read().decode('UTF-8'))
        fileobj.name = filename
        ntwk = Network(fileobj)
        return ntwk

    def write_touchstone(self, filename: str = None, dir: str = None,
                         write_z0: bool = False, skrf_comment: bool = True,
                         return_string: bool = False, to_archive: bool = None,
                         form: str = 'ri', format_spec_A: str = '{}', format_spec_B: str = '{}',
                         format_spec_freq : str = '{}', r_ref : float = None) -> Optional[str]:
        """
        Write a contents of the :class:`Network` to a touchstone file.

        Parameters
        ----------
        filename : a string, optional
            touchstone filename, without extension. if 'None', then
            will use the network's :attr:`name`.
        dir : string, optional
            the directory to save the file in.
        write_z0 : boolean
            write impedance information into touchstone as comments,
            like Ansoft HFSS does
        skrf_comment : bool, optional
            write `created by skrf` comment
        return_string : bool, optional
            return the file_string rather than write to a file
        to_archive : zipfile.Zipfile
            opened ZipFile object to place touchstone file in
        form : string
            format to write data:
            'db': db, deg. 'ma': mag, deg. 'ri': real, imag.
        format_spec_A : string, optional
            Any valid format specifying string as given by https://docs.python.org/3/library/string.html#format-string-syntax
            This specifies the formatting in the resulting touchstone file for the A part of the S parameter, (e.g. the dB magnitude for 'db' format, the linear
            magnitude for 'ma' format, or the real part for 'ri' format)
        format_spec_B : string, optional
            Any valid format specifying string as given by https://docs.python.org/3/library/string.html#format-string-syntax
            This specifies the formatting in the resulting touchstone file for the B part of the S parameter, (e.g. the angle in degrees for 'db' format,
            the angle in degrees for 'ma' format, or the imaginary part for 'ri' format)
        format_spec_freq : string, optional
            Any valid format specifying string as given by https://docs.python.org/3/library/string.html#format-string-syntax
            This specifies the formatting in the resulting touchstone file for the frequency.
        r_ref : float
            Reference impedance to renormalize the network.
            If None network port impedance is used if possible. If None and
            network port impedance is complex and not equal at all ports and
            frequency points raises ValueError.

        Note
        ----
        Format supported at the moment are [Hz/kHz/MHz/GHz] S [DB/MA/RI]
        Frequency unit can be changed by setting Network.frequency.unit property


        Note
        ----
        The functionality of this function should take place in the
        :class:`~skrf.io.touchstone.Touchstone` class.

        """
        # according to Touchstone 2.0 spec
        # [no tab, max. 4 coeffs per line, etc.]

        have_complex_ports = npy.any(self.z0.imag != 0)
        equal_z0 = npy.all(self.z0 == self.z0[0, 0])

        ntwk = self.copy()

        if r_ref is None and write_z0 == False:
            if not equal_z0:
                raise ValueError("Network has unequal port impedances but reference impedance for renormalization 'r_ref' is not specified.")
            if have_complex_ports:
                raise ValueError("Network port impedances are complex but reference impedance for renormalization 'r_ref' is not specified.")
            r_ref = ntwk.z0[0, 0]
        elif r_ref is not None:
            if not npy.isscalar(r_ref):
                raise ValueError('r_ref must be scalar')
            if r_ref.imag != 0:
                raise ValueError('r_ref must be real')
            ntwk.renormalize(r_ref)

        if write_z0 and have_complex_ports and ntwk.s_def != 'traveling':
            warnings.warn(f"Network s_def will be changed from {ntwk.s_def} to traveling to be compatible with HFSS. To silence this warning manually use Network.renormalize to change the s_def.", UserWarning)
            ntwk.renormalize(ntwk.z0, s_def='traveling')

        if filename is None:
            if ntwk.name is not None:
                filename = ntwk.name
            else:
                raise ValueError('No filename given. Network must have a name, or you must provide a filename')

        if get_extn(filename) is None:
            filename = filename + '.s%ip' % ntwk.number_of_ports

        if dir is not None:
            filename = os.path.join(dir, filename)

        # set internal variables according to form
        form = form.upper()
        if form == "RI":
            formatDic = {"labelA": "Re", "labelB": "Im"}
            funcA = npy.real
            funcB = npy.imag
        elif form == "DB":
            formatDic = {"labelA": "dB", "labelB": "ang"}
            funcA = mf.complex_2_db
            funcB = mf.complex_2_degree
        elif form == "MA":
            formatDic = {"labelA": "mag", "labelB": "ang"}
            funcA = mf.complex_2_magnitude
            funcB = mf.complex_2_degree
        else:
            raise ValueError('`form` must be either `db`,`ma`,`ri`')

        # add formatting to funcA and funcB so we don't have to write it out many many times.
        def c2str_A(c: NumberLike) -> str:
            """Function which takes a complex number for the A part of param and returns an appropriately formatted string"""
            return format_spec_A.format(funcA(c))

        def c2str_B(c: NumberLike) -> str:
            """Function which takes a complex number for B part of param and returns an appropriately formatted string"""
            return format_spec_B.format(funcB(c))

        def get_buffer() -> io.StringIO:
            if return_string is True or type(to_archive) is zipfile.ZipFile:
                from .io.general import StringBuffer  # avoid circular import
                buf = StringBuffer()
            else:
                buf = open(filename, "w")
            return buf

        with get_buffer() as output:
            # Add '!' Touchstone comment delimiters to the start of every line in ntwk.comments
            commented_header = ''
            try:
                if ntwk.comments:
                    for comment_line in ntwk.comments.split('\n'):
                        commented_header += f'!{comment_line}\n'
            except AttributeError:
                pass
            if skrf_comment:
                commented_header += '!Created with skrf (http://scikit-rf.org).\n'

            output.write(commented_header)

            # write header file.
            # the '#'  line is NOT a comment it is essential and it must be
            # exactly this format, to work
            # [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
            if write_z0:
                output.write('!Data is not renormalized\n')
                output.write(f'# {ntwk.frequency.unit} S {form} R\n')
            else:
                # Write "r_ref.real" instead of "r_ref", so we get a real number "a" instead
                # of a complex number "(a+0j)", which is unsupported by the standard Touchstone
                # format (non-HFSS). We already checked in the beginning that "r_ref" must be
                # real in this case (write_z0 == False).
                assert r_ref.imag == 0, "Complex reference impedance is encountered when " \
                                        "generating a standard Touchstone (non-HFSS), this " \
                                        "should never happen in scikit-rf."
                output.write(f'# {ntwk.frequency.unit} S {form} R {r_ref.real} \n')

            # write ports
            try:
                if ntwk.port_names and len(ntwk.port_names) == ntwk.number_of_ports:
                    ports = ''
                    for port_idx, port_name in enumerate(ntwk.port_names):
                        ports += f'! Port[{port_idx+1}] = {port_name}\n'
                    output.write(ports)
            except AttributeError:
                pass

            scaled_freq = ntwk.frequency.f_scaled

            if ntwk.number_of_ports == 1:
                # write comment line for users (optional)
                output.write('!freq {labelA}S11 {labelB}S11\n'.format(**formatDic))
                # write out data
                for f in range(len(ntwk.f)):
                    output.write(format_spec_freq.format(scaled_freq[f]) + ' ' \
                                 + c2str_A(ntwk.s[f, 0, 0]) + ' ' \
                                 + c2str_B(ntwk.s[f, 0, 0]) + '\n')
                    # write out the z0 following hfss's convention if desired
                    if write_z0:
                        output.write('! Port Impedance ')
                        for n in range(ntwk.number_of_ports):
                            output.write(f'{ntwk.z0[f, n].real:.14f} {ntwk.z0[f, n].imag:.14f} ')
                        output.write('\n')

            elif ntwk.number_of_ports == 2:
                # 2-port is a special case with
                # - single line, and
                # - S21,S12 in reverse order: legacy ?

                # write comment line for users (optional)
                output.write(
                    '!freq {labelA}S11 {labelB}S11 {labelA}S21 {labelB}S21 {labelA}S12 {labelB}S12 {labelA}S22 {labelB}S22\n'.format(
                        **formatDic))
                # write out data
                for f in range(len(ntwk.f)):
                    output.write(format_spec_freq.format(scaled_freq[f]) + ' ' \
                                 + c2str_A(ntwk.s[f, 0, 0]) + ' ' \
                                 + c2str_B(ntwk.s[f, 0, 0]) + ' ' \
                                 + c2str_A(ntwk.s[f, 1, 0]) + ' ' \
                                 + c2str_B(ntwk.s[f, 1, 0]) + ' ' \
                                 + c2str_A(ntwk.s[f, 0, 1]) + ' ' \
                                 + c2str_B(ntwk.s[f, 0, 1]) + ' ' \
                                 + c2str_A(ntwk.s[f, 1, 1]) + ' ' \
                                 + c2str_B(ntwk.s[f, 1, 1]) + '\n')
                    # write out the z0 following hfss's convention if desired
                    if write_z0:
                        output.write('! Port Impedance')
                        for n in range(2):
                            output.write(f' {ntwk.z0[f, n].real:.14f} {ntwk.z0[f, n].imag:.14f}')
                        output.write('\n')

            elif ntwk.number_of_ports == 3:
                # 3-port is written over 3 lines / matrix order

                # write comment line for users (optional)
                output.write('!freq')
                for m in range(1, 4):
                    for n in range(1, 4):
                        output.write(" {labelA}S{m}{n} {labelB}S{m}{n}".format(m=m, n=n, **formatDic))
                    output.write('\n!')
                output.write('\n')
                # write out data
                for f in range(len(ntwk.f)):
                    output.write(format_spec_freq.format(scaled_freq[f]))
                    for m in range(3):
                        for n in range(3):
                            output.write(' ' + c2str_A(ntwk.s[f, m, n]) + ' ' \
                                         + c2str_B(ntwk.s[f, m, n]))
                        output.write('\n')
                    # write out the z0 following hfss's convention if desired
                    if write_z0:
                        output.write('! Port Impedance')
                        for n in range(3):
                            output.write(f' {ntwk.z0[f, n].real:.14f} {ntwk.z0[f, n].imag:.14f}')
                        output.write('\n')

            elif ntwk.number_of_ports >= 4:
                # general n-port
                # - matrix is written line by line
                # - 4 complex numbers / 8 real numbers max. for a single line
                # - continuation lines (anything except first) go with indent
                #   this is not part of the spec, but many tools handle it this way
                #   -> allows to parse without knowledge of number of ports

                # write comment line for users (optional)
                output.write('!freq')
                for m in range(1, 1 + ntwk.number_of_ports):
                    for n in range(1, 1 + ntwk.number_of_ports):
                        if (n > 0 and (n % 4) == 0):
                            output.write('\n!')
                        output.write(" {labelA}S{m}{n} {labelB}S{m}{n}".format(m=m, n=n, **formatDic))
                    output.write('\n!')
                output.write('\n')
                # write out data
                for f in range(len(ntwk.f)):
                    output.write(format_spec_freq.format(scaled_freq[f]))
                    for m in range(ntwk.number_of_ports):
                        for n in range(ntwk.number_of_ports):
                            if (n > 0 and (n % 4) == 0):
                                output.write('\n')
                            output.write(' ' + c2str_A(ntwk.s[f, m, n]) + ' ' \
                                         + c2str_B(ntwk.s[f, m, n]))
                        output.write('\n')

                    # write out the z0 following hfss's convention if desired
                    if write_z0:
                        output.write('! Port Impedance')
                        for n in range(ntwk.number_of_ports):
                            output.write(f' {ntwk.z0[f, n].real:.14f} {ntwk.z0[f, n].imag:.14f}')
                        output.write('\n')

            if type(to_archive) is zipfile.ZipFile:
                to_archive.writestr(filename, output.getvalue())
            elif return_string is True:
                return output.getvalue()

    def write(self, file: str = None, *args, **kwargs) -> None:
        r"""
        Write the Network to disk using the :mod:`pickle` module.

        The resultant file can be read either by using the Networks
        constructor, :func:`__init__` , the read method :func:`read`, or
        the general read function :func:`skrf.io.general.read`


        Parameters
        ----------
        file : str or file-object
            filename or a file-object. If left as None then the
            filename will be set to Network.name, if its not None.
            If both are None, ValueError is raised.
        \*args, \*\*kwargs :
            passed through to :func:`~skrf.io.general.write`


        Note
        ----
        If the self.name is not None and file is  can left as None
        and the resultant file will have the `.ntwk` extension appended
        to the filename.


        Examples
        --------
        >>> n = rf.N(f=[1,2,3],s=[1,1,1],z0=50, name = 'open')
        >>> n.write()
        >>> n2 = rf.read('open.ntwk')

        See Also
        --------
        skrf.io.general.write : write any skrf object
        skrf.io.general.read : read any skrf object
        """
        # this import is delayed until here because of a circular dependency
        from .io.general import write

        if file is None:
            if self.name is None:
                raise (ValueError('No filename given. You must provide a filename, or set the name attribute'))
            file = self.name

        write(file, self, *args, **kwargs)

    def read(self, *args, **kwargs) -> None:
        r"""
        Read a Network from a 'ntwk' file

        A ntwk file is written with :func:`write`. It is just a pickled
        file.

        Parameters
        ----------
        \*args, \*\*kwargs : args and kwargs
            passed to :func:`skrf.io.general.read`


        Note
        ----
        This function calls :func:`skrf.io.general.read`.


        Examples
        --------
        >>> rf.read('myfile.ntwk')
        >>> rf.read('myfile.p')

        See Also
        --------
        skrf.io.general.read
        write
        skrf.io.general.write
        """
        from .io.general import read
        self.copy_from(read(*args, **kwargs))

    def write_spreadsheet(self, *args, **kwargs) -> None:
        """
        Write contents of network to a spreadsheet, for your boss to use.

        See Also
        --------
        skrf.io.general.network_2_spreadsheet
        """
        from .io.general import network_2_spreadsheet
        network_2_spreadsheet(self, *args, **kwargs)

    def to_dataframe(self, *args, **kwargs) -> 'pd.DataFrame':
        """
        Convert attributes of a Network to a pandas DataFrame

        Use the same parameters than :func:`skrf.io.general.network_2_dataframe`

        Parameters
        ----------
        attrs : list of string
            Network attributes to convert, like ['s_db','s_deg']
        ports : list of tuples
            list of port pairs to write. defaults to ntwk.port_tuples
            (like [[0,0]])

        Returns
        -------
        df : `pandas.DataFrame`


        See Also
        ---------
        skrf.io.general.network_2_dataframe
        """
        from .io.general import network_2_dataframe
        return network_2_dataframe(self, *args, **kwargs)

    def write_to_json_string(self) -> str:
        """
        Serialize and convert network to a JSON string.

        This is ~3x faster than writing to and reading back from touchstone
        for a 4port 20,000 point device.

        Returns
        -------
        jsonstr : string
            JSON string

        See Also
        --------
        skrf.io.general.to_json_string
        """
        from .io.general import to_json_string
        return to_json_string(self)



    # interpolation
    def interpolate(self, freq_or_n: Union[Frequency, NumberLike], basis: str = 's',
                    coords: str = 'cart', f_kwargs: dict = {}, return_array: bool = False,
                    **kwargs) -> Union['Network', npy.ndarray]:
        r"""
        Interpolate a Network along frequency axis

        The input 'freq_or_n` can be either a new
        :class:`~skrf.frequency.Frequency` or an `int`, or a new
        frequency vector (in Hz).

        This interpolates  a given `basis`, ie s, z, y, etc, in the
        coordinate system defined by `coord` like polar or cartesian.
        Different interpolation types ('linear', 'quadratic') can be used
        by passing appropriate `\*\*kwargs`. This function returns an
        interpolated Network. Alternatively :func:`~Network.interpolate_self`
        will interpolate self.

        Parameters
        ----------
        freq_or_n : :class:`~skrf.frequency.Frequency` or int or list-like
            The new frequency over which to interpolate. this arg may be
            one of the following:

            * a new :class:`~skrf.frequency.Frequency` object

            * an int: the current frequency span is resampled linearly.

            * a list-like: create a new frequency using :meth:`~skrf.frequency.Frequency.from_f`

        basis : ['s','z','y','a'], etc
            The network parameter to interpolate
        coords : string
            Coordinate system to use for interpolation: 'cart' or 'polar':
            'cart' is cartesian is Re/Im. 'polar' is unwrapped phase/mag
        return_array: bool
            return the interpolated array instead of re-assigning it to
            a given attribute
        **kwargs : keyword arguments
            passed to :func:`scipy.interpolate.interp1d` initializer.
            `kind` controls interpolation type.

            `kind` = `rational` uses interpolation by rational polynomials.

            `d` kwarg controls the degree of rational polynomials
            when `kind`=`rational`. Defaults to 4.

        Returns
        -------
        result : :class:`Network`
            an interpolated Network, or array

        Note
        ----
        The interpolation coordinate system (`coords`)  makes  a big
        difference for large amounts of interpolation. polar works well
        for duts with slowly changing magnitude. try them all.

        See  :func:`scipy.interpolate.interpolate.interp1d` for useful
        kwargs. For example:

        kind : string or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or
            as an integer specifying the order of the spline
            interpolator to use.


        See Also
        --------
        resample
        interpolate_self
        interpolate_from_f

        Examples
        --------
        .. ipython::

            @suppress
            In [21]: import skrf as rf

            In [21]: n = rf.data.ring_slot

            In [21]: n

            In [21]: new_freq = rf.Frequency(75,110,501,'ghz')

            In [21]: n.interpolate(new_freq, kind = 'cubic')

        """
        # make new network and fill with interpolated values
        result = self.copy()

        if kwargs.get('kind', None) == 'rational':
            f_interp = mf.rational_interp
            #Not supported by rational_interp
            del kwargs['kind']
        else:
            f_interp = interp1d

        # interpret input
        if isinstance(freq_or_n, Frequency):
            # input is a frequency object
            new_frequency = freq_or_n

        else:
            dim = len(shape(freq_or_n))
            if dim == 0:
                # input is a number 
                new_frequency = Frequency(start=self.frequency.start_scaled, 
                                          stop=self.frequency.stop_scaled,
                                          unit=self.frequency.unit,
                                          npoints=freq_or_n)
            elif dim == 1:
                # input is a array, or list
                new_frequency = Frequency.from_f(freq_or_n, **f_kwargs)

        # set new frequency and pull some variables
        result.frequency = new_frequency
        f = self.frequency.f
        f_new = new_frequency.f

        # interpolate z0  ( this must happen first, because its needed
        # to compute the basis transform below (like y2s), if basis!='s')
        if npy.all(self.z0 == self.z0[0]):
            # If z0 is constant we don't need to interpolate it
            z0_shape = list(self.z0.shape)
            z0_shape[0] = len(f_new)
            result._z0 = npy.ones(z0_shape) * self.z0[0]
        else:
            result._z0 = f_interp(f, self.z0, axis=0, **kwargs)(f_new)

        # interpolate  parameter for a given basis
        x = self.__getattribute__(basis)
        if coords == 'cart':
            x_new = f_interp(f, x, axis=0, **kwargs)(f_new)
        elif coords == 'polar':
            rad = npy.unwrap(npy.angle(x), axis=0)
            mag = npy.abs(x)
            interp_rad = f_interp(f, rad, axis=0, **kwargs)
            interp_mag = f_interp(f, mag, axis=0, **kwargs)
            x_new = interp_mag(f_new) * npy.exp(1j * interp_rad(f_new))
        else:
            raise ValueError(f'Unknown coords {coords}')

        # interpolate noise data too
        if self.noisy:
          f_noise = self.noise_freq.f
          f_noise_new = new_frequency.f
          noise_new = f_interp(f_noise, self.noise, axis=0, **kwargs)(f_noise_new)

        if return_array:
            return x_new
        else:
            result.__setattr__(basis, x_new)
            if self.noisy:
              result.noise = noise_new
              result.noise_freq = new_frequency
        return result

    def interpolate_self(self, freq_or_n: Union[Frequency, NumberLike], **kwargs) -> None:
        """
        Interpolate the current Network along frequency axis (inplace).

        The input 'freq_or_n` can be either a new
        :class:`~skrf.frequency.Frequency` or an `int`, or a new
        frequency vector (in Hz).

        See :func:`~Network.interpolate` for more information.

        Parameters
        ----------
        freq_or_n : :class:`~skrf.frequency.Frequency` or int or list-like
            The new frequency over which to interpolate. this arg may be
            one of the following:

            * a new :class:`~skrf.frequency.Frequency` object

            * an int: the current frequency span is resampled linearly.

            * a list-like: create a new frequency using :meth:`~skrf.frequency.Frequency.from_f`

        **kwargs : keyword arguments
                passed to :func:`scipy.interpolate.interp1d` initializer.

        Returns
        -------
        None
            The interpolation is performed inplace.

        See Also
        --------
        resample
        interpolate
        interpolate_from_f
        """
        ntwk = self.interpolate(freq_or_n, **kwargs)
        self.frequency, self.s, self.z0 = ntwk.frequency, ntwk.s, ntwk.z0
        if self.noisy:
          self.noise, self.noise_freq = ntwk.noise, ntwk.noise_freq

    ##convenience
    resample = interpolate_self

    def extrapolate_to_dc(self, points: int = None, dc_sparam: NumberLike = None,
                          kind: str = 'cubic', coords: str = 'cart',
                          **kwargs) -> 'Network':
        """
        Extrapolate S-parameters down to 0 Hz and interpolate to uniform spacing.

        If frequency vector needs to be interpolated aliasing will occur in
        time-domain. For the best results first frequency point should be a
        multiple of the frequency step so that points from DC to
        the first measured point can be added without interpolating rest of the
        frequency points.

        Parameters
        ----------
        points : int or None
            Number of frequency points to be used in interpolation.
            If None number of points is calculated based on the frequency step size
            and spacing between 0 Hz and first measured frequency point.
        dc_sparam : class:`npy.ndarray` or None
            NxN S-parameters matrix at 0 Hz.
            If None S-parameters at 0 Hz are determined by linear extrapolation.
        kind : str or int, default is 'cubic'
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or
            as an integer specifying the order of the spline
            interpolator to use for `scipy.interp1d`.

            `kind` = 'rational' uses interpolation by rational polynomials.

            `d` kwarg controls the degree of rational polynomials
            when `kind` is 'rational'. Defaults to 4.
        coords : str
            Coordinate system to use for interpolation: 'cart' or 'polar'.
            'cart' is cartesian is Re/Im, 'polar' is unwrapped phase/mag.
            Passed to :func:`Network.interpolate`

        Returns
        -------
        result : :class:`Network`
            Extrapolated Network

        See Also
        --------
        interpolate
        impulse_response
        step_response
        """
        result = self.copy()

        if self.frequency.f[0] == 0:
            return result

        if points is None:
            fstep = self.frequency.f[1] - self.frequency.f[0]
            points = len(self) + int(round(self.frequency.f[0]/fstep))
        if dc_sparam is None:
            #Interpolate DC point alone first using linear interpolation, because
            #interp1d can't extrapolate with other methods.
            #TODO: Option to enforce passivity
            x = result.s[:2]
            f = result.frequency.f[:2]
            rad = npy.unwrap(npy.angle(x), axis=0)
            mag = npy.abs(x)
            interp_rad = interp1d(f, rad, axis=0, fill_value='extrapolate')
            interp_mag = interp1d(f, mag, axis=0, fill_value='extrapolate')
            dc_sparam = interp_mag(0) * npy.exp(1j * interp_rad(0))
        else:
            #Make numpy array if argument was list
            dc_sparam = npy.array(dc_sparam)

        result.s = npy.insert(result.s, 0, dc_sparam, axis=0)
        result.frequency._f = npy.insert(result.frequency.f, 0, 0)
        result.z0 = npy.insert(result.z0, 0, result.z0[0], axis=0)

        if result.noisy:
            result.noise = npy.insert(result.noise, 0, 0, axis=0)
            result.noise_freq._f = npy.insert(result.noise_freq.f, 0, 0)

        new_f = Frequency(0, result.frequency.f_scaled[-1], points,
                unit=result.frequency.unit)
        #None of the default interpolation methods are too good
        #and cause aliasing in the time domain.
        #Best results are obtained when no interpolation is needed,
        #e.g. first frequency point is a multiple of frequency step.
        result.interpolate_self(new_f, kind=kind, coords=coords, **kwargs)
        #DC value must have zero imaginary part
        result.s[0,:,:] = result.s[0,:,:].real
        return result


    def subnetwork(self, ports: Sequence[int], offby: int = 1) -> 'Network':
        """
        Returns a subnetwork of a the Network from a list of port numbers.

        A subnetwork is Network which S-parameters corresponds to selected ports,
        with all non-selected ports considered matched.

        The resulting subNetwork is given a new Network.name property
        from the initial name and adding the kept ports indices
        (ex: 'device' -> 'device13'). Such name should make easier the use
        of functions such as n_twoports_2_nport.

        Parameters
        ----------
        ports : list of int
            List of ports to keep in the resultant Network.
            Indices are the Python indices (starts at 0)
        offby : int
            starting value for s-parameters indexes in the sub-Network name parameter.
            A value of `1`, assumes that a s21 = ntwk.s[:,1,0]. Default is 1.

        Returns
        -------
        subntw : :class:`Network` object
            Resulting subnetwork of the Network from the given ports

        See also
        --------
        subnetwork, n_twoports_2_nport

        """
        return subnetwork(self, ports, offby)

    def crop(self, f_start: float, f_stop: float, unit: str = None) -> None:
        """
        Crop Network based on start and stop frequencies.

        No interpolation is done.

        Parameters
        ----------
        f_start : number
            start frequency of crop range, in units of self.frequency.unit.
            If `f_start` is lower than the lowest frequency, no change to the network is made by the lower bound.
        f_stop : number
            stop frequency of crop range, in units of self.frequency.unit
            If `f_stop` is higher than the highest frequency, no change to the network is made by the higher bound.
        unit : string
            Units that `f_start` and `f_stop` are described in. This must be a string recognized by the Frequency
            class, e.g. 'Hz','MHz', etc. A value of `None` assumes units are same as `self`

        See Also
        --------
        cropped

        """
        if f_start is None:
            f_start = -npy.inf
        if f_stop is None:
            f_stop = npy.inf

        if f_stop<f_start:
            raise ValueError(f"`f_stop` was {f_stop}, which was smaller than `f_start`, which was {f_start}")

        if unit is not None: # if `unit` is specified, we must retranslate the frequency units
            scaleFactor = Frequency.multiplier_dict[unit.lower()]/self.frequency.multiplier# make a multiplier to put f_start and f_stop in the right units, e.g. 'GHz' -> 'MHz'
            f_start *=scaleFactor
            f_stop *=scaleFactor

        if f_start > self.frequency.f_scaled.max():
            raise ValueError(f"`f_start` was {f_start}, which was larger than the largest frequency in this Network object, which was {self.frequency.f_scaled.max()}")
        if f_stop < self.frequency.f_scaled.min():
            raise ValueError(f"`f_stop` was {f_stop}, which was smaller than the smallest frequency in this Network object, which was {self.frequency.f_scaled.min()}")

        start_idx,stop_idx = 0,self.frequency.npoints-1 # start with entire frequency range selected

        if f_start > self.frequency.f_scaled.min():
            start_idx = find_nearest_index(self.frequency.f_scaled, f_start)
            if f_start > self.frequency.f_scaled[start_idx]: # we do not want the start index to be at a frequency lower than `f_start`
                start_idx += 1
        if f_stop < self.frequency.f_scaled.max():
            stop_idx = find_nearest_index(self.frequency.f_scaled, f_stop)
            if f_stop < self.frequency.f_scaled[stop_idx]: # we don't want the stop index to be at a frequency higher than `f_stop`
                stop_idx -=1

        if stop_idx < start_idx :
            raise ValueError("Stop index/frequency lower than start: stop_idx: {}, start_idx: {}, self.frequency.f[stop_idx]: {}, self.frequency.f[start_idx]: {}"\
                                .format(stop_idx,start_idx,self.frequency.f[stop_idx],self.frequency.f[start_idx]  ))
        ntwk = self[start_idx:stop_idx + 1]
        self.frequency, self.s, self.z0 = ntwk.frequency, ntwk.s, ntwk.z0

    def cropped(self, f_start: float, f_stop: float, unit: str = None) -> 'Network':
        """
        Returns a cropped network, leaves self alone.

        Parameters
        ----------
        f_start : number
            start frequency of crop range, in units of self.frequency.unit.
            If `f_start` is lower than the lowest frequency, no change to the network is made by the lower bound.
        f_stop : number
            stop frequency of crop range, in units of self.frequency.unit
            If `f_stop` is higher than the highest frequency, no change to the network is made by the higher bound.
        unit : string
            Units that `f_start` and `f_stop` are described in. This must be a string recognized by the Frequency
            class, e.g. 'Hz','MHz', etc. A value of `None` assumes units are same as `self`

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting cropped network

        See Also
        --------
        crop

        """
        out = self.copy()
        out.crop(f_start=f_start, f_stop=f_stop,unit=unit)
        return out

    def flip(self) -> None:
        """
        Swaps the ports of a 2n-port Network (inplace).

        In case the network is 2n-port and n > 1, 'second' numbering scheme is
        assumed to be consistent with the ** cascade operator::

                 +--------+                 +--------+
               0-|0      n|-n             0-|n      0|-n
               1-|1    n+1|-n+1    flip   1-|n+1    1|-n+1
                ...      ...       =>       ...      ...
             n-1-|n-1 2n-1|-2n-1        n-1-|2n-1 n-1|-2n-1
                 +--------+                 +--------+

        See Also
        --------
        flipped
        renumber
        renumbered

        """
        if self.number_of_ports % 2 == 0:
            n = int(self.number_of_ports / 2)
            old = list(range(0, 2*n))
            new = list(range(n, 2*n)) + list(range(0, n))
            self.renumber(old, new)
        else:
            raise ValueError('you can only flip two-port Networks')

    def flipped(self) -> 'Network':
        """
        Returns a flipped network, leaves self alone.

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting flipped Network

        See Also
        --------
        flip
        renumber
        renumbered

        """
        out = self.copy()
        out.flip()
        return out

    def renormalize(self, z_new: NumberLike, s_def: str = None) -> None:
        """
        Renormalize s-parameter matrix given a new port impedances.

        Parameters
        ----------
        z_new : complex array of shape FxN, F, N or a  scalar
            new port impedances

        s_def : str -> s_def :  can be: None, 'power', 'pseudo' or 'traveling'
            None to use the definition set in the network `s_def` attribute.
            Scattering parameter definition : 'power' for power-waves definition,
            'pseudo' for pseudo-waves definition.
            'traveling' corresponds to the initial implementation.
            Default is 'power'.
            NB: results are the same for real-valued characteristic impedances.

        See Also
        --------
        renormalize_s
        fix_z0_shape
        """
        # cast any array like type (tuple, list) to a npy.array
        z_new = npy.array(z_new, dtype=complex)
        # make sure the z_new shape can be compared with self.z0
        z_new = fix_z0_shape(z_new, self.frequency.npoints, self.nports)
        if s_def is None:
            s_def = self.s_def
        # Try to avoid renormalization if possible since it goes through
        # Z-parameters which can cause numerical inaccuracies.
        # We need to renormalize if z_new is different from z0
        # or s_def is different and there is at least one complex port.
        need_to_renorm = False
        if npy.any(self.z0 != z_new):
            need_to_renorm = True
        if s_def != self.s_def and (self.z0.imag != 0).any():
            need_to_renorm = True
        if need_to_renorm:
            # We can use s2s if z0 is the same. This is numerically much more
            # accurate.
            if (self.z0 == z_new).all():
                self.s = s2s(self.s, self.z0, s_def, self.s_def)
            else:
                self.s = renormalize_s(self.s, self.z0, z_new, s_def, self.s_def)
        # Update s_def if it was changed
        self.s_def = s_def
        self.z0 = z_new

    def renumber(self, from_ports: Sequence[int], to_ports: Sequence[int]) -> None:
        """
        Renumbers ports of a Network (inplace).

        Parameters
        ----------
        from_ports : list-like
            List of port indices to change. Size between 1 and N_ports.
        to_ports : list-like
            List of desired port indices. Size between 1 and N_ports.

        NB : from_ports and to_ports must have same size.

        Returns
        -------
            None
            The reorganization of the Network's port is performed inplace.


        Examples
        --------

        In the following example, the ports of a 3-ports Network are
        reorganized. Dummy reference impedances are set only to follow more
        easily the renumbering.

        >>> f = rf.Frequency(1, 1, 1)
        >>> s = np.ones((1, 3, 3))
        >>> z0 = [10, 20, 30]
        >>> ntw = rf.Network(frequency=f, s=s, z0=z0)  # our OEM Network

        In picture, we have::

            Order in          Original Order
            skrf.Network
                        ┌───────────────────┐
                        │       OEM         │
                        │                   │
            0   ────────┤  A  (10 Ω)        │
                        │                   │
                        │                   │
            1   ────────┤  B  (20 Ω)        │
                        │                   │
                        │                   │
            2   ────────┤  C  (30 Ω)        │
                        │                   │
                        └───────────────────┘

        While after renumbering

        >>> ntw.renumber([0,1,2], [1, 2, 0])


        we now have::

            Order in          Original Order
            skrf.Network
                        ┌───────────────────┐
                        │       OEM         │
                        │                   │
            1   ────────┤  A  (10 Ω)        │
                        │                   │
                        │                   │
            2   ────────┤  B  (20 Ω)        │
                        │                   │
                        │                   │
            0   ────────┤  C  (30 Ω)        │
                        │                   │
                        └───────────────────┘

        **Other examples:**


        To flip the ports of a 2-port network 'foo':

        >>> foo.renumber( [0,1], [1,0] )

        To rotate the ports of a 3-port network 'bar' so that port 0 becomes port 1:

        >>> bar.renumber( [0,1,2], [1,2,0] )

        To swap the first and last ports of a N-port (N>=2) Network 'duck':

        >>> duck.renumber( [0,-1], [-1,0] )

        See Also
        --------
        renumbered
        flip
        flipped

        """
        from_ports = npy.array(from_ports)
        to_ports = npy.array(to_ports)
        if len(npy.unique(from_ports)) != len(from_ports):
            raise ValueError('an index can appear at most once in from_ports or to_ports')
        if any(npy.unique(from_ports) != npy.unique(to_ports)):
            raise ValueError('from_ports and to_ports must have the same set of indices')

        self.s[:, to_ports, :] = self.s[:, from_ports, :]  # renumber rows
        self.s[:, :, to_ports] = self.s[:, :, from_ports]  # renumber columns
        self.z0[:, to_ports] = self.z0[:, from_ports]

    def renumbered(self, from_ports: Sequence[int], to_ports: Sequence[int]) -> 'Network':
        """
        Returns a renumbered Network, leave self alone.

        Parameters
        ----------
        from_ports : list-like
            List of port indices to change. Size between 1 and N_ports.
        to_ports : list-like
            List of desired port indices. Size between 1 and N_ports.

        NB: from_ports and to_ports must have same size.

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting renumbered Network

        See Also
        --------
        renumber
        flip
        flipped

        """
        out = self.copy()
        out.renumber(from_ports, to_ports)
        return out

    def rotate(self, theta: NumberLike, unit: str = 'deg') -> None:
        """
        Rotate S-parameters
        """
        if unit == 'deg':
            theta = mf.degree_2_radian(theta )

        self.s = self.s * npy.exp(-1j*theta)

    def delay(self, d: float, unit: str = 'deg', port: int = 0, media: Any = None, **kw) -> 'Network':
        """
        Add phase delay to a given port.

        This will connect a transmission line of length `d/2` to the selected `port`. If no propagation properties are
        specified for the line (`media=None`), then freespace is assumed to convert a distance `d` into an electrical
        length. If a phase angle is specified for `d`, it will be evaluated at the center frequency of the network.

        Parameters
        ----------
        d : float
            The angle/length/delay of the transmission line (see `unit` argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
            The units of d.  See :func:`Media.to_meters`, for details
        port : int
            Port to add the delay to.
        media: skrf.media.Media
            Media object to use for generating the delay. If None, this will
            default to freespace.

        Returns
        -------
        ntwk : :class:`Network` object
            A delayed copy of the `Network`.

        """
        if d ==0:
            return self
        d=d/2.
        if media is None:
            from .media import Freespace
            media = Freespace(frequency=self.frequency,z0=self.z0[:,port])

        l =media.line(d=d, unit=unit,**kw)
        return connect(self, port, l, 0)

    def windowed(self, window: Union[str, float, Tuple[str, float]]=('kaiser', 6),
            normalize: bool = True, center_to_dc: bool = None) -> 'Network':
        """
        Return a windowed version of s-matrix. Used in time-domain analysis.

        When using time domain through :attr:`s_time_db`,
        or similar properties, the spectrum is usually windowed,
        before the IFFT is taken. This is done to
        compensate for the band-pass nature of a spectrum [#]_ .

        This function calls :func:`scipy.signal.get_window` which gives
        more details about the windowing.

        Parameters
        ----------
        window : string, float, or tuple
            The type of window to create. See :func:`scipy.signal.get_window`
            for details.
        normalize : bool
            Normalize the window to preserve power. ie
            sum(ntwk.s,axis=0) == sum(ntwk.windowed().s,axis=0)
        center_to_dc : bool or None
            If True only the positive half of the window is applied to the signal.
            This should be used if frequency vector begins from DC or from "close enough" to DC.
            If False full window is used which also attenuates low frequencies.
            If None then value is determined automatically based on if frequency
            vector begins from 0.

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting windowed Network

        Examples
        --------
        >>> ntwk = rf.Network('myfile.s2p')
        >>> ntwk_w = ntwk.windowed()
        >>> ntwk_w.plot_s_time_db()

        References
        ----------
        .. [#] Agilent Time Domain Analysis Using a Network Analyzer Application Note 1287-12

        """

        if center_to_dc is None:
            center_to_dc = self.frequency.f[0] == 0

        if center_to_dc:
            window = signal.get_window(window, 2*len(self))[len(self):]
        else:
            window = signal.get_window(window, len(self))

        window = window.reshape(-1, 1, 1) * npy.ones((len(self),
                                                      self.nports,
                                                      self.nports))
        windowed = self * window
        if normalize:
            # normalize the s-parameters to account for power lost in windowing
            windowed.s = windowed.s * npy.sum(self.s_mag, axis=0) / \
                         npy.sum(windowed.s_mag, axis=0)

        return windowed

    def time_gate(self, *args, **kw) -> 'Network':
        """
        Time gate this Network

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting time-gated Network

        see `skrf.time_domain.time_gate`
        """
        return time_gate(self, *args, **kw)


    # noise
    def add_noise_polar(self, mag_dev: float, phase_dev: float, **kwargs) -> None:
        """
        adds a complex zero-mean gaussian white-noise.

        adds a complex zero-mean gaussian white-noise of a given
        standard deviation for magnitude and phase

        Parameters
        ----------
        mag_dev : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase [in degrees]

        """

        phase_rv = stats.norm(loc=0, scale=phase_dev).rvs(size=self.s.shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size=self.s.shape)

        phase = (self.s_deg + phase_rv)
        mag = self.s_mag + mag_rv
        self.s = mag * npy.exp(1j * npy.pi / 180. * phase)

    def add_noise_polar_flatband(self, mag_dev: float, phase_dev: float, **kwargs) -> None:
        """
        adds a flatband complex zero-mean gaussian white-noise signal of
        given standard deviations for magnitude and phase

        Parameters
        ----------
        mag_dev : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase [in degrees]

        """
        phase_rv = stats.norm(loc=0, scale=phase_dev).rvs(size=self.s[0].shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size=self.s[0].shape)

        phase = (self.s_deg + phase_rv)
        mag = self.s_mag + mag_rv
        self.s = mag * npy.exp(1j * npy.pi / 180. * phase)

    def multiply_noise(self, mag_dev: float, phase_dev: float, **kwargs) -> None:
        """
        multiplies a complex bivariate gaussian white-noise signal
        of given standard deviations for magnitude and phase.
        magnitude mean is 1, phase mean is 0

        Parameters
        ----------
        mag_dev: float
                standard deviation of magnitude
        phase_dev: float
            standard deviation of phase [in degrees]


        """
        phase_rv = stats.norm(loc=0, scale=phase_dev).rvs( \
            size=self.s.shape)
        mag_rv = stats.norm(loc=1, scale=mag_dev).rvs( \
            size=self.s.shape)
        self.s = mag_rv * npy.exp(1j * npy.pi / 180. * phase_rv) * self.s

    def nudge(self, amount: float = 1e-12) -> 'Network':
        """
        Perturb s-parameters by small amount.

        This is useful to work-around numerical bugs.

        Parameters
        ----------
        amount : float
            amount to add to s parameters

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting renumbered Network


        Note
        ----
        This function is::

            self.s = self.s + amount

        """
        self.s = self.s + amount

    # other
    def func_on_parameter(self, func: Callable, attr: str = 's', *args, **kwargs) -> 'Network':
        r"""
        Applies a function parameter matrix, one frequency slice at a time

        This is useful for functions that can only operate on 2d arrays,
        like numpy.linalg.inv. This loops over f and calls
        `func(ntwkA.s[f,:,:], \*args, \*\*kwargs)`

        Parameters
        ----------
        func : func
            function to apply to s-parameters, on a single-frequency slice.
            (ie `func(ntwkA.s[0,:,:], \*args, \*\*kwargs)`
        attr: string
            Name of the parameter to operate on. Ex: 's', 'z', etc.
            Default is 's'.

        \*args, \*\*kwargs :
            passed to the func

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting renumbered Network

        Examples
        --------
        >>> from numpy.linalg import inv
        >>> ntwk.func_on_parameter(inv)
        """
        ntwkB = self.copy()
        p = self.__getattribute__(attr)
        ntwkB.s = npy.r_[[func(p[k, :, :], *args, **kwargs) \
                          for k in range(len(p))]]
        return ntwkB

    def nonreciprocity(self, m: int, n: int, normalize: bool = False) -> 'Network':
        r"""
        Normalized non-reciprocity metric.

        This is a port-by-port measure of how non-reciprocal a n-port
        network is. It is defined by,

        .. math::

            (S_{mn} - S_{nm}) / \sqrt ( S_{mn} S_{nm} )

        Parameters
        ----------
        m : int
            m index
        n : int
            n index
        normalize : bool
            Normalize the result. Default is False.

        Returns
        -------
        ntwk : :class:`Network` object
            Resulting renumbered Network

        """
        forward = self.__getattribute__('s%i%i' % (m, n))
        reverse = self.__getattribute__('s%i%i' % (n, m))
        if normalize:
            denom = forward * reverse
            denom.s = npy.sqrt(denom.s)
            return (forward - reverse) / denom
        else:
            return (forward - reverse)

    # generalized mixed mode transformations
    def se2gmm(self, p: int, z0_mm: npy.ndarray = None, s_def : str = None) -> None:
        """
        Transform network from single ended parameters to generalized mixed mode parameters [#]_

        Parameters
        ----------
        p : int
            number of differential ports
        z0_mm : Numpy array
            `f x 2*p x 2*p` matrix of mixed mode impedances, optional.
            If input is None, 2 * z0 Ohms differential and z0 / 2 Ohms common mode
            reference impedance is used, where z0 is average of the differential
            pair ports reference impedance.
            Single-ended ports not converted to differential mode keep their z0.
        s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
            Scattering parameter definition :
            None to use the definition set in the network `s_def` attribute.
            'power' for power-waves definition [#Kurokawa]_,
            'pseudo' for pseudo-waves definition [#Marks]_.
            All the definitions give the same result if z0 is real valued.

        Note Odd Number of Ports

            In the case where there are an odd number of ports (such as a 3-port network
            with ports 0, 1, and 2), se2gmm() assumes that the last port (port 2) remains
            single-ended and ports 0 and 1 are converted to differential mode and common
            mode, respectively. For networks in which the port ordering is not suitable,
            port renumbering can be used.

        Examples
        --------
        For example, a 3-port single-ended network is converted to mixed-mode
        parameters::

            | Port 0 (single-ended, 50 ohms) --> Port 0 (single-ended, 50 ohms)
            | Port 1 (single-ended, 50 ohms) --> Port 1 (differential mode, 100 ohms)
            | Port 2 (single-ended, 50 ohms) --> Port 2 (common mode, 25 ohms)

        >>> ntwk.renumber([0,1,2], [2,1,0])
        >>> ntwk.se2gmm(p=1)
        >>> ntwk.renumber([2,1,0], [0,1,2])

        In the resulting network, port 0 is single-ended, port 1 is
        differential mode, and port 2 is common mode.
        
        In following examples, sx is single-mode port x, dy is
        differential-mode port y, and cz is common-mode port z. The low
        insertion loss path of a transmission line is symbolized by ==.
        
        2-Port diagram::
            
              +-----+             +-----+
            0-|s0   |           0-|d0   |
              |     | =se2gmm=>   |     |
            1-|s1   |           1-|c0   |
              +-----+             +-----+
        
        3-Port diagram::
            
              +-----+             +-----+
            0-|s0   |           0-|d0   |
            1-|s1   | =se2gmm=> 1-|c0   |
            2-|s2   |           2-|s2   |
              +-----+             +-----+
              
        Note: The port s2 remain in single-mode.
        
        4-Port diagram::
              
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
              |      |   =se2gmm=>   |      |
            1-|s1==s3|-3           2-|c0==c1|-3
              +------+               +------+
        
        5-Port diagram::
            
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
            1-|s1==s3|-3 =se2gmm=> 2-|c0==c1|-3
              |    s4|-4             |    s4|-4
              +------+               +------+
              
        Note: The port s4 remain in single-mode.
        
        8-Port diagram::
            
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
            1-|s1==s3|-3           2-|d2==d3|-3
              |      |   =se2gmm=>   |      |
            4-|s4==s6|-6           4-|c0==c1|-5
            5-|s5==s7|-7           6-|c2==c3|-7
              +------+               +------+
        
        2N-Port diagram::
            
                 A                                  B
                 +------------+                     +-----------+ 
               0-|s0========s2|-2                 0-|d0=======d1|-1
               1-|s1========s3|-3                 2-|d2=======d3|-3
                ...          ...     =se2gmm=>     ...         ...
            2N-4-|s2N-4==s2N-2|-2N-2           2N-4-|cN-4===cN-3|-2N-3
            2N-3-|s2N-3==s2N-1|-2N-1           2N-2-|cN-2===cN-1|-2N-1
                 +------------+                     +-----------+
                 
        Note: The network `A` is not cascadable with the `**` operator
        along transmission path.

        References
        ----------
        .. [#] Ferrero and Pirola; Generalized Mixed-Mode S-Parameters; IEEE Transactions on
            Microwave Theory and Techniques; Vol. 54; No. 1; Jan 2006
        .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.

        See Also
        --------
        gmm2se

        """
        if 2*p > self.nports or p < 0:
            raise ValueError('Invalid number of differential ports')
        if s_def is None:
            s_def = self.s_def
        if s_def != self.s_def:
            # Need to first renormalize to the new s_def if we have complex ports
            self.renormalize(self.z0, s_def)
            self.s_def = s_def
        # Assumes 'proper' order (first differential ports, then single ended ports)
        if z0_mm is None:
            z0_mm = self.z0.copy()
            z0_avg = 0.5*(z0_mm[:, 0:2*p:2] + z0_mm[:, 1:2*p:2])
            z0_mm[:, 0:p] = 2*z0_avg  # differential mode impedance
            z0_mm[:, p:2 * p] = 0.5*z0_avg  # common mode impedance
        else:
            # Make sure shape is correct
            # Only set mixed mode ports
            _z0_mm = self.z0.copy()
            shape = [self.z0.shape[0], 2 * p]
            z0_p = npy.broadcast_to(z0_mm, shape)
            _z0_mm[:,:2*p] = z0_p
            z0_mm = _z0_mm
        Xi_tilde_11, Xi_tilde_12, Xi_tilde_21, Xi_tilde_22 = self._Xi_tilde(p, self.z0, z0_mm, s_def)
        A = Xi_tilde_21 + npy.einsum('...ij,...jk->...ik', Xi_tilde_22, self.s)
        B = Xi_tilde_11 + npy.einsum('...ij,...jk->...ik', Xi_tilde_12, self.s)
        self.s = mf.rsolve(B, A)
        self.z0 = z0_mm

    def gmm2se(self, p: int, z0_se: NumberLike = None, s_def : str = None) -> None:
        """
        Transform network from generalized mixed mode parameters [#]_ to single ended parameters

        Parameters
        ----------

        p : int
            number of differential ports
        z0_se: Numpy array
            `f x 2*p x 2*p` matrix of single ended impedances, optional
            if input is None, extract the reference impedance from the differential network
            calculated as 0.5 * (0.5 * z0_diff + 2 * z0_comm) for each differential port.
            Single-ended ports not converted to differential mode keep their z0.
        s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
            Scattering parameter definition:
            None to use the definition set in the network `s_def` attribute.
            'power' for power-waves definition [#Kurokawa]_,
            'pseudo' for pseudo-waves definition [#Marks]_.
            All the definitions give the same result if z0 is real valued.
            
        Examples
        --------
        
        In following examples, sx is single-mode port x, dy is
        differential-mode port y, and cz is common-mode port z. The low
        insertion loss path of a transmission line is symbolized by ==.
        
        2-Port diagram::
            
              +-----+             +-----+
            0-|s0   |           0-|d0   |
              |     | <=gmm2se=   |     |
            1-|s1   |           1-|c0   |
              +-----+             +-----+
        
        3-Port diagram::
            
              +-----+             +-----+
            0-|s0   |           0-|d0   |
            1-|s1   | <=gmm2se= 1-|c0   |
            2-|s2   |           2-|s2   |
              +-----+             +-----+
              
        Note: The port s2 remain in single-mode.
        
        4-Port diagram::
              
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
              |      |   <=gmm2se=   |      |
            1-|s1==s3|-3           2-|c0==c1|-3
              +------+               +------+
        
        5-Port diagram::
            
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
            1-|s1==s3|-3 <=gmm2se= 2-|c0==c1|-3
              |    s4|-4             |    s4|-4
              +------+               +------+
              
        Note: The port s4 remain in single-mode.
        
        8-Port diagram::
            
              +------+               +------+
            0-|s0==s2|-2           0-|d0==d1|-1
            1-|s1==s3|-3           2-|d2==d3|-3
              |      |   <=gmm2se=   |      |
            4-|s4==s6|-6           4-|c0==c1|-5
            5-|s5==s7|-7           6-|c2==c3|-7
              +------+               +------+
        
        2N-Port diagram::
            
                 A                                  B
                 +------------+                     +-----------+ 
               0-|s0========s2|-2                 0-|d0=======d1|-1
               1-|s1========s3|-3                 2-|d2=======d3|-3
                ...          ...    <=gmm2se=     ...         ...
            2N-4-|s2N-4==s2N-2|-2N-2           2N-4-|cN-4===cN-3|-2N-3
            2N-3-|s2N-3==s2N-1|-2N-1           2N-2-|cN-2===cN-1|-2N-1
                 +------------+                     +-----------+
                 
        Note: The network `A` is not cascadable with the `**` operator
        along transmission path.

        References
        ----------
        .. [#] Ferrero and Pirola; Generalized Mixed-Mode S-Parameters; IEEE Transactions on
            Microwave Theory and Techniques; Vol. 54; No. 1; Jan 2006
        .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.

        See Also
        --------
        se2gmm

        """
        if 2*p > self.nports or p < 0:
            raise ValueError('Invalid number of differential ports')
        if s_def is None:
            s_def = self.s_def
        if s_def != self.s_def:
            # Need to first renormalize to the new s_def if we have complex ports
            self.renormalize(self.z0, s_def)
            self.s_def = s_def
        # Assumes 'proper' order (differential ports, single ended ports)
        if z0_se is None:
            z0_se = self.z0.copy()
            z0_avg = 0.5*(0.5*z0_se[:, 0:p] + 2*z0_se[:, p:2*p])
            z0_se[:, 0:p] = z0_avg
            z0_se[:, p:2 * p] = z0_avg
        else:
            # Make sure shape is correct
            # Only set mixed mode ports
            _z0_se = self.z0.copy()
            shape = [self.z0.shape[0], 2 * p]
            z0_p = npy.broadcast_to(z0_se, shape)
            _z0_se[:,:2*p] = z0_p
            z0_se = _z0_se
        Xi_tilde_11, Xi_tilde_12, Xi_tilde_21, Xi_tilde_22 = self._Xi_tilde(p, z0_se, self.z0, s_def)
        A = Xi_tilde_22 - npy.einsum('...ij,...jk->...ik', self.s, Xi_tilde_12)
        # Note that B sign is incorrect in the paper. Inverted B here gives the
        # correct result.
        B = -Xi_tilde_21 + npy.einsum('...ij,...jk->...ik', self.s, Xi_tilde_11)
        self.s = npy.linalg.solve(A, B)  # (35)
        self.z0 = z0_se

    # generalized mixed mode supplement functions
    _T = npy.array([[1, 0, -1, 0], [0, 0.5, 0, -0.5], [0.5, 0, 0.5, 0], [0, 1, 0, 1]])  # (5)

    def _m(self, z0: npy.ndarray, s_def : str) -> npy.ndarray:
        if s_def == 'pseudo':
            scaling = npy.sqrt(z0.real) / (2 * npy.abs(z0))
            Z = npy.ones((z0.shape[0], 2, 2), dtype=npy.complex128)
            Z[:, 0, 1] = z0
            Z[:, 1, 1] = -z0
            return scaling[:, npy.newaxis, npy.newaxis] * Z
        elif s_def == 'power':
            scaling = 1 / (2 * npy.sqrt(z0.real))
            Z = npy.ones((z0.shape[0], 2, 2), dtype=npy.complex128)
            Z[:, 0, 1] = z0
            Z[:, 1, 1] = -z0.conj()
            return scaling[:, npy.newaxis, npy.newaxis] * Z
        elif s_def == 'traveling':
            Z = npy.ones((z0.shape[0], 2, 2), dtype=npy.complex128)
            sqrtz0 = npy.sqrt(z0)
            Z[:, 0, 0] = 1 / sqrtz0
            Z[:, 0, 1] = sqrtz0
            Z[:, 1, 0] = 1 / sqrtz0
            Z[:, 1, 1] = -sqrtz0
            return 0.5 * Z
        else:
            raise ValueError('Unknown s_def')

    def _M(self, j: int, k: int, z0_se: npy.ndarray, s_def : str) -> npy.ndarray:  # (14)
        M = npy.zeros((self.f.shape[0], 4, 4), dtype=npy.complex128)
        M[:, :2, :2] = self._m(z0_se[:, j], s_def)
        M[:, 2:, 2:] = self._m(z0_se[:, k], s_def)
        return M

    def _M_circle(self, l: int, p: int, z0_mm: npy.ndarray, s_def : str) -> npy.ndarray:  # (12)
        M = npy.zeros((self.f.shape[0], 4, 4), dtype=npy.complex128)
        M[:, :2, :2] = self._m(z0_mm[:, l], s_def)  # differential mode impedance of port pair
        M[:, 2:, 2:] = self._m(z0_mm[:, p + l], s_def)  # common mode impedance of port pair
        return M

    def _X(self, j: int, k: int , l: int, p: int, z0_se: npy.ndarray, z0_mm: npy.ndarray, s_def : str) -> npy.ndarray:  # (15)
        return npy.einsum('...ij,...jk->...ik', self._M_circle(l, p, z0_mm, s_def).dot(self._T),
                          npy.linalg.inv(self._M(j, k, z0_se, s_def)))  # matrix multiplication elementwise for each frequency

    def _P(self, p: int) -> npy.ndarray:  # (27) (28)
        n = self.nports

        Pda = npy.zeros((p, 2 * n), dtype=bool)
        Pdb = npy.zeros((p, 2 * n), dtype=bool)
        Pca = npy.zeros((p, 2 * n), dtype=bool)
        Pcb = npy.zeros((p, 2 * n), dtype=bool)
        Pa = npy.zeros((n - 2 * p, 2 * n), dtype=bool)
        Pb = npy.zeros((n - 2 * p, 2 * n), dtype=bool)
        for l in npy.arange(p):
            Pda[l, 4 * (l + 1) - 3 - 1] = True
            Pca[l, 4 * (l + 1) - 1 - 1] = True
            Pdb[l, 4 * (l + 1) - 2 - 1] = True
            Pcb[l, 4 * (l + 1) - 1] = True
        for l in npy.arange(n - 2 * p):
            Pa[l, 4 * p + 2 * (l + 1) - 1 - 1] = True
            Pb[l, 4 * p + 2 * (l + 1) - 1] = True
        return npy.concatenate((Pda, Pca, Pa, Pdb, Pcb, Pb))

    def _Q(self) -> npy.ndarray:  # (29) error corrected
        n = self.nports

        Qa = npy.zeros((n, 2 * n), dtype=bool)
        Qb = npy.zeros((n, 2 * n), dtype=bool)
        for l in npy.arange(n):
            Qa[l, 2 * (l + 1) - 1 - 1] = True
            Qb[l, 2 * (l + 1) - 1] = True
        return npy.concatenate((Qa, Qb))

    def _Xi(self, p: int, z0_se: npy.ndarray, z0_mm: npy.ndarray, s_def : str) -> npy.ndarray:  # (24)
        n = self.nports
        Xi = npy.ones(self.f.shape[0])[:, npy.newaxis, npy.newaxis] * npy.eye(2 * n, dtype=npy.complex128)
        for l in npy.arange(p):
            Xi[:, 4 * l:4 * l + 4, 4 * l:4 * l + 4] = self._X(l * 2, l * 2 + 1, l, p, z0_se, z0_mm, s_def)
        return Xi

    def _Xi_tilde(self, p: int, z0_se: npy.ndarray, z0_mm: npy.ndarray, s_def : str) -> npy.ndarray:  # (31)
        n = self.nports
        P = npy.ones(self.f.shape[0])[:, npy.newaxis, npy.newaxis] * self._P(p)
        QT = npy.ones(self.f.shape[0])[:, npy.newaxis, npy.newaxis] * self._Q().T
        Xi = self._Xi(p, z0_se, z0_mm, s_def)
        Xi_tilde = npy.einsum('...ij,...jk->...ik', npy.einsum('...ij,...jk->...ik', P, Xi), QT)
        return Xi_tilde[:, :n, :n], Xi_tilde[:, :n, n:], Xi_tilde[:, n:, :n], Xi_tilde[:, n:, n:]

    def impulse_response(self, window: str = 'hamming', n: int = None, pad: int = 1000,
                        bandpass: bool = None, squeeze: bool = True) -> Tuple[npy.ndarray, npy.ndarray]:
        """Calculates time-domain impulse response of one-port.

        First frequency must be 0 Hz for the transformation to be accurate and
        the frequency step must be uniform. Positions of the reflections are
        accurate even if the frequency doesn't begin from 0, but shapes will
        be distorted.

        Real measurements should be extrapolated to DC and interpolated to
        uniform frequency spacing.

        Y-axis is the reflection coefficient.

        Parameters
        ----------
        window : string
                FFT windowing function.
        n : int
                Length of impulse response output.
                If n is not specified, 2 * (m - 1) points are used,
                where m = len(self) + pad
        pad : int
                Number of zeros to add as padding for FFT.
                Adding more zeros improves accuracy of peaks.
        bandpass : bool or None
                If False window function is center on 0 Hz.
                If True full window is used and low frequencies are attenuated.
                If None value is determined automatically based on if the
                frequency vector begins from 0.
        squeeze: bool
                Squeeze impulse response to one dimension,
                if a oneport gets transformed.
                Has no effect when transforming a multiport.
                Default = True

        Returns
        -------
        t : class:`npy.ndarray`
            Time vector
        y : class:`npy.ndarray`
            Impulse response

        See Also
        --------
            step_response
            extrapolate_to_dc
        """
        if n is None:
            # Use zero-padding specification. Note that this does not allow n odd.
            n = 2 * (self.frequency.npoints + pad - 1)

        fstep = self.frequency.step
        if n % 2 == 0:
            t = npy.fft.ifftshift(npy.fft.fftfreq(n, fstep))
        else:
            t = npy.fft.ifftshift(npy.fft.fftfreq(n+1, fstep))[1:]
        if bandpass in (True, False):
            center_to_dc = not bandpass
        else:
            center_to_dc = None
        if window != None:
            w = self.windowed(window=window, normalize=False, center_to_dc=center_to_dc)
        else:
            w = self

        ir = mf.irfft(w.s, n=n)
        if squeeze:
            ir = ir.squeeze()

        return t, ir

    def step_response(self, window: str = 'hamming', n: int = None, pad: int = 1000, squeeze: bool = True) -> Tuple[npy.ndarray, npy.ndarray]:
        """Calculates time-domain step response of one-port.

        First frequency must be 0 Hz for the transformation to be accurate and
        the frequency step must be uniform.

        Real measurements should be extrapolated to DC and interpolated to
        uniform frequency spacing.

        Y-axis is the reflection coefficient.

        Parameters
        ----------
        window : string
                FFT windowing function.
        n : int
                Length of step response output.
                If n is not specified, 2 * (m - 1) points are used,
                where m = len(self) + pad
        pad : int
                Number of zeros to add as padding for FFT.
                Adding more zeros improves accuracy of peaks.
        squeeze: bool
                Squeeze step response to one dimension,
                if a oneport gets transformed.
                Has no effect when transforming a multiport.
                Default = True

        Returns
        -------
        t : class:`npy.ndarray`
            Time vector
        y : class:`npy.ndarray`
            Step response

        Raises
        ------
        ValueError
            If used with an Network with more than one port

        NotImplementedError
            If used with non equidistant sampled frequency vector

        See Also
        --------
            impulse_response
            extrapolate_to_dc
        """
        if self.frequency.sweep_type != 'lin':
            raise NotImplementedError(
                'Unable to transform non equidistant sampled points to time domain')

        if self.frequency.f[0] != 0:
            warnings.warn(
                "Frequency doesn't begin from 0. Step response will not be correct.",
                RuntimeWarning
            )

        t, y = self.impulse_response(window=window, n=n, pad=pad, bandpass=False, squeeze=squeeze)
        return t, npy.cumsum(y, axis=0)

    # Network Active s/z/y/vswr parameters
    def s_active(self, a: npy.ndarray) -> npy.ndarray:
        r"""
        Returns the active s-parameters of the network for a defined wave excitation a.

        The active s-parameter at a port is the reflection coefficients
        when other ports are excited. It is an important quantity for active
        phased array antennas.

        Active s-parameters are defined by [#]_:

        .. math::

           \mathrm{active(s)}_{m} = \sum_{i=1}^N s_{mi}\frac{a_i}{a_m}

        where :math:`s` are the scattering parameters and :math:`N` the number of ports

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude (pseudowave formulation [#]_)

        Returns
        -------
        s_act : complex array of shape (n_freqs, n_ports)
            active S-parameters for the excitation a

        See Also
        --------
            z_active : active Z-parameters
            y_active : active Y-parameters
            vswr_active : active VSWR

        References
        ----------
        .. [#] D. M. Pozar, IEEE Trans. Antennas Propag. 42, 1176 (1994).

        .. [#] D. Williams, IEEE Microw. Mag. 14, 38 (2013).

        """
        return s2s_active(self.s, a)

    def z_active(self, a: npy.ndarray) -> npy.ndarray:
        r"""
        Returns the active Z-parameters of the network for a defined wave excitation a.

        The active Z-parameters are defined by:

        .. math::

           \mathrm{active}(z)_{m} = z_{0,m} \frac{1 + \mathrm{active}(s)_m}{1 - \mathrm{active}(s)_m}

        where :math:`z_{0,m}` is the characteristic impedance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        z_act : complex array of shape (nfreqs, nports)
            active Z-parameters for the excitation a

        See Also
        --------
            s_active : active S-parameters
            y_active : active Y-parameters
            vswr_active : active VSWR
        """
        return s2z_active(self.s, self.z0, a)

    def y_active(self, a: npy.ndarray) -> npy.ndarray:
        r"""
        Returns the active Y-parameters of the network for a defined wave excitation a.

        The active Y-parameters are defined by:

        .. math::

           \mathrm{active}(y)_{m} = y_{0,m} \frac{1 - \mathrm{active}(s)_m}{1 + \mathrm{active}(s)_m}

        where :math:`y_{0,m}` is the characteristic admittance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        y_act : complex array of shape (nfreqs, nports)
            active Y-parameters for the excitation a

        See Also
        --------
            s_active : active S-parameters
            z_active : active Z-parameters
            vswr_active : active VSWR
        """
        return s2y_active(self.s, self.z0, a)

    def vswr_active(self, a: npy.ndarray) -> npy.ndarray:
        r"""
        Returns the active VSWR of the network for a defined wave excitation a.

        The active VSWR is defined by :

        .. math::

           \mathrm{active}(vswr)_{m} = \frac{1 + |\mathrm{active}(s)_m|}{1 - |\mathrm{active}(s)_m|}

        where :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        vswr_act : complex array of shape (nfreqs, nports)
            active VSWR for the excitation a

        See Also
        --------
            s_active : active S-parameters
            z_active : active Z-parameters
            y_active : active Y-parameters
        """
        return s2vswr_active(self.s, a)

COMPONENT_FUNC_DICT = Network.COMPONENT_FUNC_DICT
PRIMARY_PROPERTIES = Network.PRIMARY_PROPERTIES
Y_LABEL_DICT = Network.Y_LABEL_DICT

## Functions operating on Network[s]
def connect(ntwkA: Network, k: int, ntwkB: Network, l: int, num: int = 1) -> Network:
    """
    Connect two n-port networks together.

    Connect ports `k` thru `k+num-1` on `ntwkA` to ports
    `l` thru `l+num-1` on `ntwkB`. The resultant network has
    (ntwkA.nports+ntwkB.nports-2*num) ports. The port indices ('k','l')
    start from 0. Port impedances **are** taken into account.
    When the two networks have overlapping frequencies, the resulting
    network will contain only the overlapping frequencies.


    Note
    ----
    The effect of mis-matched port impedances is handled by inserting
    a 2-port 'mismatch' network between the two connected ports.
    This mismatch Network is calculated with the
    :func:`impedance_mismatch` function.


    Parameters
    ----------
    ntwkA : :class:`Network`
            network 'A'
    k : int
            starting port index on `ntwkA` ( port indices start from 0 )
    ntwkB : :class:`Network`
            network 'B'
    l : int
            starting port index on `ntwkB`
    num : int
            number of consecutive ports to connect (default 1)


    Returns
    -------
    ntwkC : :class:`Network`
            new network of rank (ntwkA.nports + ntwkB.nports - 2*num)


    See Also
    --------
    connect_s : actual  S-parameter connection algorithm.
    innerconnect_s : actual S-parameter connection algorithm.


    Examples
    --------
    To implement a *cascade* of two networks

    >>> ntwkA = rf.Network('ntwkA.s2p')
    >>> ntwkB = rf.Network('ntwkB.s2p')
    >>> ntwkC = rf.connect(ntwkA, 1, ntwkB,0)

    """
    # some checking
    try:
        check_frequency_equal(ntwkA, ntwkB)
    except IndexError as e:
        common_freq = npy.intersect1d(ntwkA.frequency.f, ntwkB.frequency.f, return_indices=True)
        if common_freq[0].size == 0:
            raise e
        else:
            ntwkA = ntwkA[common_freq[1]]
            ntwkB = ntwkB[common_freq[2]]
            warnings.warn("Using a frequency subset:\n" + str(ntwkA.frequency))

    if (k + num - 1 > ntwkA.nports - 1):
        raise IndexError('Port `k` out of range')
    if (l + num - 1 > ntwkB.nports - 1):
        raise IndexError('Port `l` out of range')

    have_complex_ports = (ntwkA.z0.imag != 0).any() or (ntwkB.z0.imag != 0).any()

    # If definitions aren't identical and there are complex ports renormalize first
    # Output will have ntwkA s_def if they are different.
    if ntwkA.s_def != ntwkB.s_def and have_complex_ports:
        warnings.warn('Connecting two networks with different s_def and complex ports. '
                'The resulting network will have s_def of the first network: ' + ntwkA.s_def + '. '\
                'To silence this warning explicitly convert the networks to same s_def '
                'using `renormalize` function.')
        ntwkB = ntwkB.copy()
        ntwkB.renormalize(ntwkB.z0, ntwkA.s_def)

    s_def_original = ntwkA.s_def
    if ntwkA.s_def == 'power' and have_complex_ports:
        # When port impedance is complex, power-waves are discontinuous across
        # a junction between two identical transmission lines while traveling
        # waves and pseudo-waves are continuous. The connection algorithm relies
        # on continuity and 'power' networks must be first converted to either
        # of the other definition.
        ntwkA = ntwkA.copy()
        ntwkA.renormalize(ntwkA.z0, 'pseudo')
        ntwkB = ntwkB.copy()
        ntwkB.renormalize(ntwkB.z0, 'pseudo')

    s_def = ntwkA.s_def
    if s_def == 'power':
        # Ports are real. Save some time by not copying the networks
        # and use the definition that works with real ports
        s_def = 'traveling'

    # create output Network, from copy of input
    ntwkC = ntwkA.copy()

    # if networks' z0's are not identical, then connect a impedance
    # mismatch, which takes into account the effect of differing port
    # impedances.
    # import pdb;pdb.set_trace()
    if assert_z0_at_ports_equal(ntwkA, k, ntwkB, l) == False:
        ntwkC.s = connect_s(
            ntwkA.s, k,
            impedance_mismatch(ntwkA.z0[:, k], ntwkB.z0[:, l], s_def), 0)
        # the connect_s() put the mismatch's output port at the end of
        #   ntwkC's ports.  Fix the new port's impedance, then insert it
        #   at position k where it belongs.
        ntwkC.z0[:, k:] = npy.hstack((ntwkC.z0[:, k + 1:], ntwkB.z0[:, [l]]))
        ntwkC.renumber(from_ports=[ntwkC.nports - 1] + list(range(k, ntwkC.nports - 1)),
                       to_ports=list(range(k, ntwkC.nports)))

    # call s-matrix connection function
    ntwkC.s = connect_s(ntwkC.s, k, ntwkB.s, l)

    # combine z0 arrays and remove ports which were `connected`
    ntwkC.z0 = npy.hstack(
        (npy.delete(ntwkA.z0, range(k, k + 1), 1), npy.delete(ntwkB.z0, range(l, l + 1), 1)))

    # if we're connecting more than one port, call innerconnect recursively
    # until all connections are made to finish the job
    if num > 1:
        ntwkC = innerconnect(ntwkC, k, ntwkA.nports - 1 + l, num - 1)

    # if ntwkB is a 2port, then keep port indices where you expect.
    if ntwkB.nports == 2 and ntwkA.nports > 2 and num == 1:
        from_ports = list(range(ntwkC.nports))
        to_ports = list(range(ntwkC.nports))
        to_ports.pop(k);
        to_ports.append(k)

        ntwkC.renumber(from_ports=from_ports,
                       to_ports=to_ports)

    # if ntwkA and ntwkB are both 2port, and either one has noise, calculate ntwkC's noise
    either_are_noisy = False
    try:
      either_are_noisy = ntwkA.noisy or ntwkB.noisy
    except:
      pass

    if num == 1 and ntwkA.nports == 2 and ntwkB.nports == 2 and either_are_noisy:
      if ntwkA.noise_freq is not None and ntwkB.noise_freq is not None and ntwkA.noise_freq != ntwkB.noise_freq:
          raise IndexError('Networks must have same noise frequency. See `Network.interpolate`')
      cA = ntwkA.noise
      cB = ntwkB.noise

      noise_freq = ntwkA.noise_freq
      if noise_freq is None:
        noise_freq = ntwkB.noise_freq

      if cA is None:
        cA = npy.broadcast_arrays(npy.array([[0., 0.], [0., 0.]]), ntwkB.noise)[0]
      if cB is None:
        cB = npy.broadcast_arrays(npy.array([[0., 0.], [0., 0.]]), ntwkA.noise)[0]

      if k == 0:
        # if we're connecting to the "input" port of ntwkA, recalculate the equivalent noise of ntwkA,
        # since we're modeling the noise as a pair of sources at the "input" port
        # TODO
        raise (NotImplementedError)
      if l == 1:
        # if we're connecting to the "output" port of ntwkB, recalculate the equivalent noise,
        # since we're modeling the noise as a pair of sources at the "input" port
        # TODO
        raise (NotImplementedError)

      # interpolate abcd into the set of noise frequencies


      if ntwkA.deembed :
          if ntwkA.frequency.f.size > 1 :
              a_real = interp1d(ntwkA.frequency.f, ntwkA.inv.a.real,
                      axis=0, kind=Network.noise_interp_kind)
              a_imag = interp1d(ntwkA.frequency.f, ntwkA.inv.a.imag,
                      axis=0, kind=Network.noise_interp_kind)
              a = a_real(noise_freq.f) + 1.j * a_imag(noise_freq.f)
          else :
              a_real = ntwkA.inv.a.real
              a_imag = ntwkA.inv.a.imag
              a = a_real + 1.j * a_imag

          a = npy_inv(a)
          a_H = npy.conj(a.transpose(0, 2, 1))
          cC = npy.matmul(a, npy.matmul(cB -cA, a_H))
      else :
          if ntwkA.frequency.f.size > 1 :
              a_real = interp1d(ntwkA.frequency.f, ntwkA.a.real,
                      axis=0, kind=Network.noise_interp_kind)
              a_imag = interp1d(ntwkA.frequency.f, ntwkA.a.imag,
                      axis=0, kind=Network.noise_interp_kind)
              a = a_real(noise_freq.f) + 1.j * a_imag(noise_freq.f)
          else :
              a_real = ntwkA.a.real
              a_imag = ntwkA.a.imag
              a = a_real + 1.j * a_imag

          a_H = npy.conj(a.transpose(0, 2, 1))
          cC = npy.matmul(a, npy.matmul(cB, a_H)) + cA
      ntwkC.noise = cC
      ntwkC.noise_freq = noise_freq

    if ntwkC.s_def != s_def_original:
        ntwkC.renormalize(ntwkC.z0, s_def_original)

    return ntwkC


def connect_fast(ntwkA: Network, k: int, ntwkB: Network, l: int) -> Network:
    """
    Alias for connect

    Parameters
    ----------
    ntwkA : :class:`Network`
            network 'A'
    k : int
            starting port index on `ntwkA` ( port indices start from 0 )
    ntwkB : :class:`Network`
            network 'B'
    l : int
            starting port index on `ntwkB`


    Returns
    -------
    ntwkC : :class:`Network`
            new network of rank `(ntwkA.nports + ntwkB.nports - 2)`
    """
    warnings.warn("connect_fast is deprecated. Use connect.", DeprecationWarning, stacklevel=2)
    return connect(ntwkA, k, ntwkB, l)


def innerconnect(ntwkA: Network, k: int, l: int, num: int = 1) -> Network:
    """
    connect ports of a single n-port network.

    this results in a (n-2)-port network. remember port indices start
    from 0.


    Note
    ----
    A 2-port 'mismatch' network is inserted between the connected ports
    if their impedances are not equal.


    Parameters
    ----------
    ntwkA : :class:`Network`
        network 'A'
    k,l : int
        starting port indices on ntwkA ( port indices start from 0 )
    num : int
        number of consecutive ports to connect

    Returns
    -------
    ntwkC : :class:`Network`
        new network of rank (ntwkA.nports - 2*num)

    See Also
    --------
    connect_s : actual  S-parameter connection algorithm.
    innerconnect_s : actual S-parameter connection algorithm.


    Examples
    --------
    To connect ports '0' and port '1' on ntwkA

    >>> ntwkA = rf.Network('ntwkA.s3p')
    >>> ntwkC = rf.innerconnect(ntwkA, 0,1)

    """

    if (k + num - 1 > ntwkA.nports - 1):
        raise IndexError('Port `k` out of range')
    if (l + num - 1 > ntwkA.nports - 1):
        raise IndexError('Port `l` out of range')

    # create output Network, from copy of input
    ntwkC = ntwkA.copy()

    s_def_original = ntwkC.s_def

    # 'power' is not supported, convert to supported definition and back afterwards
    if ntwkC.s_def == 'power':
        ntwkC.renormalize(ntwkC.z0, 'pseudo')

    if not (ntwkC.z0[:, k] == ntwkC.z0[:, l]).all():
        # connect a impedance mismatch, which will takes into account the
        # effect of differing port impedances
        mismatch = impedance_mismatch(ntwkC.z0[:, k], ntwkC.z0[:, l], ntwkC.s_def)
        ntwkC.s = connect_s(ntwkC.s, k, mismatch, 0)
        # the connect_s() put the mismatch's output port at the end of
        #   ntwkC's ports.  Fix the new port's impedance, then insert it
        #   at position k where it belongs.
        ntwkC.z0[:, k:] = npy.hstack((ntwkC.z0[:, k + 1:], ntwkC.z0[:, [l]]))
        ntwkC.renumber(from_ports=[ntwkC.nports - 1] + list(range(k, ntwkC.nports - 1)),
                       to_ports=list(range(k, ntwkC.nports)))

    # call s-matrix connection function
    ntwkC.s = innerconnect_s(ntwkC.s, k, l)

    # update the characteristic impedance matrix
    ntwkC.z0 = npy.delete(ntwkC.z0, list(range(k, k + 1)) + list(range(l, l + 1)), 1)

    # recur if we're connecting more than one port
    if num > 1:
        ntwkC = innerconnect(ntwkC, k, l - 1, num - 1)

    if ntwkC.s_def != s_def_original:
        ntwkC.renormalize(ntwkC.z0, s_def_original)

    return ntwkC


def cascade(ntwkA: Network, ntwkB: Network) -> Network:
    """
    Cascade two 2, 2N-ports Networks together.

    Connects ports N through 2N-1  on `ntwkA` to ports 0 through N of
    `ntwkB`. This calls `connect()`, which is a more general function.
    Use `Network.renumber` to change port order if needed.

    Note
    ----
    connection diagram::

              A                B
           +---------+   +---------+
          -|0      N |---|0      N |-
          -|1     N+1|---|1     N+1|-
          ...       ... ...       ...
          -|N-2  2N-2|---|N-2  2N-2|-
          -|N-1  2N-1|---|N-1  2N-1|-
           +---------+   +---------+

    Parameters
    ----------
    ntwkA : :class:`Network`
        network `ntwkA`
    ntwkB : :class:`Network`
        network `ntwkB`

    Returns
    -------
    C : :class:`Network`
        the resultant network of ntwkA cascaded with ntwkB

    See Also
    --------
    connect : connects two Networks together at arbitrary ports.
    Network.renumber : changes the port order of a network

    """

    if ntwkA.nports<2:
            raise ValueError('nports must be >1')


    N = int(ntwkA.nports/2 )
    if ntwkB.nports == 1:
        # we are terminating a N-port with a 1-port.
        # which port on self to use is ambiguous. choose N
        return connect(ntwkA, N, ntwkB, 0)

    elif ntwkA.nports % 2 == 0 and ntwkA.nports == ntwkB.nports:
        # we have two 2N-port balanced networks
        return connect(ntwkA, N, ntwkB, 0, num=N)

    elif ntwkA.nports % 2 == 0 and ntwkA.nports == 2 * ntwkB.nports:
        # we have a 2N-port balanced network terminated by a N-port network
        return connect(ntwkA, N, ntwkB, 0, num=N)

    else:
        raise ValueError('I dont know what to do, check port shapes of Networks')



def cascade_list(l: Sequence[Network]) -> Network:
    """
    cascade a list of 2N-port networks

    all networks must have same frequency

    Parameters
    ----------
    l : list-like
        (ordered) list of networks

    Returns
    -------
    out : 2-port Network
        the results of cascading all networks in the list `l`

    """
    return reduce(cascade, l)


def de_embed(ntwkA: Network, ntwkB: Network) -> Network:
    """
    De-embed `ntwkA` from `ntwkB`.

    This calls `ntwkA.inv ** ntwkB`. The syntax of cascading an inverse
    is more explicit, it is recommended that it be used instead of this
    function.

    Parameters
    ----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : :class:`Network`
            network `ntwkB`

    Returns
    -------
    C : Network
            the resultant network of ntwkB de-embedded from ntwkA

    See Also
    --------
    connect : connects two Networks together at arbitrary ports.

    """
    return ntwkA.inv ** ntwkB


def stitch(ntwkA: Network, ntwkB: Network, **kwargs) -> Network:
    r"""
    Stitches ntwkA and ntwkB together.

    Concatenates two networks' data. Given two networks that cover
    different frequency bands this can be used to combine their data
    into a single network.

    Parameters
    ----------
    ntwkA, ntwkB : :class:`Network` objects
        Networks to stitch together

    \*\*kwargs : keyword args
        passed to :class:`Network` constructor, for output network

    Returns
    -------
    ntwkC : :class:`Network`
        result of stitching the networks `ntwkA` and `ntwkB` together

    Examples
    --------
    >>> from skrf.data import wr2p2_line, wr1p5_line
    >>> rf.stitch(wr2p2_line, wr1p5_line)
    2-Port Network: 'wr2p2,line',  330-750 GHz, 402 pts, z0=[ 50.+0.j  50.+0.j]
    """

    A, B = ntwkA, ntwkB
    C = Network(
        frequency=Frequency.from_f(npy.r_[A.f[:], B.f[:]], unit='hz'),
        s=npy.r_[A.s, B.s],
        z0=npy.r_[A.z0, B.z0],
        name=A.name,
        **kwargs
    )
    C.frequency.unit = A.frequency.unit
    return C


def overlap(ntwkA: Network, ntwkB: Network) -> Tuple[Network, Network]:
    """
    Returns the overlapping parts of two Networks, interpolating if needed.

    If frequency vectors for each ntwk don't perfectly overlap, then
    ntwkB is interpolated so that the resultant networks have identical
    frequencies.

    Parameters
    ----------
    ntwkA : :class:`Network`
        a ntwk which overlaps `ntwkB`. (the `dominant` network)
    ntwkB : :class:`Network`
        a ntwk which overlaps `ntwkA`.

    Returns
    -------
    ntwkA_new : :class:`Network`
        part of `ntwkA` that overlapped `ntwkB`
    ntwkB_new : :class:`Network`
        part of `ntwkB` that overlapped `ntwkA`, possibly interpolated


    See Also
    --------

    :func:`skrf.frequency.overlap_freq`

    """

    new_freq = ntwkA.frequency.overlap(ntwkB.frequency)
    return ntwkA.interpolate(new_freq), ntwkB.interpolate(new_freq)


def concat_ports(ntwk_list: Sequence[Network], port_order: str = 'second',
        *args, **kw) -> Network:
    """
    Concatenate networks along the port axis


    Note
    ----
    The `port_order` ='first', means front-to-back, while
    `port_order`='second' means left-to-right. So, for example, when
    concatenating two 2-networks, `A` and `B`, the ports are ordered as follows:

    'first'
        a0 o---o a1  ->   0 o---o 1
        b0 o---o b1  ->   2 o---o 3

    'second'
        a0 o---o a1  ->   0 o---o 2
        b0 o---o b1  ->   1 o---o 3


    use `Network.renumber` to change port ordering.

    Parameters
    ----------
    ntwk_list  : list of skrf.Networks
        ntwks to concatenate
    port_order : ['first', 'second']

    Examples
    --------

    >>>concat([ntwkA,ntwkB])
    >>>concat([ntwkA,ntwkB,ntwkC,ntwkD], port_order='second')

    To put for lines in parallel
    >>> from skrf import air
    >>> l1 = air.line(100, z0=[0,1])
    >>> l2 = air.line(300, z0=[2,3])
    >>> l3 = air.line(400, z0=[4,5])
    >>> l4 = air.line(400, z0=[6,7])
    >>> concat_ports([l1,l2,l3,l4], port_order='second')

    See Also
    --------
    stitch :  concatenate two networks along the frequency axis
    renumber : renumber ports
    """
    # if ntwk list is longer than 2, recursively call myself
    # until we are done
    if len(ntwk_list) > 2:
        f = lambda x, y: concat_ports([x, y], port_order='first')
        out = reduce(f, ntwk_list)
        # if we want to renumber ports, we have to wait
        # until after the recursive calls
        if port_order == 'second':
            N = out.nports
            old_order = list(range(N))
            new_order = list(range(0, N, 2)) + list(range(1, N, 2))
            out.renumber(new_order, old_order)
        return out

    ntwkA, ntwkB = ntwk_list

    if ntwkA.frequency != ntwkB.frequency:
        raise ValueError('ntwks don\'t have matching frequencies')
    A = ntwkA.s
    B = ntwkB.s

    nf = A.shape[0]  # num frequency points
    nA = A.shape[1]  # num ports on A
    nB = B.shape[1]  # num ports on B
    nC = nA + nB  # num ports on C

    # create composite matrix, appending each sub-matrix diagonally
    C = npy.zeros((nf, nC, nC), dtype='complex')
    C[:, :nA, :nA] = A.copy()
    C[:, nA:, nA:] = B.copy()

    ntwkC = ntwkA.copy()
    ntwkC.s = C
    ntwkC.z0 = npy.hstack([ntwkA.z0, ntwkB.z0])
    if port_order == 'second':
        old_order = list(range(nC))
        new_order = list(range(0, nC, 2)) + list(range(1, nC, 2))
        ntwkC.renumber(old_order, new_order)
    return ntwkC


def average(list_of_networks: Sequence[Network], polar: bool = False) -> Network:
    """
    Calculates the average network from a list of Networks.

    This is complex average of the s-parameters for a  list of Networks.


    Parameters
    ----------
    list_of_networks : list of :class:`Network` objects
        the list of networks to average
    polar : boolean, optional
        Average the mag/phase components individually. Default is False.

    Returns
    -------
    ntwk : :class:`Network`
            the resultant averaged Network

    Note
    ----
    This same function can be accomplished with properties of a
    :class:`~skrf.networkset.NetworkSet` class.

    Examples
    --------

    >>> ntwk_list = [rf.Network('myntwk.s1p'), rf.Network('myntwk2.s1p')]
    >>> mean_ntwk = rf.average(ntwk_list)
    """
    out_ntwk = list_of_networks[0].copy()

    if polar:
        # average the mag/phase components individually
        raise NotImplementedError
    else:
        # average the re/im components individually
        for a_ntwk in list_of_networks[1:]:
            out_ntwk += a_ntwk

        out_ntwk.s = out_ntwk.s / (len(list_of_networks))

    return out_ntwk


def stdev(list_of_networks: Sequence[Network], attr: str = 's') -> npy.ndarray:
    """
    Calculates the standard deviation of a network attribute from a list of Networks.

    This is the standard deviation for complex values of the s-parameters and other related attributes
    for a list of Networks.


    Parameters
    ----------
    list_of_networks : list of :class:`Network` objects
        the list of networks to average
    attr : str, optional
        name of attribute to average

    Returns
    -------
    stdev_array : ndarray
    An array of standard deviation values computed after combining the s-parameter values of the given networks.

    Examples
    --------

    >>> ntwk_list = [rf.Network('myntwk.s1p'), rf.Network('myntwk2.s1p')]
    >>> ntwk_stdev = rf.stdev(ntwk_list)
    """
    return npy.array([getattr(network, attr) for network in list_of_networks]).std(axis=0)


def one_port_2_two_port(ntwk: Network) -> Network:
    """
    Calculates the 2-port network given a symmetric, reciprocal and lossless 1-port network.

    Parameters
    ----------
    ntwk : :class:`Network`
        a symmetric, reciprocal and lossless one-port network.

    Returns
    -------
    ntwk : :class:`Network`
        the resultant two-port Network
    """
    result = ntwk.copy()
    result.s = npy.zeros((result.frequency.npoints, 2, 2), dtype=complex)
    s11 = ntwk.s[:, 0, 0]
    result.s[:, 0, 0] = s11
    result.s[:, 1, 1] = s11
    ## HACK: TODO: verify this mathematically
    result.s[:, 0, 1] = npy.sqrt(1 - npy.abs(s11) ** 2) * \
                        npy.exp(1j * (
                        npy.angle(s11) + npy.pi / 2. * (npy.angle(s11) < 0) - npy.pi / 2 * (npy.angle(s11) > 0)))
    result.s[:, 1, 0] = result.s[:, 0, 1]

    result.z0 = npy.hstack([ntwk.z0,ntwk.z0])
    return result


def chopinhalf(ntwk: Network, *args, **kwargs) -> Network:
    r"""
    Chops a sandwich of identical, reciprocal 2-ports in half.

    Given two identical, reciprocal 2-ports measured in series,
    this returns one.


    Note
    ----
    In other words, given

    .. math::

        B = A\cdot A

    Return A, where A port2 is connected to A port1. The result may
    be found through signal flow graph analysis and is,

    .. math::

        a_{11} = \frac{b_{11}}{1+b_{12}}

        a_{22} = \frac{b_{22}}{1+b_{12}}

        a_{12}^2 = b_{21}(1-\frac{b_{11}b_{22}}{(1+b_{12})^2}

    Parameters
    ----------
    ntwk : :class:`Network`
        a 2-port  that is equal to two identical two-ports in cascade


    """
    if ntwk.nports != 2:
        raise ValueError('Only valid on 2ports')

    b11, b22, b12 = ntwk.s11, ntwk.s22, ntwk.s12
    kwargs['name'] = kwargs.get('name', ntwk.name)

    a11 = b11 / (1 + b12)
    a22 = b22 / (1 + b12)
    a21 = b12 * (1 - b11 * b22 / (1 + b12) ** 2)  # this is a21^2 here
    a21.s = mf.sqrt_phase_unwrap(a21.s)
    A = n_oneports_2_nport([a11, a21, a21, a22], *args, **kwargs)

    return A

def evenodd2delta(n: Network, z0: NumberLike = 50, renormalize: bool = True,
        doublehalf: bool = True) -> Network:
    """
    Convert ntwk's s-matrix from even/odd mode into a delta (normal) s-matrix

    This assumes even/odd ports are ordered [1e,1o,2e,2o]

    This is useful for handling coupler sims. Only 4-ports supported for now.

    Parameters
    ----------
    n : skrf.Network
        Network with an even/odd mode s-matrix
    z0: number, list of numbers
        the characteristic impedance to set output networks port impedance
        to , and used to renormalize s-matrix before conversio if
        `renormalize`=True.
    renormalize : Bool
        if impedances are in even/odd then they must be renormalized to
        get correct transformation
    doublehalf: Bool
        convert even/odd impedances to double/half their values. this is
        required if data comes from hfss waveports .

    Returns
    -------
    out: skrf.Network
        same network as `n` but with s-matrix in normal delta basis

    See Also
    --------
    Network.se2gmm, Network.gmm2se

    """

    # move even and odd ports, so we have even and odd
    # s-matrices contiguous
    n_eo = n.copy()
    n_eo.renumber([0,1,2,3],[0,2,1,3])

    if doublehalf:
        n_eo.z0 = n_eo.z0*[2,2,.5,.5]
    # if the n_eo s-matrix is given with e/o z0's we need
    # to renormalize into 50
    if renormalize:
        n_eo.renormalize(z0)

    even = n_eo.s[:,0:2,0:2]
    odd  = n_eo.s[:,2:4,2:4]

    # compute sub-networks for symmetric 4port
    s_a = .5*(even+odd)
    s_b = .5*(even-odd)

    # create output network
    n_delta = n_eo.copy()
    n_delta.s[:,0:2,0:2] = n_delta.s[:,2:4,2:4] = s_a
    n_delta.s[:,2:4,0:2] = n_delta.s[:,0:2,2:4] = s_b
    n_delta.z0=z0

    return n_delta


def subnetwork(ntwk: Network, ports: int, offby:int = 1) -> Network:
    """
    Returns a subnetwork of a given Network from a list of port numbers.

    A subnetwork is Network which S-parameters corresponds to selected ports,
    with all non-selected ports considered matched.

    The resulting subNetwork is given a new Network.name property
    from the initial name and adding the kept ports indices
    (ex: 'device' -> 'device13'). Such name should make easier the use
    of functions such as n_twoports_2_nport.

    Parameters
    ----------
    ntwk : :class:`Network` object
        Network to split into a subnetwork
    ports : list of int
        List of ports to keep in the resultant Network.
        Indices are the Python indices (starts at 0)
    offby : int
        starting value for s-parameters indexes in the sub-Network name parameter.
        A value of `1`, assumes that a s21 = ntwk.s[:,1,0]. Default is 1.

    Returns
    -------
    subntwk : :class:`Network` object
        Resulting subnetwork from the given ports

    See also
    --------
    Network.subnetwork, n_twoports_2_nport

    Examples
    --------

    >>> tee = rf.data.tee  # 3 port Network
    >>> tee12 = rf.subnetwork(tee, [0, 1])  # 2 port Network from ports 1 & 2, port 3 matched
    >>> tee23 = rf.subnetwork(tee, [1, 2])  # 2 port Network from ports 2 & 3, port 1 matched
    >>> tee13 = rf.subnetwork(tee, [0, 2])  # 2 port Network from ports 1 & 3, port 2 matched

    """
    # forging subnetwork name
    subntwk_name = (ntwk.name or 'p') + ''.join([str(index+offby) for index in ports])
    # create a dummy Network with same frequency and z0 from the original
    subntwk = Network(frequency=ntwk.frequency, z0=ntwk.z0[:,ports], name=subntwk_name)
    # keep requested rows and columns of the s-matrix. ports can be not contiguous
    subntwk.s = ntwk.s[npy.ix_(npy.arange(ntwk.s.shape[0]), ports, ports)]
    # keep port_names
    if ntwk.port_names:
        subntwk.port_names = [ntwk.port_names[idx] for idx in ports]
    return subntwk

## Building composit networks from sub-networks
def n_oneports_2_nport(ntwk_list: Sequence[Network], *args, **kwargs) -> Network:
    r"""
    Builds a N-port Network from list of N one-ports

    Parameters
    ----------
    ntwk_list : list of :class:`Network` objects
        must follow left-right, top-bottom order, ie, s11,s12,s21,s22
    \*args, \*\*kwargs :
        passed to :func:`Network.__init__` for the N-port

    Returns
    -------
    nport : n-port :class:`Network`
        result
    """
    nports = int(npy.sqrt(len(ntwk_list)))

    s_out = npy.concatenate(
        [npy.concatenate(
            [ntwk_list[(k + (l * nports))].s for k in range(nports)], 2) \
         for l in range(nports)], 1)

    z0 = npy.concatenate(
        [ntwk_list[k].z0 for k in range(0, nports ** 2, nports + 1)], 1)
    frequency = ntwk_list[0].frequency
    return Network(s=s_out, z0=z0, frequency=frequency, *args, **kwargs)


def n_twoports_2_nport(ntwk_list: Sequence[Network], nports: int,
        offby:int = 1, **kwargs) -> Network:
    r"""
    Builds a N-port Network from list of two-ports

    This  method was made to reconstruct a n-port network from 2-port
    subnetworks as measured by a 2-port VNA. So, for example, given a
    3-port DUT, you  might measure the set p12.s2p, p23.s2p, p13.s2p.
    From these measurements, you can construct p.s3p.

    By default all entries of result.s are filled with 0's, in case  you
    dont fully specify the entire s-matrix of the resultant ntwk.

    Parameters
    ----------
    ntwk_list : list of :class:`Network` objects
        the names must contain the port index, ie 'p12' or 'p43',
        ie. define the Network.name property of the :class:`Network` object.
    offby : int
        starting value for s-parameters indices. ie  a value of `1`,
        assumes that a s21 = ntwk.s[:,1,0]

    \*args, \*\*kwargs :
        passed to :func:`Network.__init__` for the N-port

    Returns
    -------
    nport : n-port :class:`Network`
        result

    See Also
    --------
    concat_ports : concatenate ntwks along their ports
    """

    frequency = ntwk_list[0].frequency
    nport = Network(frequency=frequency,
                    s=npy.zeros(shape=(frequency.npoints, nports, nports)),
                    **kwargs)

    for subntwk in ntwk_list:
        for m, n in nport.port_tuples:
            if m != n and m > n:
                if '%i%i' % (m + offby, n + offby) in subntwk.name:
                    pass
                elif '%i%i' % (n + offby, m + offby) in subntwk.name:
                    subntwk = subntwk.flipped()
                else:
                    continue

                for mn, jk in zip(product((m, n), repeat=2), product((0, 1), repeat=2)):
                    m, n, j, k = mn[0], mn[1], jk[0], jk[1]
                    nport.s[:, m, n] = subntwk.s[:, j, k]
                    nport.z0[:, m] = subntwk.z0[:, j]
    return nport


def four_oneports_2_twoport(s11: Network, s12: Network, s21: Network, s22: Network, *args, **kwargs) -> Network:
    r"""
    Builds a 2-port Network from list of four 1-ports

    Parameters
    ----------
    s11 : one-port :class:`Network`
        s11
    s12 : one-port :class:`Network`
        s12
    s21 : one-port :class:`Network`
        s21
    s22 : one-port :class:`Network`
        s22
    \*args, \*\*kwargs :
        passed to :func:`Network.__init__` for the twoport

    Returns
    -------
    twoport : two-port :class:`Network`
        result

    See Also
    --------
    n_oneports_2_nport
    three_twoports_2_threeport
    """
    return n_oneports_2_nport([s11, s12, s21, s22], *args, **kwargs)


def three_twoports_2_threeport(ntwk_triplet: Sequence[Network], auto_order:bool = True, *args,
                               **kwargs) -> Network:
    r"""
    Creates 3-port from  three 2-port Networks

    This function provides a convenient way to build a 3-port Network
    from a set of 2-port measurements. Which may occur when measuring
    a three port device on a 2-port VNA.

    Note
    ----
    if `auto_order` is False,  ntwk_triplet must be of port orderings:
         [p12, p13, p23]

    else if `auto_order`is True, then the  3 Networks in ntwk_triplet must
    contain port identification in their names.
    For example, their names may be like `me12`, `me13`, `me23`

    Parameters
    ----------
    ntwk_triplet : list of 2-port Network objects
        list of three 2-ports. see notes about order.

    auto_order : bool
        if True attempt to inspect port orderings from Network names.
        Names must be like 'p12', 'p23', etc
    contains : str
        only files containing this string will be loaded.
    \*args,\*\*kwargs :
        passed to :func:`Network.__init__` for resultant network

    Returns
    -------
    threeport : 3-port Network

    See Also
    --------
    n_oneports_2_nport

    Examples
    --------
    >>> rf.three_twoports_2_threeport(rf.read_all('.').values())
    """
    raise DeprecationWarning('Use n_twoports_2_nport instead')
    if auto_order:
        p12, p13, p23 = None, None, None
        s11, s12, s13, s21, s22, s23, s31, s32, s33 = None, None, None, None, None, None, None, None, None

        for k in ntwk_triplet:
            if '12' in k.name:
                p12 = k
            elif '13' in k.name:
                p13 = k
            elif '23' in k.name:
                p23 = k
            elif '21' in k.name:
                p12 = k.flipped()
            elif '31' in k.name:
                p31 = k.flipped()
            elif '32' in k.name:
                p23 = k.flipped()
    else:
        p12, p13, p23 = ntwk_triplet
        p21 = p12.flipped()
        p31 = p13.flipped()
        p32 = p23.flipped()

    if p12 != None:
        s11 = p12.s11
        s12 = p12.s12
        s21 = p12.s21
        s22 = p12.s22

    if p13 != None:
        s11 = p13.s11
        s13 = p13.s12
        s31 = p13.s21
        s33 = p13.s22

    if p23 != None:
        s22 = p23.s11
        s23 = p23.s12
        s32 = p23.s21
        s33 = p23.s22

    ntwk_list = [s11, s12, s13, s21, s22, s23, s31, s32, s33]

    for k in range(len(ntwk_list)):
        if ntwk_list[k] is None:
            frequency = ntwk_triplet[0].frequency
            s = npy.zeros((len(ntwk_triplet[0]), 1, 1))
            ntwk_list[k] = Network(s=s, frequency=frequency)

    threeport = n_oneports_2_nport(ntwk_list, *args, **kwargs)
    return threeport


## Functions operating on s-parameter matrices
def connect_s(A: npy.ndarray, k: int, B: npy.ndarray, l: int) -> npy.ndarray:
    """
    Connect two n-port networks' s-matrices together.

    Specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matrices. The function
    :func:`connect` operates on :class:`Network` types.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
            S-parameter matrix of `A`, shape is fxnxn
    k : int
            port index on `A` (port indices start from 0)
    B : :class:`numpy.ndarray`
            S-parameter matrix of `B`, shape is fxnxn
    l : int
            port index on `B`

    Returns
    -------
    C : :class:`numpy.ndarray`
        new S-parameter matrix


    Note
    ----
    Internally, this function creates a larger composite network
    and calls the  :func:`innerconnect_s` function. see that function for more
    details about the implementation


    See Also
    --------
    connect : operates on :class:`Network` types
    innerconnect_s : function which implements the connection algorithm


    """

    if k > A.shape[-1] - 1 or l > B.shape[-1] - 1:
        raise (ValueError('port indices are out of range'))

    nf = A.shape[0]  # num frequency points
    nA = A.shape[1]  # num ports on A
    nB = B.shape[1]  # num ports on B
    nC = nA + nB  # num ports on C

    # create composite matrix, appending each sub-matrix diagonally
    C = npy.zeros((nf, nC, nC), dtype='complex')
    C[:, :nA, :nA] = A.copy()
    C[:, nA:, nA:] = B.copy()

    # call innerconnect_s() on composit matrix C
    return innerconnect_s(C, k, nA + l)


def innerconnect_s(A: npy.ndarray, k: int, l: int) -> npy.ndarray:
    """
    connect two ports of a single n-port network's s-matrix.

    Specifically, connect port `k`  to port `l` on `A`. This results in
    a (n-2)-port network.  This     function operates on, and returns
    s-matrices. The function :func:`innerconnect` operates on
    :class:`Network` types.


    Note
    ----
    The algorithm used to calculate the resultant network is called a
    'sub-network growth',  can be found in [#]_. The original paper
    describing the  algorithm is given in [#]_.


    Parameters
    ----------
    A : :class:`numpy.ndarray`
        S-parameter matrix of `A`, shape is fxnxn
    k : int
        port index on `A` (port indices start from 0)
    l : int
        port index on `A`

    Returns
    -------
    C : :class:`numpy.ndarray`
            new S-parameter matrix


    References
    ----------
    .. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

    .. [#] Filipsson, Gunnar; , "A New General Computer Algorithm for S-Matrix Calculation of Interconnected Multiports," Microwave Conference, 1981. 11th European , vol., no., pp.700-704, 7-11 Sept. 1981. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4131699&isnumber=4131585


    """

    if k > A.shape[-1] - 1 or l > A.shape[-1] - 1:
        raise (ValueError('port indices are out of range'))

    nA = A.shape[1]  # num of ports on input s-matrix
    # create an empty s-matrix, to store the result
    C = npy.zeros(shape=A.shape, dtype='complex')

    # loop through ports and calculates resultant s-parameters
    for i in range(nA):
        for j in range(nA):
            C[:, i, j] = \
                A[:, i, j] + \
                (A[:, k, j] * A[:, i, l] * (1 - A[:, l, k]) + \
                 A[:, l, j] * A[:, i, k] * (1 - A[:, k, l]) + \
                 A[:, k, j] * A[:, l, l] * A[:, i, k] + \
                 A[:, l, j] * A[:, k, k] * A[:, i, l]) / \
                ((1 - A[:, k, l]) * (1 - A[:, l, k]) - A[:, k, k] * A[:, l, l])

    # remove ports that were `connected`
    C = npy.delete(C, (k, l), 1)
    C = npy.delete(C, (k, l), 2)

    return C


## network parameter conversion
def s2z(s: npy.ndarray, z0: NumberLike = 50, s_def: str = S_DEF_DEFAULT) -> npy.ndarray:
    r"""
    Convert scattering parameters [#]_ to impedance parameters [#]_


    For power-waves, Eq.(19) from [#Kurokawa]_:

    .. math::
        Z = F^{-1} (1 - S)^{-1} (S G + G^*) F

    where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\sqrt{|Re(Z_0)|}])`

    For pseudo-waves, Eq.(74) from [#Marks]_:

    .. math::
        Z = (1 - U^{-1} S U)^{-1}  (1 + U^{-1} S U) G

    where :math:`U = \sqrt{Re(Z_0)}/|Z_0|`

    Parameters
    ----------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number
        port impedances.
    s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition : 'power' for power-waves definition [#Kurokawa]_,
        'pseudo' for pseudo-waves definition [#Marks]_.
        'traveling' corresponds to the initial implementation.
        Default is 'power'.

    Returns
    -------
    z : complex array-like
        impedance parameters



    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
    .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.

    """
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0[z0.real == 0] += ZERO

    s = npy.array(s, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = npy.zeros_like(s)  # (nfreqs, nports, nports)
    npy.einsum('ijj->ij', Id)[...] = 1.0

    if s_def == 'power':
        # Power-waves. Eq.(19) from [Kurokawa et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = npy.zeros_like(s), npy.zeros_like(s)
        npy.einsum('ijj->ij', F)[...] = 1.0/(2*npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        z = npy.linalg.solve(mf.nudge_eig((Id - s) @ F), (s @ G + npy.conjugate(G)) @ F)

    elif s_def == 'pseudo':
        # Pseudo-waves. Eq.(74) from [Marks et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        ZR, U = npy.zeros_like(s), npy.zeros_like(s)
        npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
        npy.einsum('ijj->ij', ZR)[...] = z0
        USU = npy.linalg.solve(U, s @ U)
        z = npy.linalg.solve(mf.nudge_eig(Id - USU), (Id + USU) @ ZR)

    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrtz0 = npy.zeros_like(s)
        npy.einsum('ijj->ij', sqrtz0)[...] = npy.sqrt(z0)
        z = sqrtz0 @ npy.linalg.solve(mf.nudge_eig(Id - s), (Id + s) @ sqrtz0)
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return z

def s2y(s: npy.ndarray, z0:NumberLike = 50, s_def: str = S_DEF_DEFAULT) -> npy.ndarray:
    """
    convert scattering parameters [#]_ to admittance parameters [#]_

    Equations are the inverse of :func:`s2z`.

    Parameters
    ----------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number
        port impedances
    s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition : 'power' for power-waves definition [#]_,
        'pseudo' for pseudo-waves definition [#]_.
        'traveling' corresponds to the initial implementation.
        Default is 'power'.

    Returns
    -------
    y : complex array-like
        admittance parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    .. [#] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
    .. [#] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.
    """
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0[z0.real == 0] += ZERO

    s = npy.array(s, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = npy.zeros_like(s)  # (nfreqs, nports, nports)
    npy.einsum('ijj->ij', Id)[...] = 1.0

    if s_def == 'power':
        # Power-waves. Inverse of Eq.(19) from [Kurokawa et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = npy.zeros_like(s), npy.zeros_like(s)
        npy.einsum('ijj->ij', F)[...] = 1.0/(2*npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        y = npy.linalg.solve(mf.nudge_eig((s @ G + npy.conjugate(G)) @ F), (Id - s) @ F)

    elif s_def == 'pseudo':
        # pseudo-waves. Inverse of Eq.(74) from [Marks et al.]
        YR, U = npy.zeros_like(s), npy.zeros_like(s)
        npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
        npy.einsum('ijj->ij', YR)[...] = 1/z0
        USU = npy.linalg.solve(U, s @ U)
        y = YR @ npy.linalg.solve(mf.nudge_eig(Id + USU), Id - USU)

    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrty0 = npy.zeros_like(s)  # (nfreqs, nports, nports)
        npy.einsum('ijj->ij', sqrty0)[...] = npy.sqrt(1.0/z0)
        y = sqrty0 @ (Id - s) @ npy.linalg.solve(mf.nudge_eig(Id + s), sqrty0)
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return y

def s2t(s: npy.ndarray) -> npy.ndarray:
    """
    Converts scattering parameters [#]_ to scattering transfer parameters [#]_ .

    transfer parameters are also referred to as
    'wave cascading matrix' [#]_, this function only operates on 2N-ports
    networks with same number of input and output ports, also known as
    'balanced networks'.

    Parameters
    ----------
    s : :class:`numpy.ndarray` (shape fx2nx2n)
        scattering parameter matrix

    Returns
    -------
    t : npy.ndarray
        scattering transfer parameters (aka wave cascading matrix)

    See Also
    --------
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
    ----------
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    .. [#] Janusz A. Dobrowolski, "Scattering Parameter in RF and Microwave Circuit Analysis and Design",
           Artech House, 2016, pp. 65-68
    """
    z, y, x = s.shape
    # test here for even number of ports.
    # s-parameter networks are square matrix, so x and y are equal.
    if(x % 2 != 0):
        raise IndexError('Network don\'t have an even number of ports')
    t = npy.zeros((z, y, x), dtype=complex)
    yh = int(y/2)
    xh = int(x/2)
    # S_II,I^-1
    sinv = npy.linalg.inv(s[:, yh:y, 0:xh])
    # np.linalg.inv test for singularity (matrix not invertible)
    for k in range(len(s)):
    # T_I,I = S_I,II - S_I,I . S_II,I^-1 . S_II,II
        t[k, 0:yh, 0:xh] = s[k, 0:yh, xh:x] - s[k, 0:yh, 0:xh].dot(sinv[k].dot(s[k, yh:y, xh:x]))
        # T_I,II = S_I,I . S_II,I^-1
        t[k, 0:yh, xh:x] = s[k, 0:yh, 0:xh].dot(sinv[k])
        # T_II,I = -S_II,I^-1 . S_II,II
        t[k, yh:y, 0:xh] = -sinv[k].dot(s[k, yh:y, xh:x])
        # T_II,II = S_II,I^-1
        t[k, yh:y, xh:x] = sinv[k]
    return t

def s2s(s: NumberLike, z0: NumberLike, s_def_new: str, s_def_old: str):
    r"""
    Convert scattering parameters to scattering parameters with different
    definition.

    Calculates port voltages and currents using the old definition and
    then calculates the incoming and reflected waves from the voltages
    using the new S-parameter definition.

    Only has effect if z0 has at least one complex impedance port.

    Parameters
    ----------
    s : complex array-like
        impedance parameters
    z0 : complex array-like or number
        port impedances
    s_def_new : str -> s_def : can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition of the output network.
        'power' for power-waves definition [#Kurokawa]_,
        'pseudo' for pseudo-waves definition [#Marks]_.
        'traveling' corresponds to the initial implementation.
    s_def_old : str -> s_def : can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition of the input network.
        'power' for power-waves definition [#Kurokawa]_,
        'pseudo' for pseudo-waves definition [#Marks]_.
        'traveling' corresponds to the initial implementation.

    Returns
    -------
    s : complex array-like
        scattering parameters

    References
    ----------
    .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
    .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.
    """
    if s_def_new == s_def_old:
        # Nothing to do.
        return s
        
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    if npy.isreal(z0).all():
        # Nothing to do because all port impedances are real so the used 
        # definition (power or travelling) does not make a difference.
        return s

    # Calculate port voltages and currents using the old s_def.
    F, G = npy.zeros_like(s), npy.zeros_like(s)
    if s_def_old == 'power':
        npy.einsum('ijj->ij', F)[...] = 1.0/(npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        Id = npy.eye(s.shape[1], dtype=complex)
        v = F @ (G.conjugate() + G @ s)
        i = F @ (Id - s)
    elif s_def_old == 'pseudo':
        npy.einsum('ijj->ij', F)[...] = abs(z0)/npy.sqrt(z0.real)
        npy.einsum('ijj->ij', G)[...] = abs(z0)/(z0*npy.sqrt(z0.real))
        Id = npy.eye(s.shape[1], dtype=complex)
        v = F @ (Id + s)
        i = G @ (Id - s)
    elif s_def_old == 'traveling':
        npy.einsum('ijj->ij', F)[...] = npy.sqrt(z0)
        npy.einsum('ijj->ij', G)[...] = 1/(npy.sqrt(z0))
        Id = npy.eye(s.shape[1], dtype=complex)
        v = F @ (Id + s)
        i = G @ (Id - s)
    else:
        raise ValueError(f'Unknown s_def: {s_def_old}')

    # Calculate a and b waves from the voltages and currents.
    F, G = npy.zeros_like(s), npy.zeros_like(s)
    if s_def_new == 'power':
        npy.einsum('ijj->ij', F)[...] = 1/(2*npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        a = F @ (v + G @ i)
        b = F @ (v - G.conjugate() @ i)
    elif s_def_new == 'pseudo':
        npy.einsum('ijj->ij', F)[...] = npy.sqrt(z0.real)/(2*abs(z0))
        npy.einsum('ijj->ij', G)[...] = z0
        a = F @ (v + G @ i)
        b = F @ (v - G @ i)
    elif s_def_new == 'traveling':
        npy.einsum('ijj->ij', F)[...] = 1/(npy.sqrt(z0))
        npy.einsum('ijj->ij', G)[...] = z0
        a = F @ (v + G @ i)
        b = F @ (v - G @ i)
    else:
        raise ValueError(f'Unknown s_def: {s_def_old}')

    # New S-parameter matrix from a and b waves.
    s_new = npy.zeros_like(s)
    for n in range(nports):
        for m in range(nports):
            s_new[:, m, n] = b[:, m, n] / a[:, n, n]

    return s_new

def z2s(z: NumberLike, z0:NumberLike = 50, s_def: str = S_DEF_DEFAULT) -> npy.ndarray:
    r"""
    convert impedance parameters [#]_ to scattering parameters [#]_

    For power-waves, Eq.(18) from [#Kurokawa]_:

    .. math::
        S = F (Z – G^*) (Z + G)^{-1} F^{-1}

    where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\sqrt{|Re(Z_0)|}])`

    For pseudo-waves, Eq.(73) from [#Marks]_:

    .. math::
        S = U (Z - G) (Z + G)^{-1}  U^{-1}

    where :math:`U = \sqrt{Re(Z_0)}/|Z_0|`


    Parameters
    ----------
    z : complex array-like
        impedance parameters
    z0 : complex array-like or number
        port impedances
    s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition : 'power' for power-waves definition [#Kurokawa]_,
        'pseudo' for pseudo-waves definition [#Marks]_.
        'traveling' corresponds to the initial implementation.
        Default is 'power'.

    Returns
    -------
    s : complex array-like
        scattering parameters



    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
    .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.

    """
    nfreqs, nports, nports = z.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0[z0.real == 0] += ZERO

    z = npy.array(z, dtype=complex)

    if s_def == 'power':
        # Power-waves. Eq.(18) from [Kurokawa et al.3]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = npy.zeros_like(z), npy.zeros_like(z)
        npy.einsum('ijj->ij', F)[...] = 1.0/(2*npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        s = mf.rsolve(F @ (z + G), F @ (z - npy.conjugate(G)))

    elif s_def == 'pseudo':
        # Pseudo-waves. Eq.(73) from [Marks et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        ZR, U = npy.zeros_like(z), npy.zeros_like(z)
        npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
        npy.einsum('ijj->ij', ZR)[...] = z0
        s = mf.rsolve(U @ (z + ZR), U @ (z - ZR))

    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id, sqrty0 = npy.zeros_like(z), npy.zeros_like(z) # (nfreqs, nports, nports)
        npy.einsum('ijj->ij', Id)[...] = 1.0
        npy.einsum('ijj->ij', sqrty0)[...] = npy.sqrt(1.0/z0)
        s = mf.rsolve(sqrty0 @ z @ sqrty0 + Id, sqrty0 @ z @ sqrty0 - Id)
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return s

def z2y(z: npy.ndarray) -> npy.ndarray:
    """
    convert impedance parameters [#]_ to admittance parameters [#]_


    .. math::
        y = z^{-1}

    Parameters
    ----------
    z : complex array-like
        impedance parameters

    Returns
    -------
    y : complex array-like
        admittance parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    """
    return npy.linalg.inv(z)

def z2t(z: npy.ndarray) -> NoReturn:
    """
    Not Implemented yet

    convert impedance parameters [#]_ to scattering transfer parameters [#]_


    Parameters
    ----------
    z : complex array-like or number
        impedance parameters

    Returns
    -------
    s : complex array-like or number
        scattering parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    """
    raise (NotImplementedError)


def a2s(a: npy.ndarray, z0: NumberLike = 50) -> npy.ndarray:
    """
    convert abcd parameters to s parameters

    Parameters
    ----------
    a : complex array-like
        abcd parameters
    z0 : complex array-like or number
        port impedances

    Returns
    -------
    s : complex array-like
        abcd parameters

    """
    nfreqs, nports, nports = a.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    A = a[:,0,0]
    B = a[:,0,1]
    C = a[:,1,0]
    D = a[:,1,1]
    denom = A*z02 + B + C*z01*z02 + D*z01

    s = npy.array([
        [
            (A*z02 + B - C*z01.conj()*z02 - D*z01.conj() ) / denom,
            (2*npy.sqrt(z01.real * z02.real)) / denom,
        ],
        [
            (2*(A*D - B*C)*npy.sqrt(z01.real * z02.real)) / denom,
            (-A*z02.conj() + B - C*z01*z02.conj() + D*z01) / denom,
        ],
    ]).transpose()
    return s

    #return z2s(a2z(a), z0)



def a2z(a: npy.ndarray) -> npy.ndarray:
    """
    Converts abcd parameters to z parameters [#]_ .


    Parameters
    ----------
    a : :class:`numpy.ndarray` (shape fx2x2)
        abcd parameter matrix

    Returns
    -------
    z : npy.ndarray
        impedance parameters

    See Also
    --------
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
    .. [#] https://en.wikipedia.org/wiki/Two-port_network
    """

    return z2a(a)


def z2a(z: npy.ndarray) -> npy.ndarray:
    """
    Converts impedance parameters to abcd  parameters [#]_ .


    Parameters
    ----------
    z : :class:`numpy.ndarray` (shape fx2x2)
        impedance parameter matrix

    Returns
    -------
    abcd : npy.ndarray
        scattering transfer parameters (aka wave cascading matrix)

    See Also
    --------
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
    ----------
    .. [#] https://en.wikipedia.org/wiki/Two-port_network
    """
    abcd = npy.array([
        [z[:, 0, 0] / z[:, 1, 0],
         1. / z[:, 1, 0]],
        [(z[:, 0, 0] * z[:, 1, 1] - z[:, 1, 0] * z[:, 0, 1]) / z[:, 1, 0],
         z[:, 1, 1] / z[:, 1, 0]],
    ]).transpose()
    return abcd


def s2a(s: npy.ndarray, z0: NumberLike = 50) -> npy.ndarray:
    """
    Converts scattering parameters to abcd parameters [#]_ .

    Parameters
    ----------
    s : :class:`numpy.ndarray` (shape `fx2x2`)
        impedance parameter matrix

    z0: number or, :class:`numpy.ndarray` (shape `fx2`)
        port impedance

    Returns
    -------
    abcd : npy.ndarray
        scattering transfer parameters (aka wave cascading matrix)

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Two-port_network

    """
    nfreqs, nports, nports = s.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    denom = (2*s[:,1,0]*npy.sqrt(z01.real * z02.real))
    a = npy.array([
        [
            ((z01.conj() + s[:,0,0]*z01)*(1 - s[:,1,1]) + s[:,0,1]*s[:,1,0]*z01) / denom,
            ((1 - s[:,0,0])*(1 - s[:,1,1]) - s[:,0,1]*s[:,1,0]) / denom,
        ],
        [
            ((z01.conj() + s[:,0,0]*z01)*(z02.conj() + s[:,1,1]*z02) - s[:,0,1]*s[:,1,0]*z01*z02) / denom,
            ((1 - s[:,0,0])*(z02.conj() + s[:,1,1]*z02) + s[:,0,1]*s[:,1,0]*z02) / denom,
        ],
    ]).transpose()
    return a


def y2s(y: npy.ndarray, z0:NumberLike = 50, s_def: str = S_DEF_DEFAULT) -> Network:
    r"""
    convert admittance parameters [#]_ to scattering parameters [#]_

    For power-waves, from [#Kurokawa]_:

    .. math::
        S = F (1 – G Y) (1 + G Y)^{-1} F^{-1}

    where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\sqrt{|Re(Z_0)|}])`

    For pseudo-waves, Eq.(73) from [#Marks]_:

    .. math::
        S = U (Y^{-1} - G) (Y^{-1} + G)^{-1}  U^{-1}

    where :math:`U = \sqrt{Re(Z_0)}/|Z_0|`


    Parameters
    ----------
    y : complex array-like
        admittance parameters

    z0 : complex array-like or number
        port impedances

    s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition : 'power' for power-waves definition [#Kurokawa]_,
        'pseudo' for pseudo-waves definition [#Marks]_.
        'traveling' corresponds to the initial implementation.
        Default is 'power'.

    Returns
    -------
    s : complex array-like or number
        scattering parameters

    See Also
    --------
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
    .. [#Kurokawa] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
    .. [#Marks] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.
    """
    nfreqs, nports, nports = y.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0[z0.real == 0] += ZERO

    y = npy.array(y, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = npy.zeros_like(y)  # (nfreqs, nports, nports)
    npy.einsum('ijj->ij', Id)[...] = 1.0

    if s_def == 'power':
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = npy.zeros_like(y), npy.zeros_like(y)
        npy.einsum('ijj->ij', F)[...] = 1.0/(2*npy.sqrt(z0.real))
        npy.einsum('ijj->ij', G)[...] = z0
        s = mf.rsolve(F @ (Id + G @ y), F @ (Id - npy.conjugate(G) @ y))

    elif s_def == 'pseudo':
        # Pseudo-waves
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        ZR, U = npy.zeros_like(y), npy.zeros_like(y)
        npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
        npy.einsum('ijj->ij', ZR)[...] = z0
        # This formulation is not very good numerically
        UY = mf.rsolve(mf.nudge_eig(y, cond=1e-12), U)
        s = mf.rsolve(UY + U @ ZR, -2 * U @ ZR) + Id

    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrtz0 = npy.zeros_like(y)  # (nfreqs, nports, nports)
        npy.einsum('ijj->ij', sqrtz0)[...] = npy.sqrt(z0)
        s = mf.rsolve(Id + sqrtz0 @ y @ sqrtz0, Id - sqrtz0 @ y @ sqrtz0)
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return s

def y2z(y: npy.ndarray) -> npy.ndarray:
    """
    convert admittance parameters [#]_ to impedance parameters [#]_

    .. math::

        z = y^{-1}

    Parameters
    ----------
    y : complex array-like
        admittance parameters

    Returns
    -------
    z : complex array-like
        impedance parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    """
    return npy.linalg.inv(y)


def y2t(y: npy.ndarray) -> NoReturn:
    """
    Not Implemented Yet

    convert admittance parameters [#]_ to scattering-transfer parameters [#]_


    Parameters
    ----------
    y : complex array-like or number
        impedance parameters

    Returns
    -------
    t : complex array-like or number
        scattering parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    """
    raise (NotImplementedError)


def t2s(t: npy.ndarray) -> npy.ndarray:
    """
    converts scattering transfer parameters [#]_ to scattering parameters [#]_

    transfer parameters are also referred to as
    'wave cascading matrix', this function only operates on 2N-ports
    networks with same number of input and output ports, also known as
    'balanced networks'.

    Parameters
    ----------
    t : :class:`numpy.ndarray` (shape fx2nx2n)
            scattering transfer parameters

    Returns
    -------
    s : :class:`numpy.ndarray`
            scattering parameter matrix.

    See Also
    --------
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
    """
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


def t2z(t: npy.ndarray) -> NoReturn:
    """
    Not Implemented Yet

    Convert scattering transfer parameters [#]_ to impedance parameters [#]_


    Parameters
    ----------
    t : complex array-like or number
        impedance parameters

    Returns
    -------
    z : complex array-like or number
        scattering parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    """
    raise (NotImplementedError)


def t2y(t: npy.ndarray) -> NoReturn:
    """
    Not Implemented Yet

    Convert scattering transfer parameters to admittance parameters [#]_


    Parameters
    ----------
    t : complex array-like or number
        t-parameters

    Returns
    -------
    y : complex array-like or number
        admittance parameters

    See Also
    --------
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
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters

    """
    raise (NotImplementedError)


def h2z(h: npy.ndarray) -> npy.ndarray:
    """
    Converts hybrid parameters to z parameters [#]_ .


    Parameters
    ----------
    h : :class:`numpy.ndarray` (shape fx2x2)
        hybrid parameter matrix

    Returns
    -------
    z : npy.ndarray
        impedance parameters

    See Also
    --------
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
    .. [#] https://en.wikipedia.org/wiki/Two-port_network
    """

    return z2h(h)


def h2s(h: npy.ndarray, z0: NumberLike = 50) -> npy.ndarray:
    """
    convert hybrid parameters to s parameters

    Parameters
    ----------
    h : complex array-like
        hybrid parameters
    z0 : complex array-like or number
        port impedances

    Returns
    -------
    s : complex array-like
        scattering parameters

    """

    return z2s(h2z(h), z0)


def s2h(s: npy.ndarray, z0: NumberLike = 50) -> npy.ndarray:
    """
    Convert scattering parameters [#]_ to hybrid parameters [#]_


    Parameters
    ----------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number
        port impedances.

    Returns
    -------
    h : complex array-like
        hybrid parameters



    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/Two-port_network#Hybrid_parameters_(h-parameters)

    """
    return z2h(s2z(s, z0))


def z2h(z: npy.ndarray) -> npy.ndarray:
    """
    Converts impedance parameters to hybrid parameters [#]_ .


    Parameters
    ----------
    z : :class:`numpy.ndarray` (shape fx2x2)
        impedance parameter matrix

    Returns
    -------
    h : npy.ndarray
        hybrid parameters

    See Also
    --------
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
    .. [#] https://en.wikipedia.org/wiki/Two-port_network
    """
    h = npy.array([
        [(z[:, 0, 0] * z[:, 1, 1] - z[:, 1, 0] * z[:, 0, 1]) / z[:, 1, 1],
         -z[:, 1, 0] / z[:, 1, 1]],
        [z[:, 0, 1] / z[:, 1, 1],
         1. / z[:, 1, 1]],
    ]).transpose()
    return h


## these methods are used in the secondary properties
def passivity(s: npy.ndarray) -> npy.ndarray:
    r"""
    Passivity metric for a multi-port network.

    A metric which is proportional to the amount of power lost in a
    multiport network, depending on the excitation port. Specifically,
    this returns a matrix who's diagonals are equal to the total
    power received at all ports, normalized to the power at a single
    excitement port.

    mathematically, this is a test for unitary-ness of the
    s-parameter matrix [#]_.

    for two port this is

    .. math::

            \sqrt( |S_{11}|^2 + |S_{21}|^2 \, , \, |S_{22}|^2+|S_{12}|^2)

    in general it is

    .. math::

            \sqrt( S^H \cdot S)

    where :math:`H` is conjugate transpose of S, and :math:`\cdot`
    is dot product.

    Note
    ----
    The total amount of power dissipated in a network depends on the
    port matches. For example, given a matched attenuator, this metric
    will yield the attenuation value. However, if the attenuator is
    cascaded with a mismatch, the power dissipated will not be equivalent
    to the attenuator value, nor equal for each excitation port.


    Returns
    -------
    passivity : :class:`numpy.ndarray` of shape fxnxn

    References
    ------------
    .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Lossless_networks
    """
    if s.shape[-1] == 1:
        raise (ValueError('Doesn\'t exist for one ports'))

    pas_mat = s.copy()
    for f in range(len(s)):
        pas_mat[f, :, :] = npy.sqrt(npy.dot(s[f, :, :].conj().T, s[f, :, :]))

    return pas_mat


def reciprocity(s: npy.ndarray) -> npy.ndarray:
    """
        Reciprocity metric for a multi-port network.

        This returns the magnitude of the difference between the
        s-parameter matrix and its transpose.

        for two port this is

        .. math::

                | S - S^T |


        where :math:`T` is transpose of S

        Parameters
        ----------
        s : :class:`numpy.ndarray` of shape `fxnxn`
            s-parameter matrix

        Returns
        -------
        reciprocity : :class:`numpy.ndarray` of shape `fxnxn`
        """
    if s.shape[-1] == 1:
        raise (ValueError('Doesn\'t exist for one ports'))

    rec_mat = s.copy()
    for f in range(len(s)):
        rec_mat[f, :, :] = abs(s[f, :, :] - s[f, :, :].T)

    return rec_mat


## renormalize
def renormalize_s(s: npy.ndarray, z_old: NumberLike, z_new: NumberLike, s_def:str = S_DEF_DEFAULT, s_def_old: Union[str, None] = None) -> npy.ndarray:

    """
    Renormalize a s-parameter matrix given old and new port impedances.
    Can be also used to convert between different S-parameter definitions.


    Note
    ----
     This re-normalization assumes power-wave formulation per default.
     To use the pseudo-wave formulation, use s_def='pseudo'.
     However, results should be the same for real-valued characteristic impedances.
     See the [#Marks]_ and [#Anritsu]_ for more details.


    Note
    ----
    This just calls ::

        z2s(s2z(s, z0=z_old, s_def=s_def_old), z0=z_new, s_def=s_def)



    Parameters
    ----------
    s : complex array of shape `fxnxn`
        s-parameter matrix

    z_old : complex array of shape `fxnxn` or a scalar
        old (original) port impedances

    z_new : complex array of shape `fxnxn` or a scalar
        new port impedances

    s_def : str -> s_def :  can be: 'power', 'pseudo' or 'traveling'
        Scattering parameter definition of the output network:
        'power' for power-waves definition,
        'pseudo' for pseudo-waves definition.
        'traveling' corresponds to the initial implementation.
        Default is 'power'.

    s_def_old : str -> s_def :  can be: None, 'power', 'pseudo' or 'traveling'
        Scattering parameter definition of the input network:
        None to copy s_def.
        'power' for power-waves definition,
        'pseudo' for pseudo-waves definition.
        'traveling' corresponds to the initial implementation.

    Returns
    -------
    :class:`numpy.ndarray`
        renormalized s-parameter matrix (shape `fxnxn`)

    See Also
    --------
    Network.renormalize : method of Network to renormalize s
    fix_z0_shape
    s2z
    z2s

    References
    ----------
    .. [#Marks] R. B. Marks and D. F. Williams, "A general waveguide circuit theory," Journal of Research of the National Institute of Standards and Technology, vol. 97, no. 5, pp. 533-561, 1992.

    .. [#Anritsu] Anritsu Application Note: Arbitrary Impedance, https://web.archive.org/web/20200111134414/https://archive.eetasia.com/www.eetasia.com/ARTICLES/2002MAY/2002MAY02_AMD_ID_NTES_AN.PDF?SOURCES=DOWNLOAD

    Examples
    --------
    >>> s = zeros(shape=(101,2,2))
    >>> renormalize_s(s, 50,25)


    """
    if s_def_old not in S_DEFINITIONS and s_def_old is not None:
        raise ValueError('s_def_old parameter should be one of:', S_DEFINITIONS)
    if s_def_old is None:
        s_def_old = s_def
    if s_def not in S_DEFINITIONS:
        raise ValueError('s_def parameter should be one of:', S_DEFINITIONS)
    # thats a heck of a one-liner!
    return z2s(s2z(s, z0=z_old, s_def=s_def_old), z0=z_new, s_def=s_def)

def fix_param_shape(p: NumberLike):
    """
    This attempts to broadcast p to satisfy
        npy.shape(p) == (nfreqs, nports, nports)

    Parameters
    ----------
    p : number, array-like
        p can be:
        * a number (one frequency, one port)
        * 1D array-like (many frequencies, one port)
        * 2D array-like (one frequency, many ports)
        * 3D array-like (many frequencies, many ports)

    Returns
    -------
    p : array of shape == (nfreqs, nports, nports)
        p with the right shape for a nport Network

    """
    # Ensure input is numpy array
    p = npy.array(p, dtype=complex)
    if len(p.shape) == 0:
        # Scalar
        return p.reshape(1, 1, 1)
    if len(p.shape) == 1:
        # One port with many frequencies
        return p.reshape(p.shape[0], 1, 1)
    if p.shape[-1] != p.shape[-2]:
        raise ValueError('Input matrix must be square')
    if len(p.shape) == 2:
        # Many port with one frequency
        return p.reshape(-1, p.shape[0], p.shape[0])
    if len(p.shape) != 3:
        raise ValueError('Input array has too many dimensions. Shape: {}' \
                .format(p.shape))
    return p

def fix_z0_shape(z0: NumberLike, nfreqs: int, nports: int) -> npy.ndarray:
    """
    Make a port impedance of correct shape for a given network's matrix

    This attempts to broadcast z0 to satisfy
        npy.shape(z0) == (nfreqs,nports)

    Parameters
    ----------
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
    -------
    z0 : array of shape ==(nfreqs,nports)
        z0  with the right shape for a nport Network

    Examples
    --------
    For a two-port network with 201 frequency points, possible uses may
    be

    >>> z0 = rf.fix_z0_shape(50 , 201,2)
    >>> z0 = rf.fix_z0_shape([50,25] , 201,2)
    >>> z0 = rf.fix_z0_shape(range(201) , 201,2)


    """
    if npy.shape(z0) == (nfreqs, nports):
        # z0 is of correct shape. super duper.return it quick.
        return z0.copy()

    elif npy.ndim(z0) == 0:
        # z0 is a single number or npy.array without dimensions.
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


## cascading assistance functions
def inv(s: npy.ndarray) -> npy.ndarray:
    """
    Calculates 'inverse' s-parameter matrix, used for de-embedding

    This is not literally the inverse of the s-parameter matrix.
    Instead, it is defined such that the inverse of the s-matrix cascaded
    with itself is a unity scattering transfer parameter (T) matrix.

    .. math::

            inv(s) = t2s({s2t(s)}^{-1})

    where :math:`x^{-1}` is the matrix inverse. In words, this
    is the inverse of the scattering transfer parameters matrix
    transformed into a scattering parameters matrix.

    Parameters
    ----------
    s : :class:`numpy.ndarray` (shape fx2nx2n)
            scattering parameter matrix.

    Returns
    -------
    s' : :class:`numpy.ndarray`
            inverse scattering parameter matrix.

    See Also
    --------
    t2s : converts scattering transfer parameters to scattering parameters
    s2t : converts scattering parameters to scattering transfer parameters


    """
    # this idea is from lihan
    t = s2t(s)
    tinv = npy.linalg.inv(t)
    sinv = t2s(tinv)
    #for f in range(len(i)):
    #    i[f, :, :] = npy.linalg.inv(i[f, :, :])  # could also be written as
    #    #   npy.mat(i[f,:,:])**-1  -- Trey

    return sinv


def flip(a: npy.ndarray) -> npy.ndarray:
    """
    Invert the ports of a networks s-matrix, 'flipping' it over left and right.

    in case the network is 2n-port and n > 1, 'second' numbering scheme is
    assumed to be consistent with the ** cascade operator::

         +--------+                 +--------+
       0-|0      n|-n             0-|n      0|-n
       1-|1    n+1|-n+1    flip   1-|n+1    1|-n+1
        ...      ...       =>       ...      ...
     n-1-|n-1 2n-1|-2n-1        n-1-|2n-1 n-1|-2n-1
         +--------+                 +--------+

    Parameters
    ----------
    a : :class:`numpy.ndarray`
            scattering parameter matrix. shape should be should be `2nx2n`, or
            `fx2nx2n`

    Returns
    -------
    c : :class:`numpy.ndarray`
            flipped scattering parameter matrix

    See Also
    --------
    renumber
    """
    c = a.copy()
    n2 = a.shape[-1]
    m2 = a.shape[-2]
    n = int(n2/2)
    if (n2 == m2) and (n2 % 2 == 0):
        old = list(range(0,2*n))
        new = list(range(n,2*n)) + list(range(0,n))
        if(len(a.shape) == 2):
            c[new, :] = c[old, :] # renumber rows
            c[:, new] = c[:, old] # renumber columns
        else:

            c[:, new, :] = c[:, old, :] # renumber rows
            c[:, :, new] = c[:, :, old] # renumber columns
    else:
        raise IndexError('matrices should be 2nx2n, or kx2nx2n')
    return c


## COMMON CHECKS (raise exceptions)
def check_frequency_equal(ntwkA: Network, ntwkB: Network) -> None:
    """
    checks if two Networks have same frequency
    """
    if assert_frequency_equal(ntwkA, ntwkB) == False:
        raise IndexError('Networks don\'t have matching frequency. See `Network.interpolate`')


def check_frequency_exist(ntwk) -> None:
    """
    Check if a Network has a non-zero Frequency.
    """
    if assert_frequency_exist(ntwk) == False:
        raise ValueError('Network has no Frequency. Frequency points must be defined.')


def check_z0_equal(ntwkA: Network, ntwkB: Network) -> None:
    """
    checks if two Networks have same port impedances
    """
    # note you should check frequency equal before you call this
    if assert_z0_equal(ntwkA, ntwkB) == False:
        raise ValueError('Networks don\'t have matching z0.')


def check_nports_equal(ntwkA: Network, ntwkB: Network) -> None:
    """
    checks if two Networks have same number of ports
    """
    if assert_nports_equal(ntwkA, ntwkB) == False:
        raise ValueError('Networks don\'t have matching number of ports.')


## TESTs (return [usually boolean] values)
def assert_frequency_equal(ntwkA: Network, ntwkB: Network) -> bool:
    """
    """
    return (ntwkA.frequency == ntwkB.frequency)


def assert_frequency_exist(ntwk: Network) -> bool:
    """
    Test if the Network Frequency is defined.

    Returns
    -------
    bool: boolean

    """
    return bool(len(ntwk.frequency))


def assert_z0_equal(ntwkA: Network, ntwkB: Network) -> bool:
    """
    """
    return (ntwkA.z0 == ntwkB.z0).all()


def assert_z0_at_ports_equal(ntwkA: Network, k: int, ntwkB: Network, l: int) -> bool:
    """
    """
    return (ntwkA.z0[:, k] == ntwkB.z0[:, l]).all()


def assert_nports_equal(ntwkA: Network, ntwkB: Network) -> bool:
    """
    """
    return (ntwkA.number_of_ports == ntwkB.number_of_ports)


## Other
# don't belong here, but i needed them quickly
# this is needed for port impedance mismatches
def impedance_mismatch(z1: NumberLike, z2: NumberLike, s_def: str = 'traveling') -> npy.ndarray:
    """
    creates a two-port s-matrix for a impedance mis-match

    Parameters
    ----------
    z1 : number or array-like
            complex impedance of port 1
    z2 : number or array-like
            complex impedance of port 2
    s_def : str, optional. Default is 'traveling'.
        Scattering parameter definition:
        'power' for power-waves definition,
        'pseudo' for pseudo-waves definition.
        'traveling' corresponds to the initial implementation.
        NB: results are the same for real-valued characteristic impedances.

    Returns
    -------
    s' : 2-port s-matrix for the impedance mis-match

    References
    ----------
    .. [#] R. B. Marks et D. F. Williams, A general waveguide circuit theory,
            J. RES. NATL. INST. STAN., vol. 97, no. 5, p. 533, sept. 1992.
    """
    from .tlineFunctions import zl_2_Gamma0
    gamma = zl_2_Gamma0(z1, z2)
    result = npy.zeros(shape=(len(gamma), 2, 2), dtype='complex')

    if s_def == 'traveling':
        result[:, 0, 0] = gamma
        result[:, 1, 1] = -gamma
        result[:, 1, 0] = (1 + gamma) * npy.sqrt(1.0 * z1 / z2)
        result[:, 0, 1] = (1 - gamma) * npy.sqrt(1.0 * z2 / z1)
    elif s_def == 'pseudo':
        n = npy.abs(z2/z1) * npy.sqrt(z1.real / z2.real)
        result[:, 0, 0] = gamma
        result[:, 1, 1] = -gamma
        result[:, 1, 0] = 2 * z2 / (n * (z1 + z2))
        result[:, 0, 1] = 2 * z1 * n / (z1 + z2)
    elif s_def == 'power':
        n = npy.sqrt(z1.real / z2.real)
        result[:, 0, 0] = (z2 - z1.conjugate()) / (z1 + z2)
        result[:, 1, 1] = (z1 - z2.conjugate()) / (z1 + z2)
        result[:, 1, 0] = (2 * z1.real) / (n * (z1 + z2))
        result[:, 0, 1] = (2 * z2.real) * n / (z1 + z2)
    else:
        raise ValueError(f'Unsupported s_def: {s_def}')

    return result


def two_port_reflect(ntwk1: Network, ntwk2: Network = None) -> Network:
    """
    Generates a two-port reflective two-port, from two one-ports.


    Parameters
    ----------
    ntwk1 : one-port Network object
            network seen from port 1
    ntwk2 : one-port Network object, or None
            network seen from port 2. if None then will use ntwk1.

    Returns
    -------
    result : Network object
            two-port reflective network


    Note
    ----
    The resultant Network is copied from `ntwk1`, so its various
    properties(name, frequency, etc) are inherited from that Network.


    Examples
    --------
    >>> short,open = rf.Network('short.s1p', rf.Network('open.s1p')
    >>> rf.two_port_reflect(short,open)
    """
    result = ntwk1.copy()
    if ntwk2 is None:
        ntwk2 = ntwk1
    s11 = ntwk1.s[:, 0, 0]
    s22 = ntwk2.s[:, 0, 0]
    s21 = npy.zeros(ntwk1.frequency.npoints, dtype=complex)
    result.s = npy.array( \
        [[s11, s21], \
         [s21, s22]]). \
        transpose().reshape(-1, 2, 2)
    result.z0 = npy.hstack([ntwk1.z0, ntwk2.z0])
    try:
        result.name = ntwk1.name + '-' + ntwk2.name
    except(TypeError):
        pass
    return result

def s2s_active(s: npy.ndarray, a:npy.ndarray) -> npy.ndarray:
    r"""
    Returns active s-parameters for a defined wave excitation a.

    The active s-parameter at a port is the reflection coefficients
    when other ports are excited. It is an important quantity for active
    phased array antennas.

    Active s-parameters are defined by [#]_:

    .. math::

        \mathrm{active}(s)_{m} = \sum_{i=1}^N\left( s_{mi} a_i \right) / a_m

    where :math:`s` are the scattering parameters and :math:`N` the number of ports

    Parameters
    ----------
    s : complex array
        scattering parameters (nfreqs, nports, nports)

    a : complex array of shape (n_ports)
        forward wave complex amplitude (pseudowave formulation [#]_)

    Returns
    -------
    s_act : complex array of shape (n_freqs, n_ports)
        active S-parameters for the excitation a

    See Also
    --------
        s2z_active : active Z-parameters
        s2y_active : active Y-parameters
        s2vswr_active : active VSWR

    References
    ----------
    .. [#] D. M. Pozar, IEEE Trans. Antennas Propag. 42, 1176 (1994).

    .. [#] D. Williams, IEEE Microw. Mag. 14, 38 (2013).

    """
    a = npy.asarray(a, dtype=complex)
    a[a == 0] = 1e-12  # solve numerical singularity
    s_act = npy.einsum('fmi,i,m->fm', s, a, npy.reciprocal(a))
    return s_act  # shape : (n_freqs, n_ports)

def s2z_active(s: npy.ndarray, z0: NumberLike, a: npy.ndarray) -> npy.ndarray:
    r"""
    Returns the active Z-parameters for a defined wave excitation a.

    The active Z-parameters are defined by:

    .. math::

        \mathrm{active}(z)_{m} = z_{0,m} \frac{1 + \mathrm{active}(s)_m}{1 - \mathrm{active}(s)_m}

    where :math:`z_{0,m}` is the characteristic impedance and
    :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

    Parameters
    ----------
    s : complex array
        scattering parameters (nfreqs, nports, nports)

    z0 : complex array-like or number
        port impedances.

    a : complex array of shape (n_ports)
        forward wave complex amplitude

    Returns
    -------
    z_act : complex array of shape (nfreqs, nports)
        active Z-parameters for the excitation a

    See Also
    --------
        s2s_active : active S-parameters
        s2y_active : active Y-parameters
        s2vswr_active : active VSWR

    """
    # TODO : vectorize the for loop
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    s_act = s2s_active(s, a)
    z_act = npy.einsum('fp,fp,fp->fp', z0, 1 + s_act, npy.reciprocal(1 - s_act))
    return z_act

def s2y_active(s: npy.ndarray, z0: NumberLike, a: npy.ndarray) -> npy.ndarray:
    r"""
    Returns the active Y-parameters for a defined wave excitation a.

    The active Y-parameters are defined by:

    .. math::

        \mathrm{active}(y)_{m} = y_{0,m} \frac{1 - \mathrm{active}(s)_m}{1 + \mathrm{active}(s)_m}

    where :math:`y_{0,m}` is the characteristic admittance and
    :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

    Parameters
    ----------
    s : complex array
        scattering parameters (nfreqs, nports, nports)

    z0 : complex array-like or number
        port impedances.

    a : complex array of shape (n_ports)
        forward wave complex amplitude

    Returns
    -------
    y_act : complex array of shape (nfreqs, nports)
        active Y-parameters for the excitation a

    See Also
    --------
        s2s_active : active S-parameters
        s2z_active : active Z-parameters
        s2vswr_active : active VSWR
    """
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    s_act = s2s_active(s, a)
    y_act = npy.einsum('fp,fp,fp->fp', npy.reciprocal(z0), 1 - s_act, npy.reciprocal(1 + s_act))        
    return y_act

def s2vswr_active(s: npy.ndarray, a: npy.ndarray) -> npy.ndarray:
    r"""
    Returns the active VSWR for a defined wave excitation a..

    The active VSWR is defined by :

    .. math::

        \mathrm{active}(vswr)_{m} = \frac{1 + |\mathrm{active}(s)_m|}{1 - |\mathrm{active}(s)_m|}

    where :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.


    Parameters
    ----------
    s : complex array
        scattering parameters (nfreqs, nports, nports)

    a : complex array of shape (n_ports)
        forward wave complex amplitude

    Returns
    -------
    vswr_act : complex array of shape (nfreqs, nports)
        active VSWR for the excitation a

    See Also
    --------
        s2s_active : active S-parameters
        s2z_active : active Z-parameters
        s2y_active : active Y-parameters
    """
    s_act = s2s_active(s, a)
    vswr_act = npy.einsum('fp,fp->fp', (1 + npy.abs(s_act)), npy.reciprocal(1 - npy.abs(s_act)))
    return vswr_act

