"""
.. currentmodule:: skrf.frequency

========================================
frequency (:mod:`skrf.frequency`)
========================================

Provides a frequency object and related functions.

Most of the functionality is provided as methods and properties of the
:class:`Frequency` Class.


Frequency Class
===============
.. autosummary::
   :toctree: generated/

   Frequency

Functions
=========

.. autosummary::
    :toctree: generated/

    overlap_freq

Misc
====

.. autosummary::
    :toctree: generated/

    InvalidFrequencyWarning

"""

# from matplotlib.pyplot import gca,plot, autoscale
from typing import List
import warnings

from numbers import Number
from .constants import NumberLike, ZERO
from typing import Union
from numpy import pi, linspace, geomspace
import numpy as npy
from numpy import gradient  # used to center attribute `t` at 0
import re
from .util import slice_domain, find_nearest_index


class InvalidFrequencyWarning(UserWarning):
    """Thrown if frequency values aren't monotonously increasing
    """
    pass


class Frequency:
    """
    A frequency band.

    The frequency object provides a convenient way to work with and
    access a frequency band. It contains  a frequency vector as well as
    a frequency unit. This allows a frequency vector in a given unit
    to be available (:attr:`f_scaled`), as well as an absolute frequency
    axis in 'Hz'  (:attr:`f`).

    A Frequency object can be created from either (start, stop, npoints)
    using the default constructor, :func:`__init__`. Or, it can be
    created from an arbitrary frequency vector by using the class
    method :func:`from_f`.

    Internally, the frequency information is stored in the `f` property
    combined with the `unit` property. All other properties, `start`
    `stop`, etc are generated from these.
    """
    unit_dict = {
            'hz': 'Hz',
            'khz': 'kHz',
            'mhz': 'MHz',
            'ghz': 'GHz',
            'thz': 'THz'
            }
    """
    Dictionnary to convert unit string with correct capitalization for display.
    """

    multiplier_dict={
            'hz': 1,
            'khz': 1e3,
            'mhz': 1e6,
            'ghz': 1e9,
            'thz': 1e12
            }
    """
    Frequency unit multipliers.
    """


    def __init__(self, start: float = 0, stop: float = 0, npoints: int = 0,
        unit: str = 'ghz', sweep_type: str = 'lin') -> None:
        """
        Frequency initializer.

        Creates a Frequency object from start/stop/npoints and a unit.
        Alternatively, the class method :func:`from_f` can be used to
        create a Frequency object from a frequency vector instead.

        Parameters
        ----------
        start : number, optional
            start frequency in  units of `unit`. Default is 0.
        stop : number, optional
            stop frequency in  units of `unit`. Default is 0.
        npoints : int, optional
            number of points in the band. Default is 0.
        unit : string, optional
            Frequency unit of the band: 'hz', 'khz', 'mhz', 'ghz', 'thz'.
            This is used to create the attribute :attr:`f_scaled`.
            It is also used by the :class:`~skrf.network.Network` class
            for plots vs. frequency. Default is 'ghz'.
        sweep_type : string, optional
            Type of the sweep: 'lin' or 'log'.
            'lin' for linear and 'log' for logarithmic. Default is 'lin'.

        Note
        ----
        The attribute `unit` sets the frequency multiplier, which is used
        to scale the frequency when `f_scaled` is referenced.

        Note
        ----
        The attribute `unit` is not case sensitive.
        Hence, for example, 'GHz' or 'ghz' is the same.

        See Also
        --------
        from_f : constructs a Frequency object from a frequency
            vector instead of start/stop/npoints.
        :attr:`unit` : frequency unit of the band

        Examples
        --------
        >>> wr1p5band = Frequency(500, 750, 401, 'ghz')

        """
        self._unit = unit.lower()

        start =  self.multiplier * start
        stop = self.multiplier * stop

        if sweep_type.lower() == 'lin':
            self._f = linspace(start, stop, npoints)
        elif sweep_type.lower() == 'log' and start > 0:
            self._f = geomspace(start, stop, npoints)
        else:
            raise ValueError('Sweep Type not recognized')

    def __str__(self) -> str:
        """
        """
        try:
            output =  \
                   '%s-%s %s, %i pts' % \
                   (self.f_scaled[0], self.f_scaled[-1], self.unit, self.npoints)
        except (IndexError):
            output = "[no freqs]"

        return output

    def __repr__(self) -> str:
        """
        """
        return self.__str__()

    def __getitem__(self, key: Union[str, int, slice]) -> 'Frequency':
        """
        Slices a Frequency object based on an index, or human readable string.

        Parameters
        ----------
        key : str, int, or slice
            if int, then it is interpreted as the index of the frequency
            if str, then should be like '50.1-75.5ghz', or just '50'.
            If the frequency unit is omitted then :attr:`unit` is
            used.

        Examples
        --------
        >>> b = rf.Frequency(50, 100, 101, 'ghz')
        >>> a = b['80-90ghz']
        >>> a.plot_s_db()
        """

        output = self.copy()


        if isinstance(key, str):

            # they passed a string try and do some interpretation
            re_numbers = re.compile(r'.*\d')
            re_hyphen = re.compile(r'\s*-\s*')
            re_letters = re.compile('[a-zA-Z]+')

            freq_unit = re.findall(re_letters,key)

            if len(freq_unit) == 0:
                freq_unit = self.unit
            else:
                freq_unit = freq_unit[0]

            key_nounit = re.sub(re_letters,'',key)
            edges  = re.split(re_hyphen,key_nounit)

            edges_freq = Frequency.from_f([float(k) for k in edges],
                                        unit = freq_unit)
            if len(edges_freq) ==2:
                slicer=slice_domain(output.f, edges_freq.f)
            elif len(edges_freq)==1:
                key = find_nearest_index(output.f, edges_freq.f[0])
                slicer = slice(key,key+1,1)
            else:
                raise ValueError()
            try:
                output._f = npy.array(output.f[slicer]).reshape(-1)
                return output
            except(IndexError):
                raise IndexError('slicing frequency is incorrect')


        if output.f.shape[0] > 0:
            output._f = npy.array(output.f[key]).reshape(-1)
        else:
            output._f = npy.empty(shape=(0))

        return output


    @classmethod
    def from_f(cls, f: NumberLike, *args,**kwargs) -> 'Frequency':
        """
        Construct Frequency object from a frequency vector.

        The unit is set by kwarg 'unit'

        Parameters
        ----------
        f : scalar or array-like
            frequency vector

        *args, **kwargs : arguments, keyword arguments
            passed on to  :func:`__init__`.

        Returns
        -------
        myfrequency : :class:`Frequency` object
            the Frequency object

        Raises
        ------
        InvalidFrequencyWarning:
            If frequency points are not monotonously increasing

        Examples
        --------
        >>> f = npy.linspace(75,100,101)
        >>> rf.Frequency.from_f(f, unit='ghz')
        """
        if npy.isscalar(f):
            f = [f]
        temp_freq =  cls(0,0,0,*args, **kwargs)
        temp_freq._f = npy.array(f) * temp_freq.multiplier
        temp_freq.check_monotonic_increasing()

        return temp_freq

    def __eq__(self, other: object) -> bool:
        #return (list(self.f) == list(other.f))
        # had to do this out of practicality
        if not isinstance(other, self.__class__):
            return False
        if len(self.f) != len(other.f):
            return False
        elif len(self.f) == len(other.f) == 0:
            return True
        else:
            return (max(abs(self.f-other.f)) < ZERO)

    def __ne__(self,other: object) -> bool:
        return (not self.__eq__(other))

    def __len__(self) -> int:
        """
        The number of frequency points
        """
        return self.npoints

    def __mul__(self,other: 'Frequency') -> 'Frequency':
        out = self.copy()
        out.f = self.f*other
        return out

    def __rmul__(self,other: 'Frequency') -> 'Frequency':
        out = self.copy()
        out.f = self.f*other
        return out

    def __div__(self,other: 'Frequency') -> 'Frequency':
        out = self.copy()
        out.f = self.f/other
        return out

    def check_monotonic_increasing(self) -> None:
        """Validate the frequency values

        Raises
        ------
        InvalidFrequencyWarning:
            If frequency points are not monotonously increasing
        """
        increase = npy.diff(self.f) > 0
        if not increase.all():
            warnings.warn("Frequency values are not monotonously increasing!\n"
            "To get rid of the invalid values call `drop_non_monotonic_increasing`", 
                InvalidFrequencyWarning)

    def drop_non_monotonic_increasing(self) -> List[int]:
        """Drop duplicate and invalid frequency values and return the dropped indices

        Returns:
            list[int]: The dropped indices
        """
        invalid = npy.zeros(len(self.f), dtype=bool)
        for i, val in enumerate(self.f):
            if not i:
                last_valid = val
            else:
                if val > last_valid:
                    last_valid = val
                else:
                    invalid[i] = True
        self._f = self._f[~invalid]
        return list(npy.flatnonzero(invalid))

    @property
    def start(self) -> float:
        """
        Starting frequency in Hz.
        """
        return self.f[0]

    @property
    def start_scaled(self) -> float:
        """
        Starting frequency in :attr:`unit`'s.
        """
        return self.f_scaled[0]
    @property
    def stop_scaled(self) -> float:
        """
        Stop frequency in :attr:`unit`'s.
        """
        return self.f_scaled[-1]

    @property
    def stop(self) -> float:
        """
        Stop frequency in Hz.
        """
        return self.f[-1]

    @property
    def npoints(self) -> int:
        """
        Number of points in the frequency.
        """
        return len(self.f)

    @npoints.setter
    def npoints(self, n: int) -> None:
        """
        Set the number of points in the frequency.
        """
        warnings.warn('Possibility to set the npoints parameter will removed in the next release.',
             DeprecationWarning, stacklevel=2)
        
        if self.sweep_type == 'lin':
            self.f = linspace(self.start, self.stop, n)
        elif self.sweep_type == 'log':
            self.f = geomspace(self.start, self.stop, n)
        else:
            raise ValueError(
                'Unable to change number of points for sweep type', self.sweep_type)

    @property
    def center(self) -> float:
        """
        Center frequency in Hz.

        Returns
        -------
        center : number
            the exact center frequency in units of Hz
        """
        return self.start + (self.stop-self.start)/2.

    @property
    def center_idx(self) -> int:
        """
        Closes idx of :attr:`f` to the center frequency.
        """
        return self.npoints // 2

    @property
    def center_scaled(self) -> float:
        """
        Center frequency in :attr:`unit`'s.

        Returns
        -------
        center : number
            the exact center frequency in units of :attr:`unit`'s
        """
        return self.start_scaled + (self.stop_scaled-self.start_scaled)/2.

    @property
    def step(self) -> float:
        """
        The inter-frequency step size (in Hz) for evenly-spaced
        frequency sweeps

        See Also
        --------
        df : for general case
        """
        return self.span/(self.npoints-1.)

    @property
    def step_scaled(self) -> float:
        """
        The inter-frequency step size (in :attr:`unit`) for evenly-spaced
        frequency sweeps.

        See Also
        --------
        df : for general case
        """
        return self.span_scaled/(self.npoints-1.)

    @property
    def span(self) -> float:
        """
        The frequency span.
        """
        return abs(self.stop-self.start)

    @property
    def span_scaled(self) -> float:
        """
        The frequency span.
        """
        return abs(self.stop_scaled-self.start_scaled)

    @property
    def f(self) -> npy.ndarray:
        """
        Frequency vector in Hz.

        Returns
        ----------
        f : :class:`numpy.ndarray`
            The frequency vector  in Hz

        See Also
        ----------
        f_scaled : frequency vector in units of :attr:`unit`
        w : angular frequency vector in rad/s
        """

        return self._f
    
    @f.setter
    def f(self,new_f: NumberLike) -> None:
        """
        Sets the frequency object by passing a vector in Hz.

        Raises
        ------
        InvalidFrequencyWarning:
            If frequency points are not monotonously increasing
        """
        warnings.warn('Possibility to set the f parameter will removed in the next release.',
             DeprecationWarning, stacklevel=2)
        
        self._f = npy.array(new_f)

        self.check_monotonic_increasing()



    @property
    def f_scaled(self) -> npy.ndarray:
        """
        Frequency vector in units of :attr:`unit`.

        Returns
        -------
        f_scaled : numpy.ndarray
            A frequency vector in units of :attr:`unit`

        See Also
        --------
        f : frequency vector in Hz
        w : frequency vector in rad/s
        """
        return self.f/self.multiplier

    @property
    def w(self) -> npy.ndarray:
        r"""
        Angular frequency in radians/s.
        
        Angular frequency is defined as :math:`\omega=2\pi f` [#]_

        Returns
        -------
        w : :class:`numpy.ndarray`
            Angular frequency in rad/s
            
        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Angular_frequency

        See Also
        --------
        f_scaled : frequency vector in units of :attr:`unit`
        f : frequency vector in Hz
        """
        return 2*pi*self.f

    @property
    def df(self) -> npy.ndarray:
        """
        The gradient of the frequency vector.
        
        Note
        ----
        The gradient is calculated using::

            `gradient(self.f)`

        """
        return gradient(self.f)

    @property
    def df_scaled(self) -> npy.ndarray:
        """
        The gradient of the frequency vector (in unit of :attr:`unit`).

        Note
        ----
        The gradient is calculated using::

            `gradient(self.f_scaled)`
        """
        return gradient(self.f_scaled)

    @property
    def dw(self) -> npy.ndarray:
        """
        The gradient of the frequency vector (in radians).

        Note
        ----
        The gradient is calculated using::

            `gradient(self.w)`
        """
        return gradient(self.w)

    @property
    def unit(self) -> str:
        """
        Unit of this frequency band.

        Possible strings for this attribute are:
        'hz', 'khz', 'mhz', 'ghz', 'thz'

        Setting this attribute is not case sensitive.

        Returns
        -------
        unit : string
            lower-case string representing the frequency units
        """
        return self.unit_dict[self._unit]

    @unit.setter
    def unit(self, unit: str) -> None:
        self._unit = unit.lower()

    @property
    def multiplier(self) -> float:
        """
        Multiplier for formatting axis.

        This accesses the internal dictionary `multiplier_dict` using
        the value of :attr:`unit`

        Returns
        -------
        multiplier : number
            multiplier for this Frequencies unit
        """
        return self.multiplier_dict[self._unit]

    def copy(self) -> 'Frequency':
        """
        Returns a new copy of this frequency.
        """
        freq =  Frequency.from_f(self.f, unit='hz')
        freq.unit = self.unit
        return freq

    @property
    def t(self) -> npy.ndarray:
        """
        Time vector in s.

        t_period = 1/f_step
        """
        return linspace(-.5/self.step , .5/self.step, self.npoints)

    @property
    def t_ns(self) -> npy.ndarray:
        """
        Time vector in ns.

        t_period = 1/f_step
        """
        return self.t*1e9

    def round_to(self, val: Union[str, Number] = 'hz') -> None:
        """
        Round off frequency values to a specified precision.

        This is useful for dealing with finite precision limitations of
        VNA's and/or other software

        Parameters
        ----------
        val : string or number
            if val is a string it should be a frequency :attr:`unit`
            (ie 'hz', 'mhz',etc). if its a number, then this returns
            f = f-f%val

        Examples
        --------
        >>> f = skrf.Frequency.from_f([.1,1.2,3.5],unit='hz')
        >>> f.round_to('hz')

        """
        if isinstance(val, str):
            val = self.multiplier_dict[val.lower()]

        self.f = npy.round_(self.f/val)*val

    def overlap(self,f2: 'Frequency') -> 'Frequency':
        """
        Calculates overlapping frequency  between self and f2.

        See Also
        --------
        overlap_freq

        """
        return overlap_freq(self, f2)

    @property
    def sweep_type(self) -> str:
        """
        Frequency sweep type.

        Returns
        -------
        sweep_type: str
            'lin' if linearly increasing, 'log' or 'unknown'.

        """
        if npy.allclose(self.f, linspace(self.f[0], self.f[-1], self.npoints)):
            sweep_type = 'lin'
        elif self.f[0] and npy.allclose(self.f, geomspace(self.f[0], self.f[-1], self.npoints)):
            sweep_type = 'log'
        else:
            sweep_type = 'unknown'
        return sweep_type


def overlap_freq(f1: 'Frequency',f2: 'Frequency') -> Frequency:
    """
    Calculates overlapping frequency between f1 and f2.

    Or, put more accurately, this returns a Frequency that is the part
    of f1 that is overlapped by f2. The resultant start frequency is
    the smallest f1.f that is greater than f2.f.start, and the stop
    frequency is the largest f1.f that is smaller than f2.f.stop.

    This way the new frequency overlays onto f1.


    Parameters
    ----------
    f1 : :class:`Frequency`
        a frequency object
    f2 : :class:`Frequency`
        a frequency object

    Returns
    -------
    f3 : :class:`Frequency`
        part of f1 that is overlapped by f2

    """
    if f1.start > f2.stop:
        raise ValueError('Out of bounds. f1.start > f2.stop')
    elif f2.start > f1.stop:
        raise ValueError('Out of bounds. f2.start > f1.stop')


    start = max(f1.start, f2.start)
    stop = min(f1.stop, f2.stop)
    f = f1.f[(f1.f>=start) & (f1.f<=stop)]
    freq =  Frequency.from_f(f, unit = 'hz')
    freq.unit = f1.unit
    return freq
