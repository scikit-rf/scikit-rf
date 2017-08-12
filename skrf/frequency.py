

'''
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
=============

.. autosummary::
    :toctree: generated/

    overlap_freq

'''

# from matplotlib.pyplot import gca,plot, autoscale
from numpy import pi, linspace, logspace
import numpy as npy
from numpy import fft, shape, gradient# used to center attribute `t` at 0
import re
from .util import slice_domain,find_nearest_index
#from .constants import ZERO
global ZER0
ZERO=1e-4 # currently needed to allow frequency __eq__ method to work
# comparing 1e-4hz is very small for most applications


class Frequency(object):
    '''
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
    '''
    unit_dict = {\
            'hz':'Hz',\
            'khz':'KHz',\
            'mhz':'MHz',\
            'ghz':'GHz',\
            'thz':'THz'\
            }
    multiplier_dict={
            'hz':1,\
            'khz':1e3,\
            'mhz':1e6,\
            'ghz':1e9,\
            'thz':1e12\
            }


    def __init__(self,start=0, stop=0, npoints=0, unit='ghz', sweep_type='lin'):
        '''
        Frequency initializer.

        Creates a Frequency object from start/stop/npoints and a unit.
        Alternatively, the class method :func:`from_f` can be used to
        create a Frequency object from a frequency vector instead.

        Parameters
        ----------
        start : number
                start frequency in  units of `unit`
        stop : number
                stop frequency in  units of `unit`
        npoints : int
                number of points in the band.
        unit : ['hz','khz','mhz','ghz']
                frequency unit of the band. This is used to create the
                attribute :attr:`f_scaled`. It is also used by the
                :class:`~skrf.network.Network` class for plots vs.
                frequency.
        sweep_type : ['lin', 'log']
                type of the sweep. 'lin' for linear and 'log' for logarithmic.

        Notes
        --------
        The attribute unit sets the property freqMultiplier, which is used
        to scale the frequency when f_scaled is referenced.

        See Also
        ---------
                from_f : constructs a Frequency object from a frequency
                        vector instead of start/stop/npoints.

        Examples
        ---------

        >>> wr1p5band = Frequency(500,750,401, 'ghz')



        '''
        self._unit = unit.lower()
        self.sweep_type = sweep_type

        start =  self.multiplier * start
        stop = self.multiplier * stop

        if sweep_type.lower() == 'lin':
            self.f = linspace(start, stop, npoints)
        elif sweep_type.lower() == 'log':
            self.f = logspace(npy.log10(start), npy.log10(stop), npoints)
        else:
            raise ValueError('Sweep Type not recognized')

    def __str__(self):
        '''
        '''
        try:
            output =  \
                   '%s-%s %s, %i pts' % \
                   (self.f_scaled[0], self.f_scaled[-1], self.unit, self.npoints)
        except (IndexError):
            output = "[no freqs]"

        return output

    def __repr__(self):
        '''
        '''
        return self.__str__()

    def __getitem__(self,key):
        '''
        Slices a Frequency object based on an index, or human readable string

        Parameters
        -----------
        key : str, or slice
            if int; then it is interpreted as the index of the frequency
            if str, then should be like '50.1-75.5ghz', or just '50'.
            If the frequency unit is omited then self.frequency.unit is
            used.

        Examples
        -----------
        >>> b = rf.Frequency(50,100,101,'ghz')
        >>> a = b['80-90ghz']
        >>> a.plot_s_db()
        '''

        output = self.copy()


        if isinstance(key, str):

            # they passed a string try and do some interpretation
            re_numbers = re.compile('.*\d')
            re_hyphen = re.compile('\s*-\s*')
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
                output.f = npy.array(output.f[slicer]).reshape(-1)
                return output
            except(IndexError):
                raise IndexError('slicing frequency is incorrect')


        if output.f.shape[0] > 0:
            output.f = npy.array(output.f[key]).reshape(-1)
        else:
            output.f = npy.empty(shape=(0))

        return output


    @classmethod
    def from_f(cls,f, *args,**kwargs):
        '''
        Construct Frequency object from a frequency vector.

        The unit is set by kwarg 'unit'

        Parameters
        -----------
        f : array-like
                frequency vector

        *args, **kwargs : arguments, keyword arguments
                passed on to  :func:`__init__`.

        Returns
        --------
        myfrequency : :class:`Frequency` object
                the Frequency object

        Examples
        -----------
        >>> f = npy.linspace(75,100,101)
        >>> rf.Frequency.from_f(f, unit='ghz')
        '''
        temp_freq =  cls(0,0,0,*args, **kwargs)
        temp_freq.f = npy.array(f) * temp_freq.multiplier
        return temp_freq

    def __eq__(self, other):
        #return (list(self.f) == list(other.f))
        # had to do this out of practicality
        if len(self.f) != len(other.f):
            return False
        elif len(self.f) == len(other.f) == 0:
            return True
        else:
            return (max(abs(self.f-other.f)) < ZERO)

    def __ne__(self,other):
        return (not self.__eq__(other))

    def __len__(self):
        '''
        The number of frequeny points
        '''
        return self.npoints

    def __mul__(self,other):
        out = self.copy()
        out.f = self.f*other
        return out

    def __rmul__(self,other):
        out = self.copy()
        out.f = self.f*other
        return out

    def __div__(self,other):
        out = self.copy()
        out.f = self.f/other
        return out

    @property
    def start(self):
        '''
        starting frequency in Hz
        '''
        return self.f[0]

    @property
    def start_scaled(self):
        '''
        starting frequency in :attr:`unit`'s
        '''
        return self.f_scaled[0]
    @property
    def stop_scaled(self):
        '''
        stop frequency in :attr:`unit`'s
        '''
        return self.f_scaled[-1]


    @property
    def stop(self):
        '''
        stop frequency in Hz
        '''
        return self.f[-1]

    @property
    def npoints(self):
        '''
        number of points in the frequency
        '''
        return len(self.f)

    @npoints.setter
    def npoints(self, n):
        '''
        set the number of points in the frequency
        '''
        self.f = linspace(self.start, self.stop, n)



    @property
    def center(self):
        '''
        Center frequency in Hz

        Returns
        ---------
        center : number
                the exact center frequency in units of hz
        '''
        return self.start + (self.stop-self.start)/2.

    @property
    def center_idx(self):
        '''
        closes idx of :attr:`f` to the center frequency
        '''
        return int(self.npoints)/2

    @property
    def center_scaled(self):
        '''
        Center frequency in :attr:`unit`'s

        Returns
        ---------
        center : number
                the exact center frequency in units of :attr:`unit`'s
        '''
        return self.start_scaled + (self.stop_scaled-self.start_scaled)/2.

    @property
    def step(self):
        '''
        the inter-frequency step size (in hz) for evenly-spaced
        frequency sweeps

        see `df` for general case
        '''
        return self.span/(self.npoints-1.)

    @property
    def step_scaled(self):
        '''
        the inter-frequency step size (in self.unit) for evenly-spaced
        frequency sweeps

        see `df` for general case
        '''
        return self.span_scaled/(self.npoints-1.)

    @property
    def span(self):
        '''
        the frequency span
        '''
        return abs(self.stop-self.start)

    @property
    def span_scaled(self):
        '''
        the frequency span
        '''
        return abs(self.stop_scaled-self.start_scaled)

    @property
    def f(self):
        '''
        Frequency vector  in Hz

        Returns
        ----------
        f :  :class:`numpy.ndarray`
                The frequency vector  in Hz

        See Also
        ----------
                f_scaled : frequency vector in units of :attr:`unit`
                w : angular frequency vector in rad/s
        '''

        return self._f

    @f.setter
    def f(self,new_f):
        '''
        sets the frequency object by passing a vector in Hz
        '''
        self._f = npy.array(new_f)



    @property
    def f_scaled(self):
        '''
        Frequency vector in units of :attr:`unit`

        Returns
        ---------
        f_scaled :  :class:`numpy.ndarray`
                A frequency vector in units of :attr:`unit`

        See Also
        ---------
                f : frequency vector in Hz
                w : frequency vector in rad/s
        '''
        return self.f/self.multiplier

    @property
    def w(self):
        '''
        Frequency vector in radians/s

        The frequency vector  in rad/s

        Returns
        ----------
        w :  :class:`numpy.ndarray`
                The frequency vector  in rad/s

        See Also
        ----------
                f_scaled : frequency vector in units of :attr:`unit`
                f :  frequency vector in Hz
        '''
        return 2*pi*self.f

    @property
    def df(self):
        '''
        the gradient of the frequency vector (in hz)
        '''
        return gradient(self.f)
    @property
    def df_scaled(self):
        '''
        the gradient of the frequency vector (in self.unit)
        '''
        return gradient(self.f_scaled)
    @property
    def dw(self):
        '''
        the gradient of the frequency vector (in radians)
        '''
        return gradient(self.w)

    @property
    def unit(self):
        '''
        Unit of this frequency band.

        Possible strings for this attribute are:
         'hz', 'khz', 'mhz', 'ghz', 'thz'

        Setting this attribute is not case sensitive.

        Returns
        ---------
        unit : string
                lower-case string representing the frequency units
        '''
        return self.unit_dict[self._unit]

    @unit.setter
    def unit(self,unit):
        self._unit = unit.lower()

    @property
    def multiplier(self):
        '''
        Multiplier for formatting axis

        This accesses the internal dictionary `multiplier_dict` using
        the value of :attr:`unit`

        Returns
        ---------
        multiplier : number
                multiplier for this Frequencies unit
        '''
        return self.multiplier_dict[self._unit]

    def copy(self):
        '''
        returns a new copy of this frequency
        '''
        freq =  Frequency.from_f(self.f, unit='hz')
        freq.unit = self.unit
        return freq

    @property
    def t(self):
        '''
        time vector in s.

        t_period = 1/f_step
        '''
        return linspace(-.5/self.step , .5/self.step, self.npoints)

    @property
    def t_ns(self):
        '''
        time vector in ns.

        t_period = 1/f_step
        '''
        return self.t*1e9

    def round_to(self, val = 'hz'):
        '''
        Round off frequency values to a specified precision.

        This is useful for dealing with finite precision limitations of
        VNA's and/or other software

        Parameters
        -----------
        val : string or number
            if val is a string it should  be a frequency unit
            (ie 'hz', 'mhz',etc). if its a number, then this returns
            f = f-f%val

        Examples
        ---------
        >>>f = skrf.Frequency.from_f([.1,1.2,3.5],unit='hz')
        >>>f.round_to('hz')

        '''
        if isinstance(val, str):
            val = self.multiplier_dict[val.lower()]

        self.f = npy.round_(self.f/val)*val

    def overlap(self,f2):
        '''
        Calculates overlapping frequency  between self and f2

        See Also
        ---------

        overlap_freq

        '''
        return overlap_freq(self, f2)


def overlap_freq(f1,f2):
    '''
    Calculates overlapping frequency between f1 and f2.

    Or, put more accurately, this returns a Frequency that is the part
    of f1 that is overlapped by f2. The resultant start frequency is
    the smallest f1.f that is greater than f2.f.start, and the stop
    frequency is the largest f1.f that is smaller than f2.f.stop.

    This way the new frequency overlays onto f1.


    Parameters
    ------------
    f1 : :class:`Frequency`
        a  frequency object
    f2 : :class:`Frequency`
        a  frequency object

    Returns
    ----------
    f3 : :class:`Frequency`
        part of f1 that is overlapped by f2

    '''
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


def f_2_frequency(f):
    '''
    converts a frequency vector to a Frequency object

    Depricated
    -------------
    Use the class method :func:`Frequency.from_f`
    convenience function


    !deprecated, use classmethod from_f instead.
    '''
    return Frequency(start=f[0], stop=f[-1],npoints = len(f), unit='hz')
