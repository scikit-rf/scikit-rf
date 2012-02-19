
#       frequency.py
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
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

'''

from pylab import linspace, gca
from numpy import pi

class Frequency(object):
    '''
    A frequency band.

    The frequency object provides a convenient way to work with and
    access a frequency band. It contains  a fruequency vector as well as
    a frequency unit. This allows a frequency vector in a given unit
    to be available (:attr:`f_scaled`), as well as an absolute frquency
    axis in 'Hz'  (:attr:`f`).
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
    def __init__(self,start, stop, npoints, unit='hz', sweep_type='lin'):
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
        self.start =  self.multiplier * start
        self.stop = self.multiplier * stop
        self.npoints = npoints
        self.sweep_type = sweep_type

    @classmethod
    def from_f(cls,f, *args,**kwargs):
        '''
        Alternative constructor of a Frequency object from a frequency
        vector,

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
        >>> f = np.linspace(75,100,101)
        >>> rf.Frequency.from_f(f, unit='ghz')
        '''
        return cls(start=f[0], stop=f[-1],npoints = len(f), *args, **kwargs)

    def __eq__(self, other):
        return (list(self.f) == list(other.f))

    def __ne__(self,other):
        return (not self.__eq__(other))

    @property
    def center(self):
        '''
        Center frequency.

        Returns
        ---------
        center : number
                the exact center frequency in units of :attr:`unit`
        '''
        return self.start + (self.stop-self.start)/2.

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
        return linspace(self.start,self.stop,self.npoints)
        #return self._f

    @f.setter
    def f(self,new_f):
        '''
        sets the frequency object by passing a vector in Hz
        '''
        #self._f = new_f
        self.start = new_f[0]
        self.stop = new_f[-1]
        self.npoints = len(new_f)

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
        Multiplier for formating axis

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
        return Frequency(
            start = self.start/self.multiplier, 
            stop = self.stop/self.multiplier, 
            npoints = self.npoints, 
            unit = self.unit, 
            sweep_type = self.sweep_type)
        
    def labelXAxis(self, ax=None):
        '''
        Label the x-axis of a plot.

        Sets the labels of a plot using :func:`matplotlib.x_label` with
        string containing the frequency  unit.

        Parameters
        ---------------
        ax : :class:`matplotlib.Axes`, optional
                Axes on which to label the plot, defaults what is
                returned by :func:`matplotlib.gca()`
        '''
        if ax is None:
            ax = gca()
        ax.set_xlabel('Frequency [%s]' % self.unit )

def f_2_frequency(f):
    '''
    converts a frequency vector to a Frequency object

    Depricated
    -------------
    Use the class method :func:`Frequency.from_f`
    convienience function


    !depricated, use classmethod from_f instead.
    '''
    return Frequency(start=f[0], stop=f[-1],npoints = len(f), unit='hz')
