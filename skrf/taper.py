"""
taper (:mod:`skrf.taper`)
=========================

Taper Objects

Tapered transformers, or tapers in short, are used to match
one impedance to another (from Z1 to Z2) [#]_.

By tapering a transmission line, a very broadband impedance match (low VSWR)
can be realized over a wide bandwidth, the longer the taper, the wider the frequency band.

References
----------
.. [#] https://www.microwaves101.com/encyclopedias/tapered-transformers

.. autosummary::
    :toctree: generated/

    Taper1D
    Linear
    Exponential
    SmoothStep
    Klopfenstein

"""

from numbers import Number
from typing import Callable, List
from . network import cascade_list
from scipy  import linspace
from numpy import exp, log


class Taper1D:
    """
    Generic 1D Taper Object
    """
    def __init__(self, med, start: Number, stop: Number, n_sections: int, f: Callable,
                 length: Number, length_unit: str = 'm', param: str = 'z0', f_is_normed: bool = True,
                 med_kw: dict = {}, f_kw: dict = {}):
        """
        Generic 1D Taper Constructor.

        Parameters
        ----------
        med : :class:`~skrf.media.media.Media`
            A :class:`~skrf.media.media.Media` object or a `@classmethod` `__init__`,
            used to generate the transmission line.
            See `med_kw` for arguments.
            Examples:

            * `skrf.media.RectangularWaveguide`  # a class
            * `skrf.media.RectangularWaveguide.from_z0`  # an init

        start : number
            starting value for `param`
        stop : number
            stop value for `param`
        n_sections : int
            number of sections in taper
        f : Callable
            function defining the taper transition. must take either
            no arguments or take `(x,length, start, stop)`.
            see `f_is_normed` arguments
        length : number
            physical length of the taper (in `length_unit`)
        length_unit : str
            unit of length variable. see `skrf.to_meters`
        param : str
            name of the parameter of `med` that varies along the taper
        f_is_normed: bool
            is `f` scalable and normalized. ie can f just be scaled
            to fit different start/stop conditions? if so then f is
            called with no arguments, and must  have domain and raings
            of [0,1], and [0,1]
        f_kw : dict
            passed to `f()` when  called
        med_kw : dict
            passed to `med.__init__` when an instance is created

        Note
        ----
        The default behaviour should is to taper based on impedance.
        To do this we inspect the `med` class for a `from_z0`
        init method, and if it exists, we assign it to `med` attribute,
        in `__init__`.
        Admittedly having `med` be a class or a method is abuse,
        it makes for a intuitive operation.

        Examples
        --------
        Create a linear taper from 100 to 1000mil

        >>> from skrf import Frequency, RectangularWaveguide, Taper1D, mil, inch
        >>> taper = Taper1D(med=RectangularWaveguide,
                            param='a',
                            start=100*mil,
                            stop=1000*mil,
                            length=1*inch,
                            n_sections=20,
                            f=lambda x: x,
                            f_is_normed=True,
                            med_kw={'frequency':Frequency(75,110,101,'ghz')})
        """
        self.med = med
        self.param = param
        self.start = start
        self.stop = stop
        self.f = f
        self.f_is_normed = f_is_normed
        self.length = length
        self.length_unit = length_unit
        self.n_sections = n_sections
        self.med_kw = med_kw
        self.f_kw = f_kw

        # the default behaviour should be to taper based on impedance.
        # to do this we inspect the media class for a `from_z0`
        # init method, and if it exists, we assign it to `med` attribute
        # admittedly having `med` be a class or a method is abuse,
        # it makes for a intuitive operation
        if param == 'z0':
            if hasattr(self.med, 'from_z0'):
                self.med = getattr(self.med, 'from_z0')

    def __str__(self) -> str:
        return 'Taper: {classname}: {param} from {start}-{stop}'


    @property
    def section_length(self) -> Number:
        """
        Section length.

        Returns
        -------
        l : number
        """
        return  1.0*self.length/self.n_sections

    @property
    def value_vector(self):
        if self.f_is_normed ==True:
            x = linspace(0,1,self.n_sections)
            y = self.f(x, **self.f_kw)*(self.stop-self.start) + self.start
        else:
            x = linspace(0,self.length,self.n_sections)
            y = self.f(x,self.length, self.start, self.stop, **self.f_kw)
        return y

    def media_at(self, val: Number):
        """
        Create a media instance for the taper with parameter value `val`.

        Parameters
        ----------
        val : number
            parameter value

        Returns
        -------
        media : :class:`~skrf.media.media.Media`
            media instance for the taper for the given parameter value
        """
        med_kw = self.med_kw.copy()
        med_kw.update({self.param: val})
        return self.med(**med_kw)

    def section_at(self, val: Number):
        """
        Create a single section of the taper with parameter value `val`.

        Parameters
        ----------
        val : number
            parameter value

        Returns
        -------
        media : :class:`~skrf.network.Network`
            Network instance for the section of the taper
            for the given parameter value
        """
        return self.media_at(val).line(self.section_length,
                                       unit=self.length_unit)

    @property
    def medias(self) -> List:
        """
        List of medias.

        Returns
        -------
        medias : list of :class:`~skrf.media.media.Media`
        """
        return [self.media_at(k) for k in self.value_vector]

    @property
    def sections(self) -> List:
        """
        List of sections.

        Returns
        -------
        sections : list of :class:`~skrf.network.Network`
        """
        return [self.section_at(k) for k in self.value_vector]

    @property
    def network(self):
        """
        Resulting Network

        Returns
        -------
        ntwk : :class:`~skrf.network.Network`
        """
        return cascade_list(self.sections)



class Linear(Taper1D):
    """
    A linear Taper.

    Defined by :math:`f(x)=x`

    """
    def __init__(self, **kw):
        opts = dict(f=lambda x:x, f_is_normed=True)
        kw.update(opts)
        super().__init__(**kw)


class Exponential(Taper1D):
    r"""
    An Exponential Taper.

    Defined by :math:`f(x) = f_0 \exp\left[ \frac{x}{x_1}  \ln\left( \frac{f_1}{f_0} \right) \right]`

    where:

    *    :math:`f_0`: start param value
    *    :math:`f_1`: stop param value
    *    :math:`x`: independent variable (position along taper)
    *    :math:`x_1`: length of taper

    """
    def __init__(self, **kw):
        """
        Exponential Taper Constructor
        """

        def f(x, length, start, stop):
            return start*exp(x/length*(log(stop/start)))

        opts = dict(f=f, f_is_normed=False)
        kw.update(opts)
        super().__init__(**kw)


class SmoothStep(Taper1D):
    """
    A smoothstep Taper.

    There is no analytical basis for this in the EE world that I know
    of. it is just a reasonable smooth curve, that is easy to implement.

    :math:`f(x) = (3 x^2 - 2x^3)`

    References
    ----------
    https://en.wikipedia.org/wiki/Smoothstep

    """
    def __init__(self, **kw):
        """
        Smoothstep Taper Constructor.
        """

        f = lambda x:  3*x**2 - 2*x**3
        opts = dict(f=f, f_is_normed=True)
        kw.update(opts)
        super().__init__(**kw)


class Klopfenstein(Taper1D):
    """
    Klopfenstein Taper.

    This impedance taper was first described by R. W. Klopfenstein [#microwaves101]_ in a paper titled
    "A Transmission Line Taper of Improved Design", published in 1956 [#Klopfenstein]_ .
    A correction to Klopfenstein's math was published by in May 1973 by Darko Kajfez and Jame Prewit [#Kajfez]_ .

    References
    ----------
    .. [#microwaves101] https://www.microwaves101.com/encyclopedias/klopfenstein-taper
    .. [#Klopfenstein] R. W. Klopfenstein Proc. IRE. vol. 44 pp. 31-35 Jan. 1956.
        doi: 10.1109/JRPROC.1956.274847
        https://ieeexplore.ieee.org/document/4051841
    .. [#Kajfez] D. Kajfez and J. O. Prewitt, "Correction to "A Transmission Line Taper of Improved Design" (Letters),"
        in IEEE Transactions on Microwave Theory and Techniques, vol. 21, no. 5, pp. 364-364, May 1973, doi: 10.1109/TMTT.1973.1128003.
        https://ieeexplore.ieee.org/document/1128003
    """
    def __init__(self, **kw):
        raise NotImplementedError('Not Yet Implemented')