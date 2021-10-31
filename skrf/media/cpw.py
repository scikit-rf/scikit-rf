

"""
cpw (:mod:`skrf.media.cpw`)
========================================

.. autosummary::
   :toctree: generated/

   CPW

"""
from scipy.constants import  epsilon_0, mu_0
from scipy.special import ellipk
from numpy import pi, sqrt, log, zeros, ones
from .media import Media
from ..tlineFunctions import surface_resistivity
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .. frequency import Frequency


class CPW(Media):
    """
    A Coplanar Waveguide.

    This class was made from the technical documentation [#]_ provided
    by the qucs project [#]_ .
    The variables  and properties of this class are coincident with
    their derivations.

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object, optional
        frequency band of the media. The default is None.
    z0 : number, array-like, optional
        the port impedance for media. The default is None.
        Only needed if  its different from the characteristic impedance
        of the transmission.
    w : number, or array-like, optional
        width of center conductor, in m. Default is 70.
    s : number, or array-like
        width of gap, in m. Default is 4.
    ep_r : number, or array-like, optional
        relative permativity of substrate. Default is 3.
    t : number, or array-like, optional
        conductor thickness, in m.
        Default is None (metalization thickness neglected)
    rho : number, or array-like, optional
        resistivity of conductor. Default is None

    References
    ----------
    .. [#] http://qucs.sourceforge.net/docs/technical.pdf
    .. [#] http://www.qucs.sourceforge.net/

    """
    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 w: NumberLike = 70, s: NumberLike = 4,
                 ep_r: NumberLike = 3, t: Union[NumberLike, None] = None,
                 rho: Union[NumberLike, None] = None,
                 *args, **kwargs):
        Media.__init__(self, frequency=frequency,z0=z0)

        self.w, self.s, self.ep_r, self.t, self.rho =\
                w, s, ep_r, t, rho


    def __str__(self) -> str:
        f=self.frequency
        output =  \
                'Coplanar Waveguide Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\n W= %.2em, S= %.2em'% \
                (self.w,self.s)
        return output

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def ep_re(self) -> NumberLike:
        r"""
        Effective permittivity of the CPW (also known as Keff).

        If the thickness of the dielectric substrate is large,
        the effective dielectric constant of the even mode can be approx as:

        .. math::

                \varepsilon_{eff} = \frac{\varepsilon_r + 1}{2}

        The effective permittivity can be defined as in the case of a
        microstrip line, that is as the square of ratio
        of the capacitance per unit length to the phase velocity.
        """
        return (self.ep_r+1)/2.0

    @property
    def k1(self) -> NumberLike:
        r"""
        Intermediary parameter. see qucs docs on cpw lines.

        Defined as:

        .. math::

                k = \frac{w}{w + 2s}

        """
        return self.w/(self.w +2*self.s)

    @property
    def K_ratio(self) -> NumberLike:
        """
        Intermediary parameter. see qucs docs on cpw lines.

        K_ratio is the ratio of two elliptic integrals.
        """
        k1 = self.k1
        # k prime
        k_p = sqrt(1 - k1**2)

        if (0 <= k1 <= 1/sqrt(2)):
            return pi/log(2*(1 + sqrt(k1))/(1 - sqrt(k1)))
        elif (1/sqrt(2) < k1 <= 1):
            return (log(2*(1 + sqrt(k_p))/(1 - sqrt(k_p)) ))/pi




    @property
    def alpha_conductor(self) -> NumberLike:
        """
        Losses due to conductor resistivity.

        Returns
        -------
        alpha_conductor : array-like
                lossyness due to conductor losses

        See Also
        --------
        surface_resistivity : calculates surface resistivity
        """
        if self.rho is None or self.t is None:
            raise(AttributeError('must provide values conductivity and conductor thickness to calculate this. see initializer help'))

        t, k1, ep_re = self.t, self.k1,self.ep_re
        r_s = surface_resistivity(f=self.frequency.f, rho=self.rho, \
                mu_r=1)
        a = self.w/2.
        b = self.s+self.w/2.
        K = ellipk      # complete elliptical integral of first kind
        K_p = lambda x: ellipk(sqrt(1-x**2)) # ellipk's compliment

        return ((r_s * sqrt(ep_re)/(480*pi*K(k1)*K_p(k1)*(1-k1**2) ))*\
                (1./a * (pi+log((8*pi*a*(1-k1))/(t*(1+k1)))) +\
                 1./b * (pi+log((8*pi*b*(1-k1))/(t*(1+k1))))))



    @property
    def Z0(self) -> NumberLike:
        """
        Characteristic impedance
        """
        return (30.*pi / sqrt(self.ep_re) * self.K_ratio)*ones(len(self.frequency.f), dtype='complex')

    @property
    def gamma(self) -> NumberLike:
        """
        Propagation constant.

        See Also
        --------
        alpha_conductor : calculates losses to conductors
        """
        beta = 1j*2*pi*self.frequency.f*sqrt(self.ep_re*epsilon_0*mu_0)
        alpha = zeros(len(beta))
        if self.rho is not None and self.t is not None:
            alpha = self.alpha_conductor

        return beta+alpha
