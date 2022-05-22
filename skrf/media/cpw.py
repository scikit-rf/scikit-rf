

"""
cpw (:mod:`skrf.media.cpw`)
========================================

.. autosummary::
   :toctree: generated/

   CPW

"""
from scipy.constants import  epsilon_0, mu_0, c, pi
from scipy.special import ellipk
from numpy import sqrt, log, zeros, ones, any, tanh, sinh, exp, real
from .media import Media
from ..tlineFunctions import surface_resistivity, skin_depth
from ..constants import INF, NumberLike
from typing import Union, TYPE_CHECKING
import warnings

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
                 w: NumberLike = 3e-3, s: NumberLike = 0.3e-3,
                 h: NumberLike = 1.55,
                 ep_r: NumberLike = 3, t: Union[NumberLike, None] = None,
                 rho: Union[NumberLike, None] = 1.68e-8, tand: NumberLike = 0,
                 has_metal_backside: bool = False,
                 *args, **kwargs):
        Media.__init__(self, frequency=frequency,z0=z0)

        self.w, self.s, self.h, self.t, self.ep_r, self.tand, self.rho =\
                w, s, h, t, ep_r, tand, rho
        self.has_metal_backside = has_metal_backside
        
        # quasi-static effective permittivity of substrate + line and
        # the impedance of the microstrip line
        self.zl_eff, self.ep_reff, k1, kk1, kpk1 = self.analyse_quasi_static(
            ep_r, w, s, h, t, has_metal_backside)
        
        # analyse dispersion of impedance and relatice permittivity
        self._z_characteristic, self.ep_reff_f = self.analyse_dispersion(
            self.zl_eff, self.ep_reff, ep_r, w, s, h, self.frequency.f)
        
        self.alpha_conductor, self.alpha_dielectric = self.analyse_loss(
            ep_r, real(self.ep_reff_f), tand, rho, 1., self.frequency.f,
            w, t, s, k1, kk1, kpk1)

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
    def Z0(self) -> NumberLike:
        """
        Characteristic impedance
        """
        return self._z_characteristic

    @property
    def gamma(self) -> NumberLike:
        """
        Propagation constant.
        
        Returns
        -------
        gamma : :class:`numpy.ndarray`
        """
        ep_reff, f = real(self.ep_reff_f), self.frequency.f
        
        alpha = self.alpha_dielectric.copy()
        if self.rho is not None:
            alpha += self.alpha_conductor
            
        beta = 2. * pi * f * sqrt(ep_reff) / c

        return alpha + 1j * beta
    
    def analyse_quasi_static(self, ep_r: NumberLike, 
                           w: NumberLike, s: NumberLike,
                           h: NumberLike, t: NumberLike,
                           has_metal_backside: bool):
        """
        This function calculates the quasi-static impedance of a CPW
        line, the value of the effective permittivity as per filling factor
        and the effective width due to the finite conductor thickness for the
        given CPW line and substrate properties.
        
        References
        ----------
        .. [#] Ghione ...
            
        Returns
        -------
        zl_eff : :class:`numpy.ndarray`
        ep_reff : :class:`numpy.ndarray`
        """
        Z0 = sqrt(mu_0 / epsilon_0)
        k1 = w / (w + s + s)
        kk1 = ellipk(k1)
        kpk1 = ellipk(sqrt(1. - k1 * k1))
        q1 = kk1 / kpk1
        
        # backside is metal
        if(has_metal_backside):
            k3 = tanh((pi / 4.) * (w / h)) / tanh((pi / 4.) * (w + s + s) / h)
            q3 = ellipk(k3) / ellipk(sqrt(1. - k3 * k3))
            qz = 1. / (q1 + q3)
            e = 1. + q3 * qz * (ep_r - 1.)
            zr = Z0 / 2. * qz
            
        # backside is air
        else:
            k2 = sinh((pi / 4.) * (w / h)) / sinh((pi / 4.) * (w + s + s) / h)
            q2 = ellipk(k2) / ellipk(sqrt(1. - k2 * k2))
            e = 1. + (ep_r - 1.) / 2. * q2 / q1
            zr = Z0 / 4. / q1
            
        # effect of strip thickness
        if t is not None and t > 0.:
            d = (t * 1.25 / pi) * (1. + log(4. * pi * w / t))
            ke = k1 + (1. - k1 * k1) * d / 2. / s
            qe = ellipk(ke) / ellipk(sqrt(1. - ke * ke))
            
            # backside is metal
            if(has_metal_backside):
                qz = 1. / (qe + q3)
                zr = Z0 / 2. * qz
            # backside is air
            else:
                zr = Z0 / 4. / qe
                
            # modifies ep_re
            e = e - (0.7 * (e - 1.) * t / s) / (q1 + (0.7 * t / s))
            
        
        ep_reff = e
        zl_eff = zr / sqrt(ep_reff)
        
        return zl_eff, ep_reff, k1, kk1, kpk1
    
    def analyse_dispersion(self, zl_eff: NumberLike, ep_reff: NumberLike,
                          ep_r: NumberLike, w: NumberLike, s: NumberLike,
                          h: NumberLike, f: NumberLike):
         """
         This function compute the frequency dependent characteristic
         impedance and effective permittivity accounting for microstripline
         frequency dispersion.
         
         References
         ----------
         .. [#] Ghione...
             
         Returns
         -------
         z : :class:`numpy.ndarray`
         e : :class:`numpy.ndarray`
         """
         # cut-off frequency of the TE0 mode
         fte = ((c / 4.) / (h * sqrt(ep_r - 1.)))
         
         # dispersion factor G
         p = log(w / h)
         u = 0.54 - (0.64 - 0.015 * p) * p
         v = 0.43 - (0.86 - 0.54 * p) * p
         G = exp(u * log(w / s) + v)
         
         # add the dispersive effects to ep_reff
         sqrt_ep_reff = sqrt(ep_reff)
         sqrt_e = sqrt_ep_reff + (sqrt(ep_r) - sqrt_ep_reff) / \
             (1. + G * (f / fte)**(-1.8))
             
         e = sqrt_e**2
             
         z = zl_eff * sqrt_ep_reff / sqrt_e
         
         return z, e
     
    def analyse_loss(self, ep_r: NumberLike, ep_reff: NumberLike, 
                    tand: NumberLike, rho: NumberLike, mu_r: NumberLike,
                    f: NumberLike, w: NumberLike, t: NumberLike, s: NumberLike,
                    k1: NumberLike, kk1: NumberLike, kpk1: NumberLike):
        """
        The function calculates the conductor and dielectric losses of a
        single microstrip line using wheeler's incremental inductance rule.
        
        References
        ----------
        .. [#] H. A. Wheeler, "Formulas for the Skin Effect,"
            Proceedings of the IRE, vol. 30, no. 9, pp. 412-424, Sept. 1942.
            
        Returns
        -------
        a_conductor : :class:`numpy.ndarray`
        a_dielectric : :class:`numpy.ndarray`
        """
        Z0 = sqrt(mu_0 / epsilon_0)
        if t is not None and t > 0.:
            if rho is None:
                raise(AttributeError('must provide values conductivity and conductor thickness to calculate this. see initializer help'))
            r_s = surface_resistivity(f=f, rho=rho, \
                    mu_r=1)
            ds = skin_depth(f = f, rho = rho, mu_r = 1.)
            if(any(t < 3 * ds)):
                warnings.warn(
                    'Conductor loss calculation invalid for line'
                    'height t ({})  < 3 * skin depth ({})'.format(t, ds[0]),
                    RuntimeWarning
                    )
            n = (1. - k1) * 8. * pi / (t * (1. + k1))
            a = w / 2.
            b = a + s
            ac = (pi + log(n * a)) / a + (pi + log(n * b)) / b
            a_conductor = r_s * sqrt(ep_reff) * ac / (4. * Z0 * kk1 * kpk1 * \
                               (1. - k1 * k1)) 
        else:
            a_conductor = zeros(f.shape)
        
        l0 = c / f
        a_dielectric =  pi * ep_r / (ep_r - 1) * (ep_reff - 1) / \
            sqrt(ep_reff) * tand / l0
            
        return a_conductor, a_dielectric
