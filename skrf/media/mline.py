

"""
MLine (:mod:`skrf.media.MLine`)
========================================

.. autosummary::
   :toctree: generated/

   MLine

"""
import numpy as npy
from numpy import log, log10, tanh, sqrt, exp, real, imag, cosh, \
                            ones, zeros, arctan
from scipy.constants import  epsilon_0, mu_0, c, pi
from .media import Media
from ..tlineFunctions import skin_depth, surface_resistivity
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from .. frequency import Frequency


class MLine(Media):
    """
    Microstripline class

    This class was made from the technical documentation [#]_ and sources
    provided by the qucs project [#]_ .
    The variables  and properties of this class are coincident with
    their derivations.

    In addition, Djordjevic [#]_ /Svensson [#]_  wideband debye dielectric
    model is considered to provide more realistic modelling of broadband
    microstrip as well as causal time domain response.

    * Quasi-static characteristic impedance and dielectric constant models:

        + Hammerstad and Jensen
        + Schneider
        + Wheeler

    * Frequency dispersion of impedance and dielectric constant models:

        + Kirschning and Jansen
        + Hammerstad and Jensen
        + Yamashita
        + Kobayashi
        + No dispersion

    * Strip thickness correction model:

        + Hammerstad and Jensen, add a certain amount to W if T > 0.

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of the media
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characteristic impedance of the transmission
    w : number, or array-like
        width of conductor, in m.
    h : number, or array-like
        height of substrate between ground plane and conductor, in m.
    t : number, or array-like or None, optional
        conductor thickness, in m. Default is None.
    ep_r : number, or array-like
        relative permittivity of dielectric at frequency f_epr_tand
    mu_r : number, or array-like
        relative permeability of dielectric (assumed frequency invariant)
    model : str
        microstripline quasi-static impedance and dielectric model in:

        
        * 'hammerstadjensen' (default)
        * 'schneider'
        * 'wheeler'
        
    disp : str
        microstripline impedance and dielectric frequency dispersion model in:

        * 'hammerstadjensen'
        * 'kirschningjansen' (default)
        * 'kobayashi'
        * 'schneider'
        * 'yamashita'
        * 'none'
        
    diel : str
        dielectric frequency dispersion model in:
        
        * 'djordjevicsvensson' (default)
        * 'frequencyinvariant'
        
    rho: number, or array-like, optional
        resistivity of conductor (None)
    tand : number, or array-like
        dielectric loss factor at frequency f_epr_tand
    rough : number, or array-like
        RMS roughness of conductor in m.
    f_low : number, or array-like
        lower frequency for wideband Debye Djordjevic/Svensson dielectric model
    f_high : number, or array-like
        higher frequency for wideband Debye Djordjevic/Svensson dielectric model
    f_epr_tand : number, or array-like
        measurement frequency for ep_r and tand of dielectric
    compatibility_mode: str
        This switch the behaviour to stick with ADS or Qucs simulator
        implementations of microstriplines.
        
        * 'qucs' (default) follow QUCS behaviour
        * 'ads' : follow ADS behaviour

    References
    ----------
    .. [#] http://qucs.sourceforge.net/docs/technical.pdf
    .. [#] https://github.com/Qucs/qucsator/blob/develop/src/components/microstrip/msline.cpp
    .. [#] C. Svensson, G.E. Dermer,
        Time domain modeling of lossy interconnects,
        IEEE Trans. on Advanced Packaging, May 2001, N2, Vol. 24, pp.191-196.
    .. [#] Djordjevic, R.M. Biljic, V.D. Likar-Smiljanic, T.K. Sarkar,
        Wideband frequency-domain characterization of FR-4 and time-domain causality,
        IEEE Trans. on EMC, vol. 43, N4, 2001, p. 662-667.
    """
    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 w: NumberLike = 3, h: NumberLike = 1.6,
                 t: Union[NumberLike, None] = None,
                 ep_r: NumberLike = 4.5, mu_r: NumberLike = 1,
                 model: str = 'hammerstadjensen',
                 disp: str = 'kirschningjansen',
                 diel: str = 'djordjevicsvensson',
                 rho: NumberLike = 1.68e-8, tand: NumberLike = 0,
                 rough: NumberLike = 0.15e-6,
                 f_low: NumberLike = 1e3, f_high: NumberLike = 1e12,
                 f_epr_tand: NumberLike = 1e9,
                 compatibility_mode: str = 'qucs',
                 *args, **kwargs):
        Media.__init__(self, frequency = frequency, z0 = z0)

        self.w, self.h, self.t = w, h, t
        self.ep_r, self.mu_r = ep_r, mu_r
        self.model, self.disp, self.diel = model, disp, diel
        self.rho, self.tand, self.rough, self.disp =  rho, tand, rough, disp
        self.f_low, self.f_high, self.f_epr_tand = f_low, f_high, f_epr_tand
        self.compatibility_mode = compatibility_mode
        
        # variation of dielectric constant with frequency
        # implemented by ADS only
        self.ep_r_f, self.tand_f = self.analyse_dielectric(self.ep_r, self.tand,
            self.f_low, self.f_high, self.f_epr_tand, self.frequency.f,
            self.diel)
        
        # quasi-static effective dielectric constant of substrate + line and
        # the impedance of the microstrip line
        self.zl_eff, self.ep_reff, self.w_eff = self.analyse_quasi_static(
            real(self.ep_r_f), self.w, self.h, self.t, self.model)
        
        # analyse dispersion of Zl and Er
        # qucs use w instead of w_eff here
        if compatibility_mode == 'qucs':
            self._z_characteristic, self.ep_reff_f = self.analyse_dispersion(
                self.zl_eff, self.ep_reff, real(self.ep_r_f),
                self.w, self.h,
                self.frequency.f, self.disp)
        else:
            self._z_characteristic, self.ep_reff_f = self.analyse_dispersion(
                self.zl_eff, self.ep_reff, real(self.ep_r_f),
                self.w_eff, self.h,
                self.frequency.f, self.disp)
        
        # analyse losses of line
        # qucs use quasi-static values here, leading to a difference
        # against ads
        if compatibility_mode == 'qucs':
            self.alpha_conductor, self.alpha_dielectric = self.analyse_loss(
                real(self.ep_r_f), real(self.ep_reff), self.tand_f,
                self.rho, self.mu_r,
                real(self.zl_eff), real(self.zl_eff),
                self.frequency.f, self.w, self.t, self.rough)
        else:
            self.alpha_conductor, self.alpha_dielectric = self.analyse_loss(
                real(self.ep_r_f), real(self.ep_reff_f), self.tand_f,
                self.rho, self.mu_r,
                real(self._z_characteristic), real(self._z_characteristic),
                self.frequency.f, self.w, self.t, self.rough)

    def __str__(self) -> str:
        f=self.frequency
        output =  \
                'Microstripline Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\n W= %.2em, H= %.2em'% \
                (self.w,self.h)
        return output

    def __repr__(self) -> str:
        return self.__str__()


    @property
    def gamma(self):
        """
        Propagation constant.

        See Also
        --------
        alpha_conductor : calculates losses to conductor
        alpha_dielectric: calculates losses to dielectric
        beta            : calculates phase parameter
        """
        ep_reff, f = real(self.ep_reff_f), self.frequency.f
        
        alpha = self.alpha_dielectric.copy()
        if self.rho is not None:
            alpha += self.alpha_conductor
        
        beta  = 2 * pi * f* sqrt(ep_reff) / c
        
        return alpha + 1j*beta
    
    @property
    def Z0(self) -> npy.ndarray:
        """
        Characteristic Impedance.

        Returns
        -------
        Z0 : :class:`numpy.ndarray`
        """
        return self._z_characteristic
    
    @property
    def Z0_f(self) -> npy.ndarray:
        """
        Alias fos Characteristic Impedance for backward compatibility.
        Deprecated, do not use.

        Returns
        -------
        Z0 : :class:`numpy.ndarray`
        """
        warnings.warn(
            "`Z0_f` is deprecated, use `Z0` instead",
             DeprecationWarning
        )
        return self._z_characteristic
    
    def analyse_dielectric(self, ep_r: NumberLike, tand: NumberLike,
                          f_low: NumberLike, f_high: NumberLike,
                          f_epr_tand: NumberLike, f: NumberLike,
                          diel: str):
        """
        Frequency dependent relative permittivity of dielectric.
        """
        if diel == 'djordjevicsvensson':
            # compute the slope for a log frequency scale, tanD dependent.
            k = log((f_high + 1j * f_epr_tand) / (f_low + 1j * f_epr_tand))
            fd = log((f_high + 1j * f) / (f_low + 1j * f))
            ep_d = -tand * ep_r  / imag(k)
            # value for frequency above f_high
            ep_inf = ep_r * (1. + tand * real(k) / imag(k))
            # compute complex permitivity
            ep_r_f = ep_inf + ep_d * fd
            # get tand
            tand_f = -imag(ep_r_f) / real(ep_r_f)
        elif diel == 'frequencyinvariant':
            ep_r_f =  ep_r
            tand_f = tand
        else:
            raise ValueError('Unknown dielectric dispersion model')
        
        return real(ep_r_f), tand_f
    
    def analyse_quasi_static(self, ep_r: NumberLike, 
                           w: NumberLike, h: NumberLike, t: NumberLike,
                           model: str):
        """
        This function calculates the quasi-static impedance of a microstrip
        line, the value of the effective dielectric constant and the
        effective width due to the finite conductor thickness for the given
        microstrip line and substrate properties.
        """
        Z0 = sqrt(mu_0 / epsilon_0)
        zl_eff = Z0
        ep_reff = ep_r
        w_eff = w
        
        if model == 'wheeler':
            # compute strip thickness effect
            dw1 = 0
            if t is not None and t > 0:
                dw1 = t / pi * log(4. * exp(1.) / sqrt((t / h)**2) + \
                                   (1. / pi / (w / t + 1.1))**2)
            dwr = (1. + 1. / ep_r) / 2. * dw1
            wr = w + dwr
            w_eff = wr
            
            # compute characteristic impedance
            if (w / h) < 3.3:
                cp = log(4. * h / wr + sqrt((4 * h / wr)**2 + 2))
                b = (ep_r - 1.) / (ep_r + 1.) / \
                    2 * (log(pi / 2.) + log(4. / pi) / ep_r)
                zl_eff = (cp - b) * Z0 / pi / sqrt(2 * (ep_r + 1.))
            else:
                cp = 1 + log(pi / 2.) + log(wr / h / 2. + 0.94)
                d = 1. / pi / 2. * (1. + log(pi**2 / 16.)) * (ep_r - 1.) \
                    / ep_r**2
                x = 2. * log(2.) / pi + wr / h / 2. + (ep_r + 1.) / 2 / pi / \
                ep_r * cp + d
                zl_eff = Z0 / 2 / x / sqrt(ep_r)
            
            # compute effective dielectric constant
            if (w / h) < 1.3:
                a = log(8 * h / wr) + (wr / h)**2 / 32
                b = (ep_r - 1.) / (ep_r + 1.) / \
                    2 * (log(pi / 2.) + log(4. / pi) / ep_r)
                ep_reff = (ep_r + 1.) / 2. * (a / (a - b))**2 
            else:
                # qucsator is 4.0137 but doc 0.94 * 2 = 1.88
                d = (ep_r - 1.) / 2. / pi / ep_r * \
                    (log(2.1349 * wr / h + 4.0137) - 0.5169 / ep_r)
                e = wr / h / 2 + 1. / pi * log(8.5397 * wr / h + 16.0547)
                ep_reff = ep_r * ((e - d) / e)**2 
                
        elif model == 'schneider':
            u = w / h
            dw = 0
            
            # consider strip thickness equations
            if t is not None and t > 0:
                if t < (w / 2):
                    if u < (1. / pi / 2):
                        arg = 2 * pi * w / t
                    else:
                        arg = h / t
                    dw = t / pi * (1. + log(2 * arg))
                    if (t / dw) >= 0.75:
                        dw = 0
            w_eff = w + dw
            u = w_eff / h
                
            # effective dielectric constant
            ep_reff = (ep_r + 1.) / 2. + (ep_r - 1.) / 2. / sqrt (1. + 10. / u)
            
            # characteristic impedance
            if u < 1.:
                z = 1. / pi / 2. * log(8. / u + u / 4)
            else:
                z = 1. / (u + 2.42 - 0.44 / u + (1. - 1. / u)**6)
            zl_eff = Z0 * z / sqrt(ep_reff)
            
        elif model == 'hammerstadjensen':
            u = w / h
            if t is not None:
                t = t/h
            du1 = 0.
            
            # compute strip thickness effect
            if t is not None and t > 0:
                # Qucs formula 11.22 is wrong, normalized w has to be used instead (see Hammerstad and Jensen Article)
                # Normalized w is named u and is actually used in qucsator source code
                # coth(alpha) = 1/tanh(alpha)
                du1 = t / pi * log(1. + 4. * exp(1.) / t * tanh(sqrt(6.517 * u))**2)
            
            # sech(alpha) = 1/cosh(alpha)
            dur =  du1 * (1. + 1. / cosh(sqrt(ep_r - 1.))) / 2.
            
            u1 = u + du1
            ur = u + dur
            w_eff = ur * h
            
            # compute impedances for homogeneous medium
            zr = hammerstad_zl(ur)
            z1 = hammerstad_zl(u1)
            
            # compute effective dielectric constant
            a, b  = hammerstad_ab(ur, ep_r)
            e = hammerstad_er(ur, ep_r, a, b)
            
            # compute final characteristic impedance and dielectric constant
            #including strip thickness effects
            zl_eff = zr / sqrt(e)
            ep_reff = e * (z1 / zr)**2
            
        else:
            raise ValueError('Unknown microstripline quasi-static model')
        
        return zl_eff, ep_reff, w_eff
        
    def analyse_dispersion(self, zl_eff: NumberLike, ep_reff: NumberLike,
                          ep_r: NumberLike, wr: NumberLike, h: NumberLike, 
                          f: NumberLike, disp: str):
         """
         Frequency dependent characteristic impedance and dielectric constant
         accounting for microstripline dispersion.
         """
         u = wr/h
         if disp == 'schneider':
             k = sqrt(ep_reff / ep_r)
             fn = 4. * h * f / c * sqrt(ep_r - 1.)
             fn2 = fn**2
             e = ep_reff * ((1. + fn2) / (1. + k * fn2))**2
             z = zl_eff * sqrt(ep_reff / e)
         elif disp == 'hammerstadjensen':
             Z0 = sqrt(mu_0 / epsilon_0)
             g = pi**2 / 12 * (ep_r - 1) / ep_reff * sqrt(2 * pi * zl_eff / Z0)
             fp = (2 * mu_0 * h * f) / zl_eff
             e = ep_r - (ep_r - ep_reff) / (1 + g * fp**2)
             z =  zl_eff * sqrt(ep_reff / e) * (e - 1) / (ep_reff - 1)
         elif disp == 'kirschningjansen':
             fn = f * h * 1e-6
             e = kirsching_er(u, fn, ep_r, ep_reff)
             z, _ = kirsching_zl(u, fn, ep_r, ep_reff, e, zl_eff)
         elif disp == 'yamashita':
             k = sqrt(ep_r / ep_reff)
             fp = 4 * h * f / c * sqrt(ep_r - 1) * \
                 (0.5 + (1 + 2 * log10(1 + u))**2)
             e = ep_reff * ((1 + k * fp**1.5 / 4) / (1 + fp**1.5 / 4))**2
             z =  npy.ones(f.shape) * zl_eff
         elif disp == 'kobayashi':
             fk = c * arctan(ep_r * sqrt((ep_reff - 1) / (ep_r - ep_reff)))/ \
                 (2 * pi * h * sqrt(ep_r - ep_reff))
             fh = fk / (0.75 + (0.75 - 0.332 / (ep_r**1.73)) * u) 
             no = 1 + 1 / (1 + sqrt(u)) + 0.32 * (1 / (1 + sqrt(u)))**3 
             nc = npy.where(u < 0.7,
                 1 + 1.4 / (1 + u) * (0.15 - 0.235 * exp(-0.45 * f / fh)),
                 1)
             n = npy.where(no * nc < 2.32, no * nc, 2.32)
             e =  ep_r - (ep_r - ep_reff) / (1 + (f / fh)**n)
             z =  npy.ones(f.shape) * zl_eff
         elif disp == 'none':
             e = ones(f.shape) * ep_reff
             z = ones(f.shape) * zl_eff
             
         else:
             raise ValueError('Unknown microstripline dispersion model')
             
         return z, e
         
    def analyse_loss(self, ep_r: NumberLike, ep_reff: NumberLike, 
                    tand: NumberLike, rho: NumberLike, mu_r: NumberLike,
                    zl_eff_f1: NumberLike, zl_eff_f2: NumberLike, 
                    f: NumberLike, w: NumberLike, t: NumberLike,
                    D: NumberLike):
        """
        The function calculates the conductor and dielectric losses of a
        single microstrip line.
        """
        # limited to only Hammerstad and Jensen model
        Z0 = npy.sqrt(mu_0/epsilon_0)

        # conductor losses
        if t is not None and  t > 0:
            if rho is None:
                raise(AttributeError('must provide resistivity rho. '
                                     'see initializer help'))
            else:
                Rs  = surface_resistivity(f=f, rho=rho, mu_r=1)
                ds = skin_depth(f, rho, mu_r)
                if(npy.any(t < 3 * ds)):
                    warnings.warn(
                        'Conductor loss calculation invalid for line'
                        'height t ({})  < 3 * skin depth ({})'.format(t, ds[0]),
                        RuntimeWarning
                        )
                # current distribution factor
                Ki  = exp(-1.2 * ((zl_eff_f1 + zl_eff_f2) / 2 / Z0)**0.7)
                # D is RMS surface roughness
                Kr  = 1 + 2 / pi * arctan(1.4 * (D/ds)**2)
            a_conductor = Rs / (zl_eff_f1 * w) * Ki * Kr
        else:
            a_conductor = zeros(f.shape)
        
        # dielectric losses
        l0 = c / f
        a_dielectric =  pi * ep_r / (ep_r - 1) * (ep_reff - 1) / \
            sqrt(ep_reff) * tand / l0
            
        return a_conductor, a_dielectric

def hammerstad_ab(u: NumberLike, ep_r: NumberLike) -> NumberLike:
    """
    Intermediary parameter. see qucs docs on microstrip lines.
    """
    a = 1. + log((u**4 + (u / 52.)**2) / (u**4 + 0.432)) / 49. \
        + log(1 + (u / 18.1)**3) / 18.7
        
    b = 0.564 * ((ep_r - 0.9) / (ep_r + 3.))**0.053
    
    return a, b

def hammerstad_zl(u: NumberLike) -> NumberLike:
    """
    intermediary parameter. see qucs docs on microstrip lines.
    """
    fu = 6 + (2 * pi - 6) * exp(-(30.666 / u)**0.7528)
    Z0 = sqrt(mu_0/epsilon_0)
    return Z0 / 2. / pi * log(fu / u + sqrt(1. + (2. / u)**2))

def hammerstad_er(u: NumberLike, ep_r: NumberLike, a: NumberLike,
                  b: NumberLike) -> NumberLike:
    """
    intermediary parameter. see qucs docs on microstrip lines.
    """
    return (ep_r + 1) / 2 + (ep_r - 1) / 2 * (1. + 10. / u)**(-a * b)

def kirsching_zl(u: NumberLike, fn: NumberLike,
                 ep_r: NumberLike, ep_reff: NumberLike, ep_reff_f: NumberLike,
                 zl_eff: NumberLike):
    """
    intermediary parameter. see qucs docs on microstrip lines.
    """
    #fn = f * h * 1e-6 # GHz-mm
    R1 = npy.minimum(0.03891 * ep_r**1.4, 20.)
    R2 = npy.minimum(0.267 * u**7, 20.)
    R3 = 4.766 * exp(-3.228 * u**0.641)
    R4 = 0.016 + (0.0514 * ep_r)**4.524
    R5 = (fn / 28.843)**12
    R6 = npy.minimum(22.20 * u **1.92, 20.)
    R7 = 1.206 - 0.3144 * exp(-R1) * (1 - exp(-R2))
    R8 = 1 + 1.275 * (1 - exp(-0.004625 * R3 * ep_r**1.674 \
                              * (fn / 18.365)**2.745))
    R9 = 5.086 * R4 * R5/(0.3838 + 0.386 * R4) \
        * exp(-R6) / (1 + 1.2992 * R5) \
        * (ep_r - 1)**6 / (1 + 10 * (ep_r - 1)**6)
    R10 = 0.00044 * ep_r**2.136 + 0.0184
    R11 = (fn / 19.47)**6 / (1 + 0.0962 * (fn / 19.47)**6)
    R12 = 1 / (1 + 0.00245 * u**2)
    R13 = 0.9408 * ep_reff_f**R8 - 0.9603
    R14 = (0.9408 - R9) * ep_reff**R8 - 0.9603
    R15 = 0.707 * R10 * (fn / 12.3)**1.097
    R16 = 1 + 0.0503 * ep_r**2 * R11 * (1 - exp(-(u / 15)**6))
    R17 = R7 * (1 - 1.1241 * R12 / R16 \
        *exp(-0.026 * fn**1.15656 - R15))
    return zl_eff * (R13 / R14)**R17, R17

def kirsching_er(u: NumberLike, fn: NumberLike,
                 ep_r: NumberLike, ep_reff: NumberLike):
    """
    intermediary parameter. see qucs docs on microstrip lines.
    """
    # in the paper fn is in GHz-cm while in Qucs it is GHz-mm, thus a factor
    # 10 for all constant that multiply or divide fn
    P1 = 0.27488 + (0.6315 + 0.525 / ( 1+ 0.0157 * fn)**20) * u \
        -0.065683 * exp(-8.7513 * u)
    P2 = 0.33622 * (1  -exp(-0.03442 * ep_r))
    P3 = 0.0363 * exp(-4.6 * u) * (1 - exp(-(fn / 38.7)**4.97))
    P4 = 1 + 2.751 * (1 - exp(-(ep_r / 15.916)**8))
    Pf = P1 * P2 * ((0.1844 + P3 * P4) * fn)**1.5763
    return ep_r - (ep_r - ep_reff) / (1 + Pf)
