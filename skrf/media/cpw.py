"""
cpw (:mod:`skrf.media.cpw`)
========================================

.. autosummary::
   :toctree: generated/

   CPW

"""
from numpy import sqrt, log, zeros, any, tanh, sinh, exp, real, imag
from scipy.constants import  epsilon_0, mu_0, c, pi
from scipy.special import ellipk
from .media import Media
from ..tlineFunctions import surface_resistivity, skin_depth
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from .. frequency import Frequency


class CPW(Media):
    r"""
    Coplanar waveguide.
    
    
    A coplanar waveguide transmission line is defined in terms of width,
    spacing, and thickness on a given relative permittivity substrate of a
    certain height. The line has a conductor resistivity and a tangential loss
    factor. The backside of the strip can be made of air or metal (grounded 
    coplanar waveguide).

    This class is highly inspired by the technical documentation [QUCSa]_
    and sources provided by the qucs project [QUCSb]_ .
    
    In addition, Djordjevic [DBLS01]_ /Svensson [SvDe01]_  wideband debye dielectric
    model is considered to provide a more realistic modelling of broadband
    microstrip with causal time domain response.
    
    A compatibility mode is provided to mimic the behaviour of QUCS or of
    Keysight ADS. There is known differences in the output of these
    simulators.
    
    The quasi-static models of characteristic impedance and effective
    permittivity give the value at zero frequency. The dispersion models
    compute frequency-dependant values of these variables.
    
    * Quasi-static characteristic impedance and effective permittivity model
      use [GhNa84]_ and [GhNa83]_. The models are corrected to account for
      strip thickness using a first-order approach described in [GGBB96]_.
      A comparison shows that ADS simulator uses another thickness correction
      method that is according to ADS doc based on [Cohn60]_. This second method
      is not implemented in skrf.
    
    * Frequency dispersion of impedance and effective permittivity model use
      [FGVM91]_ and [GMDK97]_.
    
    * Loss model is computed using Wheeler's incremental inductance rule
      [Whee42]_ applied to coplanar waveguide by [OwWu58]_ and [Ghio93]_.

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object, optional
        frequency band of the media. The default is None.
    z0 : number, array-like, or None (default None)
        the port impedance for media. Only needed if different from the
        characteristic impedance Z0 of the transmission line. In ohm
    w : number, or array-like
        width of the center conductor, in m. Default is 3e-3 m.
    s : number, or array-like
        spacing (width of the gap), in m. Default is 0.3e-3 m.
    h : number, or array-like
        height of the substrate between backside and conductor, in m.
        Default is 1.55 m (equivalent to infinite height for default w and s).
    t : number, or array-like, optional
        conductor thickness, in m. Default is None (no width correction
        to account for strip thickness).
    has_metal_backside : bool, default False
        If the backside is air (False) or metal (True)
    ep_r : number, or array-like, optional
        relative permittivity of the substrate at frequency f_epr_tand,
        no unit. Default is 4.5.
    diel : str
        dielectric frequency dispersion model in:
        
        * 'djordjevicsvensson' (default)
        * 'frequencyinvariant'
        
    rho : number, or array-like, or None
        resistivity of conductor, ohm / m. Default is 1.68e-8 ohm /m (copper).
    tand : number, or array-like
        dielectric loss factor at frequency f_epr_tand. Default is 0.
    f_low : number, or array-like
        lower frequency for wideband Debye Djordjevic/Svensson dielectric
        model, in Hz. Default is 1 kHz.
    f_high : number, or array-like
        higher frequency for wideband Debye Djordjevic/Svensson dielectric
        model, in Hz. Default is 1 THz.
    f_epr_tand : number, or array-like
        measurement frequency for ep_r and tand of dielectric, in Hz.
        Default is 1 GHz.
    compatibility_mode: str or None (default)
        If set to 'qucs', following behaviour happens :
        
        * Characteristic impedance will be real (no imaginary part due to tand)
        
    \*args, \*\*kwargs : arguments, keyword arguments
            passed to :class:`~skrf.media.media.Media`'s constructor
            (:func:`~skrf.media.media.Media.__init__`
             
    Note
    ----
    When the thickness of the strip is smaller than 3 skin depth, the losses
    model gives over-optimistic results and the media will issue a warning.
    At DC, the losses of the line could be smaller than its conductor
    resistance, which is not physical.

    References
    ----------
    .. [QUCSa] http://qucs.sourceforge.net/docs/technical.pdf
    .. [QUCSb] http://www.qucs.sourceforge.net/
    .. [DBLS01] Djordjevic, R.M. Biljic, V.D. Likar-Smiljanic, T.K. Sarkar,
        Wideband frequency-domain characterization of FR-4 and time-domain
        causality,
        IEEE Trans. on EMC, vol. 43, N4, 2001, p. 662-667.
    .. [SvDe01] C. Svensson, G.E. Dermer,
        Time domain modeling of lossy interconnects,
        IEEE Trans. on Advanced Packaging, May 2001, N2, Vol. 24, pp.191-196.
    .. [GhNa84] G. Ghione and C. Naldi. "Analytical Formulas for Coplanar Lines
       in Hybrid and Monolithic MICs", Electronics Letters,
       Vol. 20, No. 4, February 16, 1984, pp. 179-181.
    .. [GhNa83] G. Ghione and C. Naldi. "Parameters of Coplanar Waveguides with
        Lower Common Planes", Electronics Letters,
        Vol. 19, No. 18, September 1, 1983, pp. 734-735.
    .. [Cohn60] S. B. Cohn, "Thickness Corrections for Capacitive obstacles and
       Strip Conductors", IRE Trans. on Microwave Theory and Techniques,
       Vol. MTT-8, November 1960, pp. 638-644.
    .. [GGBB96] K. C. Gupta, R. Garg, I. J. Bahl, and P. Bhartia, Microstrip
       Lines and Slotlines, 2nd ed.Artech House, Inc., 1996.
    .. [FGVM91] M. Y. Frankel, S. Gupta, J. A. Valdmanis, and G. A. Mourou,
       "Terahertz Attenuation and Dispersion Characteristics of Coplanar
       Transmission Lines" IEEE Trans. on Microwave Theory and Techniques,
       vol. 39, no. 6, pp. 910-916, June 1991.
    .. [GMDK97] S. Gevorgian, T. Martinsson, A. Deleniv, E. Kollberg, and
       I. Vendik, "Simple and accurate dispersion expression for the
       effective dielectric constant of coplanar waveguides" in
       Proceedings of Microwaves, Antennas and Propagation,
       vol. 144, no. 2.IEE, Apr. 1997, pp. 145-148. 
    .. [Whee42] H. A. Wheeler, "Formulas for the Skin Effect,"
        Proceedings of the IRE, vol. 30, no. 9, pp. 412-424, Sept. 1942.
    .. [OwWu58] G. H. Owyang and T. T. Wu, "The Approximate Parameters of Slot
        Lines and Their Complement" IRE Transactions on Antennas and
        Propagation, pp. 49-55, Jan. 1958.
    .. [Ghio93] G. Ghione, "A CAD-Oriented Analytical Model for the Losses of
        General Asymmetric Coplanar Lines in Hybrid and Monolithic MICs"
        IEEE Trans. on Microwave Theory and Techniques,
        vol. 41, no. 9, pp. 1499-1510, Sept. 1993. 

    """
    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 w: NumberLike = 3e-3, s: NumberLike = 0.3e-3,
                 h: NumberLike = 1.55,
                 ep_r: NumberLike = 4.5, t: Union[NumberLike, None] = None,
                 diel: str = 'djordjevicsvensson',
                 rho: Union[NumberLike, None] = 1.68e-8, tand: NumberLike = 0,
                 f_low: NumberLike = 1e3, f_high: NumberLike = 1e12,
                 f_epr_tand: NumberLike = 1e9,
                 has_metal_backside: bool = False,
                 compatibility_mode: Union[str, None] = None,
                 *args, **kwargs):
        Media.__init__(self, frequency=frequency,z0=z0)

        self.w, self.s, self.h, self.t, self.ep_r, self.tand, self.rho =\
                w, s, h, t, ep_r, tand, rho
        self.diel = diel
        self.f_low, self.f_high, self.f_epr_tand = f_low, f_high, f_epr_tand
        self.has_metal_backside = has_metal_backside
        self.compatibility_mode = compatibility_mode
        
        # variation of effective permittivity with frequency
        # Not implemented in QUCS but implemented in ADS.
        # 'frequencyinvariant' will give a constant complex value whith a real
        # part compatible with qucs and an imaginary part due to tand
        self.ep_r_f, self.tand_f = self.analyse_dielectric(
            self.ep_r, self.tand,
            self.f_low, self.f_high, self.f_epr_tand, self.frequency.f,
            self.diel)
        
        # quasi-static effective permittivity of substrate + line and
        # the impedance of the coplanar waveguide
        # qucs use real-valued ep_r giving real-valued impedance
        if compatibility_mode == 'qucs':
            self.zl_eff, self.ep_reff, k1, kk1, kpk1 = \
                self.analyse_quasi_static(
                real(self.ep_r_f), w, s, h, t, has_metal_backside)
        else:
            self.zl_eff, self.ep_reff, k1, kk1, kpk1 = \
                self.analyse_quasi_static(
                self.ep_r_f, w, s, h, t, has_metal_backside)
        
        # analyse dispersion of impedance and relatice permittivity
        if compatibility_mode == 'qucs':
            self._z_characteristic, self.ep_reff_f = self.analyse_dispersion(
                self.zl_eff, self.ep_reff, real(self.ep_r_f), w, s, h,
                self.frequency.f)
        # ads does not use frequency dispersion for cpw/cpwg (so sad)
        elif compatibility_mode == 'ads':
            self._z_characteristic = self.zl_eff
            self.ep_reff_f = self.ep_reff
        # use frequency dispersion and complex permittivity by default
        else:
            self._z_characteristic, self.ep_reff_f = self.analyse_dispersion(
                self.zl_eff, self.ep_reff, self.ep_r_f, w, s, h,
                self.frequency.f)
        
        # analyse losses of line
        self.alpha_conductor, self.alpha_dielectric = self.analyse_loss(
            real(self.ep_r_f), real(self.ep_reff_f), self.tand_f, rho, 1.,
            self.frequency.f,
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
    
    def analyse_dielectric(self, ep_r: NumberLike, tand: NumberLike,
                          f_low: NumberLike, f_high: NumberLike,
                          f_epr_tand: NumberLike, f: NumberLike,
                          diel: str):
        """
        This function calculate the frequency-dependent relative permittivity
        of dielectric and tangential loss factor.
        
        References
        ----------
        .. [#] C. Svensson, G.E. Dermer,
            Time domain modeling of lossy interconnects,
            IEEE Trans. on Advanced Packaging, May 2001, N2, Vol. 24, pp.191-196.
        .. [#] Djordjevic, R.M. Biljic, V.D. Likar-Smiljanic, T.K. Sarkar,
            Wideband frequency-domain characterization of FR-4 and time-domain
            causality,
            IEEE Trans. on EMC, vol. 43, N4, 2001, p. 662-667.
            
        Returns
        -------
        ep_r_f : :class:`numpy.ndarray`
        tand_f : :class:`numpy.ndarray`
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
            ep_r_f =  ep_r - 1j * ep_r * tand
            tand_f = tand
        else:
            raise ValueError('Unknown dielectric dispersion model')
        
        return ep_r_f, tand_f
    
    def analyse_quasi_static(self, ep_r: NumberLike, 
                           w: NumberLike, s: NumberLike,
                           h: NumberLike, t: NumberLike,
                           has_metal_backside: bool):
        """
        This function calculates the quasi-static impedance of a coplanar
        waveguide line, the value of the effective permittivity as per filling
        factor, and the effective width due to the finite conductor thickness
        for the given coplanar waveguide line and substrate properties.
        Model from [#]_ with air backside and [#]_ with a metal backside.
        The models are corrected to account for
        strip thickness using a first-order approach described in [GGBB96]_.
        ADS simulator report to use a custom correction based on [Cohn60]_.
        This second method is not implemented in skrf.
        
        References
        ----------
        .. [GhNa84] G. Ghione and C. Naldi. "Analytical Formulas for Coplanar Lines
           in Hybrid and Monolithic MICs", Electronics Letters,
           Vol. 20, No. 4, February 16, 1984, pp. 179-181.
        .. [GhNa83] G. Ghione and C. Naldi. "Parameters of Coplanar Waveguides with
            Lower Common Planes", Electronics Letters,
            Vol. 19, No. 18, September 1, 1983, pp. 734-735.
        .. [Cohn60] S. B. Cohn, "Thickness Corrections for Capacitive obstacles and
           Strip Conductors", IRE Trans. on Microwave Theory and Techniques,
           Vol. MTT-8, November 1960, pp. 638-644.
        .. [GGBB96] K. C. Gupta, R. Garg, I. J. Bahl, and P. Bhartia, Microstrip
           Lines and Slotlines, 2nd ed.Artech House, Inc., 1996.
            
        Returns
        -------
        zl_eff : :class:`numpy.ndarray`
        ep_reff : :class:`numpy.ndarray`
        """
        Z0 = sqrt(mu_0 / epsilon_0)
        a = w
        b = w + 2. * s
        
        # equation (3a) from [GhNa84] or (6) from [GhNa83]
        k1 = a / b
        kk1 = ellipk(k1)
        kpk1 = ellipk(sqrt(1. - k1 * k1))
        q1 = ellipa(k1)
        
        # backside is metal
        if has_metal_backside:
            # equation (4) from [GhNa83]
            # in qucs the 2 coefficient turn to 4 and fit better with ads
            k3 = tanh(pi * a / 4. / h) / tanh(pi * b / 4. / h)
            q3 = ellipa(k3)
            qz = 1. / (q1 + q3)
            # equation (7) from [GhNa83]
            # equivalent to e = (q1 + ep_r * q3) / (q1 + q3) and paper
            e = 1. + q3 * qz * (ep_r - 1.)
            # equation (8) from [GhNa83] with later division by sqrt(e)
            zr = Z0 / 2. * qz
            
        # backside is air
        else:
            # equation (3b) from [GhNa84]
            k2 = sinh((pi / 4.) * a / h) / sinh((pi / 4.) * b / h)
            q2 = ellipa(k2)
            # equation (2) from [GhNa84]
            e = 1. + (ep_r - 1.) / 2. * q2 / q1
            # equation (1) from [GhNa84] with later division by sqrt(e)
            zr = Z0 / 4. / q1
            
        # a posteriori effect of strip thickness
        if t is not None and t > 0.:
            # equation (7.98) from [GGBB96]
            d = 1.25 * t / pi * (1. + log(4. * pi * w / t)) 
            # equation between (7.99) and (7.100) from [GGBB96]
            #approx. equal to ke = (w + d) / (w + d + 2 * (s - d))
            ke = k1 + (1. - k1 * k1) * d / 2. / s
            qe = ellipa(ke)
            
            # backside is metal
            if has_metal_backside:
                # equation (8) from [GhNa83] with k1 -> ke
                # but keep q3 unchanged ? (not in papers)
                qz = 1. / (qe + q3)
                zr = Z0 / 2. * qz
            # backside is air
            else:
                # equation (7.99) from [GGBB96] with later division by sqrt(e)
                zr = Z0 / 4. / qe
            
            # modifies ep_re
            # equation (7.100) of [GGBB96]
            e = e - (0.7 * (e - 1.) * t / s) / (q1 + (0.7 * t / s))
        
        ep_reff = e
        # final division of (1) from [GhNa84] and (8) from [GhNa83]
        zl_eff = zr / sqrt(ep_reff)
        
        return zl_eff, ep_reff, k1, kk1, kpk1
    
    def analyse_dispersion(self, zl_eff: NumberLike, ep_reff: NumberLike,
                          ep_r: NumberLike, w: NumberLike, s: NumberLike,
                          h: NumberLike, f: NumberLike):
         """
         This function computes the frequency-dependent characteristic
         impedance and effective permittivity accounting for coplanar waveguide
         frequency dispersion.
         
         References
         ----------
         .. [#] M. Y. Frankel, S. Gupta, J. A. Valdmanis, and G. A. Mourou,
            "Terahertz Attenuation and Dispersion Characteristics of Coplanar
            Transmission Lines" IEEE Trans. on Microwave Theory and Techniques,
            vol. 39, no. 6, pp. 910-916, June 1991.
         .. [#] S. Gevorgian, T. Martinsson, A. Deleniv, E. Kollberg, and
            I. Vendik, "Simple and accurate dispersion expression for the
            effective dielectric constant of coplanar waveguides" in
            Proceedings of Microwaves, Antennas and Propagation,
            vol. 144, no. 2.IEE, Apr. 1997, pp. 145-148. 
             
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
        complanar waveguide line using wheeler's incremental inductance rule.
        
        References
        ----------
        .. [#] H. A. Wheeler, "Formulas for the Skin Effect,"
            Proceedings of the IRE, vol. 30, no. 9, pp. 412-424, Sept. 1942.
        .. [#] G. H. Owyang and T. T. Wu, "The Approximate Parameters of Slot
            Lines and Their Complement" IRE Transactions on Antennas and
            Propagation, pp. 49-55, Jan. 1958.
        .. [#] G. Ghione, "A CAD-Oriented Analytical Model for the Losses of
            General Asymmetric Coplanar Lines in Hybrid and Monolithic MICs"
            IEEE Trans. on Microwave Theory and Techniques,
            vol. 41, no. 9, pp. 1499-1510, Sept. 1993. 
            
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
            if any(t < 3 * ds):
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
    
def ellipa(k: NumberLike):
    """
    Approximation of K(k)/K'(k).
    First appeared in [#]_
    More accurate expressions can be found in the above article and in [#]_.
    
    The maximum relative error of the approximation implemented here is
    about 2 ppm, so good enough for any practical purpose.
    
    References
    ==========
    .. [#] Hilberg, W., "From Approximations to Exact Relations for
       Characteristic Impedances," IEEE Trans. MTT, May 1969.
    .. [#] Abbott, J. T., "Modeling the Capacitive Behavior of Coplanar
       Striplines and Coplanar Waveguides using Simple Functions",
       Rochester Institute of Technology, Rochester, New York, June 2011.
    """
    if k < sqrt(1. / 2.):
        kp = sqrt(1 - k * k)
        r = pi / log(2. * (1. + sqrt(kp)) / (1. - sqrt(kp)))
    else:
        r = log(2. * (1. + sqrt(k)) / (1. - sqrt(k))) / pi
    
    return r
