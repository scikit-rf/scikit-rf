

'''
.. module:: skrf.media.MLine
========================================
MLine (:mod:`skrf.media.MLine`)
========================================

Microstripline class


This class was made from the technical documentation [#]_ provided
by the qucs project [#]_ .
The variables  and properties of this class are coincident with
their derivations.

In addition, Djordjevic/Svensson widebande debye dielectric model is considered
to provide more realistic modelling of broadband microstrip as well as causal
time domain response.

* Quasi-static characteristic impedance model:
    In the case another model is used for effective dielectric constant,
    Hammerstad and Jensen method is used for the impedance.
    + Kirschning and Jansen
    + Hammerstad and Jensen
    + None
* Quasi-static effective dielectric constant:
    + Kirschning and Jansen
    + Hammerstad and Jensen
    + Yamashita
    + Kobayashi
    + None
* Strip thickness correction:
    + Hammerstad and Jensen, add a certain amount to W if T > 0.

.. [#] http://qucs.sourceforge.net/docs/technical.pdf
.. [#] http://www.qucs.sourceforge.net/
.. C. Svensson, G.E. Dermer,
    Time domain modeling of lossy interconnects,
    IEEE Trans. on Advanced Packaging, May 2001, N2, Vol. 24, pp.191-196.
.. Djordjevic, R.M. Biljic, V.D. Likar-Smiljanic, T.K. Sarkar,
    Wideband frequency-domain characterization of FR-4 and time-domain causality,
    IEEE Trans. on EMC, vol. 43, N4, 2001, p. 662-667.
'''
import numpy as npy
from scipy.constants import  epsilon_0, mu_0, c
from numpy import real, imag, pi, sqrt,log, exp, log10, zeros, ones, tanh, \
    cosh, arctan, absolute
from .media import Media
from ..tlineFunctions import skin_depth, surface_resistivity

class MLine(Media):
    '''
    Microstripline initializer

    Parameters
    -------------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of the media
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
    w : number, or array-like
        width of conductor, in m.
    h : number, or array-like
        height of subtrate between ground plane and conductor, in m.
    t : number, or array-like, optional
        conductor thickness, in m.
    ep_r : number, or array-like
        relative permittivity of dielectric at frequency f_epr_tand
    mu_r : number, or array-like
        relative permeability of dielectric (assumed frequency invariant)
    diel : number, or array-like
        dielectric dispersion model: djordjevicsvensson or frequencyinvariant.
    rho: number, or array-like, optional
        resistivity of conductor (None)
    tand: number, or array-like
        dielectric loss factor at frequency f_epr_tand
    rough: number, or array-like
        RMS rhougness of conductor in m.
    disp: number, or array-like
        microstripline dispersion model in
        * kirschningjansen
        * hammerstadjensen
        * yamashita
        * kobayashi
        * none
    f_low: number, or array-like
        lower frequency for wideband Debye Djordjevic/Svensson dielectric model
    f_high: number, or array-like
        higher frequency for wideband Debye Djordjevic/Svensson dielectric model
    f_epr_tand: number, or array-like
        measurement frequency for ep_r and tand of dielectric
    '''
    def __init__(self, frequency=None, z0=None, w=3, h=1.6, t=None,
                 ep_r=4.5, mu_r=1, diel='djordjevicsvensson',
                 rho=1.68e-8, tand=0, rough=0.15e-6, disp='kirschningjansen',
                 f_low=1e3, f_high=1e12, f_epr_tand=1e9,
                 *args, **kwargs):
        Media.__init__(self, frequency=frequency,z0=z0)
        
        self.w, self.h, self.t = w, h, t
        self.ep_r, self.mu_r, self.diel = ep_r, mu_r, diel
        self.rho, self.tand, self.rough, self.disp =  rho, tand, rough, disp
        self.f_low, self.f_high, self.f_epr_tand = f_low, f_high, f_epr_tand


    def __str__(self):
        f=self.frequency
        output =  \
                'Microstripline Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\n W= %.2em, H= %.2em'% \
                (self.w,self.h)
        return output

    def __repr__(self):
        return self.__str__()

    @property
    def delta_w1(self):
        '''
        intermediary parameter. see qucs docs on microstrip lines.
        '''
        w, h, t = self.w, self.h, self.t
        if t > 0.:
            # Qucs formula 11.22 is wrong, normalized w has to be used instead (see Hammerstad and Jensen Article)
            return t/pi * log(1. + 4*exp(1.)*tanh(sqrt(6.517*w/h))**2/t) * h
        else:
            return 0.
    
    @property
    def delta_wr(self):
        '''
        intermediary parameter. see qucs docs on microstrip lines.
        '''
        delta_w1, ep_r = self.delta_w1, real(self.ep_r_f)
        if self.t > 0.:
            return delta_w1/2 * (1+1/cosh(sqrt(ep_r-1)))
        else:
            return 0.
    
    @property
    def ep_r_f(self):
        '''
        Frequency dependant relative permittivity of dielectric
        '''
        ep_r, tand  = self.ep_r, self.tand
        f_low, f_high, f_epr_tand = self.f_low, self.f_high, self.f_epr_tand
        f = self.frequency.f
        if self.diel == 'djordjevicsvensson':
            # compute the slope for a log frequency scale, tanD dependant.
            m = (ep_r*tand)  * (pi/(2*log(10)))
            # value for frequency above f_high
            ep_inf = (ep_r - 1j*ep_r*tand - m*log((f_high + 1j*f_epr_tand)/(f_low + 1j*f_epr_tand)))
            return ep_inf + m*log((f_high + 1j*f)/(f_low + 1j*f))
        elif self.diel == 'frequencyinvariant':
            return ones(self.frequency.f.shape) * (ep_r - 1j*ep_r*tand)
        else:
            raise ValueError('Unknown dielectric dispersion model')
    
    @property
    def tand_f(self):
        '''
        Frequency dependant dielectric loss factor
        '''
        ep_r = self.ep_r_f
        return -imag(ep_r) / real(ep_r)
    
    @property
    def ep_reff(self):
        '''
        Quasistatic effective relative permittivity of dielectric.
        Accounting for the filling factor between air and substrate.
        '''
        h, ep_r  = self.h, self.ep_r_f
        w1 = self.w + self.delta_w1
        wr = self.w + self.delta_wr
        return ep_re(wr, h, ep_r) * (ZL1(w1, h)/ZL1(wr, h))**2
    
    @property
    def G(self):
        '''
        intermediary parameter. see qucs docs on microstrip lines.
        '''
        ep_r, ep_reff, ZL = self.ep_r_f, self.ep_reff, self.Z0
        ZF0 = npy.sqrt(mu_0/epsilon_0)
        return (pi**2)/12 * (ep_r-1)/ep_reff * sqrt(2*pi*ZL/ZF0)
    
    @property
    def ep_reff_f(self):
        '''
        Frequency dependant effective relative permittivity of dielectric,
        accounting for microstripline dispersion.
        '''
        ep_r, ep_reff  = self.ep_r_f, self.ep_reff
        w, h = self.w + self.delta_wr, self.h
        f = self.frequency.f
        if self.disp == 'hammerstadjensen':
            ZL, G = self.Z0, self.G
            fp = ZL/(2*mu_0*h)
            return ep_r - (ep_r - ep_reff) / (1+G*(f/fp)**2)
        elif self.disp == 'kirschningjansen':
            fn = self.frequency.f * h * 1e-6
            P1 = 0.27488+(0.6315+0.525/(1+0.0157*fn)**20)*w/h -0.065683*exp(-8.7513*w/h)
            P2 = 0.33622*(1-exp(-0.03442*ep_r))
            P3 = 0.0363*exp(-4.6*w/h)*(1-exp(-(fn/38.7)**4.97))
            P4 = 1+2.751*(1-exp(-(ep_r/15.916)**8))
            Pf = P1*P2*((0.1844+P3*P4)*fn)**1.5763
            return ep_r - (ep_r-ep_reff)/(1+Pf)
        elif self.disp == 'yamashita':
            k = sqrt(ep_r/ep_reff)
            F = 4*h*f*sqrt(ep_r-1)/c * (0.5+(1+2*log(1+w/h))**2)
            return ep_reff * ((1+1/4*k*F**1.5)/(1+1/4*F**1.5))**2
        elif self.disp == 'kobayashi':
            f50 = c/(2*pi*h*(0.75+(0.75-0.332/(ep_r**1.73)*w/h))) \
                * arctan(ep_r*sqrt((ep_reff-1)/(ep_r-ep_reff)))/ \
                sqrt(ep_r-ep_reff)
            m0 = 1+1/(1+sqrt(w/h))+0.32*(1/(1+sqrt(w/h)))**3
            mc = 1+1.4/(1+w/h)*(0.15-0.235*exp(-0.45*f/f50))
            if(w/h >= 0.7):
                mc = 1
            m = m0*mc
            return ep_r - (ep_r-ep_reff)/(1+f/f50)**m
        elif self.disp == 'none':
            return ones(self.frequency.f.shape)*self.ep_reff
        else:
            raise ValueError('Unknown microstripline dispersion model')

    @property
    def Z0(self):
        '''
        Quasistatic characteristic impedance
        '''
        h, ep_reff = self.h, real(self.ep_reff)
        wr = self.w + self.delta_wr
        return ZL1(wr, h)/sqrt(ep_reff)
    
    @property
    def Z0_f(self):
        '''
        Frequency dependant characteristic impedance
        '''
        ZL, ep_reff, ep_reff_f = self.Z0, real(self. ep_reff), real(self.ep_reff_f)
        wr, h = self.w + self.delta_wr, self.h
        if self.disp == 'hammerstadjensen':
            return ZL * sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        elif self.disp == 'kirschningjansen':
            u = wr/h
            fn = self.frequency.f * self.h * 1e-6
            ep_r = real(self.ep_r_f)
            R1 = 0.03891*ep_r**1.4
            R2 = 0.267*u**7
            R3 = 4.766*exp(-3.228*u**0.641)
            R4 = 0.016+(0.0514*ep_r)**4.524
            R5 = (fn/28.843)**12
            R6 = 22.20*u**1.92
            R7 = 1.206-0.3144*exp(-R1)*(1-exp(-R2))
            R8 = 1+1.275*(1-exp(-0.004625*R3*ep_r**1.674)*(fn/18.365)**2.745)
            R9 = 5.086*R4*R5/(0.3838+0.386*R4)*exp(-R6)/(1+1.2992*R5)*(ep_r-1)**6/(1+10*(ep_r-1)**6)
            R10 = 0.00044*ep_r**2.136+0.0184
            R11 = (fn/19.47)**6/(1+0.0962*(fn/19.47)**6)
            R12 = 1/(1+0.00245*u**2)
            R13 = 0.9408*ep_reff_f**R8-0.9603
            R14 = (0.9408-R9)*ep_reff**R8-0.9603
            R15 = 0.707*R10*(fn/12.3)**1.097
            R16 = 1+0.0503*ep_r**2*R11*(1-exp(-(u/15)**6))
            R17 = R7 * (1-1.1241*R12/R16*exp(-0.026*fn**1.15656-R15))
            return ZL * (R13/R14)**R17
        elif self.disp == 'yamashita':
            return ZL * npy.sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        elif self.disp == 'kobayashi':
            return ZL * npy.sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        elif self.disp == 'none':
            return npy.ones(self.frequency.f.shape)*self.Z0
        else:
            raise ValueError('Unknown microstripline dispersion model')

    @property
    def alpha_conductor(self):
        '''
        Losses due to conductor resistivity

        Returns
        --------
        alpha_conductor : array-like
                lossyness due to conductor losses
        See Also
        ----------
        surface_resistivity : calculates surface resistivity
        '''
        if self.rho is None or self.t is None:
            raise(AttributeError('must provide values conductivity to calculate this. see initializer help'))
        else:
            f = self.frequency.f
            ZF0 = npy.sqrt(mu_0/epsilon_0)
            ZL, rho, mu_r  = real(self.Z0_f), self.rho, self.mu_r
            ep_reff= real(self.ep_reff)
            w = self.w + self.delta_wr
            rough = self.rough
            Kr  = 1 + 2/pi*arctan(1.4*(rough/skin_depth(f, rho, mu_r))**2)
            Ki  = exp(-1.2*(ZL/ZF0)**0.7)
            Rs  = surface_resistivity(f=f, rho=rho, mu_r=1)
        return sqrt(ep_reff) * Rs*Kr*Ki/(ZL*w)
    
    @property
    def alpha_dielectric(self):
        '''
        Losses due to dielectric

        '''
        ep_r, ep_reff, tand = real(self.ep_r_f), real(self.ep_reff), self.tand_f
        f = self.frequency.f
        return pi*f/c * ep_r/sqrt(ep_reff) * (ep_reff-1)/(ep_r-1) * tand
    
    @property
    def beta_phase(self):
        '''
        Phase parameter
        '''
        ep_reff, f = real(self.ep_reff_f), self.frequency.f
        return 2*pi*f*sqrt(ep_reff)/c

    @property
    def gamma(self):
        '''
        Propagation constant


        See Also
        --------
        alpha_conductor : calculates losses to conductor
        alpha_dielectric: calculates losses to dielectric
        beta            : calculates phase parameter
        '''
        f = self.frequency.f
        alpha = zeros(len(f))
        beta  = zeros(len(f))
        beta  = self.beta_phase
        alpha = self.alpha_dielectric
        if self.rho is not None:
            alpha += self.alpha_conductor
        return alpha + 1j*beta

def a(u):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 1. + 1./49.*log((u**4+(u/52)**2)/(u**4+0.432)) + 1./18.7*log(1+(u/18.1)**3)
    
def b(ep_r):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 0.564 * ((ep_r-0.9)/(ep_r+3.))**0.053
def f(u):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 6 + (2*pi-6)*exp(-(30.666/u)**0.7528)
    
def ZL1(w, h):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    u = w/h
    ZF0 = sqrt(mu_0/epsilon_0)
    return ZF0/(2*pi)*log(f(u)/u + sqrt(1.+(2./u)**2))

def ep_re(w, h, ep_r):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    u = w/h
    return (ep_r+1)/2 + (ep_r-1)/2 * (1.+10./u)**(-a(u)*b(ep_r))
