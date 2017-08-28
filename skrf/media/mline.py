

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
    Hammerstad and Jensen, which is better than 0.001% for W/H <= 1 and better
    than 0.03% for W/H <= 1000, which is better than physical tolerances.
* Quasi-static effective dielectric constant:
    Hammerstad and Jensen, which is better than 0.2% at least for ep_r y 128
    0.01 <= W/H <= 100, which cover most use.
* Strip thickness correction:
    Hammerstad and Jensen, add a certain amount to W if T > 0.

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
from scipy.special import ellipk
from numpy import real, imag,pi,sqrt,log,zeros, ones
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
    ep_r : number, or array-like
            relative permativity of substrate
    t : number, or array-like, optional
            conductor thickness, in m.
    rho: number, or array-like, optional
            resistivity of conductor (None)

    '''
    def __init__(self, frequency=None, z0=None, w=3, h=1.6, t=None,
                 ep_r=4.5, mu_r=1, diel='DjordjevicSvensson',
                 rho=1.68e-8, tand=0, rough=0, disp='Constant',
                 f_low=1e3, f_high=1e9, f_epr_tand=1e9,
                 *args, **kwargs):
        Media.__init__(self, frequency=frequency,z0=z0)
        
        self.w, self.h, self.t = w, h, t
        self.ep_r, self.mu_r, self.rho, self.tand = ep_r, mu_r, rho, tand
        self.rough, self.disp = rough, disp
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
            return (t/h)/npy.pi * npy.log(1. + 4*npy.exp(1.)/((t/h)/(npy.tanh(npy.sqrt(6.517*(w/h)))**2))) * h
        else:
            return 0.
    
    @property
    def delta_wr(self):
        '''
        intermediary parameter. see qucs docs on microstrip lines.
        '''
        if self.t > 0.:
            return 1./2 * self.delta_w1 * (1+1/npy.cosh(npy.sqrt(self.ep_r-1)))
        else:
            return 0.
    
    @property
    def ep_r_f(self):
        '''
        Quasistatic effective relative dielectric
        constant, accounting for the filling factor between air and substrate.
        '''
        ep_r, tand  = self.ep_r, self.tand
        f_low, f_high, f_epr_tand = self.f_low, self.f_high, self.f_epr_tand
        f = self.frequency.f
        m = (ep_r * tand) * npy.log10(f_high/f_low) * npy.log(10)/(2*npy.pi)
        ep_inf = ep_r - m*npy.log10((f_high + 1j*f_epr_tand)/(f_low + 1j*f_epr_tand))

        return ep_inf + m*npy.log10((f_high + 1j*f)/(f_low + 1j*f))
    
    @property
    def tand_f(self):
        return -npy.imag(self.ep_r_f)/npy.real(self.ep_r_f)
    
    @property
    def ep_reff(self):
        '''
        Quasistatic effective relative dielectric
        constant, accounting for the filling factor between air and substrate.
        '''
        h, ep_r  = self.h, self.ep_r_f
        w1 = self.w + self.delta_w1
        wr = self.w + self.delta_wr
        return ep_re(wr, h, ep_r) * npy.power(ZL1(w1, h)/ZL1(wr, h), 2)
    
    @property
    def G(self):
        ep_r, ep_reff, ZL = self.ep_r_f, self.ep_reff, self.Z0
        ZF0 = npy.sqrt(mu_0/epsilon_0)
        '''
        intermediary parameter. see qucs docs on microstrip lines.
        '''
        return (npy.pi**2)/12 * (ep_r-1)/ep_reff * npy.sqrt(2*npy.pi*ZL/ZF0)
    
    @property
    def ep_reff_f(self):
        '''
        Frequency dispersed quasistatic effective relative dielectric
        constant, accounting for the filling factor between air and substrate.
        '''
        if self.disp == 'hammerstadjensen':
            ep_r, ep_reff, h  = self.ep_r_f, self.ep_reff, self.h
            f = self.frequency.f
            ZL, G = self.Z0, self.G
            fp = ZL/(2*mu_0*h)
            return ep_r - (ep_r - ep_reff) / (1+G*(f/fp)**2)
        elif self.disp == 'kirschningjansen':
            ep_r, ep_reff, w, h  = self.ep_r_f, self.ep_reff, self.w, self.h
            fn = self.frequency.f * h * 1e-6
            P1 = 0.27488+(0.6315+0.525/(1+0.0157*fn)**20)*w/h -0.065683*npy.exp(-8.7513*w/h)
            P2 = 0.33622*(1-npy.exp(-0.03442*ep_r))
            P3 = 0.0363*npy.exp(-4.6*w/h)*(1-npy.exp(-(fn/38.7)**4.97))
            P4 = 1+2.751*(1-npy.exp(-(ep_r/15.916)**8))
            Pf = P1*P2*((0.1844+P3*P4)*fn)**1.5763
            return ep_r - (ep_r-ep_reff)/(1+Pf)
        elif self.disp == 'yamashita':
            ep_r, ep_reff, w, h  = self.ep_r_f, self.ep_reff, self.w, self.h
            f = self.frequency.f
            k = npy.sqrt(ep_r/ep_reff)
            F = 4*h*f*npy.sqrt(ep_r-1)/c * (0.5+(1+2*npy.log(1+w/h))**2)
            return ep_reff * ((1+1/4*k*F**1.5)/(1+1/4*F**1.5))**2
        elif self.disp == 'kobayashi':
            ep_r, ep_reff, w, h  = self.ep_r_f, self.ep_reff, self.w, self.h
            f = self.frequency.f
            f50 = c/(2*npy.pi*h*(0.75+(0.75-0.332/(ep_r**1.73)*w/h))) \
                * npy.arctan(ep_r*npy.sqrt((ep_reff-1)/(ep_r-ep_reff)))/ \
                npy.sqrt(ep_r-ep_reff)
            m0 = 1+1/(1+npy.sqrt(w/h))+0.32*(1/(1+npy.sqrt(w/h)))**3
            mc = 1+1.4/(1+w/h)*(0.15-0.235*npy.exp(-0.45*f/f50))
            if(w/h >= 0.7):
                mc = 1
            m = m0*mc
            return ep_r - (ep_r-ep_reff)/(1+f/f50)**m
        else:
            return npy.ones(self.frequency.f.shape)*self.ep_reff

    @property
    def Z0(self):
        '''
        Quasistatic characteristic impedance
        '''
        h, ep_r = self.h, self.ep_r_f
        wr = self.w + self.delta_wr
        return ZL1(wr, h)/npy.sqrt(ep_re(wr, h, ep_r))
    
    @property
    def Z0_f(self):
        '''
        Frequency dispersed characteristic impedance
        '''
        if self.disp == 'hammerstadjensen':
            ZL, ep_reff, ep_reff_f = self.Z0, self. ep_reff, self.ep_reff_f
            return ZL * npy.sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        elif self.disp == 'kirschningjansen':
            ZL, ep_reff, ep_reff_f = self.Z0, self. ep_reff, self.ep_reff_f
            u = self.w/self.h
            fn = self.frequency.f * self.h * 1e-6
            ep_r = self.ep_r_f
            R1 = 0.03891*ep_r**1.4
            R2 = 0.267*u**7
            R3 = 7.766*npy.exp(-3.228*u**0.641)
            R4 = 0.016+(0.0514*ep_r)**4.524
            R5 = (fn/28.843)**12
            R6 = 22.20*u**1.92
            R7 = 1.206-0.3144*npy.exp(-R1)*(1-npy.exp(-R2))
            R8 = 1+1.275*(1-npy.exp(-0.004625*R3*ep_r**1.674)*(fn/18.365)**2.745)
            R9 = 5.086*R4*R5/(0.3838+0.386*R4)*npy.exp(-R6)/(1+1.2992*R5)*(ep_r-1)**6/(1+10*(ep_r-1)**6)
            R10 = 0.00044*ep_r**2.136+0.0184
            R11 = (fn/19.47)**6/(1+0.0962*(fn/19.47)**6)
            R12 = 1/(1+0.00245*u**2)
            R13 = 0.9408*ep_reff_f**R8-0.9603
            R14 = (0.9408-R9)*ep_reff**R8-0.9603
            R15 = 0.707*R10*(fn/12.3)**1.097
            R16 = 1+0.0503*ep_r**2*R11*(1-npy.exp(-(u/15)**6))
            R17 = R7 * (1-1.1241*R12/R16*npy.exp(-0.026*fn**1.15656-R15))
            return ZL * (R13/R14)**R17
        elif self.disp == 'yamashita':
            ZL, ep_reff, ep_reff_f = self.Z0, self. ep_reff, self.ep_reff_f
            return ZL * npy.sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        elif self.disp == 'kobayashi':
            ZL, ep_reff, ep_reff_f = self.Z0, self. ep_reff, self.ep_reff_f
            return ZL * npy.sqrt(ep_reff/ ep_reff_f) * (ep_reff_f-1)/(ep_reff-1)
        else:
            return npy.ones(self.frequency.f.shape)*self.Z0

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
            raise(AttributeError('must provide values conductivity and conductor thickness to calculate this. see initializer help'))
        else:
            f = self.frequency.f
            ZF0 = npy.sqrt(mu_0/epsilon_0)
            ZL, w, rho, mu_r  = self.Z0, self.w, self.rho, self.mu_r
            rough = self.rough
            Kr  = 1 + 2/npy.pi*npy.arctan(1.4*(rough/skin_depth(f, rho, mu_r))**2)
            Ki  = npy.exp(-1.2*(ZL/ZF0)**0.7)
            Rs  = surface_resistivity(f=f, rho=rho, mu_r=1)
        return Rs/(ZL*w) * Kr * Ki
    
    @property
    def alpha_dielectric(self):
        '''
        Losses due to dielectric

        '''
        ep_reff, ep_r, tand = self.ep_reff, self.ep_r_f, self.tand_f
        f = self.frequency.f
        return ep_r/npy.sqrt(ep_reff) * (ep_reff-1)/(ep_r-1) * npy.pi*f/c * tand
        

    @property
    def gamma(self):
        '''
        Propagation constant


        See Also
        --------
        alpha_conductor : calculates losses to conductors
        '''
        beta = 1j*2*pi*self.frequency.f*sqrt(self.ep_reff_f*epsilon_0*mu_0)
        alpha = zeros(len(beta))
        if self.rho is not None and self.t is not None:
            alpha = self.alpha_conductor + self.alpha_dielectric

        return beta+alpha

def a(u):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 1. + 1./49.*npy.log((u**4+(u/52)**2)/(u**4+0.432)) + 1./18.7*npy.log(1+(u/18.1)**3)
    
def b(ep_r):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 0.564 * npy.power((ep_r-0.9)/(ep_r+3.), 0.053)
def f(u):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    return 6 + (2*npy.pi-6)*npy.exp(-npy.power(30.666/u, 0.7528))
    
def ZL1(w, h):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    u = w/h
    ZF0 = npy.sqrt(mu_0/epsilon_0)
    return ZF0/(2*npy.pi)*npy.log(f(u)/u + npy.sqrt(1.+(2./u)**2))

def ep_re(w, h, ep_r):
    '''
    intermediary parameter. see qucs docs on microstrip lines.
    '''
    u = w/h
    return (ep_r+1)/2 + (ep_r-1)/2 * npy.power(1.+10./u, -a(u)*b(ep_r))
