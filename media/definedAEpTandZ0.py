

'''
.. module:: skrf.media.definedAEpTandZ0

========================================
DefinedAEpTandZ0 (:mod:`skrf.media.definedAEpTandZ0`)
========================================

Transmission line medium defined by A, Ep, Tand and Z0.

This medium is defined by attenuation A, relative permittivity Ep_r,
loss angle tand and characteristic impedance Z0.

Djirdjevic/Svennson dispersion model is provided for dielectric, default
behaviour is frequency invariant.
'''
from scipy.constants import  epsilon_0, c
from numpy import real, imag, sqrt, ones, zeros, pi, log
from .media import Media

class DefinedAEpTandZ0(Media):
    '''
    Transmission line medium defined by A, Ep, Tand and Z0.
    
    This medium is defined by attenuation `A`, relative permittivity `Ep_r`,
    loss angle `tand` and characteristic impedance `Z0`.
    
    Djirdjevic/Svennson dispersion model is provided for dielectric, default
    behaviour is frequency invariant.
    
    A DefinedAEpTandZ0 medium is contructed: 
     * from scalar attenuation `A`, relative permittivity `Ep_r`,
       loss angle `tand` and characteristic impedance `Z0`.
       Frequency invariant behaviour or Djirdjevic/Svennson dispersion can be
       choosed trough `model`. Default is frequency invariant.
     
     See Examples.

    Parameters
    -----------
    frequency : :class:`~skrf.frequency.Frequency` object
        Frequency band of this transmission line medium
    z0 : number, array-like, or None
        The port impedance for this medium. Only needed if different from the
        characterisitc impedance of the transmission line.
        If z0 is None then will default to Z0
    A : number, array-like, default 0.0
        Attenuation due to conductor loss in dB/m/sqrt(Hz)
        The attenuation :math:`A(f)`at frequency :math:`f` is:
        .. math::
            A(f) = A*\sqrt{\frac{f}/{f_A}}
    f_A: number, default 1.0
        Frequency scaling in Hz for the attenuation. See A.
    ep_r : number, array-like, default 1.0
        Real part of the relative permittivity of the dielectric.
        if `model='frequencyinvariant'`, the complex relative permittivity is:
        .. math::
            \epsilon_r(f) = ep\_r + j \cdot ep\_r \cdot tanD
        if `model='djordjevicsvensson'`, the complex relative permittivity is:
        .. math::
            \epsilon_r(f) = ep_\inf + m \cdot \log{\frac{f_{high} - f_0}{f_{low} - f_\epsilon}}
        In this case, the value of ep_r and tanD are given for frequency f_ep:
        .. math::
            \epsilon_r(f_\epsilon) = ep\_r + j \cdot ep\_r \cdot tanD
    tanD: number, array-like, default 0.0
        Dielectric relative permittivity loss tangent. See ep_r.
    Z0: number, array-like, default 50.0
        Quasi-static characteristic impedance of the medium.
    f_low: number, default 1e3, , optional
        Low frequency in Hz for  for Djirdjevic/Svennson dispersion model.
        See ep_r.
    f_high: number, default 1e12, optional
        High frequency in Hz for  for Djirdjevic/Svennson dispersion model.
        See ep_r.
    f_ep: number, default 1e9, , optional
        Specification frequency in Hz for  for Djirdjevic/Svennson dispersion model.
        ep_r and tanD parameters are specified for this frequency. See ep_r.
    model: string, 'frequencyinvariant' or 'djordjevicsvensson', optional
        Use Djirdjevic/Svennson wideband Debye dispersion model or not.
        Default is frequency invariant behaviour.
    \*args, \*\*kwargs : arguments and keyword arguments


    Examples
    -----------
    >>>from skrf.media.definedAEpTandZ0 import DefinedAEpTandZ0
    >>>from skrf.frequency import Frequency
    >>>f = Frequency(75,110,101,'ghz')
    >>>DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, Z0=50)   
    >>>DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, tand=0.02, Z0=50)
    >>>DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, tand=0.02, Z0=50,
                        f_low=1e3, f_high=1e12, f_Ep=1e9,
                        model='djordjevicsvensson')
    
    
    '''
    def __init__(self, frequency=None, z0=None,
                 A=0.0, f_A=1.0, ep_r=1.0, tanD=0.0, Z0=50.0,
                 f_low=1.0e3, f_high=1.0e12, f_ep=1.0e9,
                 model='frequencyinvariant', *args, **kwargs):
        
        Media.__init__(self, frequency=frequency,z0=z0)
        self.A, self.f_A = A, f_A
        self.ep_r, self.tanD = ep_r, tanD
        self._Z0 = Z0
        self.f_low, self.f_high, self.f_ep = f_low, f_high, f_ep
        self.model = model
    
    def __str__(self):
        f=self.frequency
        output = 'DefinedAEpTandZ0 medium.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)
        return output

    def __repr__(self):
        return self.__str__()
    
    @property
    def ep_r_f(self):
        '''
        Frequency dependant complex relative permittivity of dielectric
        '''
        ep_r, tand  = self.ep_r, self.tanD
        f_low, f_high, f_ep = self.f_low, self.f_high, self.f_ep
        f = self.frequency.f
        if self.model == 'djordjevicsvensson':
            # compute the slope for a log frequency scale, tanD dependant.
            m = (ep_r*tand)  * (pi/(2*log(10)))
            # value for frequency above f_high
            ep_inf = (ep_r - 1j*ep_r*tand - m*log((f_high + 1j*f_ep)/(f_low + 1j*f_ep)))
            return ep_inf + m*log((f_high + 1j*f)/(f_low + 1j*f))
        elif self.model == 'frequencyinvariant':
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
    def alpha_conductor(self):
        '''
        Losses due to conductor resistivity
        '''
        A, f_A, f = self.A, self.f_A, self.frequency.f
        return A * log(10)/20 * sqrt(f/f_A)
    
    @property
    def alpha_dielectric(self):
        '''
        Losses due to dielectric
        '''
        ep_r, tand = real(self.ep_r_f), self.tand_f
        f = self.frequency.f
        return pi*sqrt(ep_r)*f/c * tand
    
    @property
    def beta_phase(self):
        '''
        Phase parameter
        '''
        ep_r, f = real(self.ep_r_f), self.frequency.f
        return 2*pi*f*sqrt(ep_r)/c

    @property
    def gamma(self):
        '''
        Propagation constant

        See Also
        --------
        alpha_conductor : calculates conductor losses
        alpha_dielectric: calculates dielectric losses
        beta_phase      : calculates phase parameter
        '''
        beta  = self.beta_phase
        alpha = self.alpha_conductor + self.alpha_dielectric
        return alpha + 1j*beta
    
    @property
    def Z0(self):
        return self._Z0
    
    @Z0.setter
    def Z0(self, val):
        self._Z0 = val
