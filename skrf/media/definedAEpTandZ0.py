

'''
.. module:: skrf.media.definedAEpTandZ0

========================================
DefinedAEpTandZ0 (:mod:`skrf.media.definedAEpTandZ0`)
========================================

Media defined by A, Ep, Tand and Z0 class.

This media is defined by attenuation A, relative permittivity Ep_r,
loss angle tand and characteristic impedance Z0.
These values are either frequency invariant or specified at each frequencies
to account for dispersion.
Djirdjevic/Svennson dispersion model is provided, custom model may be use
by providing an array with the parameter value for each frequency.
'''
from scipy.constants import  epsilon_0, c
from numpy import real, imag, sqrt, ones, zeros, pi, log
from .media import Media

class DefinedAEpTandZ0(Media):
    '''
    Media defined by A, Ep, Tand and Z0 class.
    
    A generic physical transmission line is contructed: 
     * from complex, relative permativity and permiability OR 
     * from real relative permativity and permiability with loss tangents.  
     
    See Examples. There is also a method to initialize from a 
    existing distributed circuit, appropriately named 
    :func:`Freespace.from_distributed_circuit`
    
    

    Parameters
    -----------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of this transmission line medium
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if z0 is None then will default to Z0
    ep_r : number, array-like
        complex relative permittivity. negative imaginary is lossy.
    mu_r : number, array-like
        complex relative permeability. negative imaginary is lossy.
    ep_loss_tan: None, number, array-like
        the loss tangent of the permativity. If not None, imag(ep_r) is 
        ignored. 
    mu_loss_tan: None, number, array-like
        the loss tangent of the permeability. If not None, imag(mu_r) is 
        ignored. 
    \*args, \*\*kwargs : arguments and keyword arguments


    Examples
    -----------
    >>>from skrf.media.freespace import Freespace
    >>>from skrf.frequency import Frequency
    >>>f = Frequency(75,110,101,'ghz')
    >>>Freespace(frequency=f, ep_r=11.9)   
    >>>Freespace(frequency=f, ep_r=11.9-1.1j)
    >>>Freespace(frequency=f, ep_r=11.9, ep_loss_tan=.1)
    >>>Freespace(frequency=f, ep_r=11.9-1.1j, mu_r = 1.1-.1j)
    
    
    '''
    def __init__(self, frequency=None, z0=None, A=0, f_A=1e9, ep_r=1+0j, 
                 mu_r=1+0j, tand=0, f_low=1e3, f_high=1e12, f_epr_tand=1e9,
                 diel='djordjevicsvensson', *args, **kwargs):
        
        Media.__init__(self, frequency=frequency,z0=z0)
        self.ep_r, self.mu_r, self.tand, self.diel = ep_r, mu_r, tand, diel
        self.A, self.f_A = A, f_A
        self.f_low, self.f_high, self.f_epr_tand = f_low, f_high, f_epr_tand
    
    def __str__(self):
        f=self.frequency
        output = 'Physical line Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)
        return output

    def __repr__(self):
        return self.__str__()
    
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
    def alpha_conductor(self):
        '''
        Losses due to conductor resistivity

        Returns
        --------
        alpha_conductor : array-like
                lossyness due to conductor losses

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
        alpha_conductor : calculates losses to conductor
        alpha_dielectric: calculates losses to dielectric
        beta            : calculates phase parameter
        '''
        beta  = self.beta_phase
        alpha = self.alpha_conductor + self.alpha_dielectric
        return alpha + 1j*beta
    
    @property
    def Z0(self):
        '''
        Characteristic Impedance, :math:`Z0`

        .. math::
                Z_0 = \\sqrt{ \\frac{Z^{'}}{Y^{'}}}

        Returns
        --------
        Z0 : numpy.ndarray
                Characteristic Impedance in units of ohms
        '''
        return zeros(len(self))  
