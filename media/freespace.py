

'''
.. module:: skrf.media.freespace

========================================
freespace (:mod:`skrf.media.freespace`)
========================================

A plane-wave (TEM Mode) in Freespace.

Represents a plane-wave in a homogeneous freespace, defined by
the space's relative permittivity and relative permeability.



'''
import warnings
from scipy.constants import  epsilon_0, mu_0
from numpy import real, imag, cos, sqrt,tan,array, ones
from .distributedCircuit import DistributedCircuit
from .media import Media
from ..data import materials

class Freespace(Media):
    '''
    A plane-wave (TEM Mode) in Freespace.
    
    A Freespace media can be constructed in two ways: 
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
    def __init__(self, frequency=None, z0=None,  ep_r=1+0j, 
                 mu_r=1+0j, ep_loss_tan=None,mu_loss_tan=None, 
                 rho=None, *args, **kwargs):
        
        Media.__init__(self, frequency=frequency,z0=z0)
        self.ep_r  = ep_r
        self.mu_r  = mu_r
        self.rho=rho
        
        self.ep_loss_tan =ep_loss_tan
        self.mu_loss_tan =mu_loss_tan
    
    def __str__(self):
        f=self.frequency
        output = 'Freespace  Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)
        return output

    def __repr__(self):
        return self.__str__()
    
    @property
    def ep(self):
        if self.ep_loss_tan is not None:
            ep_r = real(self.ep_r)*(1-1j*self.ep_loss_tan)
        else:
            ep_r = self.ep_r
        return ep_r*epsilon_0
    
    @property 
    def mu(self):
        if self.mu_loss_tan is not None:
            mu_r = real(self.mu_r)*(1 -1j*self.mu_loss_tan)
        else:
            mu_r = self.mu_r
        return mu_r*mu_0
    
        
    @classmethod
    def from_distributed_circuit(cls,dc, *args, **kwargs):
        '''
        initialize a freespace  from media.DistributedCirctuit
        
        Parameters 
        -----------
        dc: `skrf.media.distributedcircuit.DistributedCircuit`
            a DistributedCircuit object
        *args, **kwargs : 
            passed to `Freespace.__init__`
        
        Notes
        --------
        Here are the details
        
            w = dc.frequency.w
            z= dc.Z/(w*mu_0)
            y= dc.Y/(w*epsilon_0)
            ep_r = -1j*y
            mu_r = -1j*z
        
        '''
        w = dc.frequency.w
        z= dc.Z/(w*mu_0)
        y= dc.Y/(w*epsilon_0)
        
    
        kw={}
        kw['ep_r'] = -1j*y
        kw['mu_r'] = -1j*z
        
        kwargs.update(kw)
        return cls(frequency=dc.frequency, *args, **kwargs)
    
    @property
    def rho(self):
        '''
        conductivty in ohm*m

        Parameters
        --------------
        val : float, array-like or str
            the resistivity in ohm*m. If array-like must be same length
            as self.frequency. if str, it must be a key in
            `skrf.data.materials`.

        Examples
        ---------
        >>> wg.rho = 2.8e-8
        >>> wg.rho = 2.8e-8 * ones(len(wg.frequency))
        >>> wg.rho = 'al'
        >>> wg.rho = 'aluminum'
        '''
        return self._rho
            
    @rho.setter
    def rho(self, val):
        if isinstance(val, str):
            self._rho = materials[val.lower()]['resistivity(ohm*m)']
        else:
            self._rho=val
            
    @property 
    def ep_with_rho(self):
        '''
        complex permativity  with resistivity absorbed into its 
        imaginary component
        '''
        if self.rho is not None:
            return self.ep -1j/(self.rho*self.frequency.w)
        else: 
            return self.ep
        
    @property
    def gamma(self):
        '''
        Propagation Constant, :math:`\\gamma`

        Defined as,

        .. math::
                \\gamma =  \\sqrt{ Z^{'}  Y^{'}}

        Returns
        --------
        gamma : numpy.ndarray
                Propagation Constant,

        Notes
        ---------
        The components of propagation constant are interpreted as follows:

        positive real(gamma) = attenuation
        positive imag(gamma) = forward propagation
        '''
        ep = self.ep_with_rho
        return 1j*self.frequency.w * sqrt(ep* self.mu)
    
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
        ep = self.ep_with_rho
        return sqrt(self.mu/ep)*ones(len(self))  
    
    def plot_ep(self):
        self.plot(self.ep_r.real, label=r'ep_r real')
        self.plot(self.ep_r.imag, label=r'ep_r imag')
    def plot_mu(self):
        self.plot(self.mu_r.real, label=r'mu_r real')
        self.plot(self.mu_r.imag, label=r'mu_r imag')
        
    def plot_ep_mu(self):
        self.plot_ep()
        self.plot_mu()
