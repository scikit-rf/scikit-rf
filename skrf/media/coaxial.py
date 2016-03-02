

'''
.. module:: skrf.media.coaxial
============================================================
coaxial (:mod:`skrf.media.coaxial`)
============================================================

A coaxial transmission line defined in terms of its inner/outer diameters and permittivity
'''

#from copy import deepcopy
from scipy.constants import  epsilon_0, mu_0, pi
from numpy import sqrt, log, imag,exp
from ..tlineFunctions import surface_resistivity
from .distributedCircuit import DistributedCircuit
from .media import Media
from ..constants import INF


class Coaxial( DistributedCircuit,Media ):
    '''
    A coaxial transmission line defined in terms of its inner/outer
    diameters and permittivity

    

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object
    
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if z0 is None then will default to Z0
    Dint : number, or array-like
        inner conductor diameter, in m
    Dout : number, or array-like
        outer conductor diameter, in m
    epsilon_r=1 : number, or array-like
        relative permittivity of the dielectric medium
    tan_delta=0 : number, or array-like
        loss tangent of the dielectric medium
    sigma=infinity : number, or array-like
        conductors electrical conductivity, in S/m

    
    TODO : different conductivity in case of different conductor kind

    Notes
    ----------
    Dint, Dout, epsilon_r, tan_delta, sigma can all be vectors as long 
    as they are the same length

    References
    ---------
    .. [#] Pozar, D.M.; , "Microwave Engineering", Wiley India Pvt. Limited, 1 sept. 2009



        '''
    ## CONSTRUCTOR
    def __init__(self, frequency=None,  z0=None, Dint=.81e-3, 
                 Dout=5e-3, epsilon_r=1, tan_delta=0, sigma=INF, 
                 *args, **kwargs):
        
        
        
        Media.__init__(self, frequency=frequency,z0=z0)
        
        self.Dint, self.Dout = Dint,Dout
        self.epsilon_r, self.tan_delta, self.sigma = epsilon_r, tan_delta, sigma
        self.epsilon_prime = epsilon_0*self.epsilon_r
        self.epsilon_second = epsilon_0*self.epsilon_r*self.tan_delta

    
    @classmethod
    def from_Z0_Dout(cls, frequency=None, z0=None,Z0=50,  epsilon_r=1, 
                     Dout=5e-3, **kw):
        '''
        Init from characteristic impedance and outer diameter
        
        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency` object

        z0 : number, array-like, or None
            the port impedance for media. Only needed if  its different
            from the characterisitc impedance of the transmission
            line. if z0 is None then will default to Z0
        Z0 : number 
            desired characteristic impedance
        Dout : number, or array-like
            outer conductor diameter, in m
        epsilon_r=1 : number, or array-like
            relative permittivity of the dielectric medium
        **kw : 
            passed to __init__
        '''
        ep= epsilon_0*epsilon_r
        
        if imag(Z0) !=0:
            raise NotImplementedError()
        
        b = Dout/2.
        b_over_a = exp(2*pi*Z0*sqrt(ep/mu_0))
        a = b/b_over_a
        Dint = 2*a
        return cls(frequency=frequency, z0 = z0, Dint=Dint, Dout=Dout, 
                    epsilon_r=epsilon_r, **kw)
    
    
    @property
    def Rs(self):
        f  = self.frequency.f
        rho = 1./self.sigma
        mu_r =1
        return surface_resistivity(f=f,rho=rho, mu_r=mu_r)
    @property
    def a(self):
        return self.Dint/2.
    
    @property
    def b(self):
        return self.Dout/2.


    # derivation of distributed circuit parameters
    @property
    def R(self):
        return self.Rs/(2.*pi)*(1./self.a + 1./self.b)
    
    @property
    def L(self):
        return mu_0/(2.*pi)*log(self.b/self.a)
    
    @property
    def C(self):
        return 2.*pi*self.epsilon_prime/log(self.b/self.a)
    
    @property
    def G(self):
        f =  self.frequency.f
        return f*self.epsilon_second/log(self.b/self.a)

    def __str__(self):
        f=self.frequency
        try:
            output =  \
                'Coaxial Transmission Line.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nDint= %.2f mm, Dout= %.2f mm '% \
                (self.Dint*1e3, self.Dout*1e3) +\
                '\nCharacteristic Impedance=%.1f-%.1f Ohm'%(self.Z0[0],self.Z0[-1]) +\
                '\nPort impedance Z0=%.1f-%.1f Ohm'%(self.z0[0],self.z0[-1]) 
        except(TypeError):
            output =  \
                'Coaxial Transmission Line.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nDint= %.2f mm, Dout= %.2f mm '% \
                (self.Dint[0]*1e3, self.Dout[0]*1e3) +\
                '\nCharacteristic Impedance=%.1f-%.1f Ohm'%(self.Z0[0],self.Z0[-1]) +\
                '\nPort impedance Z0=%.1f-%.1f Ohm'%(self.z0[0],self.z0[-1]) 
        return output
