

'''
.. module:: skrf.media.distributedCircuit
============================================================
distributedCircuit (:mod:`skrf.media.distributedCircuit`)
============================================================



'''

from copy import deepcopy
from scipy.constants import  epsilon_0, mu_0, c,pi, mil
import numpy as npy
from numpy import sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
         interp, linspace, shape,zeros, reshape

from ..tlineFunctions import electrical_length
from .media import Media, DefinedGammaZ0


from ..constants import INF, ONE, ZERO

class DistributedCircuit(Media):
    '''
    A transmission line mode defined in terms of distributed impedance
    and admittance values. 

    Parameters
    ------------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of the media
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if z0 is None then will default to Z0
    C : number, or array-like
            distributed capacitance, in F/m
    L : number, or array-like
            distributed inductance, in  H/m
    R : number, or array-like
            distributed resistance, in Ohm/m
    G : number, or array-like
            distributed conductance, in S/m


    Notes
    ----------
    if C,I,R,G are vectors they should be the same length
    
    :class:`DistributedCircuit` is `Media` object representing a 
    transmission line mode defined in terms of  distributed impedance
    and admittance values. 

    A Distributed Circuit may be defined in terms
    of the following attributes,

    ================================  ================  ================
    Quantity                          Symbol            Property
    ================================  ================  ================
    Distributed Capacitance           :math:`C^{'}`     :attr:`C`
    Distributed Inductance            :math:`L^{'}`     :attr:`L`
    Distributed Resistance            :math:`R^{'}`     :attr:`R`
    Distributed Conductance           :math:`G^{'}`     :attr:`G`
    ================================  ================  ================


    The following quantities may be calculated, which are functions of 
    angular frequency (:math:`\omega`):

    ===================================  ==================================================  ==============================
    Quantity                             Symbol                                              Property
    ===================================  ==================================================  ==============================
    Distributed Impedance                :math:`Z^{'} = R^{'} + j \\omega L^{'}`              :attr:`Z`
    Distributed Admittance               :math:`Y^{'} = G^{'} + j \\omega C^{'}`              :attr:`Y`
    ===================================  ==================================================  ==============================


    The properties which define their wave behavior:

    ===================================  ============================================  ==============================
    Quantity                             Symbol                                        Method
    ===================================  ============================================  ==============================
    Characteristic Impedance             :math:`Z_0 = \\sqrt{ \\frac{Z^{'}}{Y^{'}}}`     :func:`Z0`
    Propagation Constant                 :math:`\\gamma = \\sqrt{ Z^{'}  Y^{'}}`         :func:`gamma`
    ===================================  ============================================  ==============================

    Given the following definitions, the components of propagation
    constant are interpreted as follows:

    .. math::
        +\\Re e\\{\\gamma\\} = \\text{attenuation}

        -\\Im m\\{\\gamma\\} = \\text{forward propagation}
    

    See Also 
    --------
    from_media

    '''
    
    def __init__(self, frequency=None, z0=None, C=90e-12, L=280e-9, R=0, G=0,
                *args, **kwargs):
        super(DistributedCircuit, self).__init__(frequency=frequency, 
                                                 z0=z0)
        self.C, self.L, self.R, self.G = C,L,R,G


    def __str__(self):
        f=self.frequency
        try:
            output =  \
                'Distributed Circuit Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nL\'= %.2f, C\'= %.2f,R\'= %.2f, G\'= %.2f, '% \
                (self.L, self.C,self.R, self.G)
        except(TypeError):
            output =  \
                'Distributed Circuit Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nL\'= %.2f.., C\'= %.2f..,R\'= %.2f.., G\'= %.2f.., '% \
                (self.L[0], self.C[0],self.R[0], self.G[0])
        return output

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_media(cls, my_media, *args, **kwargs):
        '''
        Initializes a DistributedCircuit from an existing
        :class:'~skrf.media.media.Media' instance.

        Parameters
        ------------
        my_media : :class:'~skrf.media.media.Media' instance.
            the media object
        '''

        w  =  my_media.frequency.w
        gamma = my_media.gamma
        Z0 = my_media.Z0
        z0 = my_media.z0

        Y = gamma/Z0
        Z = gamma*Z0
        G,C = real(Y), imag(Y)/w
        R,L = real(Z), imag(Z)/w
        return cls(frequency = my_media.frequency, 
                   z0 = z0, C=C, L=L, R=R, G=G, *args, **kwargs)
    
    @classmethod
    def from_csv(self, *args, **kw):
        d = DefinedGammaZ0.from_csv(*args,**kw)
        return self.from_media(d)

    @property
    def Z(self):
        '''
        Distributed Impedance, :math:`Z^{'}`

        Defined as

        .. math::
                Z^{'} = R^{'} + j \\omega L^{'}


        Returns
        --------
        Z : numpy.ndarray
                Distributed impedance in units of ohm/m
        '''
        w  = self.frequency.w
        return self.R + 1j*w*self.L

    @property
    def Y(self):
        '''
        Distributed Admittance, :math:`Y^{'}`

        Defined as

        .. math::
                Y^{'} = G^{'} + j \\omega C^{'}

        Returns
        --------
        Y : numpy.ndarray
                Distributed Admittance in units of S/m
        '''

        w  = self.frequency.w
        return self.G + 1j*w*self.C

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

        return sqrt(self.Z/self.Y)

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
        return sqrt(self.Z*self.Y)
