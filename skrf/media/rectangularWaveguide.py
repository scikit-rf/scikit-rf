

'''
.. module:: skrf.media.rectangularWaveguide
================================================================
rectangularWaveguide (:mod:`skrf.media.rectangularWaveguide`)
================================================================

Rectangular Waveguide class
'''
from scipy.constants import  epsilon_0, mu_0,pi,c
from numpy import sqrt, exp
from media import Media
from ..data import materials
from ..tlineFunctions import skin_depth

class RectangularWaveguide(Media):
    '''
    Rectangular Waveguide medium.

    Represents a single mode of a homogeneously filled rectangular
    waveguide of cross-section `a` x `b`. The mode is determined by
    mode-type (te or tm) and mode indecies ( m and n ).


    ====================================  =============  ===============
    Quantity                              Symbol         Variable
    ====================================  =============  ===============
    Characteristic Wave Number            :math:`k_0`    :attr:`k0`
    Cut-off Wave Number                   :math:`k_c`    :attr:`kc`
    Longitudinal Wave Number              :math:`k_z`    :attr:`kz`
    Transverse Wave Number (a)            :math:`k_x`    :attr:`kx`
    Transverse Wave Number (b)            :math:`k_y`    :attr:`ky`
    Characteristic Impedance              :math:`Z_0`    :attr:`Z0`
    ====================================  =============  ===============

    '''
    def __init__(self, frequency, a, b=None, mode_type = 'te', m=1, \
            n=0, ep_r=1, mu_r=1, rho=None, roughness=None, *args, **kwargs):
        '''
        RectangularWaveguide initializer

        Parameters
        ----------
        frequency : class:`~skrf.frequency.Frequency` object
                frequency band for this media
        a : number
                width of waveguide, in meters.
        b : number
                height of waveguide, in meters. If `None` defaults to a/2
        mode_type : ['te','tm']
                mode type, transverse electric (te) or transverse magnetic
                (tm) to-z. where z is direction of propagation
        m : int
                mode index in 'a'-direction
        n : int
                mode index in 'b'-direction
        ep_r : number, array-like,
                filling material's relative permativity
        mu_r : number, array-like
                filling material's relative permeability
        rho : number, array-like, string
            resistivity (ohm-m) of the conductor walls. If array-like 
            must be same length as frequency. if str, it must be a key in 
            `skrf.data.materials`.
        roughness : number, or array-like
            surface roughness of the conductor walls in units of RMS 
            deviation from surface
            
        *args,**kwargs : arguments, keywrod arguments
                passed to :class:`~skrf.media.media.Media`'s constructor
                (:func:`~skrf.media.media.Media.__init__`


        Examples
        ------------
        Most common usage is standard aspect ratio (2:1) dominant
        mode, TE10 mode of wr10 waveguide can be constructed by

        >>> freq = rf.Frequency(75,110,101,'ghz')
        >>> rf.RectangularWaveguide(freq, 100*mil)
        '''
        if b is None:
            b = a/2.
        if mode_type.lower() not in ['te','tm']:
            raise ValueError('mode_type must be either \'te\' or \'tm\'')

        self.frequency = frequency
        self.a = a
        self.b = b
        self.mode_type = mode_type
        self.m = m
        self.n = n
        self.ep_r = ep_r
        self.mu_r = mu_r
        self.rho = rho
        self.roughness = roughness
        Media.__init__(self,\
                frequency = frequency,\
                propagation_constant = self.kz, \
                characteristic_impedance = self.Z0,\
                *args, **kwargs)

    def __str__(self):
        f=self.frequency
        output =  \
                'Rectangular Waveguide Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\n a= %.2em, b= %.2em'% \
                (self.a,self.b)
        return output

    def __repr__(self):
        return self.__str__()
    
    def __getstate__(self):
        '''
        method needed to allow for pickling
        '''
        return dict([(k, self.__dict__[k]) for k in \
            ['frequency','_z0','kz','a','b','mode_type','m','n','ep_r','mu_r']])
        
    
    @property
    def ep(self):
        '''
        The permativity of the filling material

        Returns
        -------
        ep : number
                filling material's relative permativity
        '''
        return self.ep_r * epsilon_0

    @property
    def mu(self):
        '''
        The permeability of the filling material

        Returns
        -------
        mu : number
                filling material's relative permeability

        '''
        return self.mu_r * mu_0

    @property
    def k0(self):
        '''
        Characteristic wave number

        Returns
        -------
        k0 : number
                characteristic wave number
        '''
        return 2*pi*self.frequency.f*sqrt(self.ep * self.mu)

    @property
    def ky(self):
        '''
        Eigen-value in the `b` direction.

        Defined as

        .. math::

                k_y = n \\frac{\pi}{b}

        Returns
        -------
        ky : number
                eigen-value in `b` direction
        '''
        return self.n*pi/self.b

    @property
    def kx(self):
        '''
        Eigen value in the 'a' direction

        Defined as

        .. math::

                k_x = m \\frac{\pi}{a}

        Returns
        -------
        kx : number
                eigen-value in `a` direction
        '''
        return self.m*pi/self.a

    @property
    def kc(self):
        '''
        Cut-off wave number

        Defined as

        .. math::

                k_c = \\sqrt {k_x^2 + k_y^2} = \\sqrt {
                {m \\frac{\pi}{a}}^2 + {n \\frac{\pi}{b}}^2}

        Returns
        -------
        kc : number
                cut-off wavenumber
        '''
        return sqrt( self.kx**2 + self.ky**2)
    
    
    @property
    def f_cutoff(self):
        '''
        cutoff frequency for this mode
        
        .. math::
        
            max ( \frac{m \cdot v}{2a} , \frac{n \cdot v}{2b})
            
        where v= sqrt(ep*mu)
            
             
        
        '''
        v = 1/sqrt(self.ep*self.mu)
        if not ( self.m==1 and self.n==0):
            print ('f_cutoff not verified as correct for this mode ')
        return max(self.m*v/(2*self.a), self.n*v/(2*self.b))
    
    @property
    def f_norm(self):
        '''
        frequency vector normalized to cuttoff
        '''
        return self.frequency.f/self.f_cutoff
    
    @property
    def rho(self):
        '''
        conductivty of sidewalls in ohm*m
        
        Parameters
        --------------
        val : float, array-like or str
            the conductivity in ohm*m. If array-like must be same length
            as self.frequency. if str, it must be a key in 
            `skrf.data.materials`.
            
        Examples
        ---------
        >>> wg.rho = 2.8e-8
        >>> wg.rho = 2.8e-8 * ones(len(wg.frequency))
        >>> wg.rho = 'al'
        >>> wg.rho = 'aluminum'
        '''
        if self.roughness != None:
            delta = skin_depth(self.frequency.f, self._rho, self.mu_r)
            k_w = 1. +exp(-(delta/(2*self.roughness))**1.6)
            return self._rho*k_w**2
        
        return self._rho
        
    @rho.setter
    def rho(self, val):
        if isinstance(val, str):
            self._rho = materials[val.lower()]['resistivity(ohm*m)']
        else:
            self._rho=val
            
    
    @property
    def lambda_cutoff(self):
        '''
        cuttoff wavelength
        
        .. math:: 
            
            f_c * v
            
         where v= sqrt(ep*mu) 
        '''
        v = 1/sqrt(self.ep*self.mu)
        return self.f_cutoff*v
    
    
    def kz(self):
        '''
        The Longitudinal wave number, aka propagation constant.

        Defined as

        .. math::

                k_z = \\pm \\sqrt {k_0^2 - k_c^2}

        This is.
                * IMAGINARY for propagating modes
                * REAL  for non-propagating modes,

        Returns
        --------
        kz :  number
                The propagation constant


        '''
        k0,kc = self.k0, self.kc
        return \
                1j*sqrt(abs(k0**2 - kc**2)) * (k0>kc) +\
                sqrt(abs(kc**2- k0**2))*(k0<kc) + \
                0*(kc==k0) + self.alpha_c *(self.rho!=None)
    
    @property
    def alpha_c(self):
        '''
        Loss due to finite conductivity and roughness of sidewalls 
        
        In units of np/m
        See property `rho` for setting conductivity.
        
        Effects of finite conductivity are taken from [#]_. If 
        :attr:`roughness` is not None, then its effects the conductivity
        by 
        
        
        .. math:: 
        
            \\sigma_c = \\frac{\\sigma}{\\k_w^2}
            
        where 
            
        .. math::
            
            k_w = 1 + e^{(-\\delta/2h)^{1.6}}
        
            \\delta = skin depth 
            h = surface roughness 
            
        This is taken from Ansoft HFSS help documents.
        
        
        
        References
        --------------
        
        .. [#] Electromagnetic Waves and Antennas by Sophocles J. Orfanidis 
        http://eceweb1.rutgers.edu/~orfanidi/ewa/
        '''
        
        if self.rho==None: 
            return 0
        
        a,b,w,ep,rho,f_n = self.a, self.b, self.frequency.w, self.ep, \
            self.rho, self.f_norm
        
         
            
        return 1./b * sqrt( (w*ep)/(2./rho) ) * (1+2.*b/a*(1/f_n)**2)/\
            sqrt(1-(1/f_n)**2)
        

    def Z0(self):
        '''
        The characteristic impedance
        '''
        omega = self.frequency.w
        impedance_dict = {\
                'tez':  omega*self.mu/(-1*self.kz()),\
                'te':   omega*self.mu/(-1*self.kz()),\
                'tmz':  -1*self.kz()/(omega*self.ep),\
                'tm':   -1*self.kz()/(omega*self.ep),\
                }

        return impedance_dict[self.mode_type]
