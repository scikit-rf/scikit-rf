"""
circularWaveguide (:mod:`skrf.media.circularWaveguide`)
================================================================

.. autosummary::
   :toctree: generated/

   CircularWaveguide

"""
from scipy.constants import  epsilon_0, mu_0, pi
from scipy.special import  jn_zeros, jnp_zeros
from numpy import sqrt, where
import numpy as npy
from .media import Media
from ..data import materials
from .freespace import Freespace
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .. frequency import Frequency
    

class CircularWaveguide(Media):
    r"""
    A single mode of a homogeneously filled Circular Waveguide
        
    Represents a single mode of a homogeneously filled circular
    waveguide of cross-section `r^2 pi`. The mode is determined by
    `mode-type` (`'te'` or `'tm'`) and mode indices ( `m` and `n` ). 
    Corrugated circular waveguides, which also support HE modes, are not 
    supported.
    
    
    ====================================  =============  ===============
    Quantity                              Symbol         Variable
    ====================================  =============  ===============
    Characteristic Wave Number            :math:`k_0`    :attr:`k0`
    Cut-off Wave Number                   :math:`k_c`    :attr:`kc`
    Longitudinal Wave Number              :math:`k_z`    :attr:`gamma`
    Transverse Wave Number (a)            :math:`k_x`    :attr:`kx`
    Transverse Wave Number (b)            :math:`k_y`    :attr:`ky`
    Characteristic Impedance              :math:`z_0`    :attr:`z0`
    ====================================  =============  ===============
    
    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of this transmission line medium
    z0 : number, array-like, or None
        the port impedance for media. Only needed if it's different
        from the characteristic impedance of the transmission
        line. if z0 is None then will default to Z0
    r : number
        radius of the waveguide, in meters.
    mode_type : ['te','tm']
        mode type, transverse electric (te) or transverse magnetic
        (tm) to-z. where z is direction of propagation
    m : int
        mode index in 'phi'-direction, the azimuthal index
    n : int
        mode index in 'r'-direction, the radial index
    ep_r : number, array-like,
        filling material's relative permittivity
    mu_r : number, array-like
        filling material's relative permeability
    rho : number, array-like, string
        resistivity (ohm-m) of the conductor walls. If array-like
        must be same length as frequency. if str, it must be a key in
        `skrf.data.materials`.
    
    \*args, \*\*kwargs : arguments, keyword arguments
        passed to :class:`~skrf.media.media.Media`'s constructor
        (:func:`~skrf.media.media.Media.__init__`


    Examples
    --------   
    In the following example an ideal waveguide of 2.39 mm diameter is 
    constructed for the high W band, operated in the fundamental TE11 mode.
    If no conductivity is provided the walls are treated as perfect 
    electric conductors.

    >>> freq = rf.Frequency(88, 110, 101, 'ghz')
    >>> rf.CircularWaveguide(freq, r=0.5 * 2.39e-3)

    """
    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 r: NumberLike = 1,
                 mode_type: str = 'te', m: int = 1, n: int = 1, 
                 ep_r: NumberLike = 1, mu_r: NumberLike = 1, 
                 rho: Union[NumberLike, str, None] = None,
                 *args, **kwargs):
        
        Media.__init__(self, frequency=frequency,z0=z0)
        
        if mode_type.lower() not in ['te','tm']:
            raise ValueError('mode_type must be either \'te\' or \'tm\'')

        
        self.r = r
        self.mode_type = mode_type.lower()
        self.m = m
        self.n = n
        self.ep_r = ep_r
        self.mu_r = mu_r
        self.rho = rho
        
    def __str__(self) -> str:
        f=self.frequency
        output =  \
                'Circular Waveguide Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints) + \
                '\n r= %.2em'% \
                (self.r)
        return output

    def __repr__(self) -> str:
        return self.__str__()

    
    @classmethod
    def from_Z0(cls, frequency: 'Frequency', Z0: NumberLike,
                f: NumberLike, ep_r: NumberLike = 1, mu_r: NumberLike = 1, 
                **kwargs):
        r"""
        Initialize from specified impedance at a given frequency, assuming the 
        fundamental TE11 mode.
        
        Parameters
        ----------
        frequency : Frequency Object
        Z0 : number /array
            characteristic impedance to create at `f`
        f : number 
            frequency (in Hz) at which the resultant waveguide has the 
            characteristic impedance Z0
        ep_r : number, array-like,
            filling material's relative permittivity
        mu_r : number, array-like
            filling material's relative permeability
        \*\*kwargs : arguments, keyword arguments
            passed to :class:`~skrf.media.media.Media`'s constructor
            (:func:`~skrf.media.media.Media.__init__`            
        """
        
        mu = mu_0*mu_r
        ep = epsilon_0*ep_r
        w = 2*pi*f
        # if self.mode_type =="te":
        u = jnp_zeros(1, 1)[-1] 
        r =u/(w*mu) * 1./sqrt(1/(Z0*1j)**2+ep/mu)
        
        kwargs.update(dict(frequency=frequency, r=r, m=1, n=1, ep_r=ep_r, mu_r=mu_r))
        
        return cls(**kwargs)
    
    @property
    def ep(self) -> NumberLike:
        """
        The permativity of the filling material.

        Returns
        -------
        ep : number
            filling material's relative permittivity
        """
        return self.ep_r * epsilon_0

    @property
    def mu(self) -> NumberLike:
        """
        The permeability of the filling material.

        Returns
        -------
        mu : number
            filling material's relative permeability

        """
        return self.mu_r * mu_0

    @property
    def k0(self) -> NumberLike:
        r"""
        Characteristic wave number.
        
        .. math::
            
            k_0 = \omega \sqrt{\varepsilon \mu}

        Returns
        -------
        k0 : number
            characteristic wave number
        """
        return 2*pi*self.frequency.f*sqrt(self.ep * self.mu)
    
    @property
    def kc(self) -> NumberLike:
        r"""
        Cut-off wave number.

        Defined as

        .. math::

                k_c = \frac{u_{mn}}{R}

        where R is the radius of the waveguide, and u_mn is:

        * the n-th root of the m-th Bessel function for 'tm' mode
        * the n-th root of the Derivative of the m-th Bessel function for 'te' mode.

        Returns
        -------
        kc : number
            cut-off wavenumber
        """
        if self.mode_type =="te":
            u = jnp_zeros(self.m, self.n)[-1] 
        elif self.mode_type =="tm":
            u = jn_zeros(self.m,self.n)[-1] 
        return u/self.r

    @property
    def f_cutoff(self) -> NumberLike:
        r"""
        cutoff frequency for this mode

        .. math::

            f_c = \frac{v}{2 \pi} \frac{u_{mn}}{R}

        where R is the radius of the waveguide, and u_mn is:

        * the n-th root of the m-th Bessel function for 'tm' mode
        * the n-th root of the Derivative of the m-th Bessel function for 'te' mode.

        and v= 1/sqrt(ep*mu) is the bulk velocity inside the filling material.
        """
        v = 1/sqrt(self.ep*self.mu)
        return v* self.kc/(2*npy.pi)
        
    @property
    def f_norm(self) -> NumberLike:
        """
        frequency vector normalized to cutoff
        """
        return self.frequency.f/self.f_cutoff

    @property
    def rho(self) -> NumberLike:
        """
        conductivity of sidewalls in ohm*m

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
        """        
        # if self.roughness != None:
        #     delta = skin_depth(self.frequency.f, self._rho, self.mu_r)
        #     k_w = 1. +exp(-(delta/(2*self.roughness))**1.6)
        #     return self._rho*k_w**2

        return self._rho

    @rho.setter
    def rho(self, val):
        if isinstance(val, str):
            self._rho = materials[val.lower()]['resistivity(ohm*m)']
        else:
            self._rho=val

    @property
    def lambda_guide(self) -> NumberLike:
        r"""
        Guide wavelength.
        
        .. math::
            
            \lambda_g = 2\pi/\beta

        the distance in which the phase of the field increases by 2 pi.
        """
        return 2*pi/self.beta

    @property
    def lambda_cutoff(self) -> NumberLike:
        r"""
        Cutoff wavelength.

        .. math::

            \lambda_c = v/f_c

        where v= 1/sqrt(ep*mu)
        """
        v = 1/sqrt(self.ep*self.mu)
        return v/self.f_cutoff

    @property
    def gamma(self) -> NumberLike:
        r"""
        The propagation constant (aka Longitudinal wave number)

        Defined as

        .. math::

                k_z = \pm j \sqrt {k_0^2 - k_c^2}

        This is:

        * IMAGINARY for propagating modes
        * REAL  for non-propagating modes,

        Returns
        -------
        gamma :  number
            The propagation constant


        """
        # This also holds for the circular waveguide
        ## haringtons form
        if False: #self.m==1 and self.n==0:
            fs = Freespace(frequency=self.frequency, 
                           ep_r=self.ep_r, 
                           mu_r=self.mu_r)
                           
            g= where(self.f_norm>1.,
                     sqrt(1-self.f_norm**(-2))*fs.gamma, # cutton
                 -1j*sqrt(1-self.f_norm**(2))*fs.gamma)# cutoff
        
        else:
            # TODO:  fix this for lossy ep/mu (remove abs?)
            k0,kc = self.k0, self.kc
            g=  1j*sqrt(abs(k0**2 - kc**2)) * (k0>kc) +\
                    sqrt(abs(kc**2- k0**2))*(k0<kc) + \
                    0*(kc==k0) 

        g = g + self.alpha_c *(self.rho is not None)
        
        return g 
        
        
    @property
    def alpha_c(self) -> NumberLike:
        """
        Loss due to finite conductivity of the sidewalls for the fundamental mode TE11. Higher order 
        modes are not implemented, as well as effects due to surface roughness.

        In units of Np/m
        
        See property `rho` for setting conductivity.

        Effects of finite conductivity are taken from [#]_, but expressed in the same terms as in [#]_.

        References
        ----------

        .. [#] Eq. (3.133), Chapter 3.4, Microwave Engineering, Pozar David, 2011
        .. [#] Eq. (9.8.1), Chapter 9, Electromagnetic Waves and Antennas by Sophocles J. Orfanidis
            http://eceweb1.rutgers.edu/~orfanidi/ewa/
            
        See Also
        --------
        rho
        """

        # TODO: Generalize to higher order modes
        if (self.mode_type != "te") or (self.m != 1) or (self.n != 1): 
            raise NotImplementedError

        if self.rho is None:
            return 0
        r, w, ep, rho, f_n = self.r, self.frequency.w, self.ep, \
            self.rho, self.f_norm
        u= self.kc*r
        return 1./r * sqrt( (w*ep)/(2./rho) ) * ( (1/f_n)**2 + 1/(u**2 - 1) ) \
            /sqrt(1-(1/f_n)**2)

    @property
    def Z0(self) -> NumberLike:
        """
        The characteristic impedance.
        """
        omega = self.frequency.w
        impedance_dict = {'te':   1j*omega*self.mu/(self.gamma),
                          'tm':   -1j*self.gamma/(omega*self.ep),\
                         }

        return impedance_dict[self.mode_type]