"""
freespace (:mod:`skrf.media.freespace`)
========================================

A plane-wave (TEM Mode) in Freespace.

Represents a plane-wave in a homogeneous freespace, defined by
the space's relative permittivity and relative permeability.

.. autosummary::
    :toctree: generated/

    Freespace


"""
from scipy.constants import epsilon_0, mu_0
from .media import Media
from ..data import materials
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING
from numpy import real, sqrt, ones

if TYPE_CHECKING:
    from .. frequency import Frequency


class Freespace(Media):
    r"""
    A plane-wave (TEM Mode) in Freespace.

    A Freespace media can be constructed in two ways:

    * from complex, relative permativity and permeability OR
    * from real relative permativity and permeability with loss tangents.

    There is also a method to initialize from a
    existing distributed circuit, appropriately named
    :func:`Freespace.from_distributed_circuit`


    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of this transmission line medium
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characteristic impedance of the transmission
        line. if z0 is None then will default to Z0
    ep_r : number, array-like
        complex relative permittivity. negative imaginary is lossy.
    mu_r : number, array-like
        complex relative permeability. negative imaginary is lossy.
    ep_loss_tan : None, number, array-like
        electric loss tangent (of the permativity).
        If not None, imag(ep_r) is ignored.
    mu_loss_tan : None, number, array-like
        magnetic loss tangent (of the permeability).
        If not None, imag(mu_r) is ignored.
    rho : number, array-like, string or None
        resistivity (ohm-m) of the conductor walls. If array-like
        must be same length as frequency. if str, it must be a key in
        :data:`skrf.data.materials`.
        Default is None (lossless).
    \*args, \*\*kwargs : arguments and keyword arguments


    Examples
    --------
    >>> from skrf.media.freespace import Freespace
    >>> from skrf.frequency import Frequency
    >>> f = Frequency(75,110,101,'ghz')
    >>> Freespace(frequency=f, ep_r=11.9)
    >>> Freespace(frequency=f, ep_r=11.9-1.1j)
    >>> Freespace(frequency=f, ep_r=11.9, ep_loss_tan=.1)
    >>> Freespace(frequency=f, ep_r=11.9-1.1j, mu_r = 1.1-.1j)

    """

    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 ep_r: NumberLike = 1+0j, mu_r: NumberLike = 1+0j,
                 ep_loss_tan: Union[NumberLike, None] = None,
                 mu_loss_tan: Union[NumberLike, None] = None,
                 rho: Union[NumberLike, str, None] = None,
                 *args, **kwargs):

        Media.__init__(self, frequency=frequency, z0=z0)
        self.ep_r = ep_r
        self.mu_r = mu_r
        self.rho = rho

        self.ep_loss_tan = ep_loss_tan
        self.mu_loss_tan = mu_loss_tan

    def __str__(self) -> str:
        f = self.frequency
        output = 'Freespace  Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints)
        return output

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def ep(self) -> NumberLike:
        r"""
        Complex dielectric permittivity.

        If :math:`\tan\delta_e` is not defined:

        .. math::

            \varepsilon = \varepsilon_0 \varepsilon_r

        otherwise,

        .. math::

            \varepsilon = \varepsilon_0 \Re[\varepsilon_r] (1 - j\tan\delta_e)

        where :math:`\tan\delta_e` is the electric loss tangent.

        Returns
        -------
        ep : number or array-like
            Complex dielectric permittivity in F/m.

        """

        if self.ep_loss_tan is not None:
            ep_r = real(self.ep_r)*(1 - 1j*self.ep_loss_tan)
        else:
            ep_r = self.ep_r
        return ep_r*epsilon_0

    @property
    def mu(self) -> NumberLike:
        r"""
        Complex dielectric permeability.

        If :math:`\tan\delta_m` is not defined:

        .. math::

            \mu = \mu_0 \mu_r

        otherwise,

        .. math::

            \mu = \mu_0 \Re[\mu_r] (1 - j\tan\delta_m)

        where :math:`\tan\delta_m` is the magnetic loss tangent.

        Returns
        -------
        mu : number
            Complex permeability in H/m.
        """
        if self.mu_loss_tan is not None:
            mu_r = real(self.mu_r)*(1 -1j*self.mu_loss_tan)
        else:
            mu_r = self.mu_r
        return mu_r*mu_0


    @classmethod
    def from_distributed_circuit(cls, dc, *args, **kwargs) -> Media:
        r"""
        Initialize a freespace from :class:`~skrf.media.distributedCircuit.DistributedCircuit`.

        Parameters
        ----------
        dc: :class:`~skrf.media.distributedCircuit.DistributedCircuit`
            a DistributedCircuit object
        \*args, \*\*kwargs :
            passed to `Freespace.__init__

        Notes
        -----
        Here are the details::

            w = dc.frequency.w
            z= dc.Z/(w*mu_0)
            y= dc.Y/(w*epsilon_0)
            ep_r = -1j*y
            mu_r = -1j*z

        See Also
        --------
        skrf.media.distributedCircuit.DistributedCircuit

        """
        w = dc.frequency.w
        z= dc.Z/(w*mu_0)
        y= dc.Y/(w*epsilon_0)


        kw={}
        kw['ep_r'] = -1j*y
        kw['mu_r'] = -1j*z

        kwargs.update(kw)
        return cls(frequency=dc.frequency, *args, **kwargs)

    @property
    def rho(self) -> NumberLike:
        """
        Conductivity in ohm*m.

        Parameters
        ----------
        val : float, array-like or str
            the resistivity in ohm*m. If array-like must be same length
            as self.frequency. if str, it must be a key in
            :data:`~skrf.data.materials`.

        Examples
        --------
        >>> wg.rho = 2.8e-8
        >>> wg.rho = 2.8e-8 * ones(len(wg.frequency))
        >>> wg.rho = 'al'
        >>> wg.rho = 'aluminum'
        """
        return self._rho

    @rho.setter
    def rho(self, val):
        if isinstance(val, str):
            self._rho = materials[val.lower()]['resistivity(ohm*m)']
        else:
            self._rho=val

    @property
    def ep_with_rho(self) -> NumberLike:
        r"""
        Complex permittivity with resistivity absorbed into its imaginary component.
                          
        .. math::
            
            \varepsilon - j \frac{1}{\rho\omega}
        
        See Also
        --------
        rho
        ep
        """
        if self.rho is not None:
            return self.ep -1j/(self.rho*self.frequency.w)
        else:
            return self.ep

    @property
    def gamma(self) -> NumberLike:
        r"""
        Propagation Constant, :math:`\gamma`.

        Defined as,

        .. math::
            
                \gamma =  \sqrt{ Z^{'}  Y^{'}}

        Returns
        -------
        gamma : npy.ndarray
            Propagation Constant,

        Note
        ----
        The components of propagation constant are interpreted as follows:

        * positive real(gamma) = attenuation
        * positive imag(gamma) = forward propagation
        """
        ep = self.ep_with_rho
        return 1j*self.frequency.w * sqrt(ep*self.mu)

    @property
    def Z0(self) -> NumberLike:
        r"""
        Characteristic Impedance, :math:`Z0`.

        .. math::

                Z_0 = \sqrt{ \frac{Z^{'}}{Y^{'}}}

        Returns
        -------
        Z0 : npy.ndarray
            Characteristic Impedance in units of ohms
        """
        ep = self.ep_with_rho
        return sqrt(self.mu/ep)*ones(len(self))

    def plot_ep(self):
        """
        Plot the real and imaginary part of the complex permittivity.
        """
        self.plot(self.ep_r.real, label=r'ep_r real')
        self.plot(self.ep_r.imag, label=r'ep_r imag')

    def plot_mu(self):
        """
        Plot the real and imaginary part of the complex permeability.
        """
        self.plot(self.mu_r.real, label=r'mu_r real')
        self.plot(self.mu_r.imag, label=r'mu_r imag')

    def plot_ep_mu(self):
        """
        Plot the real and imaginary part of the complex permittivity with resistivity.       
        """
        self.plot_ep()
        self.plot_mu()
