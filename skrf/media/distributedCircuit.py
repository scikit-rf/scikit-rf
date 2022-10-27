"""
distributedCircuit (:mod:`skrf.media.distributedCircuit`)
============================================================

A transmission line mode defined in terms of distributed impedance and admittance values.

.. autosummary::
   :toctree: generated/

   DistributedCircuit

"""

from numpy import sqrt, real, imag
from .media import Media, DefinedGammaZ0
from ..constants import NumberLike
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .. frequency import Frequency


class DistributedCircuit(Media):
    r"""
    A transmission line mode defined in terms of distributed impedance and admittance values.

    Parameters
    ------------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of the media
    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characteristic impedance of the transmission
        line. if z0 is None then will default to Z0
    C : number, or array-like
            distributed capacitance, in F/m
    L : number, or array-like
            distributed inductance, in H/m
    R : number, or array-like
            distributed resistance, in Ohm/m
    G : number, or array-like
            distributed conductance, in S/m


    Notes
    -----
    if C,I,R,G are vectors they should be the same length

    :class:`DistributedCircuit` is `Media` object representing a
    transmission line mode defined in terms of distributed impedance
    and admittance values.

    A `DistributedCircuit` may be defined in terms
    of the following attributes:

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
    Distributed Impedance                :math:`Z^{'} = R^{'} + j \omega L^{'}`              :attr:`Z`
    Distributed Admittance               :math:`Y^{'} = G^{'} + j \omega C^{'}`              :attr:`Y`
    ===================================  ==================================================  ==============================


    The properties which define their wave behavior:

    ===================================  ============================================  ==============================
    Quantity                             Symbol                                        Method
    ===================================  ============================================  ==============================
    Characteristic Impedance             :math:`Z_0 = \sqrt{ \frac{Z^{'}}{Y^{'}}}`     :func:`Z0`
    Propagation Constant                 :math:`\gamma = \sqrt{ Z^{'}  Y^{'}}`         :func:`gamma`
    ===================================  ============================================  ==============================

    Given the following definitions, the components of propagation
    constant are interpreted as follows:

    .. math::

        +\Re e\{\gamma\} = \text{attenuation}

        -\Im m\{\gamma\} = \text{forward propagation}


    See Also
    --------
    from_media

    """

    def __init__(self, frequency: Union['Frequency', None] = None,
                 z0: Union[NumberLike, None] = None,
                 C: NumberLike = 90e-12, L: NumberLike = 280e-9,
                 R: NumberLike = 0, G: NumberLike = 0,
                *args, **kwargs):
        super().__init__(frequency=frequency,
                                                 z0=z0)
        self.C, self.L, self.R, self.G = C,L,R,G


    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_media(cls, my_media: Media, *args, **kwargs) -> Media:
        """
        Initializes a `DistributedCircuit` from an existing
        :class:`~skrf.media.media.Media` instance.

        Parameters
        ----------
        my_media : :class:`~skrf.media.media.Media` instance.
            the media object
            
        See Also
        --------
        :class:`~skrf.media.media.Media`
        """

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
        """
        Create a `DistributedCircuit` from numerical values stored in a csv file.

        The csv file format must be written by the function :func:`write_csv`,
        or similar method which produces the following format::

            f[$unit], Re(Z0), Im(Z0), Re(gamma), Im(gamma), Re(port Z0), Im(port Z0)
            1, 1, 1, 1, 1, 1, 1
            2, 1, 1, 1, 1, 1, 1
            .....

        See Also
        --------
        write_csv
        """        
        d = DefinedGammaZ0.from_csv(*args,**kw)
        return self.from_media(d)

    @property
    def Z(self) -> NumberLike:
        r"""
        Distributed Impedance, :math:`Z^{'}`.

        Defined as

        .. math::

                Z^{'} = R^{'} + j \omega L^{'}

        Returns
        -------
        Z : npy.ndarray
            Distributed impedance in units of ohm/m
        """
        w  = self.frequency.w
        Z = self.R + 1j*w*self.L
        # Avoid divide by zero.
        # Needs to be imaginary to avoid all divide by zeros in the media class.
        Z[Z.imag == 0] += 1j*1e-12
        return Z

    @property
    def Y(self) -> NumberLike:
        r"""
        Distributed Admittance, :math:`Y^{'}`.

        Defined as

        .. math::

                Y^{'} = G^{'} + j \omega C^{'}

        Returns
        -------
        Y : npy.ndarray
            Distributed Admittance in units of S/m
        """

        w  = self.frequency.w
        Y = self.G + 1j*w*self.C
        # Avoid divide by zero.
        # Needs to be imaginary to avoid all divide by zeros in the media class.
        Y[Y.imag == 0] += 1j*1e-12
        return Y

    @property
    def Z0(self) -> NumberLike:
        r"""
        Characteristic Impedance, :math:`Z0`

        .. math::

                Z_0 = \sqrt{ \frac{Z^{'}}{Y^{'}}}

        Returns
        -------
        Z0 : npy.ndarray
            Characteristic Impedance in units of ohms
        """

        return sqrt(self.Z/self.Y)

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
        return sqrt(self.Z*self.Y)
