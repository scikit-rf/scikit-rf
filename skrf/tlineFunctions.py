r"""
.. module:: skrf.tlineFunctions
===============================================
tlineFunctions (:mod:`skrf.tlineFunctions`)
===============================================

This module provides functions related to transmission line theory.

Impedance and Reflection Coefficient
--------------------------------------
These functions relate basic transmission line quantities such as
characteristic impedance, input impedance, reflection coefficient, etc.
Each function has two names. One is a long-winded but readable name and
the other is a short-hand variable-like names. Below is a table relating
these two names with each other as well as common mathematical symbols.

====================  ======================  ================================
Symbol                Variable Name           Long Name
====================  ======================  ================================
:math:`Z_l`           z_l                     load_impedance
:math:`Z_{in}`        z_in                    input_impedance
:math:`\Gamma_0`      Gamma_0                 reflection_coefficient
:math:`\Gamma_{in}`   Gamma_in                reflection_coefficient_at_theta
:math:`\theta`        theta                   electrical_length
====================  ======================  ================================

There may be a bit of confusion about the difference between the load
impedance the input impedance. This is because the load impedance **is**
the input impedance at the load. An illustration may provide some
useful reference.

Below is a (bad) illustration of a section of uniform transmission line
of characteristic impedance :math:`Z_0`, and electrical length
:math:`\theta`. The line is terminated on the right with some
load impedance, :math:`Z_l`. The input impedance :math:`Z_{in}` and
input reflection coefficient :math:`\Gamma_{in}` are
looking in towards the load from the distance :math:`\theta` from the
load.

.. math::
        Z_0, \theta

        \text{o===============o=}[Z_l]

        \to\qquad\qquad\qquad\quad\qquad \qquad \to \qquad \quad

        Z_{in},\Gamma_{in}\qquad\qquad\qquad\qquad\quad Z_l,\Gamma_0

So, to clarify the confusion,

.. math::
        Z_{in}= Z_{l},\qquad\qquad
        \Gamma_{in}=\Gamma_l \text{ at }  \theta=0


Short names
+++++++++++++
.. autosummary::
        :toctree: generated/

        theta

        zl_2_Gamma0
        zl_2_zin
        zl_2_Gamma_in
        zl_2_swr
        zl_2_total_loss

        Gamma0_2_zl
        Gamma0_2_Gamma_in
        Gamma0_2_zin
        Gamma0_2_swr

Long-names
++++++++++++++
.. autosummary::
        :toctree: generated/

        electrical_length

        distance_2_electrical_length
        electrical_length_2_distance

        reflection_coefficient_at_theta
        reflection_coefficient_2_input_impedance
        reflection_coefficient_2_input_impedance_at_theta
        reflection_coefficient_2_propagation_constant

        input_impedance_at_theta
        load_impedance_2_reflection_coefficient
        load_impedance_2_reflection_coefficient_at_theta

        voltage_current_propagation



Distributed Circuit and Wave Quantities
----------------------------------------
.. autosummary::
        :toctree: generated/

        distributed_circuit_2_propagation_impedance
        propagation_impedance_2_distributed_circuit

Transmission Line Physics
---------------------------------
.. autosummary::
        :toctree: generated/

        skin_depth
        surface_resistivity
"""

from . constants import NumberLike, INF, ONE
import numpy as npy
from numpy import pi, sqrt, exp, array, imag, real

from scipy.constants import mu_0
from . import mathFunctions as mf


def skin_depth(f: NumberLike, rho: float, mu_r: float):
    r"""
    Skin depth for a material.

    The skin depth is calculated as:


    .. math::

        \delta = \sqrt{\frac{ \rho }{ \pi f \mu_r \mu_0 }}

    See www.microwaves101.com [#]_ or wikipedia [#]_ for more info.

    Parameters
    ----------
    f : number or array-like
        frequency, in Hz
    rho : number of array-like
        bulk resistivity of material, in ohm*m
    mu_r : number or array-like
        relative permeability of material

    Returns
    -------
    skin depth : number or array-like
        the skin depth, in meter

    References
    ----------
    .. [#] https://www.microwaves101.com/encyclopedias/skin-depth
    .. [#] http://en.wikipedia.org/wiki/Skin_effect

    See Also
    --------
    surface_resistivity

    """
    return sqrt(rho/(pi*f*mu_r*mu_0))


def surface_resistivity(f: NumberLike, rho: float, mu_r: float):
    r"""
    Surface resistivity.

    The surface resistivity is calculated as:


    .. math::

        \frac{ \rho }{ \delta }

    where :math:`\delta` is the skin depth from :func:`skin_depth`.

    See www.microwaves101.com [#]_ or wikipedia [#]_ for more info.

    Parameters
    ----------
    f : number or array-like
        frequency, in Hz
    rho : number or array-like
        bulk resistivity of material, in ohm*m
    mu_r : number or array-like
        relative permeability of material

    Returns
    -------
    surface resistivity : number of array-like
        Surface resistivity in ohms/square

    References
    ----------
    .. [#] https://www.microwaves101.com/encyclopedias/sheet-resistance
    .. [#] https://en.wikipedia.org/wiki/Sheet_resistance

    See Also
    --------
    skin_depth
    """
    return rho/skin_depth(rho=rho, f=f, mu_r=mu_r)


def distributed_circuit_2_propagation_impedance(distributed_admittance: NumberLike,
        distributed_impedance: NumberLike):
    r"""
    Convert distributed circuit values to wave quantities.

    This converts complex distributed impedance and admittance to
    propagation constant and characteristic impedance. The relation is

    .. math::
        Z_0 = \sqrt{ \frac{Z^{'}}{Y^{'}}}
        \quad\quad
        \gamma = \sqrt{ Z^{'}  Y^{'}}

    Parameters
    ----------
    distributed_admittance : number, array-like
        distributed admittance
    distributed_impedance :  number, array-like
        distributed impedance

    Returns
    -------
    propagation_constant : number, array-like
        distributed impedance
    characteristic_impedance : number, array-like
        distributed impedance

    See Also
    --------
        propagation_impedance_2_distributed_circuit : opposite conversion
    """
    propagation_constant = \
            sqrt(distributed_impedance*distributed_admittance)
    characteristic_impedance = \
            sqrt(distributed_impedance/distributed_admittance)
    return (propagation_constant, characteristic_impedance)


def propagation_impedance_2_distributed_circuit(propagation_constant: NumberLike,
        characteristic_impedance: NumberLike):
    r"""
    Convert wave quantities to distributed circuit values.

    Convert complex propagation constant and characteristic impedance
    to distributed impedance and admittance. The relation is,

    .. math::
        Z^{'} = \gamma  Z_0 \quad\quad
        Y^{'} = \frac{\gamma}{Z_0}

    Parameters
    ----------
    propagation_constant : number, array-like
        distributed impedance
    characteristic_impedance : number, array-like
        distributed impedance

    Returns
    -------
    distributed_admittance : number, array-like
        distributed admittance
    distributed_impedance :  number, array-like
        distributed impedance


    See Also
    --------
        distributed_circuit_2_propagation_impedance : opposite conversion
    """
    distributed_admittance = propagation_constant/characteristic_impedance
    distributed_impedance = propagation_constant*characteristic_impedance
    return (distributed_admittance, distributed_impedance)


def electrical_length(gamma: NumberLike, f: NumberLike, d: NumberLike, deg: bool = False):
    r"""
    Electrical length of a section of transmission line.

    .. math::
        \theta = \gamma(f) \cdot d

    Parameters
    ----------
    gamma : number, array-like or function
        propagation constant. See Notes.
        If passed as a function, takes frequency in Hz as a sole argument.
    f : number or array-like
        frequency at which to calculate
    d : number or array-like
        length of line, in meters
    deg : Boolean
        return in degrees or not.

    Returns
    -------
    theta : number or array-like
        electrical length in radians or degrees, depending on  value of deg.

    See Also
    --------
        electrical_length_2_distance : opposite conversion

    Note
    ----
    The convention has been chosen that forward propagation is
    represented by the positive imaginary part of the value returned by
    the gamma function.
    """
    # if gamma is not a function, create a dummy function which return gamma
    if not callable(gamma):
        _gamma = gamma
        def gamma(f0): return _gamma

    # typecast to a 1D array
    f = array(f, dtype=float).reshape(-1)
    d = array(d, dtype=float).reshape(-1)

    if deg == False:
        return  gamma(f)*d
    elif deg == True:
        return  mf.radian_2_degree(gamma(f)*d )


def electrical_length_2_distance(theta: NumberLike, gamma: NumberLike, f0: NumberLike, deg: bool = True):
    r"""
    Convert electrical length to a physical distance.

    .. math::
        d = \frac{\theta}{\gamma(f_0)}

    Parameters
    ----------
    theta : number or array-like
        electrical length. units depend on `deg` option
    gamma : number, array-like or function
        propagation constant. See Notes.
        If passed as a function, takes frequency in Hz as a sole argument.
    f0 : number or array-like
        frequency at which to calculate gamma
    deg : Boolean
        return in degrees or not.

    Returns
    -------
    d : number or array-like (real)
        physical distance in m

    Note
    ----
    The convention has been chosen that forward propagation is
    represented by the positive imaginary part of the value returned by
    the gamma function.

    See Also
    --------
        distance_2_electrical_length: opposite conversion
    """
    # if gamma is not a function, create a dummy function which return gamma
    if not callable(gamma):
        _gamma = gamma
        def gamma(f0): return _gamma

    if deg:
        theta = mf.degree_2_radian(theta)
    return real(theta / gamma(f0))


def load_impedance_2_reflection_coefficient(z0: NumberLike, zl: NumberLike):
    r"""
    Reflection coefficient from a load impedance.

    Return the reflection coefficient for a given load impedance, and
    characteristic impedance.

    For a transmission line of characteristic impedance :math:`Z_0`
    terminated with load impedance :math:`Z_l`, the complex reflection
    coefficient is given by,

    .. math::
        \Gamma = \frac {Z_l - Z_0}{Z_l + Z_0}

    Parameters
    ----------
    z0 : number or array-like
        characteristic impedance
    zl : number or array-like
        load impedance (aka input impedance)

    Returns
    -------
    gamma : number or array-like
        reflection coefficient

    See Also
    --------
        Gamma0_2_zl : reflection coefficient to load impedance

    Note
    ----
    Inputs are typecasted to 1D complex array.
    """
    # typecast to a complex 1D array. this makes everything easier
    z0 = array(z0, dtype=complex).reshape(-1)
    zl = array(zl, dtype=complex).reshape(-1)

    # handle singularity  by numerically representing inf as big number
    zl[(zl == npy.inf)] = INF

    return ((zl - z0)/(zl + z0))


def reflection_coefficient_2_input_impedance(z0: NumberLike, Gamma: NumberLike):
    r"""
    Input impedance from a load reflection coefficient.

    Calculate the input impedance given a reflection coefficient and
    characteristic impedance.

    .. math::
        Z_0 \left(\frac {1 + \Gamma}{1-\Gamma} \right)

    Parameters
    ----------
    Gamma : number or array-like
        complex reflection coefficient
    z0 : number or array-like
        characteristic impedance

    Returns
    -------
    zin : number or array-like
        input impedance

    """
    # typecast to a complex 1D array. this makes everything easier
    Gamma = array(Gamma, dtype=complex).reshape(-1)
    z0 = array(z0, dtype=complex).reshape(-1)

    # handle singularity by numerically representing inf as close to 1
    Gamma[(Gamma == 1)] = ONE

    return z0*((1.0 + Gamma)/(1.0 - Gamma))


def reflection_coefficient_at_theta(Gamma0: NumberLike, theta: NumberLike):
    r"""
    Reflection coefficient at a given electrical length.

    .. math::
            \Gamma_{in} = \Gamma_0 e^{-2 \theta}

    Parameters
    ----------
    Gamma0 : number or array-like
        reflection coefficient at theta=0
    theta : number or array-like
        electrical length (may be complex)

    Returns
    -------
    Gamma_in : number or array-like
        input reflection coefficient

    """
    Gamma0 = array(Gamma0, dtype=complex).reshape(-1)
    theta = array(theta, dtype=complex).reshape(-1)
    return Gamma0 * exp(-2*theta)


def input_impedance_at_theta(z0: NumberLike, zl: NumberLike, theta: NumberLike):
    """
    Input impedance from load impedance at a given electrical length.

    Input impedance of load impedance zl at a given electrical length,
    given characteristic impedance z0.

    Parameters
    ----------
    z0 : number or array-like
        characteristic impedance
    zl : number or array-like
        load impedance
    theta : number or array-like
        electrical length of the line (may be complex)

    Returns
    -------
    zin : number or array-like
        input impedance at theta

    """
    Gamma0 = load_impedance_2_reflection_coefficient(z0=z0, zl=zl)
    Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
    return reflection_coefficient_2_input_impedance(z0=z0, Gamma=Gamma_in)


def load_impedance_2_reflection_coefficient_at_theta(z0: NumberLike, zl: NumberLike, theta: NumberLike):
    """
    Reflection coefficient of load at a given electrical length.

    Reflection coefficient of load impedance zl at a given electrical length,
    given characteristic impedance z0.

    Parameters
    ----------
    z0 : number or array-like
        characteristic impedance.
    zl : number or array-like
        load impedance
    theta : number or array-like
        electrical length of the line (may be complex).

    Returns
    -------
    Gamma_in : number or array-like
        input reflection coefficient at theta

    """
    Gamma0 = load_impedance_2_reflection_coefficient(z0=z0, zl=zl)
    Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
    return Gamma_in


def reflection_coefficient_2_input_impedance_at_theta(z0: NumberLike, Gamma0: NumberLike, theta: NumberLike):
    """
    Input impedance from load reflection coefficient at a given electrical length.

    Calculate the input impedance at electrical length theta, given a
    reflection coefficient and characteristic impedance of the medium.

    Parameters
    ----------
    z0 : number or array-like
        characteristic impedance.
    Gamma: number or array-like
        reflection coefficient
    theta: number or array-like
        electrical length of the line, (may be complex)

    Returns
    -------
    zin: number or array-like
        input impedance at theta

    """
    Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
    zin = reflection_coefficient_2_input_impedance(z0=z0, Gamma=Gamma_in)
    return zin


def reflection_coefficient_2_propagation_constant(Gamma_in: NumberLike, Gamma_l: NumberLike, d: NumberLike):
    r"""
    Propagation constant from line input and load reflection coefficients.

    Calculate the propagation constant of a line of length d, given the
    reflection coefficient and characteristic impedance of the medium.

    .. math::
        \Gamma_{in} = \Gamma_l e^{-2 j \gamma \cdot d}
        \to \gamma = -\frac{1}{2 d} \ln \left ( \frac{ \Gamma_{in} }{ \Gamma_l } \right )

    Parameters
    ----------
    Gamma_in : number or array-like
        input reflection coefficient
    Gamma_l :  number or array-like
        load reflection coefficient
    d : number or array-like
        length of line, in meters

    Returns
    -------
    gamma : number (complex) or array-like
        propagation constant (see notes)

    Note
    ----
    The convention has been chosen that forward propagation is
    represented by the positive imaginary part of gamma.

    """
    gamma = -1/(2*d) * npy.log(Gamma_in/Gamma_l)
    # the imaginary part of gamma (=beta) cannot be negative with the given
    # definition of gamma. Thus one should take the first modulo positive value
    gamma.imag = gamma.imag % (pi/d)

    return gamma


def Gamma0_2_swr(Gamma0: NumberLike):
    r"""
    Standing Wave Ratio (SWR) for a given reflection coefficient.

    Standing Wave Ratio value is defined by:

    .. math::
        VSWR = \frac{1 + |\Gamma_0|}{1 - |\Gamma_0|}

    Parameters
    ----------
    Gamma0 : number or array-like
        Reflection coefficient

    Returns
    -------
    swr : number or array-like
        Standing Wave Ratio.

    """
    return (1 + npy.abs(Gamma0)) / (1 - npy.abs(Gamma0))


def zl_2_swr(z0: NumberLike, zl: NumberLike):
    r"""
    Standing Wave Ratio (SWR) for a given load impedance.

    Standing Wave Ratio value is defined by:

    .. math::
        VSWR = \frac{1 + |\Gamma|}{1 - |\Gamma|}

    where

    .. math::
        \Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}

    Parameters
    ----------
    z0 : number or array-like
        line characteristic impedance [Ohm]
    zl : number or array-like
        load impedance [Ohm]

    Returns
    -------
    swr : number or array-like
        Standing Wave Ratio.

    """
    Gamma0 = load_impedance_2_reflection_coefficient(z0, zl)
    return Gamma0_2_swr(Gamma0)


def voltage_current_propagation(v1: NumberLike, i1: NumberLike, z0: NumberLike, theta: NumberLike):
    """
    Voltages and currents calculated on electrical length theta of a transmission line.

    Give voltage v2 and current i1 at theta, given voltage v1 
    and current i1 at theta=0 and given characteristic parameters gamma and z0.

    ::

        i1                          i2
        ○-->---------------------->--○

        v1         gamma,z0         v2

        ○----------------------------○

        <------------ d ------------->

        theta=0                   theta

    Uses (inverse) ABCD parameters of a transmission line.

    Parameters
    ----------
    v1 : array-like (nfreqs,)
        total voltage at z=0
    i1 : array-like (nfreqs,)
        total current at z=0, directed toward the transmission line
    z0: array-like (nfreqs,)
        characteristic impedance
    theta : number or array-like (nfreq, ntheta)
        electrical length of the line (may be complex).

    Return
    ------
    v2 : array-like (nfreqs, ntheta)
        total voltage at z=d
    i2 : array-like (nfreqs, ndtheta
        total current at z=d, directed outward the transmission line
    """
    # outer product by broadcasting of the electrical length
    # theta = gamma[:, npy.newaxis] * d  # (nbfreqs x nbd)
    # ABCD parameters of a transmission line (gamma, z0)
    A = npy.cosh(theta)
    B = z0*npy.sinh(theta)
    C = npy.sinh(theta)/z0
    D = npy.cosh(theta)
    # transpose and de-transpose operations are necessary
    # for linalg.inv to inverse square matrices
    ABCD = npy.array([[A, B],[C, D]]).transpose()
    inv_ABCD = npy.linalg.inv(ABCD).transpose()

    v2 = inv_ABCD[0,0] * v1 + inv_ABCD[0,1] * i1
    i2 = inv_ABCD[1,0] * v1 + inv_ABCD[1,1] * i1
    return v2, i2


def zl_2_total_loss(z0: NumberLike, zl: NumberLike, theta: NumberLike):
    r"""
    Total loss of a terminated transmission line (in natural unit).

    The total loss expressed in terms of the load impedance is [#]_ :

    .. math::
        TL = \frac{R_{in}}{R_L} \left| \cosh \theta  + \frac{Z_L}{Z_0} \sinh\theta \right|^2

    Parameters
    ----------
    z0 : number or array-like
        characteristic impedance.
    zl : number or array-like
        load impedance
    theta : number or array-like
        electrical length of the line (may be complex).

    Returns
    -------
    total_loss: number or array-like
        total loss in natural unit

    References
    ----------
    .. [#] Steve Stearns (K6OIK), Transmission Line Power Paradox and Its Resolution.
        ARRL PacificonAntenna Seminar, Santa Clara, CA, October 10-12, 2014.
        https://www.fars.k6ya.org/docs/K6OIK-A_Transmission_Line_Power_Paradox_and_Its_Resolution.pdf

    """
    Rin = npy.real(zl_2_zin(z0, zl, theta))
    total_loss = Rin/npy.real(zl)*npy.abs(npy.cosh(theta) + zl/z0*npy.sinh(theta))**2
    return total_loss


# short hand convenience.
# admittedly these follow no logical naming scheme, but they closely
# correspond to common symbolic conventions, and are convenient
theta = electrical_length
distance_2_electrical_length = electrical_length

zl_2_Gamma0 = load_impedance_2_reflection_coefficient
Gamma0_2_zl = reflection_coefficient_2_input_impedance

zl_2_zin = input_impedance_at_theta
zl_2_Gamma_in = load_impedance_2_reflection_coefficient_at_theta

Gamma0_2_Gamma_in = reflection_coefficient_at_theta
Gamma0_2_zin = reflection_coefficient_2_input_impedance_at_theta
