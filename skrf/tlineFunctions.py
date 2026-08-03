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
        surface_impedance
"""

import warnings

import numpy as np
from numpy import array, exp, pi, real, sqrt

from . import constants as _const
from . import mathFunctions as mf
from .constants import INF, ONE, NumberLike


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
    surface_impedance

    """
    return sqrt(rho/(pi*f*mu_r*_const.mu_0))


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
    surface_impedance
    """
    return rho/skin_depth(rho=rho, f=f, mu_r=mu_r)


def surface_impedance(f: NumberLike, material_properties: list[dict] | dict,
                      rms_roughness: NumberLike, boundary_loc: NumberLike = 0,
                      distribution='norm', rtol: float = 1e-6):
    r"""
    Surface impedance of a rough conductor boundary.

    The rough boundary between two materials -- and, more generally, each
    boundary of a stack of coated conductors -- is described by the cumulative
    distribution function (CDF) of its surface height, which grades the
    material properties across the transition [#tegowski]_ [#gold]_:

    .. math::

        \mu(x) = \mu_1 + \sum_k \left( \mu_{k+1} - \mu_k \right) \mathrm{CDF}_k(x)

    and likewise for :math:`\epsilon(x)`, with :math:`x` the depth into the
    stack. The graded profile is discretized into short transmission line
    segments with propagation constant :math:`\gamma_i = j\omega\sqrt{\mu_i \epsilon_i}`
    and intrinsic impedance :math:`\eta_i = \sqrt{\mu_i / \epsilon_i}`, and the
    input impedance is transformed from the deepest material up to the surface:

    .. math::

        Z_i = \eta_i \frac{Z_{i+1} + \eta_i \tanh(\gamma_i \Delta x_i)}
                          {\eta_i + Z_{i+1} \tanh(\gamma_i \Delta x_i)}

    starting at the intrinsic impedance of the deepest material. The result is
    the surface impedance seen by a wave impinging on the stack: its real part
    is the effective surface resistance and its imaginary part the internal
    reactance.

    Parameters
    ----------
    f : number or array-like
        frequency, in Hz. Must be non-zero.
    material_properties : list of dict or dict
        Material stack, ordered from the outside medium the wave impinges from
        (vacuum, dielectric) down to the deepest conductor. Each dict takes
        any of the keys ``'sigma'`` (conductivity in S/m), ``'ep_r'``
        (relative permittivity) and ``'mu_r'`` (relative permeability), with
        defaults 0, 1 and 1; values are scalars or arrays matching ``f``.
        Describe a conductor by ``'sigma'`` (its ``'ep_r'``, if given, must be
        real-valued) and a lossy dielectric by a complex ``'ep_r'``. Passing a
        single dict is a shorthand for that material below vacuum, i.e.
        ``[{}, {...}]``. Consecutive materials are separated by one rough
        boundary each.
    rms_roughness : number or array-like
        rms roughness (Rq) of each boundary, in meter. One value per boundary;
        a scalar applies to all. Use 0 for an ideally smooth (step) boundary.
    boundary_loc : number or array-like, optional
        Mean depth of each boundary, in meter, in increasing order. The
        spacing of consecutive boundaries is the coating thickness.
        Default is 0.
    distribution : str, frozen scipy.stats distribution, callable, or list thereof, optional
        Height distribution of each boundary: the name of a scipy.stats
        continuous distribution without shape parameters (``'norm'``
        (default), ``'rayleigh'``, ``'uniform'``, ...), a frozen distribution
        for those with shape parameters (e.g. ``scipy.stats.gamma(2)``), or a
        function ``distribution(x, rms_roughness, boundary_loc)`` returning
        the CDF at depths ``x``. Named and frozen distributions are shifted
        and scaled to the given mean and standard deviation.
    rtol : float, optional
        Relative tolerance to which the discretization is refined. Default is 1e-6.

    Returns
    -------
    Zs : complex number or array-like
        surface impedance, in ohm (per square), at each frequency.

    Notes
    -----
    The material variation inside each segment is handled by a 4th-order
    Magnus scheme (Eqs. (252)-(254) in [#blanes]_), and the segment count over the
    rough transitions is doubled -- combining the last refinement levels by
    Richardson extrapolation -- until ``Zs`` changes by less than ``rtol``; a
    ``RuntimeWarning`` is issued if the refinement does not converge. ``Zs``
    is referenced 5 rms outside the outermost boundary, where the material is
    still the unperturbed outside medium.

    With ``rms_roughness=0`` a single smooth conductor recovers the classic
    :math:`(1 + j)\sqrt{\pi f \mu \rho}` known from :func:`surface_resistivity`.

    See Also
    --------
    skin_depth
    surface_resistivity

    References
    ----------
    .. [#tegowski] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob, "A Transmission
       Line Approach for Rough Conductor Surface Impedance Analysis," IEEE Trans.
       Microw. Theory Techn., vol. 71, no. 2, pp. 471-479, 2023.
       https://doi.org/10.1109/TMTT.2022.3206440
    .. [#gold] G. Gold and K. Helmreich, "Modeling of transmission lines with multiple
       coated conductors," 46th European Microwave Conference (EuMC), 2016.
       https://doi.org/10.1109/EuMC.2016.7824423
    .. [#blanes] S. Blanes, F. Casas, J. A. Oteo and J. Ros, "The Magnus expansion and
       some of its applications," Physics Reports, vol. 470, no. 5-6, pp. 151-238,
       2009. https://doi.org/10.1016/j.physrep.2008.11.001

    Examples
    --------
    Copper with 1 um rms roughness at 10 GHz, against the smooth value
    :math:`(1 + j) R_s`:

    >>> import skrf as rf
    >>> Zs = rf.tlineFunctions.surface_impedance(10e9, {'sigma': 58e6}, rms_roughness=1e-6)
    >>> Rs = rf.tlineFunctions.surface_resistivity(10e9, rho=1/58e6, mu_r=1)
    >>> print(f"rough: {Zs:.4f} ohm, smooth: {(1 + 1j)*Rs:.4f} ohm")
    rough: 0.0630+0.3356j ohm, smooth: 0.0261+0.0261j ohm

    The top side of a microstrip trace with an ENIG surface finish; a nearly
    flat (5 nm rms) stack of 0.05 um gold and 4 um nickel on copper:

    >>> mats = [{}, {'sigma': 41.1e6}, {'sigma': 14.5e6, 'mu_r': 20}, {'sigma': 58e6}]
    >>> Zs = rf.tlineFunctions.surface_impedance(10e9, mats, rms_roughness=5e-9,
    ...                                          boundary_loc=[0, 0.05e-6, 4.05e-6])
    >>> print(f"{Zs:.4f} ohm")
    0.1895+0.1012j ohm

    The bottom side of the trace; copper against the substrate
    (:math:`\epsilon_r = 3.2`), rough at the copper-substrate boundary, from
    1 to 100 GHz:

    >>> import numpy as np
    >>> f = np.logspace(9, 11, 201)
    >>> Zs = rf.tlineFunctions.surface_impedance(f, [{'ep_r': 3.2}, {'sigma': 58e6}],
    ...                                          rms_roughness=0.5e-6)
    >>> print(f"{Zs[-1]:.4f} ohm")  # at 100 GHz
    0.2743+1.5154j ohm
    """
    f = np.asarray(f, dtype=float)
    is_scalar = f.ndim == 0
    f = np.atleast_1d(f)
    if np.any(f <= 0):
        raise ValueError("Frequency must be non-zero and positive.")
    omega = 2*np.pi*f

    if isinstance(material_properties, dict):
        material_properties = [{}, material_properties]  # a single material below vacuum
    if len(material_properties) < 2:
        raise ValueError("material_properties needs at least two materials (outside medium and conductor).")
    mu_r, ep_r = _parse_material_properties(material_properties, omega)  # (M, nf) each

    # one value per boundary between consecutive materials; a single value applies to all
    K = len(material_properties) - 1
    rms_roughness = np.broadcast_to(np.asarray(rms_roughness, dtype=float), K)
    boundary_loc = np.broadcast_to(np.asarray(boundary_loc, dtype=float), K)
    distribution = list(np.broadcast_to(np.asarray(distribution, dtype=object), K))
    if np.any(np.diff(boundary_loc) < 0):
        raise ValueError("boundary_loc must be sorted in increasing depth order (matching material_properties).")

    # computation span: reach 5 rms past every boundary (the -5 Rq convention of Tegowski
    # et al.), not just the outermost one -- a deeper boundary that is rougher reaches
    # further out. Zs is referenced to x_start, where the material is still the outside medium.
    # Zero roughness is floored to 1e-14 m, making that boundary a practically ideal step.
    reach = 5*np.maximum(rms_roughness, 1e-14)
    x_start = np.min(boundary_loc - reach)
    x_end = np.max(boundary_loc + reach)

    # ------- surface impedance, doubling the segments until it converges -------
    # Doubling is the standard geometric refinement: total work stays about twice the
    # finest level, and the fixed 2:1 ratio gives the clean Richardson factor
    # 1/(2**4 - 1) below.
    eta_bulk = np.sqrt(_const.mu_0*mu_r[-1]/(_const.epsilon_0*ep_r[-1]))
    Zs_levels = []  # surface impedance at each refinement level
    converged = False
    for n_segments in (64*2**k for k in range(9)):  # 64, 128, ..., 16384
        edges = _segment_edges(rms_roughness, boundary_loc, x_start, x_end, n_segments)
        dx = np.diff(edges)
        n = dx.size  # actual segment count: n_segments in the transitions, plus the bulk gaps
        x = edges[:-1] + 0.5*dx  # segment midpoints

        # material at the two Gauss-Legendre points of each segment, x -+ dx/(2*sqrt(3))
        x_gauss = np.concatenate([x - np.sqrt(3)/6*dx, x + np.sqrt(3)/6*dx])
        cdf = np.array([_roughness_cdf(x_gauss, r, loc, dist)
                        for r, loc, dist in zip(rms_roughness, boundary_loc, distribution)])
        mu_gauss = _const.mu_0*(mu_r[0][:, None] + np.einsum('kf,kn->fn', np.diff(mu_r, axis=0), cdf))
        ep_gauss = _const.epsilon_0*(ep_r[0][:, None] + np.einsum('kf,kn->fn', np.diff(ep_r, axis=0), cdf))
        mu_1, mu_2 = mu_gauss[:, :n], mu_gauss[:, n:]
        ep_1, ep_2 = ep_gauss[:, :n], ep_gauss[:, n:]

        # Segment quantities of the 4th-order Magnus method (Eqs. (252)-(254) of Blanes et
        # al., see docstring references): the electrical length s (= gamma*dx if uniform),
        # the products eta*tanh(gamma*dx) and tanh(gamma*dx)/eta, and the correction d for
        # the material variation inside the segment (d = 0 when mu_1 = mu_2 and ep_1 = ep_2).
        mu_sum, ep_sum = mu_1 + mu_2, ep_1 + ep_2
        d = -np.sqrt(3)/12*dx**2*omega[:, None]**2*(ep_1*mu_2 - mu_1*ep_2)
        s = np.sqrt(d**2 - 0.25*dx**2*omega[:, None]**2*mu_sum*ep_sum)
        tanh_over_s = np.tanh(s)/s  # even in s, so the branch of the sqrt above does not matter
        eta_tanh = tanh_over_s*0.5j*dx*omega[:, None]*mu_sum       # eta*tanh(gamma*dx)
        tanh_over_eta = tanh_over_s*0.5j*dx*omega[:, None]*ep_sum  # tanh(gamma*dx)/eta
        d = tanh_over_s*d

        # transform from the deepest material up to the surface: the impedance recursion
        # of Tegowski et al. (Eq. (9)) divided by eta, with the correction d
        Zs = eta_bulk.copy()
        for i in range(n - 1, -1, -1):
            Zs = ((1 - d[:, i])*Zs + eta_tanh[:, i])/((1 + d[:, i]) + tanh_over_eta[:, i]*Zs)
        Zs_levels.append(Zs)

        # Richardson extrapolation: the 4th-order error drops by 2**4 when the segment
        # count doubles, so 1/(2**4 - 1) of the change cancels it. Stop once the
        # extrapolated Zs settles within rtol.
        if len(Zs_levels) >= 3:
            Zs = Zs_levels[-1] + (Zs_levels[-1] - Zs_levels[-2])/(2**4 - 1)
            Zs_before = Zs_levels[-2] + (Zs_levels[-2] - Zs_levels[-3])/(2**4 - 1)
            change = np.max(np.abs(Zs - Zs_before)/np.abs(Zs))
            if change < rtol:
                converged = True
                break
    if not converged:
        warnings.warn(f"surface_impedance did not converge to rtol={rtol:.1e} "
                      f"(last change {change:.1e}); returning the last refinement.",
                      RuntimeWarning, stacklevel=2)
    return Zs[0] if is_scalar else Zs


def _roughness_cdf(x: NumberLike, rms_roughness: float, boundary_loc: float, distribution):
    """
    Cumulative height distribution of a rough boundary, evaluated at depths x.

    ``distribution`` is the name of a scipy.stats continuous distribution
    without shape parameters ('norm', 'rayleigh', 'uniform', ...), a frozen
    one for those with shape parameters (e.g. ``scipy.stats.gamma(2)``), or a
    function ``distribution(x, rms_roughness, boundary_loc)`` returning the
    CDF. Named and frozen distributions are shifted and scaled to mean
    ``boundary_loc`` and standard deviation ``rms_roughness``.
    """
    import scipy.stats  # heavy import: deferred so that `import skrf` stays lean

    x = np.asarray(x, dtype=float)
    rms_roughness = max(float(rms_roughness), 1e-14)  # zero roughness --> practically a step transition

    if isinstance(distribution, str):
        dist = getattr(scipy.stats, distribution, None)
        if not isinstance(dist, scipy.stats.rv_continuous):
            raise ValueError(f"Unknown distribution '{distribution}'. Use the name of a "
                             "scipy.stats continuous distribution, e.g. 'norm', 'rayleigh', 'uniform'.")
        try:
            m, v = dist.stats(moments='mv')
        except TypeError as e:
            raise ValueError(f"Distribution '{distribution}' requires shape parameters. Pass a frozen "
                             f"distribution instead, e.g. scipy.stats.{distribution}(...).") from e
        scale = rms_roughness/np.sqrt(float(v))
        return dist.cdf(x, loc=boundary_loc - float(m)*scale, scale=scale)

    if hasattr(distribution, 'cdf'):  # frozen scipy.stats distribution
        m, v = distribution.stats(moments='mv')
        s = np.sqrt(float(v))
        return distribution.cdf((x - boundary_loc)*(s/rms_roughness) + float(m))

    if callable(distribution):  # user-supplied CDF function
        return np.asarray(distribution(x, rms_roughness, boundary_loc))

    raise ValueError(f"Invalid distribution: {distribution!r}")


def _parse_material_properties(material_properties: list, omega: np.ndarray):
    """
    Turn the list of material dicts into (M, nf) arrays of complex mu_r and ep_r.

    A scalar value applies to all frequencies; an array must match the length
    of the frequency vector. Conduction loss enters the relative permittivity
    as -1j*sigma/(omega*epsilon_0).
    """
    ones = np.ones_like(omega)
    mu_r_list, ep_r_list = [], []
    for k, mat in enumerate(material_properties):
        unknown = set(mat) - {'sigma', 'mu_r', 'ep_r'}
        if unknown:  # catch typos, which would otherwise silently fall back to the defaults
            raise ValueError(f"material_properties[{k}] has unknown key(s) {sorted(unknown)}; "
                             "allowed keys are 'sigma', 'mu_r' and 'ep_r'.")
        sigma = np.asarray(mat.get('sigma', 0), dtype=float)*ones
        ep_r = np.asarray(mat.get('ep_r', 1), dtype=complex)*ones
        # a conductor's loss is described by sigma; allowing a complex ep_r next to it
        # would count the loss twice
        if np.any(sigma != 0) and np.any(ep_r.imag != 0):
            raise ValueError(f"material_properties[{k}]: a conductor (nonzero 'sigma') must have a "
                             "real-valued 'ep_r'; for a lossy dielectric give a complex 'ep_r' and no 'sigma'.")
        mu_r_list.append(np.asarray(mat.get('mu_r', 1), dtype=complex)*ones)
        ep_r_list.append(ep_r - 1j*sigma/(omega*_const.epsilon_0))
    return np.array(mu_r_list), np.array(ep_r_list)


def _segment_edges(rms_roughness: np.ndarray, boundary_loc: np.ndarray,
                   x_start: float, x_end: float, n_segments: int):
    """
    Segment edges for the impedance recursion: about n_segments spread uniformly
    over the rough transitions [boundary_loc -+ 5*rms_roughness], merged where
    they overlap. Each uniform bulk gap in between gets a single segment, which
    is exact for a uniform line.
    """
    intervals = sorted([b - 5*max(r, 1e-14), b + 5*max(r, 1e-14)]
                       for r, b in zip(rms_roughness, boundary_loc))
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # spread n_segments uniformly over the total transition width; each bulk gap gets one segment
    dx = sum(b - a for a, b in merged)/n_segments
    edges = {x_start, x_end}
    for a, b in merged:
        edges.update(np.linspace(a, b, max(1, round((b - a)/dx)) + 1))
    return np.array(sorted(edges))


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

    if not deg:
        return  gamma(f)*d
    else:
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
    zl[(zl == np.inf)] = INF

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
    gamma = -1/(2*d) * np.log(Gamma_in/Gamma_l)
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
    return (1 + np.abs(Gamma0)) / (1 - np.abs(Gamma0))


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
    # theta = gamma[:, np.newaxis] * d  # (nbfreqs x nbd)
    # ABCD parameters of a transmission line (gamma, z0)
    A = np.cosh(theta)
    B = z0*np.sinh(theta)
    C = np.sinh(theta)/z0
    D = np.cosh(theta)
    # transpose and de-transpose operations are necessary
    # for linalg.inv to inverse square matrices
    ABCD = np.array([[A, B],[C, D]]).transpose()
    inv_ABCD = np.linalg.inv(ABCD).transpose()

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
    Rin = np.real(zl_2_zin(z0, zl, theta))
    total_loss = Rin/np.real(zl)*np.abs(np.cosh(theta) + zl/z0*np.sinh(theta))**2
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
