"""
.. currentmodule:: skrf.constants

========================================
constants (:mod:`skrf.constants`)
========================================

This module contains constants, numerical approximations, and unit conversions

.. data:: c

    Velocity of light constant (from scipy)

.. data:: INF

    A very very large value (1e99)

.. data:: ONE

    1 + epsilon where epsilon is small. Used to avoid numerical error.

.. data:: ZERO

    0 + epsilon where epsilon is small. Used to avoid numerical error.

.. data:: LOG_OF_NEG

    Very low but minus infinity value for numerical purposes.

.. data:: K_BOLTZMANN

    Boltzmann constant (1.38064852e-23)

.. data:: S_DEFINITIONS

    S-parameter definition labels:
        - 'power' for power-waves definition,
        - 'pseudo' for pseudo-waves definition.
        - 'traveling' corresponds to the initial implementation.

.. data:: S_DEF_DEFAULT

    Default S-parameter definition: 'power', for power-wave definition.

.. autodata:: S_DEF_HFSS_DEFAULT

    Default ANSYS HFSS S-parameter definition: 'traveling'

.. autodata:: SweepTypeT

.. autodata:: FrequencyUnitT

    Frequency units: "Hz", "kHz", "MHz", "GHz", "THz" (case-insensitive).

.. autosummary::
   :toctree: generated/

   to_meters

"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Any, Literal, get_args

import numpy as np
import scipy as _scipy

scipy_consts = ["c", "inch", "mil", "mu_0", "epsilon_0"]

# used as substitutes to handle mathematical singularities.
INF = 1e99
"""
High but not infinite value for numerical purposes.
"""

ALMOST_ZERO = 1e-12
"""
Very tiny but not zero value to handle mathematical singularities.
"""

ZERO = 1e-4
"""
A very small values, often used for numerical comparisons.
"""

ONE = 1.0 + 1/1e14
"""
Almost one but not one to handle mathematical singularities.
"""

LOG_OF_NEG = -100
"""
Very low but minus infinity value for numerical purposes.
"""

K_BOLTZMANN = 1.38064852e-23
"""
Boltzmann constant (1.38064852e-23)
"""

T0 = 290.
"""
Room temperature (kind of)
"""

EIG_COND = 1e-9
"""
Eigenvalue ratio compared to the maximum eigenvalue in :meth:`~skrf.mathFunctions.nudge_eig`.

EIG_COND * max(eigenvalue)
"""

EIG_MIN = 1e-12
"""
Minimum eigenvalue used in :meth:`~skrf.mathFunctions.nudge_eig`
"""

# S-parameter definition labels and default definition
SdefT = Literal["power", "pseudo", "traveling"]
S_DEFINITIONS: list[SdefT] = list(get_args(SdefT))
S_DEF_DEFAULT = 'power'
S_DEF_HFSS_DEFAULT = 'traveling'

FrequencyUnitT = Literal["Hz", "kHz", "MHz", "GHz", "THz"]
"""
Frequency units: "Hz", "kHz", "MHz", "GHz", "THz" (case-insensitive).
"""
FREQ_UNITS: dict[FrequencyUnitT, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

SweepTypeT = Literal["lin", "log"]
"""
Frequency sweep type, either "lin" or "log".
"""

CoordT = Literal["cart", "polar"]
InterpolKindT = Literal["linear", "cubic", "nearest", "zero", "slinear", "quadratic", "rational"]
PrimaryPropertiesT = Literal['s', 'z', 'y', 'a', 'g', 'h', 't']
ComponentFuncT = Literal["re", "im", "mag", "db", "db10", "rad", "deg", "arcl", "rad_unwrap", "deg_unwrap",
                         "arcl_unwrap", "vswr", "time", "time_db", "time_mag", "time_impulse", "time_step"]
SparamFormatT = Literal["db", "ri", "ma"]
PortOrderT = Literal["first", "second", "third", "last", "auto"]
CircuitComponentT = Literal["_is_circuit_port", "_is_circuit_ground", "_is_circuit_open"]
MemoryLayoutT = Literal["C", "F"]
ErrorFunctionsT = Literal["average_l1_norm", "average_l2_norm", "maximum_l1_norm", "average_normalized_l1_norm"]

NumberLike = Number | Sequence[Number] | np.ndarray


def get_distance_dict() -> dict[str, float]:
    """Get a dictionary of distance units to meters conversion factors.

    Returns
    -------
    distance_dict : dict
        A dictionary where keys are unit strings and values are conversion factors to meters.
    """
    distance_dict = {
        'm': 1.,
        'cm': 1e-2,
        'mm': 1e-3,
        'um': 1e-6,
        'in': _scipy.constants.inch,
        'mil': _scipy.constants.mil,
        's': _scipy.constants.c,
        'us': 1e-6 * _scipy.constants.c,
        'ns': 1e-9 * _scipy.constants.c,
        'ps': 1e-12 * _scipy.constants.c,
    }
    return distance_dict


def to_meters(d: NumberLike, unit: str = 'm', v_g: float | None = None) -> NumberLike:
    """
    Translate various units of distance into meters.

    Parameters
    ----------
    d : number or array-like
        value(s) to convert
    unit : str
        the unit to that x is in:
        ['m','cm','um','in','mil','s','us','ns','ps']
    v_g : float
        group velocity in m/s

    Returns
    -------
    d_m : number of array-like
        distance in meters
    """
    if v_g is None:
        v_g = _scipy.constants.c

    _distance_dict = {
        'm': 1.,
        'cm': 1e-2,
        'mm': 1e-3,
        'um': 1e-6,
        'in': _scipy.constants.inch,
        'mil': _scipy.constants.mil,
        's': v_g,
        'us': 1e-6*v_g,
        'ns': 1e-9*v_g,
        'ps': 1e-12*v_g,
    }

    unit = unit.lower()
    try:
        return _distance_dict[unit]*d
    except KeyError as err:
        raise(ValueError('Incorrect unit')) from err


def __getattr__(name: str) -> Any:
    if name in scipy_consts:
        return getattr(_scipy.constants, name)
    if name == "distance_dict":
        return get_distance_dict()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
