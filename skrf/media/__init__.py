"""
.. module:: skrf.media
========================================
media (:mod:`skrf.media`)
========================================

This package provides objects representing transmission line mediums.

The :class:`~media.Media` object is the base-class that is inherited
by specific transmission line instances, such as
:class:`~freespace.Freespace`, or
:class:`~rectangularWaveguide.RectangularWaveguide`. The
:class:`~media.Media` object provides generic methods to produce
:class:`~skrf.network.Network`'s for any transmission line medium, such
as :func:`~media.Media.line` and :func:`~media.Media.delay_short`. These
methods are inherited by the specific transmission line classes,
which internally define relevant quantities such as propagation constant (`gamma`),
and characteristic impedance (`Z0`). This allows the specific transmission line
mediums to produce networks without re-implementing methods for
each specific media instance.



Media base-class
-------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~media.Media





"""

from warnings import warn

from . import (
    circularWaveguide,
    coaxial,
    cpw,
    definedAEpTandZ0,
    device,
    distributedCircuit,
    freespace,
    media,
    mline,
    rectangularWaveguide,
)
from .circularWaveguide import CircularWaveguide
from .coaxial import Coaxial
from .cpw import CPW
from .definedAEpTandZ0 import DefinedAEpTandZ0
from .device import (
    Device,
    DualCoupler,
    Hybrid,
    Hybrid180,
    MatchedSymmetricCoupler,
    QuadratureHybrid,
)
from .distributedCircuit import DistributedCircuit
from .freespace import Freespace
from .media import (
    DefinedGammaZ0,
    Media,
    get_z0_load,
    has_len,
    parse_z0,
    splitter_s,
)
from .mline import MLine
from .rectangularWaveguide import RectangularWaveguide

__all__ = [
      "CPW",
      "CircularWaveguide",
      "Coaxial",
      "DefinedAEpTandZ0",
      "DefinedGammaZ0",
      "Device",
      "DistributedCircuit",
      "DualCoupler",
      "Freespace",
      "Hybrid",
      "Hybrid180",
      "MLine",
      "MatchedSymmetricCoupler",
      "Media",
      "QuadratureHybrid",
      "RectangularWaveguide",
      "get_z0_load",
      "has_len",
      "parse_z0",
      "splitter_s",
]


def __getattr__(name: str):
    result = getattr(media, name, None)
    if result is not None:
        warn(f"skrf.media.{name} is deprecated. Please import {name} from skrf.media.media instead.",
             FutureWarning, stacklevel=2)
        return result
    raise AttributeError(f"module 'skrf.media' has no attribute '{name}'")
