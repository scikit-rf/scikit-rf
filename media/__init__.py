
'''
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
which interally define relevant quantities such as propagation constant (`gamma`),
and characteristic impedance (`Z0`). This allows the specific transmission line
mediums to produce networks without re-implementing methods for
each specific media instance.



Media base-class
-------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~media.Media

Transmission Line Classes
-------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    ~media.DefinedGammaZ0
    ~distributedCircuit.DistributedCircuit
    ~rectangularWaveguide.RectangularWaveguide
    ~cpw.CPW
    ~freespace.Freespace
    ~coaxial.Coaxial
    ~mline.MLine

'''

from .media import Media, DefinedGammaZ0
from .distributedCircuit import DistributedCircuit
from .freespace import Freespace
from .cpw import CPW
from .rectangularWaveguide import RectangularWaveguide
from .coaxial import Coaxial
from .mline import MLine
from .definedAEpTandZ0 import DefinedAEpTandZ0
