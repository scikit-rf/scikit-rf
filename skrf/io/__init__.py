"""
.. module:: skrf.io
========================================
io (:mod:`skrf.io`)
========================================


This Package provides functions and objects for input/output.

The general functions :func:`~general.read` and :func:`~general.write`
can be used to read and write [almost] any skrf object to disk, using the
:mod:`pickle` module.

Reading and writing touchstone files is supported through the
:class:`~touchstone.Touchstone` class, which can be more easily used
through the Network constructor, :func:`~skrf.network.Network.__init__`



.. automodule:: skrf.io.general

.. automodule:: skrf.io.touchstone

.. automodule:: skrf.io.csv


"""

from . import citi, csv, general, mdif, metas, touchstone
from .citi import Citi
from .general import load_all_touchstones, read, read_all, write, write_all
from .mdif import Mdif
from .metas import ns_2_sdatcv
from .touchstone import (
    Touchstone,
    hfss_touchstone_2_gamma_z0,
    hfss_touchstone_2_media,
    hfss_touchstone_2_network,
    read_zipped_touchstones,
)
