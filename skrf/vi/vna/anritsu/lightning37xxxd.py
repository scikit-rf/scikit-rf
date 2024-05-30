"""
.. module:: skrf.vi.vna.anritsu.lightning37xxxd
=================================================
HP 8510C (:mod:`skrf.vi.vna.hp.hp8510c`)
=================================================

HP8510C Class
================

.. autosummary::
    :nosignatures:
    :toctree: generated/
"""

import time

import numpy as np
import pyvisa

import skrf
import skrf.network
from skrf.vi.vna import VNA

class L37XXXD(VNA):
    '''
    Anritsu Lightning Family VNA

    Initially developed for 37369D VNA although should work with all 37XXXD
    Series and possibily other Lightning Family models.
    '''
    def __init__(self, address : str, backend : str = "@py", **kwargs):
        super().__init__(address, backend, **kwargs)