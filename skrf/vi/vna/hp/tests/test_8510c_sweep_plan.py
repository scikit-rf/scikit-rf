"""
The user requests a big sweep with different spacings in different frequency
blocks, but the HP8510 instrument only supports smaller sweeps with fixed
spacing. What to do? Break up the big sweep, naturally.

SweepPlan.from_hz(array_of_hz)
  ->
    SweepPlan
        SweepSection
        SweepSection
        SweepSection

Each SweepSection represents a sweep the instrument can actually perform.
Together, the SweepSections in a SweepPlan satisfy the user's request.

There is room for cleverness: conducting an 800 point sweep by running an
801 point sweep and discarding a point saved 20 minutes in a recent script.

Cleverness is only good if it works, and these tests make sure that
SweepPlan.from_hz returns a plan that hits exactly the frequencies it is
supposed to.
"""

import sys

import numpy as np
import pytest

try:
    from skrf.vi.vna.hp.hp8510c_sweep_plan import SweepPlan
except ImportError:
    pass

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

def test_800pt_swp():
    """ Test: magic size - 1
    """
    test_hz_0 = np.linspace(100,1000,800)
    test_sp_0 = SweepPlan.from_hz(test_hz_0)
    assert test_sp_0._matches_f_list(test_hz_0)

def test_801pt_swp():
     """ Test: magic size
     """
     test_hz_0 = np.linspace(100,1000,801)
     test_sp_0 = SweepPlan.from_hz(test_hz_0)
     assert test_sp_0._matches_f_list(test_hz_0)

def test_802pt_swp():
     """ Test: magic size + 1
     """
     test_hz_0 = np.linspace(100,1000,802)
     test_sp_0 = SweepPlan.from_hz(test_hz_0)
     assert test_sp_0._matches_f_list(test_hz_0)

def test_1001pt_swp():
    """ Test: typical size
    """
    test_hz_1 = np.linspace(100,1000,1001)
    test_sp_1 = SweepPlan.from_hz(test_hz_1)
    assert test_sp_1._matches_f_list(test_hz_1)

def test_multi_swp():
    """ Test: multiple frequency blocks in a single sweep
    """
    test_hz_2 = np.concatenate(([1,2,3], np.linspace(100,1000,1001)))
    test_sp_2 = SweepPlan.from_hz(test_hz_2)
    assert test_sp_2._matches_f_list(test_hz_2)

def test_multi_swp_with_single():
    """ Test: multiple frequency blocks, one of them has size one
    """
    test_hz_3 = np.concatenate(([1,2,3], np.linspace(100,1000,1001), [9999]))
    test_sp_3 = SweepPlan.from_hz(test_hz_3)
    assert test_sp_3._matches_f_list(test_hz_3)
