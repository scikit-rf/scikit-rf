from pathlib import Path

import pytest

import skrf as rf

testdir = Path(__file__).parent

@pytest.fixture()
def ntwk1() -> rf.Network:
    return rf.Network(testdir / "ntwk1.s2p")

@pytest.fixture()
def ntwk1_dc(ntwk1: rf.Network) -> rf.Network:
    return ntwk1.extrapolate_to_dc()

@pytest.fixture()
def ntwk_set_zip() -> rf.NetworkSet:
    return rf.NetworkSet.from_zip(testdir / "ntwks.zip")
