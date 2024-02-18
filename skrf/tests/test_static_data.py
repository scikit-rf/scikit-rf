import pytest

import skrf as rf

networks = [
    "ntwk1",
    "line",
    "open_2p",
    "short_2p",
    "ind",
    "ring_slot",
    "tee",
    "ring_slot_meas",
    "wr2p2_line" ,
    "wr2p2_line1",
    "wr2p2_delayshort",
    "wr2p2_short",
    "wr1p5_line",
    "wr1p5_short",
    "ro_1",
    "ro_2",
    "ro_3",
]

airs = [
    "air",
    "air50"
]


frequencies = [
    "f_wr51",
    "f_wr42",
    "f_wr34",
    "f_wr28",
    "f_wr22p4",
    "f_wr18p8",
    "f_wr14p8",
    "f_wr12p2",
    "f_wr10",
    "f_wr8",
    "f_wr6p5",
    "f_wr5p1",
    "f_wr4p3",
    "f_wr3p4",
    "f_wr2p8",
    "f_wr2p2",
    "f_wr1p9",
    "f_wr1p5",
    "f_wr1p2",
    "f_wr1",
    "f_wr0p8",
    "f_wr0p65",
    "f_wr0p51",
    "f_wm1295",
    "f_wm1092",
    "f_wm864",
    "f_wm710",
    "f_wm570",
    "f_wm470",
    "f_wm380",
    "f_wm310",
    "f_wm250",
    "f_wm200",
    "f_wm164",
    "f_wm130",
    "f_wm106",
    "f_wm86",
]


waveguides = [
    "wr51",
    "wr42",
    "wr34",
    "wr28",
    "wr22p4",
    "wr18p8",
    "wr14p8",
    "wr12p2",
    "wr10",
    "wr8",
    "wr6p5",
    "wr5p1",
    "wr4p3",
    "wr3p4",
    "wr2p8",
    "wr2p2",
    "wr1p9",
    "wr1p5",
    "wr1p2",
    "wr1",
    "wr0p8",
    "wr0p65",
    "wr0p51",
    "wm1295",
    "wm1092",
    "wm864",
    "wm710",
    "wm570",
    "wm470",
    "wm380",
    "wm310",
    "wm250",
    "wm200",
    "wm164",
    "wm130",
    "wm106",
    "wm86"
]

@pytest.mark.parametrize("name", networks)
def test_static_data(name):
    getattr(rf.data, name)


@pytest.mark.parametrize("name", airs)
def test_static_airs(name):
    getattr(rf, name)

@pytest.mark.parametrize("name", frequencies)
def test_static_frequencies(name):
    getattr(rf, name)

@pytest.mark.parametrize("name", waveguides)
def test_static_waveguides(name):
    getattr(rf, name)
