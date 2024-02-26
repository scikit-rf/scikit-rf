import sys

import pytest

try:
    from skrf.vi.vna import vna
except ImportError:
    pass

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

@pytest.mark.parametrize(
    "cmd,kwargs,expected",
    [
        ("*IDN?", {}, "*IDN?"),
        ("SENS<self:cnum>", {}, "SENS1"),
        ("SENS<self:cnum>:STAR?", {}, "SENS1:STAR?"),
        ("SENS<self:cnum>:STAR <arg>", {"arg": 100}, "SENS1:STAR 100"),
    ],
)
def test_format_cmd(cmd, kwargs, expected):
    class TestDevice:
        cnum = 1
        name = "channel"

    assert vna._format_cmd(cmd, self=TestDevice(), **kwargs) == expected


def test_vna_add_channel_support():
    class TestVNA(vna.VNA):
        pass

    assert not hasattr(TestVNA, "create_channel")
    assert not hasattr(TestVNA, "delete_channel")

    TestVNA._add_channel_support()

    assert hasattr(TestVNA, "create_channel")
    assert hasattr(TestVNA, "delete_channel")


def test_vna_create_delete_channels():
    class TestVNA(vna.VNA):
        class Channel(vna.Channel):
            pass

        def __init__(self):
            pass

    TestVNA._add_channel_support()
    instr = TestVNA()

    assert hasattr(instr, "create_channel")
    assert hasattr(instr, "delete_channel")

    instr.create_channel(1, "channel 1")
    assert hasattr(instr, "ch1")
    instr.delete_channel(1)
    assert not hasattr(instr, "ch1")
