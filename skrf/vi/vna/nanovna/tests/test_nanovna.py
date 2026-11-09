import pytest

try:
    from skrf.vi.vna.nanovna.nanovna import REG_ADDR, NanoVNAv2
except ImportError:
    pytest.skip("pyvisa not installed", allow_module_level=True)


@pytest.fixture
def mocked_nanovna(mocker):
    mocker.patch("skrf.vi.vna.nanovna.nanovna.NanoVNAv2.__init__", return_value=None)
    mocker.patch("skrf.vi.vna.nanovna.nanovna.NanoVNAv2.query")
    yield NanoVNAv2("TEST")


def test_device_info_firmware_version(mocked_nanovna):
    registers = {
        REG_ADDR.DEVICE_VARIANT: b"\x02",
        REG_ADDR.PROTOCOL_VERSION: b"\x01",
        REG_ADDR.HARDWARE_REV: b"\x02",
        REG_ADDR.FIRMWARE_MAJOR: b"\x01",
        REG_ADDR.FIRMWARE_MINOR: b"\x03",
    }
    mocked_nanovna.query.side_effect = lambda op, addr, nbytes: registers[addr]

    assert mocked_nanovna.device_info.endswith("Firmware Version: 1.3")
