from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

import skrf

pyvisa = pytest.importorskip("pyvisa")

nanovna = pytest.importorskip("skrf.vi.vna.nanovna")


@pytest.fixture(params=["Hz", "GHz"])
def instrument(request):
    resource = MagicMock(spec=pyvisa.resources.SerialInstrument)
    with patch.object(skrf.constants, "FREQ_UNIT_DEFAULT", request.param):
        with patch("pyvisa.ResourceManager") as manager:
            manager.return_value.open_resource.return_value = resource
            yield nanovna.NanoVNAv2("TEST"), resource


def test_initial_sweep_uses_hz(instrument):
    vna, resource = instrument
    np.testing.assert_array_equal(vna.frequency.f, np.linspace(1e6, 10e6, 201))
    assert vna.frequency.unit == "Hz"
    assert resource.write_raw.call_args_list == [
        call(b"\x00" * 8),
        call(b"\x23\x00" + (1_000_000).to_bytes(8, "little")),
        call(b"\x23\x10" + (45_000).to_bytes(8, "little")),
        call(b"\x21\x20" + (201).to_bytes(2, "little")),
    ]


@pytest.mark.parametrize(
    "attribute,value,start,stop,points,command",
    [
        ("freq_start", 2_000_000, 2e6, 10e6, 201,
         b"\x23\x00" + (2_000_000).to_bytes(8, "little")),
        ("freq_stop", 11_000_000, 1e6, 11e6, 201,
         b"\x23\x10" + (50_000).to_bytes(8, "little")),
        ("freq_step", 90_000, 1e6, 10e6, 101,
         b"\x21\x20" + (101).to_bytes(2, "little")),
        ("npoints", 101, 1e6, 10e6, 101,
         b"\x21\x20" + (101).to_bytes(2, "little")),
    ],
)
def test_sweep_updates_use_hz(instrument, attribute, value, start, stop, points, command):
    vna, resource = instrument
    # Start with a GHz-labelled object whose absolute frequencies are in Hz.
    vna.frequency = skrf.Frequency(0.001, 0.01, 201, unit="GHz")
    resource.write_raw.reset_mock()

    setattr(vna, attribute, value)

    assert vna.frequency.unit == "Hz"
    np.testing.assert_array_equal(vna.frequency.f, np.linspace(start, stop, points))
    resource.write_raw.assert_called_once_with(command)


def test_measurement_results_keep_frequency_axis(instrument):
    vna, resource = instrument
    vna.frequency = skrf.Frequency(1, 3, 3, unit="MHz")
    # Three protocol records: incident=4, reflected=1, transmitted=2.
    payload = bytearray()
    for index in range(3):
        for value in (4, 0, 1, 0, 2, 0):
            payload.extend(value.to_bytes(4, "little", signed=True))
        payload.extend(index.to_bytes(2, "little"))
        payload.extend(bytes(6))
    resource.read_bytes.return_value = bytes(payload)

    s11, s21 = vna.get_s11_s21()

    resource.read_bytes.assert_called_once_with(96)
    for network, expected in ((s11, 0.25), (s21, 0.5)):
        np.testing.assert_array_equal(network.f, [1e6, 2e6, 3e6])
        np.testing.assert_array_equal(network.s[:, 0, 0], [expected] * 3)
        assert network.frequency.unit == "MHz"


def test_communication_timeout_is_propagated(instrument):
    vna, resource = instrument
    resource.read_bytes.side_effect = pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_timeout)
    with pytest.raises(pyvisa.errors.VisaIOError):
        vna.get_s11_s21()
    np.testing.assert_array_equal(vna.frequency.f, np.linspace(1e6, 10e6, 201))
