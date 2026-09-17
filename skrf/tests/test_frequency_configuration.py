"""Unit, module and process-level contracts for configurable frequency units."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import skrf as rf
from skrf.frequency import InvalidFrequencyWarning
from skrf.networkSet import NetworkSet


@pytest.mark.parametrize("unit,multiplier", [("Hz", 1), ("kHz", 1e3), ("MHz", 1e6),
                                            ("GHz", 1e9), ("THz", 1e12)])
@pytest.mark.parametrize("sweep", ["lin", "log"])
def test_default_units_and_sweep_types(monkeypatch, unit, multiplier, sweep):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", unit)
    expected = [1, 5.5, 10] if sweep == "lin" else [1, np.sqrt(10), 10]
    freq = rf.Frequency(1, 10, 3, sweep_type=sweep)
    assert freq.unit == unit
    np.testing.assert_allclose(freq.f, np.array(expected) * multiplier, rtol=1e-12, atol=0)
    np.testing.assert_allclose(freq.f_scaled, expected, rtol=1e-12, atol=0)
    from_f = rf.Frequency.from_f(expected)
    np.testing.assert_allclose(from_f.f, freq.f, rtol=1e-12, atol=0)


@pytest.mark.parametrize("values", [[2, 1], [1, 1]])
def test_non_monotonic_input_still_warns(monkeypatch, values):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", "GHz")
    with pytest.warns(InvalidFrequencyWarning):
        freq = rf.Frequency.from_f(values)
    np.testing.assert_array_equal(freq.f, np.array(values) * 1e9)


def test_invalid_default_matches_explicit_unit_validation(monkeypatch):
    # Characterize existing validation; this change does not add a new exception API.
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", "invalid")
    for kwargs in ({}, {"unit": "invalid"}, {"unit": "Hz"}):
        with pytest.raises(KeyError):
            rf.Frequency(1, 2, 2, **kwargs)
        with pytest.raises(KeyError):
            rf.Frequency.from_f([1, 2], **kwargs)


def test_copy_slice_and_unit_change_preserve_absolute_frequency(monkeypatch):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", None)
    freq = rf.Frequency.from_f([1, 2, 3], unit="MHz")
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", "GHz")
    copied = freq.copy()
    assert copied is not freq
    assert copied.unit == "MHz"
    np.testing.assert_array_equal(copied.f, [1e6, 2e6, 3e6])
    np.testing.assert_array_equal(freq[1:].f, [2e6, 3e6])
    copied.unit = "GHz"
    np.testing.assert_array_equal(copied.f, freq.f)
    np.testing.assert_allclose(copied.f_scaled, [0.001, 0.002, 0.003], rtol=1e-12, atol=0)
    assert freq.unit == "MHz"


def test_network_raw_frequency_keeps_existing_hz_contract(monkeypatch):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", "GHz")
    network = rf.Network(f=[1e6, 2e6], s=[0.1, 0.2], z0=50)
    np.testing.assert_array_equal(network.f, [1e6, 2e6])
    assert network.frequency.unit == "GHz"
    explicit = rf.Network(f=[1, 2], f_unit="MHz", s=[0.1, 0.2], z0=50)
    np.testing.assert_array_equal(explicit.f, network.f)


@pytest.mark.parametrize("default", ["Hz", "GHz"])
def test_user_workflow_in_fresh_process(tmp_path, default):
    """Exercise real import, configuration, network math, disk IO and plotting."""
    pytest.importorskip("matplotlib")
    program = textwrap.dedent('''
        import sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import skrf as rf

        assert rf.constants.FREQ_UNIT_DEFAULT is None
        rf.constants.FREQ_UNIT_DEFAULT = sys.argv[2]
        multiplier = 1 if sys.argv[2] == "Hz" else 1e9
        freq = rf.Frequency.from_f([1, 2, 3])
        s = np.zeros((3, 2, 2), dtype=complex)
        s[:, 0, 1] = s[:, 1, 0] = [0.2, 0.4, 0.6]
        network = rf.Network(frequency=freq, s=s, z0=50)

        # Changing the default must not reinterpret an existing network.
        rf.constants.FREQ_UNIT_DEFAULT = "THz"
        interpolated = network.interpolate(5)
        expected_f = np.linspace(1, 3, 5) * multiplier
        np.testing.assert_allclose(interpolated.f, expected_f, rtol=1e-12, atol=0)
        cascaded = interpolated ** interpolated
        np.testing.assert_allclose(cascaded.s[:, 1, 0], np.linspace(0.2, 0.6, 5)**2,
                                   rtol=1e-12, atol=1e-15)
        cascaded.frequency.unit = "MHz"
        cascaded.write_touchstone("roundtrip", dir=".", form="ri")
        restored = rf.Network("roundtrip.s2p")
        assert restored.frequency.unit == "THz"
        np.testing.assert_allclose(restored.f, expected_f, rtol=1e-12, atol=0)
        np.testing.assert_allclose(restored.s, cascaded.s, rtol=1e-12, atol=1e-15)

        fig, ax = plt.subplots()
        try:
            restored.plot_s_mag(m=1, n=0, ax=ax)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_f,
                                       rtol=1e-12, atol=0)
            formatter = ax.xaxis.get_major_formatter()
            np.testing.assert_allclose(float(formatter(expected_f[0], 0)), expected_f[0] / 1e12,
                                       rtol=1e-12, atol=0)
            assert "THz" in ax.get_xlabel()
            fig.savefig("response.png")
        finally:
            plt.close(fig)
        assert Path("response.png").stat().st_size > 0
    ''')
    root = str(Path(rf.__file__).resolve().parent.parent)
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", program, root, default],
            cwd=tmp_path, env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("configured", [None, "Hz", "kHz", "MHz", "GHz", "THz"])
@pytest.mark.parametrize("input_unit", [None, "Hz", "kHz", "MHz", "GHz", "THz"])
@pytest.mark.parametrize("sweep", ["lin", "log"])
def test_input_units_and_output_units_are_independent(monkeypatch, configured, input_unit, sweep):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", configured)
    source_unit = input_unit or configured or "Hz"
    target_unit = configured or source_unit
    source_scale = rf.constants.FREQ_UNITS[source_unit]
    target_scale = rf.constants.FREQ_UNITS[target_unit]
    values = np.array([1, 5.5, 10] if sweep == "lin" else [1, np.sqrt(10), 10])
    for freq in (rf.Frequency(1, 10, 3, unit=input_unit, sweep_type=sweep),
                 rf.Frequency.from_f(values, unit=input_unit)):
        assert freq.unit == target_unit
        np.testing.assert_allclose(freq.f, values * source_scale)
        np.testing.assert_allclose(freq.f_scaled, values * source_scale / target_scale)


@pytest.mark.parametrize("configured", [None, "Hz", "GHz"])
@pytest.mark.parametrize("file_unit", ["Hz", "kHz", "MHz", "GHz"])
def test_touchstone_preserves_values_and_coerces_units(monkeypatch, configured, file_unit):
    from io import StringIO

    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", configured)
    stream = StringIO(f"# {file_unit} S RI R 50\n1 .1 0\n2 .2 0\n")
    stream.name = "units.s1p"
    network = rf.Network(stream)
    expected = np.array([1, 2]) * rf.constants.FREQ_UNITS[file_unit]
    assert network.frequency.unit == (configured or file_unit)
    np.testing.assert_array_equal(network.f, expected)
    np.testing.assert_allclose(network.frequency.f_scaled,
                               expected / rf.constants.FREQ_UNITS[configured or file_unit])
    np.testing.assert_allclose(network.s[:, 0, 0], [.1, .2])


def test_noise_and_network_constructor_respect_configured_unit(monkeypatch):
    path = Path(__file__).parent / "ntwk_noise.s2p"
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", None)
    baseline = rf.Network(path)
    original = rf.Frequency.from_f([1, 2], unit="MHz")
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", "GHz")
    loaded = rf.Network(path, f_unit="MHz")
    assert loaded.frequency.unit == loaded.noise_freq.unit == "GHz"
    np.testing.assert_array_equal(loaded.f, baseline.f)
    np.testing.assert_array_equal(loaded.noise_freq.f, baseline.noise_freq.f)
    np.testing.assert_allclose(loaded.noise, baseline.noise)
    network = rf.Network(frequency=original, noise_freq=original, s=[.1, .2])
    assert network.frequency.unit == network.noise_freq.unit == "GHz"
    assert network.noise_freq is not original
    assert original.unit == "MHz"
    np.testing.assert_array_equal(network.f, original.f)
    np.testing.assert_array_equal(network.noise_freq.f, original.f)


@pytest.mark.parametrize("configured", ["Hz", "MHz", "GHz"])
def test_mdif_noise_input_units_are_not_taken_from_display_unit(monkeypatch, tmp_path, configured):
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", None)
    baseline = rf.Network(Path(__file__).parents[1] / "io/tests/ts/ex_18.s2p")
    path = tmp_path / "noise.mdf"
    NetworkSet([baseline]).write_mdif(str(path))
    monkeypatch.setattr(rf.constants, "FREQ_UNIT_DEFAULT", configured)
    restored = NetworkSet.from_mdif(str(path))[0]
    assert restored.frequency.unit == restored.noise_freq.unit == configured
    np.testing.assert_allclose(restored.f, baseline.f)
    np.testing.assert_allclose(restored.noise_freq.f, baseline.noise_freq.f)
    np.testing.assert_allclose(restored.noise, baseline.noise)
