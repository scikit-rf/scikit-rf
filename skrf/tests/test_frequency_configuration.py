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
    for kwargs in ({}, {"unit": "invalid"}):
        with pytest.raises(KeyError):
            rf.Frequency(1, 2, 2, **kwargs)
        with pytest.raises(KeyError):
            rf.Frequency.from_f([1, 2], **kwargs)
    for freq in (rf.Frequency(1, 2, 2, unit="Hz"), rf.Frequency.from_f([1, 2], unit="Hz")):
        np.testing.assert_array_equal(freq.f, [1, 2])


def test_copy_slice_and_unit_change_preserve_absolute_frequency(monkeypatch):
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
    assert network.frequency.unit == "Hz"
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

        assert rf.constants.FREQ_UNIT_DEFAULT == "Hz"
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
        assert restored.frequency.unit == "MHz"
        np.testing.assert_allclose(restored.f, expected_f, rtol=1e-12, atol=0)
        np.testing.assert_allclose(restored.s, cascaded.s, rtol=1e-12, atol=1e-15)

        fig, ax = plt.subplots()
        try:
            restored.plot_s_mag(m=1, n=0, ax=ax)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_f,
                                       rtol=1e-12, atol=0)
            formatter = ax.xaxis.get_major_formatter()
            np.testing.assert_allclose(float(formatter(expected_f[0], 0)), expected_f[0] / 1e6,
                                       rtol=1e-12, atol=0)
            assert "MHz" in ax.get_xlabel()
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
