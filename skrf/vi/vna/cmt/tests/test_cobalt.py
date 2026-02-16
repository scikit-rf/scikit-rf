# This test expects the CMT S2 Analyzer Control Software to be running, see https://coppermountaintech.com/s2-vna-linux-soft/
# One can test using real hardware, or the control software's demo/simulation mode.

import os

import numpy as np
import pytest

import skrf
from skrf.vi.validators import ValidationError
from skrf.vi.vna import Channel, ValuesFormat
from skrf.vi.vna.cmt.cobalt import Cobalt, SweepType, TraceParameter, TriggerScope, TriggerSource


@pytest.fixture(scope="function")
def analyzer():
    addr = os.getenv("PYTEST_CMT_COBALT_VI_ADDR", "TCPIP0::127.0.0.1::5025::SOCKET")
    try:
        cobalt = Cobalt(addr)
        yield cobalt
        cobalt.reset()
    except Exception as e:
        pytest.skip(f"Skipping tests: could not connect to S2 Control Software at {addr} ({e})")


def test_get_id(analyzer):
    idn = analyzer.id
    assert "Cobalt" in idn or "CMT" in idn


def test_model_check(analyzer):
    assert analyzer.model == "C4220"


def test_freq_start_stop(analyzer):
    ch = analyzer.ch1
    start = 1e6
    stop = 20e9
    npoints = 101

    ch.freq_start = start
    ch.freq_stop = stop
    ch.npoints = npoints

    f = ch.frequency
    assert np.isclose(f.start, start)
    assert np.isclose(f.stop, stop)
    assert f.npoints == npoints


def test_freq_span_center(analyzer):
    center = 10e9
    span = 1e9
    ch = analyzer.ch1
    ch.freq_center = center
    ch.freq_span = span
    ch.npoints = 401

    f = ch.frequency
    assert np.isclose(f.center, center)
    assert np.isclose(f.span, span)


def test_freq_step(analyzer):
    ch = analyzer.ch1
    ch.freq_start = 1e6
    ch.freq_stop = 20e9
    ch.npoints = 401
    assert np.isclose(ch.freq_step, 49997500)

    step = 25e6
    ch.freq_step = step
    assert np.isclose(ch.freq_step, 24998750)
    assert ch.npoints == 801


def test_sweep_type(analyzer):
    ch = analyzer.ch1

    ch.sweep_type = SweepType.LOG
    assert ch.sweep_type == SweepType.LOG

    ch.sweet_type = SweepType.LINEAR
    assert ch.sweet_type == SweepType.LINEAR


def test_averaging(analyzer):
    ch = analyzer.ch1

    assert not ch.averaging_on

    ch.averaging_on = True
    assert ch.averaging_on

    ch.averaging_count = 5
    assert ch.averaging_count == 5


def test_allocate_channels(analyzer):
    # test exception on too many channels
    with pytest.raises(ValueError):
        analyzer.allocate_channels(analyzer.max_chan + 1)

    # allocate two channels and check
    analyzer.allocate_channels(2)

    assert hasattr(analyzer, "ch2")


def test_active_channel(analyzer):
    num_ch = 2
    assert isinstance(analyzer.active_channel, Channel)
    # This should fail as the channel hasn't been initialized
    analyzer.allocate_channels(1)
    with pytest.raises(AttributeError):
        analyzer.active_channel = num_ch

    analyzer.allocate_channels(num_ch)
    analyzer.active_channel = num_ch

    assert analyzer.active_channel.cnum == num_ch


def test_set_active_channel_props(analyzer):
    analyzer.active_channel = 1
    freq = skrf.Frequency(start=1e6, stop=20e9, npoints=401, unit="Hz")
    analyzer.ch1.frequency = freq

    assert analyzer.frequency == freq


def test_sweep_and_get_sdata(analyzer):
    # should default to active channel
    freq = skrf.Frequency(start=1e6, stop=20e9, npoints=401, unit="Hz")
    analyzer.ch1.frequency = freq
    analyzer.ch1.if_bandwidth = 10e3
    ntwk = analyzer.get_sdata(1, 1)

    # basic checks
    assert isinstance(ntwk, skrf.Network)
    assert ntwk.s.shape[1:] == (1, 1)

    # check agrees with known output
    ntwk_sim = skrf.Network("skrf/vi/vna/cmt/tests/sim_CMT_C4220_1MHz_20GHz_401pt.s1p")
    assert np.allclose(ntwk_sim.s, ntwk.s, atol=1e-2)


def test_trigger_modes(analyzer):
    analyzer.trigger_source = TriggerSource.BUS
    assert analyzer.trigger_source == TriggerSource.BUS

    analyzer.trigger_scope = TriggerScope.ALL
    assert analyzer.trigger_scope == TriggerScope.ALL


def test_channel_param_def(analyzer):
    analyzer.ch1.param_def = "S11"
    assert analyzer.param_def == TraceParameter.S11

    with pytest.raises(ValidationError):
        analyzer.ch1.param_def = "S31"


def test_query_format(analyzer):
    freq = skrf.Frequency(start=1e6, stop=20e9, npoints=401, unit="Hz")
    analyzer.ch1.frequency = freq
    analyzer.ch1.if_bandwidth = 10e3
    ntwk_sim = skrf.Network("skrf/vi/vna/cmt/tests/sim_CMT_C4220_1MHz_20GHz_401pt.s1p")

    assert analyzer.query_format == ValuesFormat.ASCII

    analyzer.query_format = ValuesFormat.BINARY_32
    assert analyzer.query_format == ValuesFormat.BINARY_32

    ntwk = analyzer.get_sdata(1, 1)
    assert np.allclose(ntwk_sim.s, ntwk.s, atol=1e-2)

    analyzer.query_format = ValuesFormat.BINARY_64
    assert analyzer.query_format == ValuesFormat.BINARY_64

    ntwk = analyzer.get_sdata(1, 1)
    assert np.allclose(ntwk_sim.s, ntwk.s, atol=1e-2)


def test_clear_and_error_queue(analyzer):
    analyzer.clear()
    err = analyzer.read_next_error()
    assert err[0] == 0
