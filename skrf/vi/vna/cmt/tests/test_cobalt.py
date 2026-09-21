# This test expects the CMT S2 Analyzer Control Software to be running, see https://coppermountaintech.com/s2-vna-linux-soft/
# One can test using real hardware, or the control software's demo/simulation mode.

import os

import numpy as np
import pytest

import skrf
from skrf.vi.validators import ValidationError

try:
    from pyvisa.errors import VisaIOError

    from skrf.vi.vna import Channel, ValuesFormat
    from skrf.vi.vna.cmt.cobalt import Cobalt, SweepType, TraceParameter, TriggerScope, TriggerSource
except ImportError:
    pytest.skip("pyvisa not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def analyzer():
    addr = os.getenv("PYTEST_CMT_COBALT_VI_ADDR", "TCPIP0::127.0.0.1::5025::SOCKET")
    try:
        cobalt = Cobalt(addr)
    except (OSError, VisaIOError) as e:
        pytest.skip(f"Skipping tests: could not connect to S2 Control Software at {addr} ({e})")

    try:
        yield cobalt
    finally:
        try:
            cobalt.reset()
        finally:
            cobalt._resource.close()


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

    ch.sweep_type = SweepType.LINEAR
    assert ch.sweep_type == SweepType.LINEAR


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

    analyzer.trigger_scope = TriggerScope.ACTIVE
    assert analyzer.trigger_scope == TriggerScope.ACTIVE


def test_channel_param_def(analyzer):
    for parameter in TraceParameter:
        analyzer.ch1.param_def = parameter.value
        assert analyzer.param_def == parameter
        if parameter in (TraceParameter.A, TraceParameter.B, TraceParameter.R1, TraceParameter.R2):
            for port in (1, 2):
                analyzer.ch1.stimulus_port = port
                assert analyzer.ch1.stimulus_port == port
                assert analyzer.param_def == parameter
                assert analyzer.query("CALC1:PAR1:DEF?") == f"{parameter.value}({port})"

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


@pytest.mark.parametrize("query_format, ntraces", [
    (ValuesFormat.ASCII, 1), (ValuesFormat.BINARY_32, 16), (ValuesFormat.BINARY_64, 2),
])
def test_get_snp_network(analyzer, query_format, ntraces):
    analyzer.allocate_channels(2)
    ch = analyzer.ch1
    ch.frequency = skrf.Frequency(1e6, 20e9, 11, unit="Hz")
    ch.sweep_type = SweepType.LOG
    ch.ntraces = ntraces
    ch.active_trace = 1
    ch.param_def = TraceParameter.A
    ch.stimulus_port = 2
    ch.active_trace = ntraces
    ch.trigger_cont = False
    ch.averaging_on = ntraces > 1
    ch.averaging_count = 3
    analyzer.query_format = query_format
    analyzer.trigger_source = TriggerSource.BUS
    analyzer.trigger_scope = TriggerScope.ALL
    analyzer.active_channel = 2
    commands = [
        "SERV:CHAN:ACT?", "TRIG:SOUR?", "TRIG:SCOP?", "TRIG:AVER?", "INIT1:CONT?",
        "SENS1:AVER:STATE?", "SENS1:AVER:COUN?",
        "CALC1:PAR:COUN?", "SERV:CHAN1:TRAC:ACT?", "FORM:DATA?", "FORM:BORD?",
        "CALC1:TRAC1:FORM?", "DISP:WIND1:TRAC1:Y:SCAL:PDIV?",
    ]
    commands += [f"CALC1:PAR{trace}:{setting}?"
                 for trace in range(1, ntraces + 1) for setting in ("DEF", "SPOR")]
    original = [analyzer.query(cmd) for cmd in commands]

    network = ch.get_snp_network()

    assert network.s.shape == (11, 2, 2)
    np.testing.assert_allclose(network.f, np.geomspace(1e6, 20e9, 11), rtol=1e-6)
    assert [analyzer.query(cmd) for cmd in commands] == original
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ch.active_trace = 1
        ch.param_def = f"S{row + 1}{col + 1}"
        np.testing.assert_allclose(network.s[:, row, col], ch.active_trace_sdata)
    assert analyzer.read_next_error()[0] == 0
