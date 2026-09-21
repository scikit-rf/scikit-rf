import numpy as np
import pytest

try:
    import pyvisa

    from skrf.vi.vna.cmt.cobalt import Cobalt, TraceParameter
except ImportError:
    pytest.skip("pyvisa not installed", allow_module_level=True)


@pytest.fixture
def resource(mocker):
    resource = mocker.MagicMock(spec=pyvisa.resources.MessageBasedResource)
    mocker.patch("pyvisa.ResourceManager").return_value.open_resource.return_value = resource
    responses = {
        "SERV:CHAN:COUN?": "16",
        "SERV:CHAN:ACT?": "1",
        "SERV:CHAN1:TRAC:ACT?": "1",
        "*OPC?": "1",
        "*IDN?": "Copper Mountain Technologies, C4220, 12345, 1.0",
    }
    resource.query.side_effect = responses.__getitem__
    return resource


@pytest.mark.parametrize("command", ["SERV:CHAN:COUN?", "*IDN?"])
def test_init_closes_resource_on_failure(resource, command):
    query = resource.query.side_effect

    def failed_query(cmd):
        if cmd == command:
            raise ConnectionError("connection lost")
        return query(cmd)

    resource.query.side_effect = failed_query
    with pytest.raises(ConnectionError, match="connection lost"):
        Cobalt("TEST")

    resource.close.assert_called_once_with()


@pytest.mark.parametrize("parameter", ["A", "B", "R1", "R2"])
def test_receiver_parameter(resource, parameter):
    analyzer = Cobalt("TEST")
    resource.write.reset_mock()
    query = resource.query.side_effect
    resource.query.side_effect = lambda cmd: f"{parameter}(2)" if cmd == "CALC1:PAR1:DEF?" else query(cmd)

    analyzer.ch1.param_def = parameter

    resource.write.assert_called_once_with(f"CALC1:PAR1:DEF {parameter}")
    assert analyzer.ch1.param_def == TraceParameter(parameter)


def test_trigger_single_enables_active_channel(resource, mocker):
    analyzer = Cobalt("TEST")
    resource.write.reset_mock()

    analyzer.trigger_single()

    assert resource.write.call_args_list == [
        mocker.call("TRIG:SOUR BUS"),
        mocker.call("INIT1:CONT 1"),
        mocker.call("TRIG:SING"),
    ]


@pytest.fixture
def snp_analyzer(resource):
    analyzer = Cobalt("TEST")
    responses = {
        "SERV:CHAN:ACT?": "2",
        "TRIG:SOUR?": "EXT",
        "TRIG:SCOP?": "ALL",
        "TRIG:AVER?": "0",
        "INIT1:CONT?": "0",
        "CALC1:PAR:COUN?": "16",
        "SERV:CHAN1:TRAC:ACT?": "7",
        "CALC1:PAR1:DEF?": "A(2)",
        "CALC1:PAR1:SPOR?": "2",
        "CALC1:PAR2:DEF?": "S21",
        "CALC1:PAR2:SPOR?": "1",
    }
    query = resource.query.side_effect
    resource.query.side_effect = lambda cmd: responses[cmd] if cmd in responses else query(cmd)
    def write(cmd):
        if cmd.startswith("DISP:WIND"):
            responses["SERV:CHAN:ACT?"] = cmd.removeprefix("DISP:WIND").split(":")[0]

    resource.write.side_effect = write
    s = np.arange(12).reshape(3, 2, 2) + 1j * np.arange(12, 24).reshape(3, 2, 2)
    values = {"SENS1:FREQ:DATA?": [1e6, 4e6, 10e6]}
    for parameter, row, col in [("S11", 0, 0), ("S12", 0, 1), ("S21", 1, 0), ("S22", 1, 1)]:
        values[f"SENS1:DATA:CORR? {parameter}"] = s[:, row, col].copy().view(float)
    resource.query_ascii_values.side_effect = values.__getitem__
    resource.write.reset_mock()
    return analyzer, s


@pytest.mark.parametrize("ports, indices", [(None, [0, 1]), ((2, 1), [1, 0]), ((1,), [0]), ((2,), [1])])
def test_get_snp_network(snp_analyzer, resource, ports, indices):
    analyzer, expected = snp_analyzer

    network = analyzer.ch1.get_snp_network(ports)

    np.testing.assert_array_equal(network.f, [1e6, 4e6, 10e6])
    np.testing.assert_array_equal(network.s, expected[:, indices][:, :, indices])
    commands = [call.args[0] for call in resource.write.call_args_list]
    assert commands.index("TRIG:AVER ON") < commands.index("TRIG:SING")
    assert commands.count("TRIG:SING") == 1
    for trace, port in enumerate(indices, 1):
        assert commands.index(f"CALC1:PAR{trace}:DEF S{port + 1}{port + 1}") < commands.index("TRIG:SING")


@pytest.mark.parametrize("command", ["SENS1:FREQ:DATA?", "SENS1:DATA:CORR? S21"])
def test_get_snp_network_restores_after_failure(snp_analyzer, resource, command):
    analyzer, _ = snp_analyzer
    query = resource.query_ascii_values.side_effect

    def failed_query(cmd):
        if cmd == command:
            raise TimeoutError("data transfer failed")
        return query(cmd)

    resource.query_ascii_values.side_effect = failed_query
    with pytest.raises(TimeoutError, match="data transfer failed"):
        analyzer.ch1.get_snp_network()

    commands = [call.args[0] for call in resource.write.call_args_list]
    assert commands[-11:] == [
        "CALC1:PAR1:DEF A", "CALC1:PAR1:SPOR 2",
        "CALC1:PAR2:DEF S21", "CALC1:PAR2:SPOR 1",
        "CALC1:PAR:COUN 16", "CALC1:PAR7:SEL", "INIT1:CONT 0",
        "TRIG:AVER 0", "TRIG:SCOP ALL", "DISP:WIND2:ACT", "TRIG:SOUR EXT",
    ]


@pytest.mark.parametrize("ports", [(), (1, 1), (0,), (3,), (1.5,)])
def test_get_snp_network_rejects_invalid_ports(resource, ports):
    analyzer = Cobalt("TEST")
    resource.write.reset_mock()

    with pytest.raises((ValueError, TypeError)):
        analyzer.ch1.get_snp_network(ports)

    resource.write.assert_not_called()
