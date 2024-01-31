
import sys

import numpy as np
import pytest

import skrf

try:
    from skrf.vi.vna import ValuesFormat, keysight
    from skrf.vi.vna.keysight.pna import SweepMode, SweepType
except ImportError:
    pass

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.keysight.PNA.__init__', return_value=None)
    mocker.patch('skrf.vi.vna.keysight.PNA.write')
    mocker.patch('skrf.vi.vna.keysight.PNA.write_values')
    mocker.patch('skrf.vi.vna.keysight.PNA.query')
    mocker.patch('skrf.vi.vna.keysight.PNA.query_values')
    mock = keysight.PNA('TEST')
    mock.model = "TEST"

    # This gets done in init, but we are mocking init to prevent super().__init__, so just call here
    mock.create_channel(1, 'Channel 1')

    yield mock

@pytest.mark.parametrize(
    'param,expected_query,expected_write,query_response,expected_val,write_val',
    [
        ('freq_start', 'SENS1:FREQ:STAR?', 'SENS1:FREQ:STAR 100', '100', 100, 100),
        ('freq_stop', 'SENS1:FREQ:STOP?', 'SENS1:FREQ:STOP 100', '100', 100, 100),
        ('freq_span', 'SENS1:FREQ:SPAN?', 'SENS1:FREQ:SPAN 100', '100', 100, 100),
        ('freq_center', 'SENS1:FREQ:CENT?', 'SENS1:FREQ:CENT 100', '100', 100, 100),
        ('npoints', 'SENS1:SWE:POIN?', 'SENS1:SWE:POIN 100', '100', 100, 100),
        ('if_bandwidth', 'SENS1:BWID?', 'SENS1:BWID 100', '100', 100, 100),
        ('sweep_time', 'SENS1:SWE:TIME?', 'SENS1:SWE:TIME 1.0', '1.0', 1.0, 1),
        ('sweep_type', 'SENS1:SWE:TYPE?', 'SENS1:SWE:TYPE LIN', 'LIN', SweepType.LINEAR, SweepType.LINEAR),
        ('sweep_mode', 'SENS1:SWE:MODE?', 'SENS1:SWE:MODE SING', 'SING', SweepMode.SINGLE, SweepMode.SINGLE),
        ('measurement_numbers', 'SYST:MEAS:CAT? 1', None, '1,2,3', [1, 2, 3], None),
    ]
)
def test_params(
    mocker,
    mocked_ff,
    param,
    expected_query,
    expected_write,
    query_response,
    expected_val,
    write_val
):
    if expected_query is not None:
        mocked_ff.query.return_value = query_response
        test_query = getattr(mocked_ff.ch1, param)
        mocked_ff.query.assert_called_once_with(expected_query)
        assert test_query == expected_val

    if expected_write is not None:
        setattr(mocked_ff.ch1, param, write_val)
        mocked_ff.write.assert_called_once_with(expected_write)

def test_frequency_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = [
        '100', '200', '11'
    ]
    test = mocked_ff.ch1.frequency
    assert test == skrf.Frequency(100, 200, 11, unit='hz')

def test_frequency_write(mocker, mocked_ff):
    test_f = skrf.Frequency(100, 200, 11, unit='hz')
    mocked_ff.ch1.frequency = test_f
    calls = [
        mocker.call("SENS1:FREQ:STAR 100"),
        mocker.call("SENS1:FREQ:STOP 200"),
        mocker.call("SENS1:SWE:POIN 11"),
    ]
    mocked_ff.write.assert_has_calls(calls)

# def test_create_channel(mocker, mocked_ff):
    # mocked_ff.create_channel(2, 'Channel 2')
    # assert hasattr(mocked_ff, 'ch2')
    # assert mocked_ff.ch2.cnum == 2
    # assert mocked_ff.ch2.name == "Channel 2"

def test_active_channel_query(mocker, mocked_ff):
    mocked_ff.query.return_value = 1
    test = mocked_ff.active_channel
    assert isinstance(test, keysight.PNA.Channel)
    assert test.cnum == 1

def test_active_channel_setter(mocker, mocked_ff):
    mocked_ff.query.side_effect = ['1', '1', '1', '1,2,3', '2']
    mocked_ff.active_channel = mocked_ff.ch1
    mocked_ff.write.assert_not_called()

    mocked_ff.create_channel(2, 'Test')
    mocked_ff.active_channel = mocked_ff.ch2

    assert mocked_ff.active_channel.cnum == 2

def test_query_fmt_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = ['ASC,0', 'REAL,32', 'REAL,64']
    test = mocked_ff.query_format
    assert test == ValuesFormat.ASCII
    test = mocked_ff.query_format
    assert test == ValuesFormat.BINARY_32
    test = mocked_ff.query_format
    assert test == ValuesFormat.BINARY_64

def test_query_fmt_write(mocker, mocked_ff):
    mocked_ff.query_format = ValuesFormat.ASCII
    mocked_ff.write.assert_called_with('FORM ASC,0')
    mocked_ff.query_format = ValuesFormat.BINARY_32
    calls = [
        mocker.call("FORM:BORD SWAP"),
        mocker.call("FORM REAL,32"),
    ]
    mocked_ff.write.assert_has_calls(calls)
    mocked_ff.query_format = ValuesFormat.BINARY_64
    calls = [
        mocker.call("FORM:BORD SWAP"),
        mocker.call("FORM REAL,64"),
    ]
    mocked_ff.write.assert_has_calls(calls)

def test_measurements_query(mocker, mocked_ff):
    mocked_ff.query.return_value = 'CH1_S11_1,S11,CH1_S12_1,S12'
    test = mocked_ff.ch1.measurements
    assert test == [('CH1_S11_1', 'S11'), ('CH1_S12_1', 'S12')]

def test_measurement_names_query(mocker, mocked_ff):
    mocked_ff.query.return_value = 'CH1_S11_1,S11,CH1_S12_1,S12'
    test = mocked_ff.ch1.measurement_names
    assert test == ['CH1_S11_1', 'CH1_S12_1']

def test_clear_averaging(mocker, mocked_ff):
    mocked_ff.ch1.clear_averaging()
    mocked_ff.write.assert_called_once_with('SENS1:AVER:CLE')

def test_create_measurement(mocker, mocked_ff):
    mocked_ff.query.return_value = '1'
    mocked_ff.ch1.create_measurement('CH1_S11_1', 'S11')
    write_calls = [
        mocker.call("CALC1:PAR:EXT 'CH1_S11_1',S11"),
        mocker.call("DISP:WIND:TRAC2:FEED 'CH1_S11_1'"),
    ]

    mocked_ff.write.assert_has_calls(write_calls)

def test_delete_measurement(mocker, mocked_ff):
    mocked_ff.ch1.delete_measurement('CH1_S11_1')
    mocked_ff.write.assert_called_once_with("CALC1:PAR:DEL 'CH1_S11_1'")

def test_get_measurement(mocker, mocked_ff):
    mocked_ff.ch1.get_active_trace = mocker.MagicMock(return_value=skrf.Network())
    mocked_ff.query.side_effect = [
        'CH1_S11_1,S11,CH1_S12_1,S12',
        'CH1_S11_1,S11,CH1_S12_1,S12',
    ]
    test = mocked_ff.ch1.get_measurement('CH1_S11_1')
    mocked_ff.write.assert_called_once_with("CALC1:PAR:SEL 'CH1_S11_1',fast")
    assert isinstance(test, skrf.Network)

def test_get_active_trace(mocker, mocked_ff):
    mock_sdata = np.array([1.,]*22)
    query_responses = [
        'ASC,0',
        '100','200','11'
    ]
    expected_writes = [
        mocker.call('FORM:BORD SWAP'),
        mocker.call('FORM REAL,64'),
        mocker.call('FORM ASC,0')
    ]
    mocked_ff.query.side_effect = query_responses
    mocked_ff.ch1.sweep = mocker.MagicMock()
    mocker.patch('skrf.vi.vna.keysight.PNA.Channel.active_trace_sdata',
                return_value=mock_sdata, new_callable=mocker.PropertyMock)
    test = mocked_ff.ch1.get_active_trace()
    assert isinstance(test, skrf.Network)
    mocked_ff.ch1.sweep.assert_called_once()
    mocked_ff.write.assert_has_calls(expected_writes)

# Getting the query responses right is annoying. A lot of checks and queries
# happen when getting the snp network, and specifying them here is difficult
# and error-prone especially when changes are made. Something to figure out
# for the future
# def test_get_snp_network(mocker, mocked_ff):
#     mock_sdata = np.array([1.,]*22*4)
#     query_responses = [
#         '4',
#         'ASC,0',
#         '1', '1', '1', '1', '1', '1', '1', '1', '1',
#         '1', '1', '1', '1', '1', '1', '1', '1',
#         '11',
#         'TEST1', '1',
#         'TEST2', '2',
#         'TEST3', '3',
#         'TEST4', '4',
#         '1',
#         '100','200','11'
#     ]
#     expected_writes = [
#         mocker.call('FORM:BORD SWAP;FORM REAL,64'),
#         mocker.call("CALC1:PAR:EXT 'TEST1',S11"),
#         mocker.call("DISP:WIND:TRAC:FEED1 'TEST1'"),
#         mocker.call("CALC1:PAR:EXT 'TEST2',S22"),
#         mocker.call("DISP:WIND:TRAC:FEED2 'TEST2'"),
#         mocker.call("CALC1:PAR:EXT 'TEST3',S33"),
#         mocker.call("DISP:WIND:TRAC:FEED3 'TEST3'"),
#         mocker.call("CALC1:PAR:EXT 'TEST4',S44"),
#         mocker.call("DISP:WIND:TRAC:FEED4 'TEST4'"),
#         mocker.call("CALC1:PAR:DEL 'TEST1'"),
#         mocker.call("CALC1:PAR:DEL 'TEST2'"),
#         mocker.call("CALC1:PAR:DEL 'TEST3'"),
#         mocker.call("CALC1:PAR:DEL 'TEST4'"),
#         mocker.call('FORM ASC,0')
#     ]
#     mocked_ff.wait_for_complete = mocker.MagicMock()
#     mocked_ff.query.side_effect = query_responses
#     mocked_ff.ch1.sweep = mocker.MagicMock()
#     mocked_ff.query_values.return_value=mock_sdata
#     test = mocked_ff.ch1.get_snp_network()
#     assert isinstance(test, skrf.Network)
#     mocked_ff.ch1.sweep.assert_called_once()
#     mocked_ff.write.assert_has_calls(expected_writes)
