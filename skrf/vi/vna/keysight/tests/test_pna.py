
import numpy as np
import pytest

import skrf
from skrf.vi import vna
from skrf.vi.vna import ValuesFormat, keysight
from skrf.vi.vna.keysight.pna import SweepType


@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.keysight.PNA.__init__', return_value=None)
    mocker.patch('skrf.vi.vna.keysight.PNA.write')
    mocker.patch('skrf.vi.vna.keysight.PNA.write_values')
    mocker.patch('skrf.vi.vna.keysight.PNA.query')
    mocker.patch('skrf.vi.vna.keysight.PNA.query_values')
    mock = keysight.PNA('TEST')

    # This gets done in init, but we are mocking init to prevent super().__init__, so just call here
    mock.create_channel(1, 'Channel 1') 
    yield mock

@pytest.mark.parametrize(
    'param,expected_query,expected_write,query_response,expected_val,write_val',
    [
        ('freq_start', 'SENS1:FREQ:STAR?', 'SENS1:FREQ:STAR 100', '100', 100, 100),
        ('freq_stop', 'SENS1:FREQ:STOP?', 'SENS1:FREQ:STOP 100', '100', 100, 100),
        ('freq_step', 'SENS1:SWE:STEP?', 'SENS1:SWE:STEP 100', '100', 100, 100),
        ('freq_span', 'SENS1:FREQ:SPAN?', 'SENS1:FREQ:SPAN 100', '100', 100, 100),
        ('freq_center', 'SENS1:FREQ:CENT?', 'SENS1:FREQ:CENT 100', '100', 100, 100),
        ('npoints', 'SENS1:SWE:POIN?', 'SENS1:SWE:POIN 100', '100', 100, 100),
        ('if_bandwidth', 'SENS1:BWID?', 'SENS1:BWID 100', '100', 100, 100),
        ('sweep_time', 'SENS1:SWE:TIME?', 'SENS1:SWE:TIME 1.0', '1.0', 1.0, 1),
        ('sweep_type', 'SENS1:SWE:TYPE?', 'SENS1:SWE:TYPE LIN', 'LIN', SweepType.LINEAR, SweepType.LINEAR),
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

def test_create_channel(mocker, mocked_ff):
    mocked_ff.create_channel(2, 'Channel 2')
    assert hasattr(mocked_ff, 'ch2')
    assert mocked_ff.ch2.cnum == 2
    assert mocked_ff.ch2.name == "Channel 2"

def test_active_channel_query(mocker, mocked_ff):
    mocked_ff.query.return_value = 1
    test = mocked_ff.active_channel
    assert isinstance(test, keysight.PNA.Channel)
    assert test.cnum == 1

def test_active_channel_setter(mocker, mocked_ff):
    mocked_ff.query.side_effect = ['1', '1', '1,2,3', '2']
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
    mocked_ff.write.assert_called_with('FORM REAL,32')
    mocked_ff.query_format = ValuesFormat.BINARY_64
    mocked_ff.write.assert_called_with('FORM REAL,64')
