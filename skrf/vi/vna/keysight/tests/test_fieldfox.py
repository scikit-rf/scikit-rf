import sys

import numpy as np
import pytest

import skrf

try:
    from skrf.vi.vna import ValuesFormat, keysight
    from skrf.vi.vna.keysight.fieldfox import WindowFormat
except ImportError:
    pytest.skip(allow_module_level=True)

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.__init__', return_value=None)
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write_values')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query_values')

    yield keysight.FieldFox('TEST')

@pytest.mark.parametrize(
    'param,expected_query,expected_write,query_response,expected_val,write_val',
    [
        ('freq_start', 'SENS:FREQ:STAR?', 'SENS:FREQ:STAR 100', '100', 100, 100),
        ('freq_stop', 'SENS:FREQ:STOP?', 'SENS:FREQ:STOP 100', '100', 100, 100),
        ('freq_center', 'SENS:FREQ:CENT?', 'SENS:FREQ:CENT 100', '100', 100, 100),
        ('freq_span', 'SENS:FREQ:SPAN?', 'SENS:FREQ:SPAN 100', '100', 100, 100),
        ('npoints', 'SENS:SWE:POIN?', 'SENS:SWE:POIN 100', '100', 100, 100),
        ('if_bandwidth', 'SENS:BWID?', 'SENS:BWID 100', '100', 100, 100),
        ('window_configuration', 'DISP:WIND:SPL?', 'DISP:WIND:SPL D1', 'D1',
         WindowFormat.ONE_TRACE, WindowFormat.ONE_TRACE),
        ('ntraces', 'CALC:PAR:COUN?', 'CALC:PAR:COUN 1', '1', 1, 1),
        ('active_trace', None, 'CALC:PAR1:SEL', None, None, 1),
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
        test_query = getattr(mocked_ff, param)
        mocked_ff.query.assert_called_once_with(expected_query)
        assert test_query == expected_val

    if expected_write is not None:
        setattr(mocked_ff, param, write_val)
        mocked_ff.write.assert_called_once_with(expected_write)

def test_freq_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = [
        '100', '200', '11'
    ]
    test = mocked_ff.frequency
    assert test == skrf.Frequency(100, 200, 11, unit='hz')

def test_freq_write(mocker, mocked_ff):
    test_f = skrf.Frequency(100, 200, 11, unit='hz')
    mocked_ff.frequency = test_f
    calls = [
        mocker.call("SENS:FREQ:STAR 100"),
        mocker.call("SENS:FREQ:STOP 200"),
        mocker.call("SENS:SWE:POIN 11"),
    ]
    mocked_ff.write.assert_has_calls(calls)

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

def test_define_msmnt(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.ntraces', return_value=1, new_callable=mocker.PropertyMock)
    mocked_ff.define_measurement(1, 'S11')
    mocked_ff.write.assert_called_once_with('CALC:PAR1:DEF S11')

# def test_calibration_write(mocker, mocked_ff):
    # cal_terms = mocked_ff._cal_term_map
    # mock_array = np.array([1+1j, 1+1j])
    # cal_dict = {k: mock_array.copy() for k in cal_terms.keys()}
    # mock_cal = skrf.Calibration.from_coefs(skrf.Frequency(100, 200, 11, unit='hz'), cal_dict)
    # mocked_ff.calibration = mock_cal

    # numpy defines == in a way that makes mocker.assert_has_calls not work
    # Instead, we can just do the individual comparisons ourselves
    # expected_calls = [
        # mocker.call(f'SENS:CORR:COEF {term},', np.array([1.,1.,1.,1.]))
        # for term in cal_terms.values()
    # ]
    # actual_calls = mocked_ff.write_values.call_args_list
    # for actual, expected in zip(actual_calls, expected_calls):
        # assert actual[0][0] == expected[1][0]
        # np.testing.assert_array_almost_equal(actual[0][1], expected[1][1])

def test_get_measurement_parameter(mocker, mocked_ff):
    mocked_ff.query.return_value='S11'
    test = mocked_ff.get_measurement_parameter(1)
    assert test == 'S11'

    with pytest.raises(ValueError):
        test = mocked_ff.get_measurement_parameter(5)

def test_sweep(mocker, mocked_ff):
    mocked_ff._resource = mocker.Mock()
    mocked_ff.query.return_value='1'
    mocked_ff.sweep()
    calls = [
        mocker.call('INIT:CONT 0'),
        mocker.call('INIT'),
        mocker.call('INIT:CONT 1'),
    ]
    mocked_ff.write.assert_has_calls(calls)
    mocked_ff._resource.clear.assert_called_once()

def test_get_snp_network(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.sweep')
    query_ret_vals = [
        '1',
        WindowFormat.ONE_TRACE,
        '1',
        'S11',
        '4', '4', '4', '4',
        '100', '200', '11',
        '4'
    ]
    mocked_ff.query.side_effect=query_ret_vals
    mock_s_data = np.array([1. + 1.j]*11)
    mocker.patch('skrf.vi.vna.keysight.FieldFox.active_trace_sdata',
                 return_value=mock_s_data.copy(), new_callable=mocker.PropertyMock)

    test = mocked_ff.get_snp_network()

    mocked_ff.sweep.assert_called_once()
    assert isinstance(test, skrf.Network)
    assert test.s.shape == (11,2,2)
    expected = np.array([1+1j]*11)
    for s in [getattr(test, f'{param}') for param in ['s11', 's12', 's21', 's22']]:
        actual = s.s.reshape((-1,))
        np.testing.assert_array_almost_equal(actual, expected)
