import numpy as np
import pytest

import skrf
from skrf.vi import vna
from skrf.vi.vna import keysight
from skrf.vi.vna.keysight.fieldfox import WindowFormat


@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.__init__', return_value=None)
    yield keysight.FieldFox('TEST')


def test_freq_start_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='100')
    test = mocked_ff.freq_start
    mocked_ff.query.assert_called_once_with('SENS:FREQ:STAR?')
    assert test == 100

def test_freq_start_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.freq_start = 100
    mocked_ff.write.assert_called_once_with('SENS:FREQ:STAR 100')

def test_freq_stop_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='100')
    test = mocked_ff.freq_stop
    mocked_ff.query.assert_called_once_with('SENS:FREQ:STOP?')
    assert test == 100

def test_freq_stop_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.freq_stop = 100
    mocked_ff.write.assert_called_once_with('SENS:FREQ:STOP 100')

def test_freq_step_query(mocker, mocked_ff):
    f = skrf.Frequency(100, 200, 11, unit='Hz')
    mocker.patch(
        'skrf.vi.vna.keysight.FieldFox.frequency', 
        return_value=f, 
        new_callable=mocker.PropertyMock
    )
    assert mocked_ff.freq_step == 10

def test_freq_step_write(mocker, mocked_ff):
    f = skrf.Frequency(100, 200, 11, unit='Hz')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocker.patch(
        'skrf.vi.vna.keysight.FieldFox.frequency', 
        return_value=f, 
        new_callable=mocker.PropertyMock
    )
    mocked_ff.freq_step = 10
    mocked_ff.write.assert_called_once_with('SENS:SWE:POIN 11')

def test_npoints_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='10')
    test = mocked_ff.npoints
    mocked_ff.query.assert_called_once_with('SENS:SWE:POIN?')
    assert test == 10

def test_npoints_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.npoints = 100
    mocked_ff.write.assert_called_once_with('SENS:SWE:POIN 100')

def test_sweep_time_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='0.250')
    test = mocked_ff.sweep_time
    mocked_ff.query.assert_called_once_with('SENS:SWE:TIME?')
    assert test == 0.25

def test_sweep_time_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.sweep_time = 0.25
    mocked_ff.write.assert_called_once_with('SENS:SWE:TIME 0.25')

def test_if_bw_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='30')
    mocked_ff.if_bandwidth
    mocked_ff.query.assert_called_once_with('SENS:BWID?')

def test_if_bw_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.if_bandwidth = 30
    mocked_ff.write.assert_called_once_with('SENS:BWID 30')

def test_window_config_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='D11_23')
    mocked_ff.window_configuration
    mocked_ff.query.assert_called_once_with('DISP:WIND:SPL?')

def test_window_config_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.window_configuration = WindowFormat.ONE_FIRST_ROW_TWO_SECOND_ROW
    mocked_ff.write.assert_called_once_with('DISP:WIND:SPL D11_23')

def test_n_traces_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='1')
    mocked_ff.n_traces
    mocked_ff.query.assert_called_once_with('CALC:PAR:COUN?')

def test_n_traces_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.n_traces = 1
    mocked_ff.write.assert_called_once_with('CALC:PAR:COUN 1')

def test_freq_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.freq_start', return_value=100, new_callable=mocker.PropertyMock)
    mocker.patch('skrf.vi.vna.keysight.FieldFox.freq_stop', return_value=200, new_callable=mocker.PropertyMock)
    mocker.patch('skrf.vi.vna.keysight.FieldFox.npoints', return_value=11, new_callable=mocker.PropertyMock)

    test = mocked_ff.frequency
    assert test == skrf.Frequency(start=100, stop=200, npoints=11, unit='Hz')

def test_freq_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.frequency = skrf.Frequency(100, 200, 11, unit='Hz')
    calls = [
        mocker.call('SENS:FREQ:STAR 100'), 
        mocker.call('SENS:FREQ:STOP 200'), 
        mocker.call('SENS:SWE:POIN 11')
    ]
    mocked_ff.write.assert_has_calls(calls)

def test_query_fmt_query(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='ASC,0')
    test = mocked_ff.query_format
    mocked_ff.query.assert_called_once_with('FORM?')
    assert test == vna.ValuesFormat.ASCII

    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='REAL,32')
    test = mocked_ff.query_format
    assert test == vna.ValuesFormat.BINARY_32

def test_query_fmt_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocked_ff.query_format = vna.ValuesFormat.ASCII
    mocked_ff.write.assert_called_with('FORM ASC,0')
    mocked_ff.query_format = vna.ValuesFormat.BINARY_32
    mocked_ff.write.assert_called_with('FORM REAL,32')

def test_define_msmnt(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.n_traces', return_value=1, new_callable=mocker.PropertyMock)
    mocked_ff.define_measurement(1, 'S11')
    mocked_ff.write.assert_called_once_with('CALC:PAR1:DEF S11')

def test_calibration_write(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write_values')
    cal_terms = mocked_ff._cal_term_map
    mock_array = np.array([1+1j, 1+1j])
    cal_dict = {k: mock_array.copy() for k in cal_terms.keys()}
    mock_cal = skrf.Calibration.from_coefs(skrf.Frequency(100, 200, 11, unit='hz'), cal_dict)
    mocked_ff.calibration = mock_cal

    # numpy defines == in a way that makes mocker.assert_has_calls not work
    # Instead, we can just do the individual comparisons ourselves
    expected_calls = [
        mocker.call(f'SENS:CORR:COEF {term},', np.array([1.,1.,1.,1.]))
        for term in cal_terms.values()
    ]
    actual_calls = mocked_ff.write_values.call_args_list
    for actual, expected in zip(actual_calls, expected_calls):
        assert actual[0][0] == expected[1][0]
        np.testing.assert_array_almost_equal(actual[0][1], expected[1][1])

def test_get_measurement_parameter(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='S11')
    test = mocked_ff.get_measurement_parameter(1)
    assert test == 'S11'

    with pytest.raises(ValueError):
        test = mocked_ff.get_measurement_parameter(5)

def test_sweep(mocker, mocked_ff):
    mocked_ff._resource = mocker.Mock()
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', return_value='1')
    mocked_ff.sweep()
    calls = [
        mocker.call('INIT:CONT 0'),
        mocker.call('INIT'),
        mocker.call('INIT:CONT 1'),
    ]
    mocked_ff.write.assert_has_calls(calls)
    mocked_ff._resource.clear.assert_called_once()

def test_get_snp_network(mocker, mocked_ff):
    mocker.patch('skrf.vi.vna.keysight.FieldFox.write')
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
    mocker.patch('skrf.vi.vna.keysight.FieldFox.query', side_effect=query_ret_vals)
    mock_s_data = np.array([1.]*22) # our test has 11 frequency points and we expect two values per point (re,im)
    mocker.patch('skrf.vi.vna.keysight.FieldFox.active_trace_sdata', return_value=mock_s_data.copy(), new_callable=mocker.PropertyMock)

    test = mocked_ff.get_snp_network()

    mocked_ff.sweep.assert_called_once()
    assert isinstance(test, skrf.Network)
    assert test.s.shape == (11,2,2)
    expected = np.array([1+1j]*11)
    for s in [getattr(test, f'{param}') for param in ['s11', 's12', 's21', 's22']]:
        actual = s.s.reshape((-1,))
        np.testing.assert_array_almost_equal(actual, expected)