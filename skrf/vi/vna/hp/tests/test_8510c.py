import pytest
import sys

import skrf
try:
    from skrf.vi.validators import ValidationError
    from skrf.vi.vna import ValuesFormat, hp
except ImportError:
    pass

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

@pytest.fixture
def mocked_ff(mocker):
    mocker.patch('skrf.vi.vna.hp.hp8510c.HP8510C.__init__', return_value=None)
    mocker.patch('skrf.vi.vna.hp.hp8510c.HP8510C.write')
    mocker.patch('skrf.vi.vna.hp.hp8510c.HP8510C.write_values')
    mocker.patch('skrf.vi.vna.hp.hp8510c.HP8510C.query')
    mocker.patch('skrf.vi.vna.hp.hp8510c.HP8510C.query_values')

    yield hp.HP8510C('TEST')


@pytest.mark.parametrize(
    'param,expected_query,expected_write,query_response,expected_val,write_val',
    [
        ('id', 'OUTPIDEN;', None, 'HP8510C.07.14: Aug 26  1998', 'HP8510C.07.14: Aug 26  1998', None),
        ('freq_start', 'STAR;OUTPACTI;', 'STEP; STAR 100;', '100', 100, 100),
        ('freq_stop', 'STOP;OUTPACTI;', 'STEP; STOP 100;', '100', 100, 100),
        ('npoints', 'POIN;OUTPACTI;', 'STEP; POIN 101;', '101', 101, 101),
        ('is_continuous', 'GROU?', 'CONT;', '"CONTINUAL"', True, True),
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
        test_val = getattr(mocked_ff, param)
        mocked_ff.query.assert_called_once_with(expected_query)
        assert test_val == expected_val

    if expected_write is not None:
        setattr(mocked_ff, param, write_val)
        mocked_ff.write.assert_called_once_with(expected_write)

def test_npoint_raises(mocker, mocked_ff):
    with pytest.raises(ValidationError):
        mocked_ff.npoints = 100

def test_freq_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = [
        '100', '200', '51'
    ]
    test = mocked_ff.frequency
    assert test == skrf.Frequency(100, 200, 51, unit='hz')

def test_freq_write(mocker, mocked_ff):
    mocked_ff._resource = mocker.MagicMock()
    test_f = skrf.Frequency(100, 200, 51, unit='hz')
    mocked_ff.frequency = test_f
    mocked_ff.write.assert_called_once_with("STEP; STAR 100; STOP 200; POIN51;")

def test_query_fmt_query(mocker, mocked_ff):
    mocked_ff._values_fmt = ValuesFormat.ASCII
    test = mocked_ff.query_format
    assert test == ValuesFormat.ASCII

def test_query_fmt_write(mocker, mocked_ff):
    mocked_ff.query_format = ValuesFormat.BINARY_32
    mocked_ff.write.assert_called_once_with("FORM2;")

def test_reset(mocker, mocked_ff):
    mocked_ff.query.return_value = "TEST"
    expected_calls = [
        mocker.call("FACTPRES;"),
        mocker.call("FORM4;")
    ]
    mocked_ff.reset()
    mocked_ff.write.assert_has_calls(expected_calls)

def test_wait_until_finished(mocker, mocked_ff):
    mocked_ff.wait_until_finished()
    mocked_ff.query.assert_called_once()