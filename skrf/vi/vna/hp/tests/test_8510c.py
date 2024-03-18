import sys

import pytest

import skrf

try:
    from skrf.vi.vna import hp
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
    mocked_ff._resource = None
    if expected_query is not None:
        mocked_ff.query.return_value = query_response
        test_val = getattr(mocked_ff, param)
        mocked_ff.query.assert_called_once_with(expected_query)
        assert test_val == expected_val

    if expected_write is not None:
        setattr(mocked_ff, param, write_val)
        mocked_ff.write.assert_called_once_with(expected_write)

def test_freq_query(mocker, mocked_ff):
    mocked_ff.query.side_effect = [
        '100', '200', '51'
    ]
    test = mocked_ff.frequency
    assert test == skrf.Frequency(100, 200, 51, unit='hz')

def test_freq_write(mocker, mocked_ff):
    mocked_ff._resource = mocker.MagicMock()
    mocked_ff.min_hz = 100
    mocked_ff.max_hz = 100000
    test_f = skrf.Frequency(100, 200, 51, unit='hz')
    mocked_ff.frequency = test_f
    assert mocked_ff.frequency==test_f

def test_reset(mocker, mocked_ff):
    mocked_ff.reset()
    mocked_ff.write.assert_called_once_with("FACTPRES;")

def test_wait_until_finished(mocker, mocked_ff):
    mocked_ff.wait_until_finished()
    mocked_ff.query.assert_called_once()
