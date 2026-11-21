from pathlib import Path

import numpy as np
import pytest

import skrf as rf
from skrf.calibration import EightTerm, OnePort, SixteenTerm, TwelveTerm
from skrf.io.xarray import PARAMS, PROJECTIONS, add_params, add_proj, from_xarray, read_zarr, to_xarray, write_zarr
from skrf.networkSet import NetworkSet

xr = pytest.importorskip('xarray')
zarr = pytest.importorskip('zarr')


@pytest.fixture
def frequency():
    return rf.Frequency(1, 2, 11, 'GHz')


def random_network(frequency, nports, rng):
    shape = (frequency.npoints, nports, nports)
    return rf.Network(frequency=frequency, s=rng.normal(size=shape) + 1j * rng.normal(size=shape))


def assert_networks_equal(a, b):
    np.testing.assert_array_equal(a.f, b.f)
    np.testing.assert_array_equal(a.s, b.s)
    np.testing.assert_array_equal(a.z0, b.z0)
    assert a.frequency.unit == b.frequency.unit
    for attr in ('name', 'comments', 'params', 's_def', 'port_names', 'noise_interp_kind'):
        assert getattr(a, attr) == getattr(b, attr)
    np.testing.assert_array_equal(a.port_modes, b.port_modes)
    np.testing.assert_array_equal(a.noise_fill_value, b.noise_fill_value)
    assert a.noisy == b.noisy
    if a.noisy:
        np.testing.assert_array_equal(a.noise, b.noise)
        np.testing.assert_array_equal(a.noise_freq.f, b.noise_freq.f)


@pytest.fixture
def network(frequency):
    rng = np.random.default_rng(0)
    ntwk = random_network(frequency, 2, rng)
    ntwk.z0 = 50 + rng.uniform(-5, 5, size=(frequency.npoints, 2)) + 1j * rng.normal(size=(frequency.npoints, 2))
    ntwk.name = 'dut'
    ntwk.comments = 'measured on bench 2'
    ntwk.params = {'temperature': 25.0, 'bias': 'on', 'run': 3}
    ntwk.port_names = ['in', 'out']
    ntwk.s_def = 'pseudo'
    return ntwk


def test_frequency(frequency, tmp_path):
    write_zarr(tmp_path / 'f.zarr', frequency)
    assert read_zarr(tmp_path / 'f.zarr') == frequency
    assert from_xarray(to_xarray(frequency)).unit == 'GHz'


def test_network_dataset(network):
    ds = to_xarray(network)
    assert ds['s'].dims == ('f_hz', 'm', 'n')
    assert ds['z0'].dims == ('f_hz', 'port')
    np.testing.assert_array_equal(ds['s'].sel(m=2, n=1).values, network.s[:, 1, 0])
    np.testing.assert_array_equal(ds['f_hz'].values, network.f)
    assert_networks_equal(from_xarray(ds), network)


def test_network_zarr(network, tmp_path):
    write_zarr(tmp_path / 'dut.zarr', network)
    assert zarr.open_group(tmp_path / 'dut.zarr', mode='r')['s'].dtype == np.complex128
    assert_networks_equal(read_zarr(tmp_path / 'dut.zarr'), network)


def test_network_overwrite(network, tmp_path):
    write_zarr(tmp_path / 'dut.zarr', rf.data.ring_slot)
    write_zarr(tmp_path / 'dut.zarr', network)
    assert_networks_equal(read_zarr(tmp_path / 'dut.zarr'), network)


def test_network_defaults(tmp_path):
    ntwk = rf.data.ring_slot
    write_zarr(tmp_path / 'ring_slot.zarr', ntwk)
    assert_networks_equal(read_zarr(tmp_path / 'ring_slot.zarr'), ntwk)


@pytest.fixture
def noisy():
    return rf.Network(Path(rf.__file__).parent / 'tests' / 'ntwk_noise.s2p')


def test_network_noise(noisy, tmp_path):
    ds = to_xarray(noisy)
    assert ds['noise'].dims == ('noise_f_hz', 'm', 'n')
    write_zarr(tmp_path / 'noisy.zarr', noisy)
    result = read_zarr(tmp_path / 'noisy.zarr')
    assert_networks_equal(result, noisy)
    np.testing.assert_allclose(result.nfmin, noisy.nfmin)


def test_network_noise_fill_value(noisy):
    noisy.noise_interp_kind = 'cubic'
    noisy.noise_fill_value = 1 + 2j
    assert_networks_equal(from_xarray(to_xarray(noisy)), noisy)


def test_f_scaled(network):
    ds = to_xarray(network)
    np.testing.assert_array_equal(ds['f_scaled'].values, network.frequency.f_scaled)
    assert ds['f_scaled'].attrs['units'] == 'GHz'
    assert ds['s'].sel(m=1, n=1).swap_dims(f_hz='f_scaled').sel(f_scaled=1.5).item() == \
        network.s[5, 0, 0]


def test_network_params_not_implemented(network):
    network.params = {'sweep': [1, 2, 3]}
    with pytest.raises(NotImplementedError):
        to_xarray(network)


def test_networkset(frequency, tmp_path):
    rng = np.random.default_rng(1)
    ntwks = []
    for temperature in (25.0, 50.0):
        for bias in ('on', 'off'):
            ntwk = random_network(frequency, 2, rng)
            ntwk.name = f'dut_{temperature}_{bias}'
            ntwk.params = {'temperature': temperature, 'bias': bias}
            ntwks.append(ntwk)
    ns = NetworkSet(ntwks, name='sweep')

    ds = to_xarray(ns)
    assert ds['s'].dims == ('network', 'f_hz', 'm', 'n')
    hot_off = ds['s'].where((ds.temperature == 50.0) & (ds.bias == 'off'), drop=True)
    np.testing.assert_array_equal(hot_off.values[0], ntwks[3].s)

    hot = from_xarray(ds.query(network='temperature == 50'))
    assert [n.name for n in hot.ntwk_set] == [n.name for n in ntwks[2:]]
    assert_networks_equal(hot.ntwk_set[1], ntwks[3])

    write_zarr(tmp_path / 'sweep.zarr', ns)
    result = read_zarr(tmp_path / 'sweep.zarr')
    assert result.name == 'sweep'
    assert len(result) == len(ns)
    for a, b in zip(result.ntwk_set, ns.ntwk_set):
        assert_networks_equal(a, b)


@pytest.mark.parametrize('key', ['z0', 'f_hz', 'm', 'network_attrs'])
def test_networkset_params_name_not_implemented(frequency, key):
    rng = np.random.default_rng(7)
    ntwks = [random_network(frequency, 2, rng) for _ in range(2)]
    for i, ntwk in enumerate(ntwks):
        ntwk.params = {key: i}
    with pytest.raises(NotImplementedError):
        to_xarray(NetworkSet(ntwks))


def test_networkset_noise(noisy, tmp_path):
    ns = NetworkSet([noisy, noisy.copy()])
    write_zarr(tmp_path / 'set.zarr', ns)
    for a, b in zip(read_zarr(tmp_path / 'set.zarr').ntwk_set, ns.ntwk_set):
        assert_networks_equal(a, b)


def test_networkset_noise_not_implemented(noisy):
    quiet = noisy.copy()
    quiet.noise = quiet.noise_freq = None
    with pytest.raises(NotImplementedError):
        to_xarray(NetworkSet([noisy, quiet]))


def test_networkset_without_params(frequency, tmp_path):
    rng = np.random.default_rng(2)
    ns = NetworkSet([random_network(frequency, 1, rng) for _ in range(3)])
    write_zarr(tmp_path / 'set.zarr', ns)
    for a, b in zip(read_zarr(tmp_path / 'set.zarr').ntwk_set, ns.ntwk_set):
        assert_networks_equal(a, b)


def calibration(cls, nports, nstandards, frequency):
    rng = np.random.default_rng(4)
    measured = [random_network(frequency, nports, rng) for _ in range(nstandards)]
    ideals = [random_network(frequency, nports, rng) for _ in range(nstandards)]
    kwargs = {}
    if cls is TwelveTerm:
        kwargs['n_thrus'] = 1
    elif nports == 2:
        kwargs['switch_terms'] = (random_network(frequency, 1, rng), random_network(frequency, 1, rng))
    cal = cls(measured, ideals, name='bench 2', **kwargs)
    cal.run()
    return cal


@pytest.mark.parametrize('cls, nports, nstandards', [
    (OnePort, 1, 4),
    (TwelveTerm, 2, 4),
    (EightTerm, 2, 5),
    (SixteenTerm, 2, 8),
])
def test_calibration(cls, nports, nstandards, frequency, tmp_path):
    cal = calibration(cls, nports, nstandards, frequency)
    write_zarr(tmp_path / 'cal.zarr', cal)
    result = read_zarr(tmp_path / 'cal.zarr')

    assert type(result) is cls
    assert result.name == 'bench 2'
    assert result.coefs.keys() == cal.coefs.keys()
    for key in cal.coefs:
        np.testing.assert_array_equal(result.coefs[key], cal.coefs[key])
    dut = random_network(frequency, nports, np.random.default_rng(5))
    np.testing.assert_allclose(result.apply_cal(dut).s, cal.apply_cal(dut).s)


@pytest.mark.parametrize('cls', ['SOLT', 'TRL', 'LRRM'])
def test_calibration_subclass(cls, frequency):
    coefs = calibration(TwelveTerm if cls == 'SOLT' else EightTerm, 2, 5, frequency).coefs
    cal = getattr(rf.calibration, cls).from_coefs(frequency, coefs)
    result = from_xarray(to_xarray(cal))
    assert type(result) is type(cal)
    dut = random_network(frequency, 2, np.random.default_rng(6))
    np.testing.assert_allclose(result.apply_cal(dut).s, cal.apply_cal(dut).s)


def test_calibration_not_implemented(frequency):
    coefs = calibration(TwelveTerm, 2, 4, frequency).coefs
    cal = rf.calibration.TwoPortOnePath.from_coefs(frequency, coefs, n_thrus=1)
    with pytest.raises(NotImplementedError):
        to_xarray(cal)


def test_object_not_implemented():
    with pytest.raises(NotImplementedError):
        to_xarray([1, 2, 3])


def test_newer_format(network, tmp_path):
    ds = to_xarray(network)
    ds.attrs['skrf_format'] += 1
    with pytest.raises(NotImplementedError):
        from_xarray(ds)
    with pytest.warns(UserWarning, match='newer|format'):
        assert_networks_equal(from_xarray(ds, ignore_version=True), network)

    ds.to_zarr(tmp_path / 'newer.zarr', consolidated=False)
    with pytest.raises(NotImplementedError):
        read_zarr(tmp_path / 'newer.zarr')
    with pytest.warns(UserWarning):
        read_zarr(tmp_path / 'newer.zarr', ignore_version=True)


def test_add_params(network):
    ds = add_params(to_xarray(network), 'z', 'a')
    np.testing.assert_array_equal(ds['z'].values, network.z)
    np.testing.assert_array_equal(ds['a'].values, network.a)
    assert ds['z'].dims == ds['s'].dims
    assert set(PARAMS) <= set(add_params(to_xarray(network), 'all').data_vars)


def test_add_params_all_one_port(network):
    ds = add_params(to_xarray(network.s11), 'all')
    assert set(ds.data_vars) == {'s', 'z0', 'z', 'y'}


def test_add_params_unknown(network):
    with pytest.raises(ValueError):
        add_params(to_xarray(network), 'q')


def test_add_proj(network):
    ds = add_params(to_xarray(network), 'z')
    ds = add_proj(ds, 'db', 'deg_unwrap')
    for name in ('s_db', 's_deg_unwrap', 'z_db', 'z_deg_unwrap'):
        np.testing.assert_array_equal(ds[name].values, getattr(network, name))
    assert ds['s_db'].attrs['long_name'] == 'S Magnitude (dB)'

    ds = add_proj(to_xarray(network), ['re', 'im'], params='s')
    assert {'s_re', 's_im'} <= set(ds.data_vars)
    assert len(add_proj(to_xarray(network), 'all').data_vars) == 2 + len(PROJECTIONS)


def test_add_proj_unknown(network):
    with pytest.raises(ValueError):
        add_proj(to_xarray(network), 'time_db')


def test_add_proj_networkset(frequency):
    rng = np.random.default_rng(8)
    ns = NetworkSet([random_network(frequency, 2, rng) for _ in range(3)])
    ds = add_proj(to_xarray(ns), 'deg_unwrap')
    for i, ntwk in enumerate(ns.ntwk_set):
        np.testing.assert_array_equal(ds['s_deg_unwrap'].values[i], ntwk.s_deg_unwrap)


def test_add_proj_calibration_not_implemented(frequency):
    with pytest.raises(NotImplementedError):
        add_proj(to_xarray(calibration(OnePort, 1, 4, frequency)), 'db')


def test_write_dataset(network, tmp_path):
    write_zarr(tmp_path / 'dut.zarr', add_proj(to_xarray(network), 'db'))
    np.testing.assert_array_equal(xr.open_zarr(tmp_path / 'dut.zarr', consolidated=False)['s_db'].values, network.s_db)
    assert_networks_equal(read_zarr(tmp_path / 'dut.zarr'), network)


def test_to_xarray_params_projs(network):
    ds = to_xarray(network, params='z', projs=['db', 'deg'])
    expected = add_proj(add_params(to_xarray(network), 'z'), 'db', 'deg')
    assert set(ds.data_vars) == set(expected.data_vars) == {'s', 'z0', 'z', 's_db', 's_deg', 'z_db', 'z_deg'}
    for name in ds.data_vars:
        np.testing.assert_array_equal(ds[name].values, expected[name].values)


def test_to_xarray_projs_networkset(frequency):
    rng = np.random.default_rng(9)
    ns = NetworkSet([random_network(frequency, 2, rng) for _ in range(2)])
    ds = to_xarray(ns, projs='deg_unwrap')
    for i, ntwk in enumerate(ns.ntwk_set):
        np.testing.assert_array_equal(ds['s_deg_unwrap'].values[i], ntwk.s_deg_unwrap)


def test_to_xarray_projs_calibration_not_implemented(frequency):
    with pytest.raises(NotImplementedError):
        to_xarray(calibration(OnePort, 1, 4, frequency), projs='db')
