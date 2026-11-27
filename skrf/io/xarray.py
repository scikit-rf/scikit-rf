"""
.. module:: skrf.io.xarray

========================================
xarray (:mod:`skrf.io.xarray`)
========================================

Convert skrf objects to and from :class:`xarray.Dataset`, and store them
in `Zarr <https://zarr.dev/>`_.

Unlike :func:`~skrf.io.general.write`, which uses :mod:`pickle`, Zarr files
are safe to share, do not depend on the skrf version, and can be read by
any tool that reads Zarr. Complex values are stored as complex numbers.

This needs the ``xarray`` extra: ``pip install scikit-rf[xarray]``.

Supported objects are :class:`~skrf.frequency.Frequency`,
:class:`~skrf.network.Network`, :class:`~skrf.networkSet.NetworkSet` and
the calibrations listed in :data:`CALIBRATIONS`.

.. autosummary::
   :toctree: generated/

   to_xarray
   from_xarray
   add_params
   add_proj
   write_zarr
   read_zarr

Examples
--------
>>> import skrf as rf
>>> from skrf.io.xarray import read_zarr, to_xarray, write_zarr
>>> ds = to_xarray(rf.data.ring_slot)
>>> ds['s'].sel(m=2, n=1)
>>> write_zarr('ring_slot.zarr', rf.data.ring_slot)
>>> ntwk = read_zarr('ring_slot.zarr')

"""
from __future__ import annotations

import json
from numbers import Number
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

from .. import __version__
from ..frequency import Frequency
from ..network import Network
from ..networkSet import NetworkSet

if TYPE_CHECKING:
    import xarray as xr

FORMAT_VERSION = 1
"""Version of the layout written by this module."""

PARAMS = tuple(p for p in Network.PRIMARY_PROPERTIES if p != 's')
"""Network parameters :func:`add_params` can add, such as ``'z'``, ``'y'`` and ``'a'`` (ABCD)."""

PROJECTIONS = tuple(p for p in Network.COMPONENT_FUNC_DICT if not p.startswith('time'))
"""Projections :func:`add_proj` can add, such as ``'db'`` and ``'deg'``."""

CALIBRATIONS = (
    'OnePort', 'TwelveTerm', 'SOLT', 'EightTerm', 'TRL', 'NISTMultilineTRL', 'TUGMultilineTRL',
    'UnknownThru', 'LRM', 'LRRM', 'MRC', 'SixteenTerm', 'LMR16',
)
"""Calibration classes that can be rebuilt from their error coefficients."""


def _xarray():
    try:
        import xarray as xr
    except ImportError as err:
        raise ImportError('xarray is required: pip install scikit-rf[xarray]') from err
    return xr


def _attrs(skrf_type: str, **kwargs) -> dict:
    return {'skrf_type': skrf_type, 'skrf_format': FORMAT_VERSION, 'skrf_version': __version__, **kwargs}


def _check_value(value: Any, what: str) -> Any:
    """Return `value` as a plain Python value that Zarr attributes can hold."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, Number)) and not isinstance(value, complex):
        return value
    raise NotImplementedError(f'{what} must be a number, string, bool or None, not {type(value).__name__}.')


def _check_params(params: dict | None) -> dict | None:
    if params is None:
        return None
    return {str(k): _check_value(v, f'Network.params[{k!r}]') for k, v in params.items()}


def _frequency_coords(frequency: Frequency, dim: str = 'f_hz') -> dict:
    """Coordinates for `frequency` along `dim`: in Hz, and scaled to its display unit."""
    scaled = dim.replace('f_hz', 'f_scaled')
    long_name = 'Noise frequency' if dim.startswith('noise') else 'Frequency'
    return {
        dim: (dim, frequency.f, {'units': 'Hz', 'long_name': long_name, 'display_unit': frequency.unit}),
        scaled: (dim, frequency.f_scaled, {'units': frequency.unit, 'long_name': long_name}),
    }


def _frequency(ds: xr.Dataset, dim: str = 'f_hz') -> Frequency:
    coord = ds.coords[dim]
    frequency = Frequency.from_f(coord.values, unit='Hz')
    frequency.unit = coord.attrs.get('display_unit', 'Hz')
    return frequency


def _network_attrs(ntwk: Network) -> dict:
    fill_value = complex(ntwk.noise_fill_value)
    return {
        'name': _check_value(ntwk.name, 'Network.name'),
        'comments': _check_value(ntwk.comments, 'Network.comments'),
        's_def': ntwk.s_def,
        'port_names': None if ntwk.port_names is None else [str(p) for p in ntwk.port_names],
        'port_modes': [str(m) for m in ntwk.port_modes],
        'noise_interp_kind': _check_value(ntwk.noise_interp_kind, 'Network.noise_interp_kind'),
        'noise_fill_value': [fill_value.real, fill_value.imag],
    }


def _network(frequency: Frequency, s: np.ndarray, z0: np.ndarray, attrs: dict, params: dict | None,
             noise_freq: Frequency | None = None, noise: np.ndarray | None = None) -> Network:
    real, imag = attrs['noise_fill_value']
    ntwk = Network(frequency=frequency, s=s, z0=z0, name=attrs['name'], comments=attrs['comments'],
                   params=params, s_def=attrs['s_def'], f_unit=frequency.unit, noise=noise, noise_freq=noise_freq,
                   noise_interp_kind=attrs['noise_interp_kind'], noise_fill_value=complex(real, imag) if imag else real)
    ntwk.port_modes = np.array(attrs['port_modes'])
    if attrs['port_names'] is not None:
        ntwk.port_names = attrs['port_names']
    return ntwk


def _ports(nports: int) -> dict:
    ports = np.arange(1, nports + 1)
    return {'port': ports, 'm': ports, 'n': ports}


def _network_to_xarray(ntwk: Network) -> xr.Dataset:
    xr = _xarray()
    data_vars = {
        's': (('f_hz', 'm', 'n'), ntwk.s),
        'z0': (('f_hz', 'port'), ntwk.z0),
    }
    coords = {**_frequency_coords(ntwk.frequency), **_ports(ntwk.nports)}
    if ntwk.noisy:
        data_vars['noise'] = (('noise_f_hz', 'm', 'n'), ntwk.noise)
        coords.update(_frequency_coords(ntwk.noise_freq, 'noise_f_hz'))
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=_attrs('Network', params=_check_params(ntwk.params), **_network_attrs(ntwk)),
    )


def _network_from_xarray(ds: xr.Dataset) -> Network:
    noise = {}
    if 'noise' in ds:
        noise = {'noise_freq': _frequency(ds, 'noise_f_hz'), 'noise': ds['noise'].values}
    return _network(_frequency(ds), ds['s'].values, ds['z0'].values, ds.attrs, ds.attrs['params'], **noise)


def _networkset_to_xarray(ns: NetworkSet) -> xr.Dataset:
    xr = _xarray()
    ntwks = ns.ntwk_set
    if not ntwks:
        raise NotImplementedError('Empty NetworkSets are not supported.')
    first = ntwks[0]
    data_vars = {
        's': (('network', 'f_hz', 'm', 'n'), np.stack([n.s for n in ntwks])),
        'z0': (('network', 'f_hz', 'port'), np.stack([n.z0 for n in ntwks])),
    }
    coords = {
        **_frequency_coords(first.frequency), **_ports(first.nports),
        'network': np.arange(len(ntwks)),
        # each Network's own properties, as JSON along `network` so that they follow any selection of Networks
        'network_attrs': ('network', [json.dumps(_network_attrs(n)) for n in ntwks]),
    }

    if all(n.noisy for n in ntwks) and all(n.noise_freq == first.noise_freq for n in ntwks):
        data_vars['noise'] = (('network', 'noise_f_hz', 'm', 'n'), np.stack([n.noise for n in ntwks]))
        coords.update(_frequency_coords(first.noise_freq, 'noise_f_hz'))
    elif any(n.noisy for n in ntwks):
        raise NotImplementedError('Either all Networks of a NetworkSet must have noise parameters at the same '
                                  'frequencies, or none.')

    if ns.has_params():
        for key in first.params:
            if str(key) in data_vars or str(key) in coords:
                raise NotImplementedError(f'A params key cannot be named {key!r}, which the Dataset already uses.')
            values = [_check_value(n.params[key], f'Network.params[{key!r}]') for n in ntwks]
            coords[str(key)] = ('network', values)
    elif any(n.params for n in ntwks):
        raise NotImplementedError('All Networks of a NetworkSet must have the same params keys.')

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=_attrs('NetworkSet', name=_check_value(ns.name, 'NetworkSet.name'),
                     params=[str(k) for k in first.params] if ns.has_params() else []),
    )


def _networkset_from_xarray(ds: xr.Dataset) -> NetworkSet:
    frequency = _frequency(ds)
    noise_freq = _frequency(ds, 'noise_f_hz') if 'noise' in ds else None
    params = ds.attrs['params']
    ntwks = []
    for i, attrs in enumerate(ds.coords['network_attrs'].values):
        attrs = json.loads(str(attrs))
        p = {k: ds.coords[k].values[i].item() for k in params} if params else None
        noise = {'noise_freq': noise_freq, 'noise': ds['noise'].values[i]} if noise_freq else {}
        ntwks.append(_network(frequency, ds['s'].values[i], ds['z0'].values[i], attrs, p, **noise))
    return NetworkSet(ntwks, name=ds.attrs['name'])


def _calibration_to_xarray(cal) -> xr.Dataset:
    xr = _xarray()
    cls = type(cal).__name__
    if cls not in CALIBRATIONS:
        raise NotImplementedError(f'{cls} calibrations are not supported yet. Supported: {", ".join(CALIBRATIONS)}.')
    frequency = cal.frequency
    data_vars = {}
    for key, value in cal.coefs.items():
        value = np.asarray(value)
        if value.shape != (frequency.npoints,):
            raise NotImplementedError(f'Error coefficient {key!r} of shape {value.shape} is not supported.')
        data_vars[key] = ('f_hz', value)
    return xr.Dataset(
        data_vars=data_vars,
        coords=_frequency_coords(frequency),
        attrs=_attrs('Calibration', calibration=cls, name=_check_value(cal.name, 'Calibration.name')),
    )


def _calibration_from_xarray(ds: xr.Dataset):
    from .. import calibration

    name = ds.attrs['calibration']
    if name not in CALIBRATIONS:
        raise NotImplementedError(f'{name} calibrations are not supported yet.')
    cls = getattr(calibration, name)
    coefs = {key: ds[key].values for key in ds.data_vars}
    # the coefficients replace whatever run() would find, so n_thrus only keeps TwelveTerm from guessing it
    kwargs = {'n_thrus': 1} if issubclass(cls, calibration.TwelveTerm) else {}
    cal = cls.from_coefs(_frequency(ds), coefs, **kwargs)
    cal.name = ds.attrs['name']
    return cal


def to_xarray(obj, params: str | list[str] | None = None, projs: str | list[str] | None = None) -> xr.Dataset:
    """
    Convert an skrf object to an :class:`xarray.Dataset`.

    Parameters
    ----------
    obj : :class:`~skrf.frequency.Frequency`, :class:`~skrf.network.Network`, \
:class:`~skrf.networkSet.NetworkSet` or :class:`~skrf.calibration.calibration.Calibration`
        Object to convert.
    params : str or list of str, optional
        Network parameters to add besides S, such as ``'z'`` or ``['z', 'y']``,
        or ``'all'``. See :func:`add_params`. Only for a Network or NetworkSet.
    projs : str or list of str, optional
        Projections to add, such as ``'db'`` or ``['db', 'deg']``, of S and of
        the `params`. See :func:`add_proj`. Only for a Network or NetworkSet.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        For a Network, the variables are ``s`` with dimensions
        ``(f_hz, m, n)`` and ``z0`` with dimensions
        ``(f_hz, port)``, plus ``noise`` along ``noise_f_hz`` for
        a noisy Network. ``f_hz`` is the frequency in Hz, like ``Network.f``,
        and the coordinate ``f_scaled`` is in the display unit, like
        ``Network.frequency.f_scaled``. ``m`` and ``n`` are the port indices
        of Smn and count from 1, so ``sel(m=2, n=1)`` is S21.
        A NetworkSet adds a leading ``network`` dimension, with the Networks'
        ``params`` as coordinates along it. A Calibration has one variable per
        error coefficient. Other properties are kept in ``ds.attrs``.

    Raises
    ------
    NotImplementedError
        If the object, or some of its contents, is not supported yet.

    See Also
    --------
    from_xarray
    add_params
    add_proj
    write_zarr

    Examples
    --------
    >>> ds = to_xarray(rf.data.ring_slot)
    >>> ds['s'].sel(m=2, n=1, f_hz=80e9, method='nearest')

    With Z-parameters, and the dB and phase of S and Z

    >>> ds = to_xarray(rf.data.ring_slot, params='z', projs=['db', 'deg'])
    >>> ds['s_db'].sel(m=2, n=1).plot(x='f_scaled')

    """
    from ..calibration import Calibration

    xr = _xarray()
    if isinstance(obj, (Network, NetworkSet)):
        ds, ntwks = (_network_to_xarray(obj), [obj]) if isinstance(obj, Network) else \
            (_networkset_to_xarray(obj), obj.ntwk_set)
        if params is not None:
            ds = _add_params(ds, ntwks, (params,))
        if projs is not None:
            ds = _add_proj(ds, ntwks, (projs,), None)
        return ds
    if params is not None or projs is not None:
        raise NotImplementedError(f'params and projs are only for Networks and NetworkSets, not {type(obj).__name__}.')
    if isinstance(obj, Frequency):
        return xr.Dataset(coords=_frequency_coords(obj), attrs=_attrs('Frequency'))
    if isinstance(obj, Calibration):
        return _calibration_to_xarray(obj)
    raise NotImplementedError(f'{type(obj).__name__} objects are not supported.')


def from_xarray(ds: xr.Dataset, ignore_version: bool = False):
    """
    Convert an :class:`xarray.Dataset` made by :func:`to_xarray` back to an skrf object.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Dataset to convert.
    ignore_version : bool, optional
        If True, try to read a Dataset written in a newer format than this
        version of skrf knows, with a warning instead of an error. The result
        may be wrong or incomplete. Default is False.

    Returns
    -------
    obj : :class:`~skrf.frequency.Frequency`, :class:`~skrf.network.Network`, \
:class:`~skrf.networkSet.NetworkSet` or :class:`~skrf.calibration.calibration.Calibration`
        The object, of the type given in ``ds.attrs['skrf_type']``.

    Raises
    ------
    NotImplementedError
        If the Dataset is not one this version of skrf can read, or was
        written in a newer format and `ignore_version` is False.

    See Also
    --------
    to_xarray
    read_zarr

    """
    skrf_type = ds.attrs.get('skrf_type')
    skrf_format = ds.attrs.get('skrf_format', FORMAT_VERSION)
    if skrf_format > FORMAT_VERSION:
        msg = (f'This Dataset is in format {skrf_format}, written by skrf {ds.attrs.get("skrf_version")}, '
               f'and this version of skrf reads format {FORMAT_VERSION}.')
        if not ignore_version:
            raise NotImplementedError(f'{msg} Upgrade skrf, or pass ignore_version=True to try anyway.')
        warn(f'{msg} Reading it anyway, the result may be wrong or incomplete.', stacklevel=2)
    readers = {
        'Frequency': _frequency,
        'Network': _network_from_xarray,
        'NetworkSet': _networkset_from_xarray,
        'Calibration': _calibration_from_xarray,
    }
    if skrf_type not in readers:
        raise NotImplementedError(f'Unknown skrf_type {skrf_type!r}.')
    return readers[skrf_type](ds)


def _networks(ds: xr.Dataset) -> list[Network]:
    obj = from_xarray(ds)
    if isinstance(obj, Network):
        return [obj]
    if isinstance(obj, NetworkSet):
        return obj.ntwk_set
    raise NotImplementedError('Only Datasets of a Network or a NetworkSet have network parameters.')


def _choose(names: tuple, choices: tuple, what: str) -> tuple:
    if len(names) == 1 and not isinstance(names[0], str):
        names = tuple(names[0])
    if names == ('all',):
        return choices
    unknown = [n for n in names if n not in choices]
    if unknown:
        raise ValueError(f'Unknown {what} {unknown}. Choose from {", ".join(choices)}, or \'all\'.')
    return names


def _with(ds: xr.Dataset, ntwks: list[Network], attrs: dict[str, dict]) -> xr.Dataset:
    """Add the Network properties named by `attrs` to `ds`, with the dimensions of ``s``."""
    dims = ds['s'].dims
    stack = np.stack if 'network' in dims else (lambda values: values[0])
    return ds.assign({name: (dims, stack([getattr(n, name) for n in ntwks]), a) for name, a in attrs.items()})


def add_params(ds: xr.Dataset, *params: str) -> xr.Dataset:
    """
    Add other network parameters, such as Z or Y, to a Network's Dataset.

    Each is computed by the :class:`~skrf.network.Network` property of the
    same name, from ``s``, ``z0`` and ``s_def``.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Dataset of a Network or a NetworkSet, made by :func:`to_xarray`.
    \\*params : str or list of str
        Parameters to add, from :data:`PARAMS`: ``'z'``, ``'y'``, ``'a'`` (ABCD),
        ``'g'``, ``'h'`` or ``'t'``, or ``'all'``. ``'a'``, ``'g'``, ``'h'`` and
        ``'t'`` only exist for two-ports, and ``'all'`` leaves them out for
        other Networks.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        A copy of `ds` with a variable for each parameter, named like the
        property and with the same dimensions as ``s``.

    See Also
    --------
    add_proj

    Examples
    --------
    >>> ds = add_params(to_xarray(rf.data.ring_slot), 'z', 'y')
    >>> ds['z'].sel(m=1, n=1)

    """
    return _add_params(ds, _networks(ds), params)


def add_proj(ds: xr.Dataset, *projections: str, params: str | list[str] | None = None) -> xr.Dataset:
    """
    Add projections, such as dB or phase, of a Network's parameters to its Dataset.

    Each is computed by the :class:`~skrf.network.Network` property of the
    same name, such as ``s_db`` or ``z_deg``.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Dataset of a Network or a NetworkSet, made by :func:`to_xarray`.
    \\*projections : str or list of str
        Projections to add, from :data:`PROJECTIONS`, such as ``'re'``, ``'im'``,
        ``'mag'``, ``'db'``, ``'deg'`` or ``'deg_unwrap'``, or ``'all'``.
    params : str or list of str, optional
        Parameters to project, such as ``'s'`` or ``['s', 'z']``. Default is
        every parameter in `ds`: ``s``, and those added by :func:`add_params`.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        A copy of `ds` with a variable for each parameter and projection,
        named like the property, such as ``s_db``, and with the same
        dimensions as ``s``.

    See Also
    --------
    add_params

    Examples
    --------
    >>> ds = add_proj(to_xarray(rf.data.ring_slot), 'db', 'deg')
    >>> ds['s_db'].sel(m=2, n=1).plot(x='f_scaled')

    All projections of S and Z

    >>> ds = add_proj(add_params(to_xarray(rf.data.ring_slot), 'z'), 'all')

    """
    return _add_proj(ds, _networks(ds), projections, params)


def _add_params(ds: xr.Dataset, ntwks: list[Network], params: tuple) -> xr.Dataset:
    params = _choose(params, PARAMS, 'params')
    if params == PARAMS and ntwks[0].nports != 2:
        params = tuple(p for p in params if p not in ('a', 'g', 'h', 't'))
    return _with(ds, ntwks, {p: {'long_name': f'{p.upper()}-parameters'} for p in params})


def _add_proj(ds: xr.Dataset, ntwks: list[Network], projections: tuple, params: str | list[str] | None) -> xr.Dataset:
    projections = _choose(projections, PROJECTIONS, 'projections')
    if params is None:
        params = [p for p in Network.PRIMARY_PROPERTIES if p in ds]
    elif isinstance(params, str):
        params = [params]
    _choose(tuple(params), Network.PRIMARY_PROPERTIES, 'params')
    return _with(ds, ntwks, {
        f'{p}_{proj}': {'long_name': f'{p.upper()} {Network.Y_LABEL_DICT[proj]}'}
        for p in params for proj in projections
    })


def write_zarr(store, obj, **kwargs) -> None:
    """
    Write an skrf object to a Zarr store.

    Parameters
    ----------
    store : str, :class:`pathlib.Path` or Zarr store
        Where to write, usually a directory path such as ``'dut.zarr'``.
        An existing store is overwritten.
    obj : skrf object or :class:`xarray.Dataset`
        Any object supported by :func:`to_xarray`, or a Dataset made by it,
        for example with variables added by :func:`add_params` or
        :func:`add_proj`.
    \\*\\*kwargs :
        Passed to :meth:`xarray.Dataset.to_zarr`.

    See Also
    --------
    read_zarr
    to_xarray

    Examples
    --------
    >>> write_zarr('ring_slot.zarr', rf.data.ring_slot)

    """
    xr = _xarray()
    kwargs.setdefault('mode', 'w')
    kwargs.setdefault('consolidated', False)
    ds = obj if isinstance(obj, xr.Dataset) else to_xarray(obj)
    ds.to_zarr(store, **kwargs)


def read_zarr(store, ignore_version: bool = False, **kwargs):
    """
    Read an skrf object from a Zarr store written by :func:`write_zarr`.

    Parameters
    ----------
    store : str, :class:`pathlib.Path` or Zarr store
        Where to read from.
    ignore_version : bool, optional
        Passed to :func:`from_xarray`. Default is False.
    \\*\\*kwargs :
        Passed to :func:`xarray.open_zarr`.

    Returns
    -------
    obj : skrf object
        The object that was written.

    See Also
    --------
    write_zarr
    from_xarray

    Examples
    --------
    >>> ntwk = read_zarr('ring_slot.zarr')

    """
    xr = _xarray()
    kwargs.setdefault('consolidated', False)
    with xr.open_zarr(store, **kwargs) as ds:
        return from_xarray(ds.load(), ignore_version=ignore_version)
