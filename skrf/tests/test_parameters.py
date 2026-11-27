import copy
import pickle

import numpy as np
import pytest

import skrf as rf
from skrf.parameters import NetworkParameter, Projection

# the components that have values, time_impulse and time_step are only plots
CONVERSIONS = [k for k, v in rf.Network.COMPONENT_FUNC_DICT.items() if v is not None]


@pytest.fixture(params=rf.Network.PRIMARY_PROPERTIES)
def prop_name(request):
    return request.param


@pytest.fixture(params=CONVERSIONS)
def conversion(request):
    return request.param


@pytest.fixture()
def mpl_ax():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_parameter_is_an_ndarray(ntwk1: rf.Network, prop_name: str):
    param = getattr(ntwk1, prop_name)
    assert isinstance(param, NetworkParameter)
    assert isinstance(param, np.ndarray)
    assert param.shape == (len(ntwk1), 2, 2)
    assert param.dtype == complex


def test_s_is_a_view_of_the_stored_matrix(ntwk1: rf.Network):
    assert np.shares_memory(ntwk1.s, ntwk1._s)
    ntwk1.s[:, 0, 0] = 0.5
    np.testing.assert_array_equal(ntwk1._s[:, 0, 0], 0.5)


def test_assigning_back_stores_a_plain_array(ntwk1: rf.Network):
    ntwk1.s = ntwk1.s * 2
    assert type(ntwk1._s) is np.ndarray
    ntwk1.s *= 2
    assert type(ntwk1._s) is np.ndarray
    ntwk1.z = ntwk1.z
    assert type(ntwk1._s) is np.ndarray


@pytest.mark.parametrize("derive", [
    lambda p: p[:, 0, 0],
    lambda p: p[0],
    lambda p: p * 2,
    lambda p: np.abs(p),
    lambda p: p @ p,
    lambda p: np.linalg.inv(p),
    lambda p: p.conj(),
    lambda p: p.copy(),
    lambda p: copy.copy(p),
    lambda p: copy.deepcopy(p),
    lambda p: pickle.loads(pickle.dumps(p)),
    lambda p: p.db[:, 1, 0],
    lambda p: p.db + 1,
], ids=["slice", "index", "mul", "abs", "matmul", "inv", "conj", "copy", "copy.copy", "deepcopy", "pickle",
        "projection slice", "projection add"])
def test_derived_arrays_are_plain(ntwk1: rf.Network, derive):
    assert type(derive(ntwk1.s)) is np.ndarray


def test_reductions_are_plain_scalars(ntwk1: rf.Network):
    assert type(ntwk1.s.sum()) is np.complex128
    assert type(ntwk1.s.db.max()) is np.float64


def test_repr_is_the_arrays(ntwk1: rf.Network):
    assert repr(ntwk1.s) == repr(ntwk1._s)
    assert repr(ntwk1.s.db) == repr(ntwk1.s_db)


def test_projection_matches_flat_attribute(ntwk1: rf.Network, prop_name: str, conversion: str):
    projection = getattr(getattr(ntwk1, prop_name), conversion)
    assert isinstance(projection, Projection)
    np.testing.assert_array_equal(projection, getattr(ntwk1, f"{prop_name}_{conversion}"))


def test_flat_attributes_stay_plain(ntwk1: rf.Network, conversion: str):
    assert type(getattr(ntwk1, f"s_{conversion}")) is np.ndarray


@pytest.mark.parametrize("method, flat", [
    ("plot", "plot_s_smith"),
    ("plot_smith", "plot_s_smith"),
    ("plot_complex", "plot_s_complex"),
    ("plot_polar", "plot_s_polar"),
])
def test_s_plots_like_the_flat_method(ntwk1: rf.Network, method: str, flat: str):
    plt = pytest.importorskip("matplotlib.pyplot")
    try:
        # each on a new figure, the polar plots make their own polar Axes
        plt.figure()
        getattr(ntwk1.s, method)()
        ax = plt.gca()
        plt.figure()
        getattr(ntwk1, flat)()
        expected = plt.gca()
    finally:
        plt.close("all")
    assert ax.name == expected.name
    assert len(ax.lines) == len(expected.lines) > 0
    for line, expected_line in zip(ax.lines, expected.lines):
        np.testing.assert_array_equal(line.get_xydata(), expected_line.get_xydata())
        assert line.get_label() == expected_line.get_label()


def test_other_parameters_plot_in_the_complex_plane(ntwk1: rf.Network, mpl_ax):
    ntwk1.z.plot(m=1, n=0, ax=mpl_ax)
    assert len(mpl_ax.lines) == 1
    np.testing.assert_array_equal(mpl_ax.lines[0].get_xdata(), ntwk1.z_re[:, 1, 0])


def test_projection_plots_like_the_flat_method(ntwk1: rf.Network, mpl_ax, conversion: str):
    getattr(ntwk1.s, conversion).plot(m=1, n=0, ax=mpl_ax)
    expected = pytest.importorskip("matplotlib.pyplot").figure().add_subplot()
    getattr(ntwk1, f"plot_s_{conversion}")(m=1, n=0, ax=expected)
    assert len(mpl_ax.lines) == len(expected.lines) == 1
    np.testing.assert_array_equal(mpl_ax.lines[0].get_xydata(), expected.lines[0].get_xydata())
    assert mpl_ax.get_ylabel() == expected.get_ylabel()


def test_only_s_has_a_smith_chart(ntwk1: rf.Network):
    with pytest.raises(ValueError, match="Smith chart"):
        ntwk1.z.plot_smith()


def test_detached_arrays_cannot_plot(ntwk1: rf.Network):
    detached = ntwk1.s.T
    assert detached._network is None
    with pytest.raises(ValueError, match="not attached"):
        detached.plot()
    with pytest.raises(ValueError, match="not attached"):
        detached.db.plot()
