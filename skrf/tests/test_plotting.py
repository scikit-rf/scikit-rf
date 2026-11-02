import subprocess
import sys

try:
    import matplotlib as mpl
    import matplotlib.pyplot
except ImportError:
    pass

import numpy as np
import pytest

import skrf as rf
from skrf.networkSet import NetworkSet

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)


@pytest.fixture(params=rf.Network.PRIMARY_PROPERTIES)
def primary_properties(request):
    return request.param

@pytest.fixture(params=["polar", "complex"])
def primary_methods(request):
    return request.param

@pytest.fixture(params=rf.Network._generated_functions().keys())
def generated_functions(request):
    return request.param

def test_primary_plotting(ntwk1_dc: rf.Network, primary_properties: str, primary_methods: str):
    method = f"plot_{primary_properties}_{primary_methods}"
    fig = getattr(ntwk1_dc, method)()

def test_generated_function_plots(ntwk1_dc: rf.Network, generated_functions: str):
    method = f"plot_{generated_functions}"
    fig = getattr(ntwk1_dc, method)()

def test_plot_passivity(ntwk1_dc: rf.Network):
    ntwk1_dc.plot_passivity()

def test_plot_reciprocity(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_reciprocity()

def test_plot_reciprocity2(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_reciprocity2()

def test_plot_s_db_time(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_s_db_time()
    ntwk1_dc.plot_s_db_time(m=0, n=0)

def test_plot_s_error(ntwk1_dc: rf.Network):
    ntwk1_dc.plot_s_error(ntwk1_dc)
    ntwk1_dc.plot_s_error(ntwk1_dc, error_function='average_l1_norm')
    ntwk1_dc.plot_s_error(ntwk1_dc, db=True)
    rf.plotting.plot_s_error(ntwk1_dc, ntwk1_dc, error_function='maximum_l1_norm')

def test_plot_s_smith(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_s_smith()

def test_z_time_impulse_step_z0(ntwk1_dc: rf.Network, ):
    se_diff = rf.network.concat_ports([ntwk1_dc, ntwk1_dc], port_order = 'second')
    mm_diff = se_diff.copy()
    mm_diff.se2gmm(p = 2)
    fig, ax = mpl.pyplot.subplots(1, 1)
    mm_diff.plot_z_time_step(0, 0, ax = ax)
    mm_diff.plot_z_time_impulse(0, 1, ax = ax)
    mm_diff.plot_z_time_step(2, 2, ax = ax)
    mm_diff.plot_z_time_impulse(2, 3, ax = ax)
    # test that start impedance is almost port z0 to 0.1%
    z0_start = [
        ax.lines[0].get_ydata()[0],
        ax.lines[1].get_ydata()[0],
        ax.lines[2].get_ydata()[0],
        ax.lines[3].get_ydata()[0]]
    np.testing.assert_allclose(z0_start, mm_diff.z0[0],
        rtol = 1e-3,
        err_msg = "plot_z_time_xxx start impedance does not match port impedance")

def test_x_y_labels(ntwk1_dc: rf.Network):
    fig, ax = mpl.pyplot.subplots(1, 1)
    ntwk1_dc.plot_s_db(ax = ax)
    # check that there are 4 traces
    np.testing.assert_equal(len(ax.lines), ntwk1_dc.nports**2,
                                    err_msg = "missing trace")
    # check x label
    np.testing.assert_string_equal(ax.get_xlabel(), f"Frequency ({ntwk1_dc.frequency.unit})")
    # check y label
    np.testing.assert_string_equal(ax.get_ylabel(), rf.Network.Y_LABEL_DICT['db'])



@pytest.mark.parametrize("usetex", [True, False])
def test_plot_it_all(ntwk1_dc: rf.Network, usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1_dc.plot_it_all()

@pytest.mark.parametrize("usetex", [True, False])
def test_plot_polar(ntwk1_dc: rf.Network, usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1_dc.plot_s_polar()

def test_plot_uncertainty_bounds_s(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s()

def test_plot_uncertainty_bounds_s_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s_db()

def test_plot_uncertainty_bounds_s_time_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s_time_db()

def test_plot_uncertainty_decomposition(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_decomposition()

def test_plot_minmax_bounds_s_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_db()

def test_plot_minmax_bounds_s_db10(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_db10()

def test_plot_minmax_bounds_s_time_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_time_db()

def test_plot_signature(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.signature()

def test_generated_violin_plots(ntwk_set_zip: NetworkSet, generated_functions: str):
    if "time" not in generated_functions:
        fig = ntwk_set_zip.plot_violin(generated_functions)
    else:
        with pytest.raises(NotImplementedError):
            fig = ntwk_set_zip.plot_violin(generated_functions)


# Plotting backend selection (skrf.plotting.options.backend) and the plotly backend

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

needs_plotly = pytest.mark.skipif(go is None, reason="plotly not installed")


@pytest.fixture(autouse=True)
def reset_plotting_backend():
    if go is not None:  # other test modules call rf.stylely
        import plotly.io as pio

        pio.templates.default = "plotly"
    yield
    rf.plotting.options.backend = "matplotlib"

@pytest.fixture
def mpl_axes():
    fig, ax = mpl.pyplot.subplots(1, 1)
    yield ax
    mpl.pyplot.close("all")

def _modules_after(code: str) -> set:
    out = subprocess.run([sys.executable, "-c", f"{code}; import sys; print(' '.join(sys.modules))"],
                         capture_output=True, text=True, check=True)
    return {m.split(".")[0] for m in out.stdout.split()}

def assert_plotly_parity(fig, ax, x_scale: float = 1.0):
    """Plotly traces carry the same data, names and y label as the matplotlib lines on `ax`."""
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(ax.lines)
    for trace, line in zip(fig.data, ax.lines):
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float) * x_scale, line.get_xdata(), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), line.get_ydata(), rtol=1e-12)
        if not line.get_label().startswith("_"):
            assert trace.name == line.get_label()
    assert fig.layout.yaxis.title.text == (ax.get_ylabel() or None)

def test_backend_option():
    assert rf.plotting.options.backend == "matplotlib"
    rf.plotting.options.backend = "plotly"
    assert rf.plotting.options.backend == "plotly"
    with pytest.raises(ValueError, match="matplotlib.*plotly"):
        rf.plotting.options.backend = "bokeh"
    assert rf.plotting.options.backend == "plotly"

def test_plotly_missing_raises_import_error(ntwk1: rf.Network, monkeypatch):
    for mod in [m for m in sys.modules if m.split(".")[0] == "plotly"]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match=r"scikit-rf\[plotly\]"):
        ntwk1.plot_s_db(backend="plotly")

@needs_plotly
def test_plotly_not_implemented_does_not_fall_back(ntwk_set_zip: NetworkSet):
    rf.plotting.options.backend = "plotly"
    for call in (ntwk_set_zip.signature, ntwk_set_zip.plot_logsigma, ntwk_set_zip.plot_uncertainty_bounds_s_db,
                 lambda: ntwk_set_zip.plot_violin("s_db"), ntwk_set_zip[0].plot_it_all,
                 lambda: rf.plotting.subplot_params(ntwk_set_zip[0])):
        with pytest.raises(NotImplementedError, match="plotly"):
            call()
    rf.plotting.options.backend = "matplotlib"
    with pytest.raises(NotImplementedError, match="plotly"):
        ntwk_set_zip.plot_logsigma(backend="plotly")

def test_matplotlib_unchanged_under_global_plotly(ntwk1: rf.Network, ntwk_set_zip: NetworkSet, mpl_axes):
    rf.plotting.options.backend = "plotly"
    # per-call override keeps the matplotlib return values
    assert ntwk1.plot_s_db(backend="matplotlib") is None
    assert ntwk1.frequency.plot(ntwk1.s_db[:, 0, 0], backend="matplotlib") is None
    assert ntwk1.plot_passivity(backend="matplotlib") is None
    assert ntwk_set_zip.plot_s_db(backend="matplotlib") == [None] * len(ntwk_set_zip)
    # a matplotlib ax needs backend="matplotlib" when the global option is plotly
    assert ntwk1.plot_s_db(0, 0, ax=mpl_axes, backend="matplotlib") is None
    assert ntwk1.plot_s_smith(ax=mpl_axes, backend="matplotlib") is None
    assert ntwk1.plot_s_complex(0, 0, mpl_axes, backend="matplotlib") is None
    assert len(mpl_axes.lines) > 0
    # without it the ax conflicts with the global backend
    for call in (lambda: ntwk1.plot_s_db(ax=mpl_axes), lambda: ntwk1.plot_s_smith(ax=mpl_axes),
                 lambda: ntwk_set_zip.plot_s_db(ax=mpl_axes)):
        with pytest.raises(TypeError, match="cannot be used with backend='plotly'"):
            call()

@needs_plotly
def test_plotly_fig_argument(ntwk1: rf.Network, ntwk_set_zip, mpl_axes):
    fig = go.Figure()
    # on any path to the plotting code
    for call in (lambda **kw: ntwk1.plot_s_db(fig=fig, **kw), lambda **kw: ntwk1.plot_s_db(0, 0, None, fig=fig, **kw),
                 lambda **kw: ntwk1.plot_s_smith(fig=fig, **kw), lambda **kw: ntwk_set_zip.plot_s_db(fig=fig, **kw)):
        assert call(backend="plotly") is fig
        # the backend comes from the global option (matplotlib), never from fig
        with pytest.raises(TypeError, match="cannot be used with backend='matplotlib'"):
            call()
    # ax and fig take the figures of their own backend only
    with pytest.raises(TypeError, match="pass a plotly Figure as fig"):
        ntwk1.plot_s_db(ax=fig, backend="plotly")
    with pytest.raises(TypeError, match="pass a plotly Figure as fig"):
        ntwk1.plot_s_db(0, 0, fig)
    with pytest.raises(TypeError, match="pass a plotly Figure as fig"):
        ntwk_set_zip.plot_s_db(ax=fig)
    with pytest.raises(TypeError, match="pass matplotlib Axes as ax"):
        ntwk1.plot_s_db(fig=mpl_axes)
    with pytest.raises(TypeError, match="not both"):
        ntwk1.plot_s_db(ax=mpl_axes, fig=fig)

@needs_plotly
@pytest.mark.parametrize("keywords", [{"color": "r"}, {"c": "r"}, {"linestyle": "--"}, {"ls": "--"}, {"linewidth": 2},
                                      {"lw": 2}, {"marker": "o"}, {"markersize": 3}, {"ms": 3}, {"alpha": 0.5}])
def test_plotly_rejects_matplotlib_keywords(ntwk1: rf.Network, ntwk_set_zip, keywords: dict):
    for call in (ntwk1.plot_s_db, ntwk1.plot_s_smith, ntwk1.plot_s_polar, ntwk_set_zip.plot_s_db):
        with pytest.raises(TypeError, match="Matplotlib keywords"):
            call(backend="plotly", **keywords)
    rf.plotting.options.backend = "plotly"
    with pytest.raises(TypeError, match="Matplotlib keywords"):
        ntwk1.plot_s_db(**keywords)

@needs_plotly
def test_plotly_keywords(ntwk1: rf.Network):
    trace = ntwk1.plot_s_db(0, 0, line=dict(width=3, dash="dash", color="red"), backend="plotly").data[0]
    assert (trace.line.width, trace.line.dash, trace.line.color) == (3, "dash", "red")
    assert ntwk1.plot_s_db(0, 0, name="plotly name", backend="plotly").data[0].name == "plotly name"

def test_matplotlib_rejects_plotly_keywords(ntwk1: rf.Network, mpl_axes):
    for keywords in ({"name": "a"}, {"line": {"width": 3}}):
        with pytest.raises(AttributeError):
            ntwk1.plot_s_db(0, 0, ax=mpl_axes, **keywords)

@pytest.mark.parametrize("backend", ["matplotlib", pytest.param("plotly", marks=needs_plotly)])
def test_label_works_with_both_backends(ntwk1: rf.Network, mpl_axes, backend: str):
    out = ntwk1.plot_s_db(0, 0, label="both", backend=backend, **({"ax": mpl_axes} if backend == "matplotlib" else {}))
    assert (mpl_axes.lines[0].get_label() if backend == "matplotlib" else out.data[0].name) == "both"

def test_plot_prop_keyword_args(ntwk1: rf.Network, mpl_axes):
    ntwk1.plot_prop_complex(prop_name="s", m=0, n=0, ax=mpl_axes)
    assert len(mpl_axes.lines) == 1

@needs_plotly
def test_plotly_does_not_import_matplotlib():
    loaded = _modules_after(
        "import skrf as rf; rf.plotting.options.backend = 'plotly'; n = rf.data.ring_slot; "
        "n.plot_s_db(); n.plot_s_time_db(); n.frequency.plot(n.s_db); n.plot_passivity(); "
        "n.plot_s_smith(); n.plot_s_polar(); n.plot_s_complex(); "
        "ns = rf.networkSet.NetworkSet([n, n]); ns.plot_s_db(); s = n.s[:, 0, 0]; "
        "rf.plotting.plot_rectangular(n.f, n.s_db[:, 0, 0]); rf.plotting.plot_complex_rectangular(s); "
        "rf.plotting.plot_complex_polar(s); rf.plotting.plot_smith(s); n.frequency.labelXAxis()")
    assert "plotly" in loaded
    assert "matplotlib" not in loaded

@needs_plotly
def test_plotly_generated_function_parity(ntwk1_dc: rf.Network, generated_functions: str, mpl_axes):
    method = f"plot_{generated_functions}"
    fig = getattr(ntwk1_dc, method)(backend="plotly")
    getattr(ntwk1_dc, method)(ax=mpl_axes)
    is_freq = "time" not in generated_functions
    assert len(fig.data) == ntwk1_dc.nports**2
    assert_plotly_parity(fig, mpl_axes, x_scale=ntwk1_dc.frequency.multiplier if is_freq else 1.0)
    assert fig.layout.xaxis.title.text == (f"Frequency ({ntwk1_dc.frequency.unit})" if is_freq else "Time (ns)")

@needs_plotly
@pytest.mark.parametrize("call", [
    lambda n, ns, **kw: n.plot_passivity(**kw),
    lambda n, ns, **kw: n.plot_reciprocity2(**kw),
    lambda n, ns, **kw: n.plot_s_error(n, **kw),
    lambda n, ns, **kw: n.frequency.plot(n.s_db, **kw),
    lambda n, ns, **kw: ns.plot_s_db(**kw),
], ids=["plot_passivity", "plot_reciprocity2", "plot_s_error", "Frequency.plot", "NetworkSet.plot_s_db"])
def test_plotly_rectangular_parity(ntwk1_dc: rf.Network, ntwk_set_zip: NetworkSet, call, mpl_axes):
    fig = call(ntwk1_dc, ntwk_set_zip, backend="plotly")
    call(ntwk1_dc, ntwk_set_zip, ax=mpl_axes)
    assert_plotly_parity(fig, mpl_axes, x_scale=ntwk1_dc.frequency.multiplier)

@needs_plotly
def test_plotly_figure_overlay_and_units(ntwk1: rf.Network):
    fig = go.Figure()
    assert ntwk1.plot_s_db(0, 0, fig=fig, backend="plotly") is fig
    mhz = ntwk1.copy()
    mhz.frequency.unit = "MHz"
    assert mhz.plot_s_db(0, 0, fig=fig, backend="plotly") is fig
    assert len(fig.data) == 2
    np.testing.assert_allclose(fig.data[1].x, fig.data[0].x)  # rescaled to the figure's first unit
    assert fig.layout.xaxis.title.text == f"Frequency ({ntwk1.frequency.unit})"

@needs_plotly
def test_plotly_logx(ntwk1: rf.Network, mpl_axes):
    assert ntwk1.plot_s_db(0, 0, logx=True, backend="plotly").layout.xaxis.type == "log"
    ntwk1.plot_s_db(0, 0, logx=True, ax=mpl_axes)
    assert mpl_axes.get_xscale() == "log"

@needs_plotly
def test_plotly_errors(ntwk1: rf.Network, mpl_axes):
    with pytest.raises(TypeError, match="Positional"):
        ntwk1.frequency.plot(ntwk1.s_db[:, 0, 0], "r--", backend="plotly")
    with pytest.raises(TypeError, match="cannot be used with backend"):
        ntwk1.plot_s_db(ax=mpl_axes, backend="plotly")

@needs_plotly
def test_plotly_kwargs_translation(ntwk1: rf.Network):
    # the keywords of skrf's own plots, not available to users
    fig = rf.plotting._get_plotly_figure()
    rf.plotting._plotly_rect(ntwk1.f, ntwk1.s_db[:, 0, 0], fig=fig, color="r", linestyle="--", marker="o", label="a")
    trace = fig.data[0]
    assert (trace.line.color, trace.line.dash) == ("red", "dash")
    assert (trace.marker.symbol, trace.mode) == ("circle", "lines+markers")
    assert trace.name == "a"
    trace = ntwk1.plot_s_db(0, 0, marker=dict(size=5), backend="plotly").data[0]
    assert (trace.marker.size, trace.mode) == (5, "lines+markers")
    trace = ntwk1.plot_s_db(0, 0, line=go.scatter.Line(color="red"), marker=go.scatter.Marker(symbol="x"),
                            backend="plotly").data[0]
    assert (trace.line.color, trace.marker.symbol, trace.mode) == ("red", "x", "lines+markers")

@needs_plotly
def test_plotly_rectangular_complex_x():
    with pytest.warns(np.exceptions.ComplexWarning):
        fig = rf.plotting.plot_rectangular(np.array([1 + 1j, 2 + 1j]), np.array([1, 2]), backend="plotly")
    assert list(fig.data[0].x) == [1, 2]
    fig.to_json()

def _smith_grid_traces(fig) -> list:
    return [t for t in fig.data if t.meta == "skrf_smith_chart"]

@needs_plotly
def test_plotly_complex_parity(ntwk1_dc: rf.Network, primary_properties: str, mpl_axes):
    method = f"plot_{primary_properties}_complex"
    fig = getattr(ntwk1_dc, method)(backend="plotly")
    getattr(ntwk1_dc, method)(ax=mpl_axes)
    assert len(fig.data) == ntwk1_dc.nports**2
    assert_plotly_parity(fig, mpl_axes)
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.scaleanchor) == ("Real", "x")
    assert (fig.layout.xaxis.constrain, fig.layout.yaxis.constrain) == ("domain", "domain")
    assert (fig.layout.width, fig.layout.height) == (700, 500)
    z = getattr(ntwk1_dc, primary_properties)
    np.testing.assert_allclose(fig.layout.yaxis.range, [z.imag.min() - 0.05 * np.ptp(z.imag),
                                                        z.imag.max() + 0.05 * np.ptp(z.imag)])

@needs_plotly
def test_plotly_polar(ntwk1_dc: rf.Network, primary_properties: str):
    fig = getattr(ntwk1_dc, f"plot_{primary_properties}_polar")(backend="plotly")
    z = getattr(ntwk1_dc, primary_properties)
    ports = [(m, n) for m in range(ntwk1_dc.nports) for n in range(ntwk1_dc.nports)]
    assert len(fig.data) == len(ports)
    for trace, (m, n) in zip(fig.data, ports):
        assert (trace.type, trace.thetaunit) == ("scatterpolar", "radians")
        np.testing.assert_allclose(trace.r, np.abs(z[:, m, n]))
        np.testing.assert_allclose(trace.theta, np.angle(z[:, m, n]))
        assert trace.name == rf.plotting._get_label_str(ntwk1_dc, primary_properties.upper(), m, n, usetex=False)

@needs_plotly
@pytest.mark.parametrize("chart_type", ["z", "y", "zy", "yz"])
def test_plotly_smith(ntwk1_dc: rf.Network, chart_type: str):
    r = 0.5
    fig = ntwk1_dc.plot_s_smith(r=r, chart_type=chart_type, label_axes=True, backend="plotly")
    grid, data = _smith_grid_traces(fig), [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert len(data) == ntwk1_dc.nports**2
    np.testing.assert_allclose(data[1].x, ntwk1_dc.s[:, 0, 1].real)
    np.testing.assert_allclose(data[1].y, ntwk1_dc.s[:, 0, 1].imag)
    grid_points = np.concatenate([np.asarray(t.x, dtype=float) + 1j * np.asarray(t.y, dtype=float) for t in grid])
    assert np.nanmax(np.abs(grid_points)) <= r * (1 + 1e-9)
    assert ("whitesmoke" in [t.line.color for t in grid]) == (chart_type in ("zy", "yz"))
    np.testing.assert_allclose(fig.layout.xaxis.range, [-1.1 * r, 1.1 * r])
    assert (fig.layout.xaxis.constrain, fig.layout.yaxis.constrain) == ("domain", "domain")
    assert (fig.layout.width, fig.layout.height) == (700, 550)
    assert ntwk1_dc.plot_s_smith(fig=go.Figure(layout_height=300), backend="plotly").layout.width is None
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("Real", "Imaginary")

@needs_plotly
def test_plotly_smith_labels(ntwk1_dc: rf.Network):
    fig = ntwk1_dc.plot_s_smith(draw_labels=True, draw_vswr=[2, 5], backend="plotly")
    texts = [a.text for a in fig.layout.annotations]
    assert {"∞", "0.0", "1.0", "1.0j", "2", "5"} <= set(texts)
    assert ntwk1_dc.plot_s_smith(draw_labels=True, chart_type="y", backend="plotly").layout.annotations[0].text \
        == "5.0"  # admittance labels, as matplotlib

@needs_plotly
def test_plotly_smith_grid(mpl_axes):
    fig = rf.plotting.smith(draw_labels=True, ref_imm=50.0, backend="plotly")
    rf.plotting.smith(draw_labels=True, ref_imm=50.0, ax=mpl_axes)
    assert len(_smith_grid_traces(fig)) > 0 and not [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert [a.text for a in fig.layout.annotations] == [t.get_text().replace(r"$\infty$", "∞") for t in mpl_axes.texts]

@needs_plotly
@pytest.mark.parametrize("kind", ["complex", "polar", "smith"])
def test_plotly_geometric_overlay(ntwk1: rf.Network, kind: str):
    fig = getattr(ntwk1, f"plot_s_{kind}")(0, 0, backend="plotly")
    n_traces, n_grid = len(fig.data), len(_smith_grid_traces(fig))
    assert getattr(ntwk1, f"plot_s_{kind}")(1, 1, fig=fig, backend="plotly") is fig
    assert len(fig.data) == n_traces + 1
    assert len(_smith_grid_traces(fig)) == n_grid  # the smith chart is drawn once

@needs_plotly
def test_plotly_networkset_smith(ntwk1: rf.Network, ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_s_smith(backend="plotly")
    assert len(fig.data) - len(_smith_grid_traces(fig)) == sum(n.nports**2 for n in ntwk_set_zip)
    assert len(_smith_grid_traces(fig)) == len(_smith_grid_traces(ntwk1.plot_s_smith(backend="plotly")))

@needs_plotly
@pytest.mark.parametrize("func", ["plot_rectangular", "plot_complex_rectangular", "plot_smith"])
def test_plotly_public_plot_functions(ntwk1: rf.Network, func: str, mpl_axes):
    args = (ntwk1.f, ntwk1.s_db[:, 0, 0]) if func == "plot_rectangular" else (ntwk1.s[:, 0, 0],)
    fig = getattr(rf.plotting, func)(*args, label="a", backend="plotly")
    getattr(rf.plotting, func)(*args, label="a", ax=mpl_axes)
    (trace,) = [t for t in fig.data if t.meta != "skrf_smith_chart"]
    np.testing.assert_allclose(trace.x, mpl_axes.lines[-1].get_xdata())
    np.testing.assert_allclose(trace.y, mpl_axes.lines[-1].get_ydata())
    assert trace.name == "a"
    n_traces = len(fig.data)
    assert getattr(rf.plotting, func)(*args, fig=fig, backend="plotly") is fig
    assert len(fig.data) == n_traces + 1  # the smith chart is drawn once

@needs_plotly
def test_plotly_plot_polar():
    fig = rf.plotting.plot_complex_polar(np.array([1j, -2]), y_label="r", title="t", backend="plotly")
    (trace,) = fig.data
    assert (trace.type, trace.thetaunit) == ("scatterpolar", "radians")
    np.testing.assert_allclose(trace.r, [1, 2])
    np.testing.assert_allclose(trace.theta, [np.pi / 2, np.pi])
    assert (fig.layout.polar.radialaxis.title.text, fig.layout.title.text) == ("r", "t")

@needs_plotly
def test_plotly_title_centered(ntwk1: rf.Network):
    # centered over the plot area like matplotlib, on single plots and smith charts
    for fig in (rf.plotting.plot_rectangular(ntwk1.f, ntwk1.s_db[:, 0, 0], title="t", backend="plotly"),
                rf.plotting.plot_smith(ntwk1.s[:, 0, 0], title="t", backend="plotly")):
        assert (fig.layout.title.x, fig.layout.title.xanchor, fig.layout.title.xref) == (0.5, "center", "paper")

@needs_plotly
def test_label_x_axis(ntwk1: rf.Network, mpl_axes):
    other = mpl_axes.figure.add_subplot(2, 1, 2)  # now the current axes, not the one passed in
    ntwk1.frequency.labelXAxis(mpl_axes)
    assert mpl_axes.get_xlabel() == f"Frequency ({ntwk1.frequency.unit})" and other.get_xlabel() == ""
    fig = go.Figure()
    assert ntwk1.frequency.labelXAxis(fig=fig, backend="plotly") is fig
    assert fig.layout.xaxis.title.text == f"Frequency ({ntwk1.frequency.unit})"
    # the matplotlib plots of a Network keep labelling their own axes under a global plotly backend
    rf.plotting.options.backend = "plotly"
    ntwk1.frequency.plot(ntwk1.s_db[:, 0, 0], ax=other, backend="matplotlib")
    assert other.get_xlabel() == f"Frequency ({ntwk1.frequency.unit})"

@needs_plotly
def test_plotly_frequency_plot_complex(ntwk1: rf.Network):
    with pytest.warns(match="imaginary part"):
        fig = ntwk1.frequency.plot(ntwk1.s[:, 0, 0], backend="plotly")
    np.testing.assert_array_equal(fig.data[0].y, ntwk1.s[:, 0, 0].real)
    fig.to_json()
    with pytest.warns(match="imaginary part"):  # also for a list
        fig = rf.plotting.plot_rectangular([0, 1], [1 + 1j, 2], backend="plotly")
    np.testing.assert_array_equal(fig.data[0].y, [1, 2])

@pytest.fixture
def match() -> rf.Network:
    freq = rf.Frequency(1, 2, 5, "GHz")
    return rf.media.DefinedGammaZ0(freq, z0=50, gamma=1j).match(name="match")

def test_networkset_creation_adds_methods_once(ntwk1: rf.Network, monkeypatch):
    NetworkSet([ntwk1])
    calls = []
    monkeypatch.setattr(rf.plotting, "_validate_plot_args", lambda func: calls.append(func) or func)
    NetworkSet([ntwk1])
    assert not calls

def test_networkset_non_plot_method_not_wrapped():
    assert hasattr(NetworkSet.plot_s_db, "__wrapped__") and not hasattr(NetworkSet.write_touchstone, "__wrapped__")

@needs_plotly
@pytest.mark.parametrize("label", [None, "_nolegend_"])
def test_plotly_label_without_legend_entry(ntwk1: rf.Network, label):
    fig = ntwk1.plot_s_db(0, 0, label=label, backend="plotly")
    assert fig.data[0].name is None and not fig.data[0].showlegend

def test_frequency_plot_uses_ax(ntwk1: rf.Network, mpl_axes):
    other = mpl_axes.figure.add_subplot(2, 1, 2)  # now the current axes, not the one passed in
    y = np.ones((len(ntwk1), 2, 2))
    ntwk1.frequency.plot(y, ax=mpl_axes, backend="matplotlib")
    assert (len(mpl_axes.lines), len(other.lines)) == (4, 0)

def test_frequency_plot_keeps_errors_of_the_matrix_loop(ntwk1: rf.Network, mpl_axes):
    with pytest.raises(AttributeError, match="not_a_keyword"):
        ntwk1.frequency.plot(np.ones((len(ntwk1), 2, 2)), ax=mpl_axes, backend="matplotlib", not_a_keyword=1)
    mpl_axes.plot = lambda *a, **k: (_ for _ in ()).throw(TypeError("from plot"))
    with pytest.raises(TypeError, match="from plot"):
        ntwk1.frequency.plot(np.ones((len(ntwk1), 2, 2)), ax=mpl_axes, backend="matplotlib")

@pytest.fixture
def three_port() -> rf.Network:
    freq = rf.Frequency(1, 2, 5, "GHz")
    return rf.Network(frequency=freq, s=np.ones((5, 3, 3)) * 0.1, z0=50)

@pytest.mark.parametrize("method", ["plot_reciprocity", "plot_reciprocity2"])
@pytest.mark.parametrize("backend", ["matplotlib", pytest.param("plotly", marks=needs_plotly)])
def test_plot_reciprocity_labels(three_port: rf.Network, mpl_axes, method: str, backend: str):
    fig = getattr(rf.plotting, method)(three_port, backend=backend)
    labels = [t.name for t in fig.data] if backend == "plotly" else [line.get_label() for line in mpl_axes.lines]
    assert labels == ["ports 21", "ports 31", "ports 32"]

@needs_plotly
def test_plotly_all_matplotlib_markers(ntwk1: rf.Network):
    markers = [m for m in mpl.lines.Line2D.markers if isinstance(m, str) and len(m) == 1 and m != " "]
    assert set(markers) <= rf.plotting._PLOTLY_MARKERS.keys()
    for marker in markers:
        fig = ntwk1.plot_s_db(0, 0, marker=dict(symbol=rf.plotting._PLOTLY_MARKERS[marker]), backend="plotly")
        assert fig.data[0].marker.symbol == rf.plotting._PLOTLY_MARKERS[marker]

@needs_plotly
def test_plotly_network_set_positional_ax(ntwk_set_zip: NetworkSet):
    n = len(ntwk_set_zip)
    assert len(ntwk_set_zip.plot_s_db(0, 0, None, False, backend="plotly").data) == n
    assert len([t for t in ntwk_set_zip.plot_s_smith(0, 0, 1, None, backend="plotly").data if t.name]) == n




@needs_plotly
def test_plotly_fig_on_network_set_plot(ntwk_set_zip: NetworkSet):
    fig = go.Figure()
    assert ntwk_set_zip.plot_s_db(0, 0, fig=fig, backend="plotly") is fig
    assert len(fig.data) == len(ntwk_set_zip)


@needs_plotly
def test_plot_functions_take_fig_as_fig():
    received = {}

    @rf.plotting._validate_plot_args
    def plot(ax=None, *, fig=None):
        received.update(ax=ax, fig=fig)

    fig = go.Figure()
    plot(fig=fig)
    assert received["ax"] is None
    assert received["fig"] is fig
    with pytest.raises(TypeError, match="pass a plotly Figure as fig"):
        plot(ax=fig)
