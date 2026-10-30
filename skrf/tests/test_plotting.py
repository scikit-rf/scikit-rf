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

def test_plot_uncertainty_decomposition_title(ntwk1: rf.Network):
    NetworkSet([ntwk1, ntwk1], name="set").plot_uncertainty_decomposition()
    assert mpl.pyplot.gca().get_title() == "Uncertainty Decomposition: set $S_{11}$"

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
    with pytest.raises(NotImplementedError, match="plotly"):
        ntwk_set_zip.plot_violin("s_db", quantiles=[0.5])

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
    assert ntwk_set_zip.plot_uncertainty_bounds_s_db(ax=mpl_axes, backend="matplotlib") is None
    assert ntwk_set_zip.plot_violin("s_db", ax=mpl_axes, backend="matplotlib") is None
    assert len(mpl_axes.lines) > 0
    n_lines = len(mpl_axes.lines)
    assert rf.media.Freespace(ntwk1.frequency).plot_ep_mu(ax=mpl_axes, backend="matplotlib") is None
    assert len(mpl_axes.lines) == n_lines + 4
    # without it the ax conflicts with the global backend
    for call in (lambda: ntwk1.plot_s_db(ax=mpl_axes), lambda: ntwk1.plot_s_smith(ax=mpl_axes),
                 lambda: ntwk1.plot_it_all(ax=mpl_axes), lambda: ntwk_set_zip.plot_s_db(ax=mpl_axes),
                 lambda: ntwk_set_zip.plot_uncertainty_bounds_s_db(ax=mpl_axes)):
        with pytest.raises(TypeError, match="cannot be used with backend='plotly'"):
            call()

@needs_plotly
def test_plotly_fig_argument(ntwk1: rf.Network, ntwk_set_zip, mpl_axes):
    fig = go.Figure()
    # on any path to the plotting code
    for call in (lambda **kw: ntwk1.plot_s_db(fig=fig, **kw), lambda **kw: ntwk1.plot_s_db(0, 0, None, fig=fig, **kw),
                 lambda **kw: ntwk1.plot_s_smith(fig=fig, **kw), lambda **kw: ntwk_set_zip.plot_s_db(fig=fig, **kw),
                 lambda **kw: ntwk_set_zip.plot_uncertainty_bounds_s_db(fig=fig, **kw)):
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
        "n.plot_s_smith(); n.plot_s_polar(); n.plot_s_complex(); n.plot_it_all(); "
        "ns = rf.networkSet.NetworkSet([n, n]); ns.plot_uncertainty_bounds_s_db(); ns.plot_minmax_bounds_s_db(); "
        "ns.plot_violin('s_db'); ns.plot_uncertainty_decomposition(); s = n.s[:, 0, 0]; "
        "rf.plotting.plot_rectangular(n.f, n.s_db[:, 0, 0]); rf.plotting.plot_complex_rectangular(s); "
        "rf.plotting.plot_complex_polar(s); rf.plotting.plot_smith(s); rf.plotting.subplot_params(n); "
        "rf.media.Freespace(n.frequency).plot_ep_mu(); ns.signature(); ns.animate('s_db'); "
        "rf.plotting.shade_bands([1e9, 2e9], fig=n.plot_s_db()); rf.plotting.plot_vector(1j); "
        "rf.plotting.colors(); rf.plotting.scale_frequency_ticks(fig=n.plot_s_db(), funit='MHz'); "
        "n.frequency.labelXAxis()")
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
    lambda n, ns, **kw: ns.plot_logsigma(**kw),
    lambda n, ns, **kw: ns.plot_uncertainty_decomposition(**kw),
], ids=["plot_passivity", "plot_reciprocity2", "plot_s_error", "Frequency.plot", "NetworkSet.plot_s_db",
        "plot_logsigma", "plot_uncertainty_decomposition"])
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
    fig = rf.plotting._plotly_figure()
    rf.plotting._plotly_rect(ntwk1.f, ntwk1.s_db[:, 0, 0], ax=fig, c="r", ls="--", lw=2, marker="o", ms=4, alpha=0.5,
                             label="a")
    trace = fig.data[0]
    assert (trace.line.color, trace.line.dash, trace.line.width) == ("red", "dash", 2)
    assert (trace.marker.symbol, trace.marker.size, trace.mode) == ("circle", 4, "lines+markers")
    assert (trace.opacity, trace.name) == (0.5, "a")
    trace = ntwk1.plot_s_db(0, 0, marker=dict(size=5), backend="plotly").data[0]
    assert (trace.marker.size, trace.mode) == (5, "lines+markers")

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
@pytest.mark.parametrize("method", ["plot_uncertainty_bounds_s_db", "plot_minmax_bounds_s_db",
                                    "plot_uncertainty_bounds_s"])
def test_plotly_bounds_parity(ntwk_set_zip: NetworkSet, method: str, mpl_axes):
    fig = getattr(ntwk_set_zip, method)(backend="plotly")
    getattr(ntwk_set_zip, method)(ax=mpl_axes)
    means, fills = fig.data[::2], fig.data[1::2]
    assert len(means) == len(mpl_axes.lines) == len(mpl_axes.collections)
    for mean, fill, line, poly in zip(means, fills, mpl_axes.lines, mpl_axes.collections):
        np.testing.assert_allclose(np.asarray(mean.x) * ntwk_set_zip[0].frequency.multiplier, line.get_xdata())
        np.testing.assert_allclose(mean.y, line.get_ydata())
        assert mean.name == line.get_label()
        np.testing.assert_allclose(np.unique(fill.y), np.unique(poly.get_paths()[0].vertices[:, 1]))
        assert (fill.fill, fill.fillcolor, fill.showlegend) == ("toself", mean.line.color, False)
    assert fig.layout.yaxis.title.text == mpl_axes.get_ylabel()
    assert getattr(ntwk_set_zip, method)(fig=fig, backend="plotly") is fig
    assert len(fig.data) == 4 * len(means)  # mean and fill traces added again

@needs_plotly
def test_plotly_violin(ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_violin("s_db", m=0, n=1, backend="plotly")
    freq = ntwk_set_zip[0].frequency
    (trace,) = fig.data
    assert trace.type == "violin"
    np.testing.assert_allclose(trace.y, np.array([ntwk.s_db[:, 0, 1] for ntwk in ntwk_set_zip]).ravel())
    np.testing.assert_allclose(trace.x, np.tile(freq.f_scaled, len(ntwk_set_zip)))
    np.testing.assert_allclose(trace.width, 0.75 * (freq.f_scaled[1] - freq.f_scaled[0]))
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == (f"Frequency ({freq.unit})",
                                                                          rf.Network.Y_LABEL_DICT["db"])

@needs_plotly
def test_plotly_plot_it_all(ntwk1_dc: rf.Network):
    fig = ntwk1_dc.plot_it_all(backend="plotly")
    data = [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert len(data) == 4 * ntwk1_dc.nports**2
    assert {t.xaxis for t in data} == {"x", "x2", "x3", "x4"}
    assert (fig.layout.yaxis3.scaleanchor, fig.layout.yaxis4.scaleanchor) == ("x3", "x4")
    # a legend of its own in each panel, as matplotlib
    for k, xaxis in enumerate(["x", "x2", "x3", "x4"]):
        traces = [t for t in data if t.xaxis == xaxis]
        assert {t.legend for t in traces} == {f"legend{k + 1 if k else ''}"} and all(t.showlegend for t in traces)
        x0, x1 = fig.layout[xaxis.replace("x", "xaxis")].domain
        assert x0 <= fig.layout[f"legend{k + 1 if k else ''}"].x <= x1

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
def test_plotly_subplot_params(ntwk1: rf.Network):
    fig = rf.plotting.subplot_params(ntwk1, backend="plotly")
    assert len(fig.data) == ntwk1.nports**2
    np.testing.assert_allclose(fig.data[1].y, ntwk1.s_db[:, 1, 0])  # in port_tuples order, as matplotlib
    assert {t.xaxis for t in fig.data} == {"x", "x2", "x3", "x4"}
    assert [a.text for a in fig.layout.annotations] == ["S11", "S21", "S12", "S22"]
    assert all(t.showlegend for t in fig.data)
    assert [t.legend for t in fig.data] == ["legend", "legend2", "legend3", "legend4"]  # one per panel
    assert (fig.layout.width, fig.layout.height) == (800, 800)

def test_shade_bands(ntwk1: rf.Network, mpl_axes):
    ntwk1.plot_s_db(0, 0, ax=mpl_axes)
    rf.plotting.shade_bands(ntwk1.f[[0, 40, 80]], alpha=0.2)
    assert len(mpl_axes.collections) == 2
    other = mpl_axes.figure.add_subplot(2, 1, 2)  # now the current axes, not the one passed in
    rf.plotting.shade_bands(ntwk1.f[[0, 40, 80]], ax=mpl_axes)
    assert len(mpl_axes.collections) == 4 and len(other.collections) == 0

def test_plot_vector(mpl_axes):
    quiver = rf.plotting.plot_vector(1 + 2j, 0.5j, ax=mpl_axes, color="r")
    assert quiver in mpl_axes.collections
    np.testing.assert_allclose([quiver.U[0], quiver.V[0]], [1, 2])

@needs_plotly
def test_plotly_shade_bands(ntwk1: rf.Network):
    fig = ntwk1.plot_s_db(0, 0, backend="plotly")
    edges = ntwk1.f[[0, 40, 80]]
    assert rf.plotting.shade_bands(edges, opacity=0.2, fig=fig, backend="plotly") is fig
    bands = fig.layout.shapes
    assert len(bands) == 2 and len({b.fillcolor for b in bands}) == 2
    # edges in Hz drawn in the unit of the figure, behind the data and over its whole height
    np.testing.assert_allclose([b.x0 for b in bands] + [bands[-1].x1], edges / ntwk1.frequency.multiplier)
    assert {(b.layer, b.opacity, b.yref, b.y0, b.y1) for b in bands} == {("below", 0.2, "y domain", 0, 1)}
    rf.plotting.shade_bands(edges, y_range=(-10, 0), cmap="Viridis", fig=fig, backend="plotly")
    assert (fig.layout.shapes[-1].yref, fig.layout.shapes[-1].y0, fig.layout.shapes[-1].y1) == ("y", -10, 0)
    with pytest.raises(TypeError, match="cannot be used with backend='matplotlib'"):
        rf.plotting.shade_bands(edges, fig=fig)

@needs_plotly
def test_plotly_plot_vector(ntwk1: rf.Network):
    fig = ntwk1.plot_s_smith(0, 0, backend="plotly")
    assert rf.plotting.plot_vector(1 + 2j, 0.5j, arrowcolor="red", fig=fig, backend="plotly") is fig
    (arrow,) = fig.layout.annotations[-1:]
    # from off to off + a, like quiver
    assert (arrow.ax, arrow.ay, arrow.x, arrow.y) == (0, 0.5, 1, 2.5)
    assert (arrow.axref, arrow.ayref, arrow.showarrow, arrow.arrowcolor) == ("x", "y", True, "red")
    rf.plotting.options.backend = "plotly"
    assert isinstance(rf.plotting.plot_vector(1j), go.Figure)

@needs_plotly
def test_plotly_plot_contour(monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig: shown.append(fig))
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.6, 0.6, (2, 100))
    z = np.hypot(x - 0.1, y)
    gamma, value = rf.plotting.plot_contour(1e9, x, y, z, min0max1=0,
                                            title="t", backend="plotly")
    assert value == z.min()
    (fig,) = shown
    assert [t.type for t in fig.data] == ["contour"] + ["scatter"] * 4
    assert np.isnan(fig.data[0].z).any() and fig.layout.title.text == "t"

@pytest.fixture
def one_port_cal() -> rf.calibration.OnePort:
    """Overdetermined one-port calibration, with named standards for the error NetworkSets."""
    media = rf.media.DefinedGammaZ0(rf.Frequency(1, 10, 11, "GHz"))
    ideals = [media.short(name="short"), media.open(name="open"), media.load(0.2, name="load")]
    measured = [rf.Network(frequency=i.frequency, s=i.s + 1e-3 * (k + 1), name=i.name)
                for k, i in enumerate(ideals * 2)]
    cal = rf.calibration.OnePort(ideals=ideals * 2, measured=measured)
    cal.run()
    return cal

def test_calibration_plots_matplotlib(one_port_cal, mpl_axes):
    other = mpl_axes.figure.add_subplot(2, 1, 2)  # now the current axes, not the one passed in
    assert one_port_cal.plot_calibration_errors(ax=mpl_axes) is None
    assert len(mpl_axes.lines) == 3 and len(other.lines) == 0
    assert mpl_axes.get_title() == "S11" and mpl_axes.get_ylim() == (-100, 0)
    assert one_port_cal.plot_residuals(ax=other) is None
    assert len(other.lines) == len(one_port_cal.residual_ntwks)

@needs_plotly
def test_plotly_calibration_errors(one_port_cal, mpl_axes):
    one_port_cal.plot_calibration_errors(ax=mpl_axes)
    fig = one_port_cal.plot_calibration_errors(backend="plotly")
    assert_plotly_parity(fig, mpl_axes, x_scale=1e9)
    assert fig.layout.title.text == "S11" and tuple(fig.layout.yaxis.range) == (-100, 0)
    # global option, and adding to an existing figure
    rf.plotting.options.backend = "plotly"
    assert one_port_cal.plot_calibration_errors(fig=fig) is fig
    assert len(fig.data) == 6
    with pytest.raises(TypeError):
        one_port_cal.plot_calibration_errors(ax=mpl_axes, backend="plotly")

@needs_plotly
@pytest.mark.parametrize("attr", ["s_db", "s_smith"])
def test_plotly_calibration_residuals(one_port_cal, attr: str, mpl_axes):
    one_port_cal.plot_residuals(attr, ax=mpl_axes)
    fig = one_port_cal.plot_residuals(attr, backend="plotly")
    assert isinstance(fig, go.Figure)
    assert len([t for t in fig.data if t.name]) == len(one_port_cal.residual_ntwks)
    if attr == "s_db":
        assert_plotly_parity(fig, mpl_axes, x_scale=1e9)

@needs_plotly
@pytest.mark.parametrize("method", ["plot_ep", "plot_mu", "plot_ep_mu"])
def test_plotly_freespace(ntwk1: rf.Network, method: str, mpl_axes):
    media = rf.media.Freespace(ntwk1.frequency, ep_r=4 - 0.1j, mu_r=1.2)
    assert getattr(media, method)(ax=mpl_axes) is None
    fig = getattr(media, method)(backend="plotly")
    assert_plotly_parity(fig, mpl_axes, x_scale=1e9)
    rf.plotting.options.backend = "plotly"
    assert isinstance(getattr(media, method)(), go.Figure)

@needs_plotly
def test_plotly_signature(ntwk_set_zip: NetworkSet, mpl_axes):
    img = rf.plotting.signature(ntwk_set_zip, component="s_db", cbar_label="dB")
    assert ntwk_set_zip.signature() is None
    fig = ntwk_set_zip.signature(component="s_db", vmax=0, cbar_label="dB", backend="plotly")
    (heatmap,) = fig.data
    np.testing.assert_allclose(np.asarray(heatmap.z), img.get_array())
    np.testing.assert_allclose(heatmap.x, ntwk_set_zip[0].frequency.f_scaled)
    assert heatmap.zmax == 0 and heatmap.colorbar.title.text == "dB"
    assert fig.layout.yaxis.autorange == "reversed" and fig.layout.yaxis.title.text == "Network #"
    with pytest.raises(TypeError, match="imshow"):
        ntwk_set_zip.signature(0, 0, "s_mag", None, False, None, "viridis", backend="plotly")
    rf.plotting.options.backend = "plotly"
    assert isinstance(ntwk_set_zip.signature(), go.Figure)

@needs_plotly
def test_plotly_signature_vs_time(ntwk_set_zip: NetworkSet):
    for k, ntwk in enumerate(ntwk_set_zip):
        ntwk.name = f"2024.01.02.03.04.{k:02d}.000000"
    fig = ntwk_set_zip.signature(vs_time=True, backend="plotly")
    assert [y.second for y in fig.data[0].y] == list(range(len(ntwk_set_zip)))
    assert fig.layout.yaxis.title.text == "Time"

@needs_plotly
def test_plotly_animate(ntwk_set_zip: NetworkSet, monkeypatch):
    mhz = ntwk_set_zip.copy()
    mhz[1].frequency.unit = "MHz"
    fig = mhz.animate("s_db", ylims=(-40, 0), xlims=(1e9, 1.5e9), backend="plotly")
    assert len(fig.frames) == len(mhz) and fig.layout.yaxis.range == (-40, 0)
    assert fig.layout.xaxis.range == (1, 1.5)  # xlims in Hz, like matplotlib
    # each frame shows its Network in the unit of the first one
    for frame, ntwk in zip(fig.frames, mhz):
        np.testing.assert_allclose(frame.data[0].y, ntwk.s_db[:, 0, 0])
        np.testing.assert_allclose(frame.data[0].x, ntwk.f / 1e9)
    assert [step.label for step in fig.layout.sliders[0].steps] == [ntwk.name for ntwk in mhz]
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["Play", "Pause"]
    assert len(mhz.animate("s_time_db", ylims=None, backend="plotly").frames) == len(mhz)
    saved = []
    monkeypatch.setattr(go.Figure, "write_image", lambda fig, path: saved.append(path))
    mhz.animate("s_db", savefigs=True, dir_="out", backend="plotly")
    assert len(saved) == len(mhz) and saved[0].endswith("out_00000.png")
    rf.plotting.options.backend = "plotly"
    assert isinstance(mhz.animate("s_db"), go.Figure)

def test_animate_matplotlib(ntwk_set_zip: NetworkSet, mpl_axes):
    assert ntwk_set_zip.animate("s_db", show=False) is None
    assert len(mpl_axes.figure.axes[0].lines) == 4  # the last Network

@pytest.fixture(scope="module")
def vector_fit():
    vf = rf.vectorFitting.VectorFitting(rf.data.ring_slot)
    with pytest.warns(UserWarning, match="not passive"):
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0)
    return vf

@needs_plotly
@pytest.mark.parametrize("component", ["db", "mag", "deg", "deg_unwrap", "re", "im"])
def test_plotly_vector_fitting(vector_fit, component: str, mpl_axes):
    fig = vector_fit.plot(component, backend="plotly")
    assert vector_fit.plot(component, ax=mpl_axes) is mpl_axes
    samples, fits = fig.data[:4], fig.data[4:]
    for trace, points in zip(samples, mpl_axes.collections):
        assert (trace.mode, trace.marker.color) == ("markers", "red")
        np.testing.assert_allclose(np.column_stack([trace.x, trace.y]), points.get_offsets())
    assert_plotly_parity(go.Figure(fits, layout=fig.layout), mpl_axes)
    assert [t.name for t in fig.data if t.showlegend] == ["Samples", "Fit"]
    assert fig.layout.xaxis.title.text == mpl_axes.get_xlabel()
    assert vector_fit.plot(component, 0, 1, backend="plotly").layout.title.text == "Response i=0, j=1"

@needs_plotly
def test_plotly_vector_fitting_plots(vector_fit, mpl_axes):
    fig = vector_fit.plot_s_singular(backend="plotly")
    vector_fit.plot_s_singular(ax=mpl_axes)
    for trace, line in zip(fig.data, mpl_axes.lines, strict=True):
        np.testing.assert_allclose(np.column_stack([trace.x, trace.y]), line.get_xydata())
    assert [t.name for t in fig.data] == ["σ1", "σ2"]
    fig = vector_fit.plot_convergence(backend="plotly")
    np.testing.assert_allclose(fig.data[0].y, vector_fit.delta_max_history)
    np.testing.assert_allclose(fig.data[1].y, vector_fit.d_res_history)
    assert (fig.data[1].yaxis, fig.layout.yaxis.type, fig.layout.yaxis2.overlaying) == ("y2", "log", "y")
    assert (fig.layout.yaxis.title.text, fig.layout.yaxis2.title.text) == ("Max. relative change", "Residue")
    vf = rf.vectorFitting.VectorFitting(None)
    vf.poles = np.array([-1, -5 + 6j])
    vf.residues = np.array([[0.3, 4 + 5j], [0.1, 2 + 3j], [0.1, 2 + 3j], [0.4, 3 + 4j]])
    vf.constant_coeff = np.array([0.2, 0.1, 0.1, 0.3])
    vf.proportional_coeff = np.zeros(4)
    vf.passivity_enforce(f_max=2)
    fig = vf.plot_passivation(backend="plotly")
    vf.plot_passivation(ax=mpl_axes.figure.add_subplot(2, 1, 2))
    np.testing.assert_allclose(fig.data[0].y, vf.history_max_sigma)
    assert fig.layout.yaxis.title.text == "Max. singular value"
    # the backend comes from the global option, never from ax
    rf.plotting.options.backend = "plotly"
    assert isinstance(vector_fit.plot_s_db(), go.Figure)
    with pytest.raises(TypeError, match="cannot be used with backend='plotly'"):
        vector_fit.plot_s_db(ax=mpl_axes)

@needs_plotly
def test_plotly_title_centered(ntwk1: rf.Network):
    # centered over the plot area like matplotlib, on single plots, smith charts and subplots
    for fig in (rf.plotting.plot_rectangular(ntwk1.f, ntwk1.s_db[:, 0, 0], title="t", backend="plotly"),
                rf.plotting.plot_smith(ntwk1.s[:, 0, 0], title="t", backend="plotly"),
                rf.plotting.subplot_params(ntwk1, backend="plotly")):
        assert (fig.layout.title.x, fig.layout.title.xanchor, fig.layout.title.xref) == (0.5, "center", "paper")

@pytest.fixture
def two_port_cal() -> rf.calibration.TwelveTerm:
    """Overdetermined two-port calibration."""
    media = rf.media.DefinedGammaZ0(rf.Frequency(1, 10, 11, "GHz"))
    ideals = [rf.network.two_port_reflect(media.short(), media.short()),
              rf.network.two_port_reflect(media.open(), media.open()),
              rf.network.two_port_reflect(media.load(0.2), media.load(0.2)), media.thru()]
    measured = [rf.Network(frequency=i.frequency, s=i.s + 1e-3 * (k + 1)) for k, i in enumerate(ideals * 2)]
    cal = rf.calibration.TwelveTerm(ideals=ideals * 2, measured=measured, n_thrus=1)
    cal.run()
    return cal

@needs_plotly
@pytest.mark.parametrize("cal", ["one_port_cal", "two_port_cal"])
@pytest.mark.parametrize("attr", ["s_smith", "s_db"])
def test_plotly_caled_ntwks(cal: str, attr: str, request):
    cal = request.getfixturevalue(cal)
    n_cal = len(cal.caled_ntwks)
    assert cal.plot_caled_ntwks(attr) is None
    axes = mpl.pyplot.gcf().axes
    assert [ax.get_title() for ax in axes] == [f"S{cal.caled_ntwks[0]._fmt_trace_name(m, n)}"
                                                for m, n in cal.caled_ntwks[0].port_tuples]
    n_grid = 3 if attr == "s_smith" else 0  # the smith chart axes are lines too
    assert all(len(ax.lines) == n_cal + n_grid for ax in axes) and axes[0].get_legend() is None
    mpl.pyplot.close("all")
    fig = cal.plot_caled_ntwks(attr, backend="plotly")
    data = [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert len(data) == n_cal * len(axes)
    assert [a.text for a in fig.layout.annotations if a.text.startswith("S")] == [ax.get_title() for ax in axes]
    assert not any(t.showlegend for t in data)
    rf.plotting.options.backend = "plotly"
    assert isinstance(cal.plot_caled_ntwks(attr), go.Figure)

@needs_plotly
def test_plotly_calibration_set_uncertainty(one_port_cal, mpl_axes):
    measured_sets = [NetworkSet(one_port_cal.measured[k::3]) for k in range(3)]
    cal_set = rf.calibration.calibrationSet.Dot(rf.calibration.OnePort, one_port_cal.ideals[:3], measured_sets)
    assert cal_set.plot_uncertainty_per_standard(ax=mpl_axes) is None
    fig = cal_set.plot_uncertainty_per_standard(backend="plotly")
    assert_plotly_parity(fig, mpl_axes, x_scale=1e9)
    assert cal_set.plot_uncertainty_per_standard(fig=fig, backend="plotly") is fig
    with pytest.raises(TypeError, match="cannot be used with backend='plotly'"):
        cal_set.plot_uncertainty_per_standard(ax=mpl_axes, backend="plotly")

@pytest.fixture(scope="module")
def p370_networks():
    """The test 2x-thru and FIX-DUT-FIX, and four-ports of two of each for the mixed-mode de-embeddings."""
    test_dir = rf.calibration.__path__[0] + "/tests/deembed/"
    s2x, fdf = rf.Network(test_dir + "s2xthru.s2p"), rf.Network(test_dir + "fdf.s2p")
    return (s2x, fdf, *(rf.network.concat_ports([n, n], port_order="second") for n in (s2x, fdf)))

@pytest.fixture(scope="module")
def deembedding(p370_networks):
    """Single-ended and mixed-mode IEEE P370 de-embeddings of the test 2x-thru and FIX-DUT-FIX."""
    from skrf.calibration import deembedding as de

    s2x, fdf, se2x, sefdf = p370_networks
    se_nzc, se_zc = de.IEEEP370_SE_NZC_2xThru(s2x, name="nzc"), de.IEEEP370_SE_ZC_2xThru(s2x, fdf, name="zc")
    mm_nzc = de.IEEEP370_MM_NZC_2xThru(se2x, name="mm nzc")
    mm_zc = de.IEEEP370_MM_ZC_2xThru(dummy_2xthru=se2x, dummy_fix_dut_fix=sefdf, name="mm zc")
    fer = de.IEEEP370_FER()
    return {
        "se_nzc_residuals": lambda **k: se_nzc.plot_check_residuals(**k),
        "se_nzc_impedance": lambda **k: se_nzc.plot_check_impedance(fdf, **k),
        "se_zc_residuals": lambda **k: se_zc.plot_check_residuals(**k),
        "se_zc_impedance": lambda **k: se_zc.plot_check_impedance(**k),
        "mm_nzc_residuals": lambda **k: mm_nzc.plot_check_residuals(**k),
        "mm_zc_impedance": lambda **k: mm_zc.plot_check_impedance(**k),
        "fd_se_fer": lambda **k: fer.plot_fd_se_fer(s2x, **k),
        "fd_mm_fer": lambda **k: fer.plot_fd_mm_fer(se2x, **k),
        "td_se_fer": lambda **k: fer.plot_td_se_fer(s2x, fdf, **k),
        "td_mm_fer": lambda **k: fer.plot_td_mm_fer(se2x, sefdf, **k),
    }

@needs_plotly
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")  # no mode conversion in the mixed-mode data
@pytest.mark.parametrize("plot", ["se_nzc_residuals", "se_nzc_impedance", "se_zc_residuals", "se_zc_impedance",
                                  "mm_nzc_residuals", "mm_zc_impedance", "fd_se_fer", "fd_mm_fer", "td_se_fer",
                                  "td_mm_fer"])
def test_plotly_deembedding(deembedding, plot: str):
    out = deembedding[plot]()
    mpl_fig = out[0] if isinstance(out, tuple) else out
    fig = deembedding[plot](backend="plotly")
    # each panel has the traces of its matplotlib Axes, with the same title and y data
    assert [a.text for a in fig.layout.annotations] == [ax.get_title() for ax in mpl_fig.axes]
    assert fig.layout.title.text == mpl_fig.get_suptitle()
    for i, ax in enumerate(mpl_fig.axes):
        traces = [t for t in fig.data if t.xaxis == f"x{i + 1 if i else ''}"]
        assert len(traces) == len(ax.lines)
        for trace, line in zip(traces, ax.lines):
            np.testing.assert_allclose(np.asarray(trace.y, dtype=float), line.get_ydata())
            if not line.get_label().startswith("_"):
                assert trace.name == line.get_label()
        layout = fig.layout[f"yaxis{i + 1 if i else ''}"]
        assert layout.title.text == (ax.get_ylabel() or None)
        # a legend of its own in the plot area of each panel, as matplotlib
        legend = f"legend{i + 1 if i else ''}"
        assert {t.legend for t in traces} == {legend}
        assert any(t.showlegend for t in traces) == (ax.get_legend() is not None)
        if ax.get_legend() is not None:
            x0, x1 = fig.layout[f"xaxis{i + 1 if i else ''}"].domain
            y0, y1 = layout.domain
            assert x0 <= fig.layout[legend].x <= x1 and y0 <= fig.layout[legend].y <= y1
    mpl.pyplot.close("all")
    # a figure from an earlier call can be drawn over, as matplotlib's axes or figure
    n_traces = len(fig.data)
    assert deembedding[plot](fig=fig, backend="plotly") is fig
    assert len(fig.data) == 2 * n_traces
    with pytest.raises(TypeError, match="cannot be used with backend='matplotlib'"):
        deembedding[plot](fig=fig)

@needs_plotly
def test_plotly_fer_limits(deembedding):
    from skrf.calibration.deembedding import IEEEP370_FER

    freq = rf.Frequency(1, 10, 11, "GHz")
    fig = go.Figure()
    IEEEP370_FER().plot_limit_fer2(freq, fig=fig, backend="plotly")
    assert [t.name for t in fig.data] == ["Maximum A", "Maximum B", "Maximum C"]
    np.testing.assert_allclose(fig.data[0].x, [1, 10])  # in the unit of the figure
    IEEEP370_FER().plot_limit_fer5(np.arange(3), np.ones(3), fig=fig, backend="plotly")
    assert [t.showlegend for t in fig.data[3:5]] == [True, False]  # one legend entry for the +/- lines

@needs_plotly
def test_plotly_legend_position():
    # the legend goes where it covers the least data, like matplotlib's 'best'
    fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 1], name="a", showlegend=True))
    fig.add_scatter(x=[0, 1], y=[0, 1], showlegend=False)
    assert rf.plotting._plotly_legend_position(fig, 300, 300) == (1, 0)  # the diagonal passes the other corners
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0.1, 0.1], name="a", showlegend=True))
    fig.update_yaxes(range=[0.05, 1])  # the line at the bottom of the plot area
    assert rf.plotting._plotly_legend_position(fig, 300, 300) == (1, 1)
    fig.update_yaxes(range=[-1, 0.12])  # the line at the top
    assert rf.plotting._plotly_legend_position(fig, 300, 300) == (1, 0)

@needs_plotly
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")  # no mode conversion in the mixed-mode data
@pytest.mark.parametrize("make", [  # each returns the object that drew, and keeps, the verbose figures
    lambda de, s2x, fdf, se2x, sefdf: (qm := de.IEEEP370_FD_QM(), qm.check_se_quality(s2x, verbose=True))[0],
    lambda de, s2x, fdf, se2x, sefdf: (qm := de.IEEEP370_TD_QM(1e9, 32, 0.4, 2, 2),
                                       qm.check_se_quality(s2x, verbose=True))[0],
    lambda de, s2x, fdf, se2x, sefdf: de.IEEEP370_SE_NZC_2xThru(s2x, verbose=True),
    lambda de, s2x, fdf, se2x, sefdf: de.IEEEP370_SE_ZC_2xThru(s2x, fdf, bandwidth_limit=5e9, leadin=1, verbose=True),
    lambda de, s2x, fdf, se2x, sefdf: de.IEEEP370_MM_NZC_2xThru(se2x, verbose=True),
], ids=["fd_qm", "td_qm", "se_nzc", "se_zc", "mm_nzc"])
def test_plotly_verbose_figures(p370_networks, make, monkeypatch):
    from skrf.calibration import deembedding as de

    mpl_figs = make(de, *p370_networks).verbose_figures
    assert mpl_figs and all(isinstance(f, mpl.figure.Figure) for f in mpl_figs)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig, *a, **k: shown.append(fig))
    rf.plotting.options.backend = "plotly"
    figs = make(de, *p370_networks).verbose_figures
    # each figure is shown and kept, with the panels, titles and traces of its matplotlib figure
    assert shown == figs and len(figs) == len(mpl_figs)
    for fig, mpl_fig in zip(figs, mpl_figs):
        assert len(fig.data) == sum(len(ax.lines) for ax in mpl_fig.axes)
        titles = [ax.get_title() for ax in mpl_fig.axes if ax.get_title()]
        assert [a.text for a in fig.layout.annotations if a.text in titles] == titles
    mpl.pyplot.close("all")

def test_legend_helpers(ntwk1: rf.Network, mpl_axes):
    ntwk1.plot_s_db(ax=mpl_axes)
    rf.plotting.add_markers_to_lines(ax=mpl_axes, markevery=5)
    assert [line.get_marker() for line in mpl_axes.lines] == ["o", "D", "s", "+"]
    rf.plotting.scrape_legend(2, ax=mpl_axes)
    assert [t.get_text() for t in mpl_axes.get_legend().get_texts()] == ["ntwk1, S11", "ntwk1, S22"]
    rf.plotting.legend_off(ax=mpl_axes)
    assert not mpl_axes.get_legend().get_visible()

@needs_plotly
def test_plotly_legend_helpers(ntwk1: rf.Network):
    fig = ntwk1.plot_s_db(backend="plotly")
    assert rf.plotting.add_markers_to_lines(fig=fig, markevery=5, backend="plotly") is fig
    assert [t.marker.symbol for t in fig.data] == ["circle", "diamond", "square", "cross"]
    assert {(t.mode, t.marker.maxdisplayed) for t in fig.data} == {("lines+markers", -(-len(ntwk1) // 5))}
    assert rf.plotting.scrape_legend(2, fig=fig, backend="plotly") is fig
    assert [t.name for t in fig.data if t.showlegend] == ["ntwk1, S11", "ntwk1, S22"]
    assert rf.plotting.legend_off(fig=fig, backend="plotly").layout.showlegend is False
    with pytest.raises(TypeError, match="cannot be used with backend='matplotlib'"):
        rf.plotting.legend_off(fig=fig)
    # the smith chart grid is not a line to mark
    fig = ntwk1.plot_s_smith(0, 0, backend="plotly")
    rf.plotting.add_markers_to_lines(fig=fig, backend="plotly")
    assert [t.mode for t in fig.data if t.meta == "skrf_smith_chart"] == ["lines"] * len(_smith_grid_traces(fig))

@needs_plotly
@pytest.mark.parametrize("style_file", ["skrf.mplstyle", "skrf2.mplstyle", "ieeetrans.mplstyle"])
def test_plotly_stylely(style_file: str, monkeypatch):
    import plotly.io as pio

    monkeypatch.setattr(pio.templates, "default", pio.templates.default)  # restored after the test
    mpl.pyplot.style.use("default")
    # one call styles both: matplotlib, and with the settings that plotly has, plotly's default template
    rf.stylely(style_file=style_file)
    rc = mpl.pyplot.rcParams
    assert pio.templates.default == style_file.split(".")[0]
    layout = pio.templates[pio.templates.default].layout
    tick_size = mpl.font_manager.FontProperties(size=rc["xtick.labelsize"]).get_size()  # also for 'small' etc.
    assert layout.font.size == rc["font.size"] and layout.xaxis.tickfont.size == pytest.approx(tick_size, rel=1e-3)
    assert mpl.colors.to_hex(layout.plot_bgcolor) == mpl.colors.to_hex(rc["axes.facecolor"])
    assert mpl.colors.to_hex(layout.xaxis.gridcolor) == mpl.colors.to_hex(rc["grid.color"])
    if style_file != "ieeetrans.mplstyle":  # whose old axes.color_cycle setting matplotlib ignores
        assert ([mpl.colors.to_hex(c) for c in rf.plotting.colors(backend="plotly")]
                == [mpl.colors.to_hex(c) for c in rf.plotting.colors()])
    assert pio.templates[pio.templates.default].data.scatter[0].line.width == rc["lines.linewidth"]
    rf.stylely(style_file=style_file, rc_dict={"font.size": "14"})
    assert pio.templates[pio.templates.default].layout.font.size == 14 == mpl.pyplot.rcParams["font.size"]
    mpl.pyplot.style.use("default")

@needs_plotly
def test_plotly_colors(monkeypatch):
    import plotly.io as pio
    from plotly.colors import qualitative

    monkeypatch.setattr(pio.templates, "default", "plotly")
    assert rf.plotting.colors(backend="plotly") == list(pio.templates["plotly"].layout.colorway)
    monkeypatch.setattr(pio.templates, "default", "none")
    assert rf.plotting.colors(backend="plotly") == list(qualitative.Plotly)

@needs_plotly
def test_plotly_scale_frequency_ticks(ntwk1: rf.Network):
    fig = ntwk1.plot_s_db(0, 0, backend="plotly")
    rf.plotting.shade_bands(ntwk1.f[[0, 40]], fig=fig, backend="plotly")
    fig.update_xaxes(range=[2, 4])
    assert rf.plotting.scale_frequency_ticks(fig=fig, funit="mhz", backend="plotly") is fig
    # the frequencies, bands and range of the figure in MHz, also for traces added later
    scale = ntwk1.frequency.multiplier / 1e6
    np.testing.assert_allclose(fig.data[0].x, ntwk1.f / 1e6)
    np.testing.assert_allclose([fig.layout.shapes[0].x0, fig.layout.shapes[0].x1], ntwk1.f[[0, 40]] / 1e6)
    assert fig.layout.xaxis.range == (2 * scale, 4 * scale) and fig.layout.xaxis.title.text == "Frequency (MHz)"
    np.testing.assert_allclose(ntwk1.plot_s_db(1, 0, fig=fig, backend="plotly").data[-1].x, ntwk1.f / 1e6)
    with pytest.raises(ValueError, match="invalid funit"):
        rf.plotting.scale_frequency_ticks(fig=fig, funit="furlongs", backend="plotly")
    with pytest.raises(ValueError, match="no frequency axis"):
        rf.plotting.scale_frequency_ticks(fig=rf.plotting.plot_complex_rectangular(ntwk1.s[:, 0, 0], backend="plotly"),
                                          funit="MHz", backend="plotly")

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

def test_save_all_figs(ntwk1: rf.Network, mpl_axes, tmp_path):
    ntwk1.plot_s_db(ax=mpl_axes)
    mpl_axes.set_title("open figure")
    rf.plotting.save_all_figs(str(tmp_path), format=["png", "pdf"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["open_figure.pdf", "open_figure.png"]
    fig = mpl.pyplot.figure()
    fig.add_subplot().set_title("given")
    rf.plotting.save_all_figs(str(tmp_path), figs=[fig])  # only the figures passed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["given.png", "open_figure.pdf", "open_figure.png"]

@needs_plotly
def test_plotly_save_all_figs(ntwk1: rf.Network, mpl_axes, tmp_path, monkeypatch):
    fig = ntwk1.plot_s_db(backend="plotly").update_layout(title_text="ring slot")
    mpl_axes.set_title("matplotlib")
    images = []
    monkeypatch.setattr(go.Figure, "write_image", lambda self, path, format: images.append((path, format)))
    # plotly and matplotlib figures in one call, a plotly figure as html by default
    rf.plotting.save_all_figs(str(tmp_path), figs=[fig, mpl_axes.figure])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["matplotlib.png", "ring_slot.html"]
    rf.plotting.save_all_figs(str(tmp_path), format=["html", "png"], figs=[fig])
    assert images == [(f"{tmp_path}/ring_slot.png", "png")]

@needs_plotly
def test_plotly_func_on_all_figs(ntwk1: rf.Network, mpl_axes):
    fig = ntwk1.plot_s_db(backend="plotly")
    ntwk1.plot_s_db(ax=mpl_axes)
    # a plotly figure is passed to func, a matplotlib figure is made current
    rf.plotting.func_on_all_figs(lambda f, **k: f.update_xaxes(**k), showgrid=False, figs=[fig])
    assert fig.layout.xaxis.showgrid is False
    rf.plotting.func_on_all_figs(mpl.pyplot.xlabel, "x", figs=[mpl_axes.figure])
    assert mpl_axes.get_xlabel() == "x"

def test_func_on_all_figs_makes_each_figure_current(mpl_axes):
    mpl.pyplot.figure().add_subplot()
    seen = []
    rf.plotting.func_on_all_figs(lambda: seen.append(mpl.pyplot.gcf().number))
    assert seen == mpl.pyplot.get_fignums()

@needs_plotly
def test_plotly_stylely_rcparams_types(monkeypatch):
    import plotly.io as pio
    from cycler import cycler

    monkeypatch.setattr(pio.templates, "default", pio.templates.default)  # restored after the test
    rc = {"font.size": 12, "axes.grid": False, "grid.color": (1, 0, 0),
          "axes.prop_cycle": cycler(color=["r", "#00ff00"], linestyle=["-", "--"])}
    with mpl.rc_context():
        rf.stylely(rc)
    layout = pio.templates[pio.templates.default].layout
    assert layout.xaxis.showgrid is False
    assert layout.xaxis.gridcolor == "rgb(255,0,0)"
    assert layout.colorway == ("red", "#00ff00")

@needs_plotly
def test_plotly_signature_vmax(ntwk_set_zip: NetworkSet):
    trace = ntwk_set_zip.signature(vmax=0.5, backend="plotly").data[0]
    assert (trace.zmin, trace.zmax) == (np.nanmin(trace.z), 0.5)
    assert ntwk_set_zip.signature(vmax=0.5, zmin=0.1, backend="plotly").data[0].zmin == 0.1

@needs_plotly
def test_plotly_subplots_frequency_unit(ntwk1: rf.Network):
    fig = ntwk1.plot_it_all(backend="plotly")
    edges = ntwk1.f[[0, 40]]
    rf.plotting.shade_bands(edges, fig=fig, backend="plotly")
    assert fig.layout.shapes[0].x0 == pytest.approx(ntwk1.frequency.f_scaled[0])
    others = [np.array(t.x) for t in fig.data if t.xaxis in ("x3", "x4") and t.x is not None]
    x = {t.xaxis: t.x[0] for t in fig.data if t.xaxis in ("x", "x2")}
    rf.plotting.scale_frequency_ticks(funit="MHz", fig=fig, backend="plotly")
    # both panels vs frequency, not the smith chart and the complex plane
    assert {t.xaxis: t.x[0] for t in fig.data if t.xaxis in ("x", "x2")} == pytest.approx(
        {k: v * 1e3 for k, v in x.items()})
    assert (fig.layout.xaxis.title.text, fig.layout.xaxis2.title.text) == ("Frequency (MHz)", "Frequency (MHz)")
    for before, after in zip(others, [t.x for t in fig.data if t.xaxis in ("x3", "x4") and t.x is not None]):
        np.testing.assert_array_equal(before, after)
    assert ntwk1.plot_s_db(0, 0, backend="plotly").layout.meta["skrf_frequency_unit"] == ntwk1.frequency.unit

@needs_plotly
def test_plotly_scale_frequency_ticks_subplot_shapes_and_units(ntwk1: rf.Network):
    fig = rf.plotting.subplot_params(ntwk1, proj="deg", backend="plotly")
    fig.add_shape(type="rect", x0=1, x1=2, y0=0, y1=1, row=1, col=2)
    fig.update_xaxes(range=[1, 2], row=1, col=2)
    fig.update_xaxes(title_text="Frequency (kHz)", row=2, col=2)  # a panel in another unit
    x4 = np.array(fig.data[3].x)
    rf.plotting.scale_frequency_ticks(funit="MHz", fig=fig, backend="plotly")
    np.testing.assert_allclose(fig.data[1].x, ntwk1.f / 1e6)
    np.testing.assert_allclose(fig.data[3].x, x4 * 1e-3)
    assert (fig.layout.shapes[0].x0, fig.layout.shapes[0].x1) == pytest.approx((1e3, 2e3))
    assert {a: fig.layout[a].title.text for a in ("xaxis", "xaxis2", "xaxis3", "xaxis4")} == dict.fromkeys(
        ("xaxis", "xaxis2", "xaxis3", "xaxis4"), "Frequency (MHz)")

@needs_plotly
def test_plotly_scale_frequency_ticks_animation(ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.animate("s_db", m=0, n=0, backend="plotly")
    x = np.array(fig.data[0].x)
    rf.plotting.scale_frequency_ticks(funit="MHz", fig=fig, backend="plotly")
    np.testing.assert_allclose(fig.data[0].x, x * 1e3)
    for frame in fig.frames:
        np.testing.assert_allclose(frame.data[0].x, x * 1e3)

@needs_plotly
def test_plotly_scale_frequency_ticks_log_axis(ntwk1: rf.Network):
    fig = ntwk1.plot_s_db(0, 0, logx=True, backend="plotly").update_xaxes(range=[0.5, 1])
    rf.plotting.scale_frequency_ticks(funit="MHz", fig=fig, backend="plotly")
    assert fig.layout.xaxis.range == pytest.approx((3.5, 4))

@needs_plotly
def test_plotly_scale_frequency_ticks_violin(ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_violin("s_db", backend="plotly")
    width = fig.data[0].width
    rf.plotting.scale_frequency_ticks(funit="MHz", fig=fig, backend="plotly")
    assert fig.data[0].width == pytest.approx(width * 1e3)

@needs_plotly
def test_plotly_bounds_follow_colorway(ntwk_set_zip: NetworkSet):
    for colorway in (dict(colorway=["#111111"]), dict(template=dict(layout=dict(colorway=["#111111"])))):
        fig = go.Figure(layout=colorway)
        ntwk_set_zip.plot_uncertainty_bounds_component("s_mag", 0, 0, fig=fig, backend="plotly")
        assert fig.data[0].line.color == "#111111"

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

@needs_plotly
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")  # -inf dB
def test_plotly_legend_without_finite_data(match: rf.Network):
    fig = match.plot_it_all(backend="plotly")  # S11 is -inf dB
    assert fig.layout.legend.x is not None
    line = rf.media.DefinedGammaZ0(match.frequency, z0=50, gamma=1j).line(30, "deg")
    assert len(rf.plotting.subplot_params(line, proj="db", backend="plotly").data) == 4
    empty = go.Figure(go.Scatter(x=[1, 2], y=[-np.inf, -np.inf], name="a", showlegend=True))
    assert rf.plotting._plotly_legend_position(empty, 300, 300) == rf.plotting._PLOTLY_LEGEND_SPOTS[0]

@needs_plotly
@pytest.mark.parametrize("rc, attr, expected", [
    ({"axes.facecolor": "none"}, lambda t: t.layout.plot_bgcolor, "rgba(0,0,0,0)"),
    ({"axes.titlesize": "larger", "font.size": 10}, lambda t: t.layout.title.font.size, 12),
    ({"font.family": "serif", "font.serif": ["Times"]}, lambda t: t.layout.font.family, "Times, serif"),
    ({"font.family": ["serif"], "font.serif": ["Times"]}, lambda t: t.layout.font.family, "Times, serif"),
])
def test_plotly_stylely_rcparams(rc: dict, attr, expected, monkeypatch):
    import plotly.io as pio

    monkeypatch.setattr(pio.templates, "default", pio.templates.default)  # restored after the test
    with mpl.rc_context():
        rf.stylely(rc)
    assert attr(pio.templates[pio.templates.default]) == pytest.approx(expected)

@needs_plotly
def test_plotly_stylely_skips_what_plotly_lacks(monkeypatch, caplog):
    import plotly.io as pio
    from cycler import cycler

    monkeypatch.setattr(pio.templates, "default", pio.templates.default)  # restored after the test
    default = pio.templates["plotly"].layout.colorway
    with mpl.rc_context():
        rf.stylely({"axes.prop_cycle": cycler(linestyle=["-", "--"]), "grid.color": "tab:blue", "font.size": 12})
    layout = pio.templates[pio.templates.default].layout
    assert layout.colorway == default and layout.font.size == 12
    assert layout.xaxis.gridcolor == mpl.colors.to_hex("tab:blue")  # matplotlib's own colors are resolved
    # the linestyles of a style file are no colors either
    assert rf.plotting._plotly_template({"axes.prop_cycle": "cycler('linestyle', ['-', '--'])"}).layout.colorway == \
        default
    # a value plotly can't take is skipped, and the other settings still apply
    template = rf.plotting._plotly_template({"grid.color": "notacolor", "font.size": 12})
    assert template.layout.font.size == 12 and "grid.color" in caplog.text

@needs_plotly
def test_plotly_stylely_settings(ntwk1: rf.Network, monkeypatch):
    import plotly.io as pio

    monkeypatch.setattr(pio.templates, "default", pio.templates.default)  # restored after the test
    with mpl.rc_context():
        rf.stylely(style_file="skrf2.mplstyle", rc_dict={"lines.markeredgewidth": 2, "legend.frameon": False})
    template = pio.templates["skrf2"]
    layout = template.layout
    for axis in (layout.xaxis, layout.yaxis):
        assert (axis.showline, axis.mirror, axis.linecolor, axis.linewidth) == (True, True, "black", 1)
        assert (axis.griddash, axis.ticks, axis.ticklen, axis.layer) == ("dash", "inside", 5, "below traces")
    assert (layout.xaxis.minor.ticklen, layout.xaxis.minor.tickwidth) == (3, 1)
    assert (template.data.scatter[0].marker.size, template.data.scatter[0].marker.line.width) == (3, 2)
    assert layout.legend.borderwidth == 0
    assert (layout.width, layout.height) == (600, 400)
    assert (layout.margin.l, layout.margin.r, layout.margin.b, layout.margin.t) == pytest.approx((72, 60, 48, 40))
    # new figures take the size of the template, and keep the centered title
    fig = ntwk1.plot_s_db(0, 0, backend="plotly")
    assert (fig.layout.width, fig.layout.height, fig.layout.margin.t) == (None, None, None)
    assert fig.layout.title.x == 0.5
    # a smith chart stays without frame and ticks
    fig = ntwk1.plot_s_smith(0, 0, backend="plotly")
    assert (fig.layout.xaxis.showline, fig.layout.yaxis.showline) == (False, False)
    assert (fig.layout.xaxis.ticks, fig.layout.yaxis.ticks) == ("", "")

def test_networkset_creation_adds_methods_once(ntwk1: rf.Network, monkeypatch):
    NetworkSet([ntwk1])
    calls = []
    monkeypatch.setattr(rf.plotting, "_fig_kwarg", lambda func: calls.append(func) or func)
    NetworkSet([ntwk1])
    assert not calls

def test_networkset_non_plot_method_not_wrapped():
    assert hasattr(NetworkSet.plot_s_db, "__wrapped__") and not hasattr(NetworkSet.write_touchstone, "__wrapped__")

@needs_plotly
@pytest.mark.parametrize("label", [None, "_nolegend_"])
def test_plotly_label_without_legend_entry(ntwk1: rf.Network, ntwk_set_zip: NetworkSet, label):
    for fig in (ntwk1.plot_s_db(0, 0, label=label, backend="plotly"),
                ntwk_set_zip.animate("s_db", m=0, n=0, label=label, backend="plotly")):
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

@needs_plotly
@pytest.mark.parametrize("draw", [
    lambda ntwk, fig: ntwk.plot_it_all(fig=fig, backend="plotly"),
    lambda ntwk, fig: rf.plotting.subplot_params(ntwk, fig=fig, backend="plotly"),
], ids=["plot_it_all", "subplot_params"])
def test_plotly_new_figure_plots_reject_fig(ntwk1: rf.Network, draw):
    fig = go.Figure()
    with pytest.raises(TypeError, match="fig is not supported"):
        draw(ntwk1, fig)
    assert not fig.data

@needs_plotly
def test_plotly_subplot_params_polar(ntwk1: rf.Network):
    fig = rf.plotting.subplot_params(ntwk1, proj="polar", backend="plotly")
    assert {t.type for t in fig.data} == {"scatterpolar"}
    assert [t.subplot for t in fig.data] == ["polar", "polar2", "polar3", "polar4"]
    # specs of the caller are kept
    fig = rf.plotting.subplot_params(ntwk1, backend="plotly", subplot_kw=dict(specs=[[{}, {"b": 0.1}], [{}, {}]]))
    assert len(fig.data) == 4

def test_save_all_figs_unique_names(tmp_path):
    figs = [mpl.pyplot.figure() for _ in range(2)]
    for fig in figs:
        fig.add_subplot()
    rf.plotting.save_all_figs(str(tmp_path), figs=figs, format=["png"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unnamedPlot.png", "unnamedPlot_1.png"]
    mpl.pyplot.close("all")

@needs_plotly
def test_plotly_save_all_figs_unique_names(ntwk1: rf.Network, tmp_path):
    figs = [ntwk1.plot_s_db(backend="plotly"), ntwk1.plot_s_deg(backend="plotly"),
            ntwk1.plot_s_db(backend="plotly").update_layout(title_text="a b"),
            ntwk1.plot_s_deg(backend="plotly").update_layout(title_text="a b")]
    rf.plotting.save_all_figs(str(tmp_path), figs=figs)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_b.html", "a_b_1.html", "unnamedPlot.html",
                                                          "unnamedPlot_1.html"]

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
        fig = ntwk1.plot_s_db(0, 0, backend="plotly")
        rf.plotting.add_markers_to_lines(fig=fig, marker_list=[marker], backend="plotly")
        assert fig.data[0].marker.symbol == rf.plotting._PLOTLY_MARKERS[marker]

@needs_plotly
@pytest.mark.parametrize("kind", ["shade", "bar"])
def test_plotly_uncertainty_color_error(ntwk_set_zip: NetworkSet, kind: str):
    fig = ntwk_set_zip.plot_uncertainty_bounds_component("s_mag", 0, 0, type=kind, color_error="r", backend="plotly")
    assert "red" in (fig.data[-1].fillcolor, fig.data[-1].marker.color)

@needs_plotly
def test_plotly_minmax_color_error(ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_minmax_bounds_component("s_mag", 0, 0, color_error="0.5", backend="plotly")
    assert fig.data[-1].fillcolor == "rgb(128,128,128)"
