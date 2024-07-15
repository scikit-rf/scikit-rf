import sys

try:
    import matplotlib as mpl
except ImportError:
    pass

import numpy as np
import pytest

import skrf as rf

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

ntwk1 = rf.Network("skrf/tests/ntwk1.s2p")
ntwk1 = ntwk1.extrapolate_to_dc()

ntwk_set = rf.NetworkSet.from_zip("skrf/tests/ntwks.zip")

@pytest.fixture(params=rf.Network.PRIMARY_PROPERTIES)
def primary_properties(request):
    return request.param

@pytest.fixture(params=["polar", "complex"])
def primary_methods(request):
    return request.param

@pytest.fixture(params=rf.Network._generated_functions().keys())
def generated_functions(request):
    return request.param

def test_primary_plotting(primary_properties, primary_methods):
    method = f"plot_{primary_properties}_{primary_methods}"
    fig = getattr(ntwk1, method)()

def test_generated_function_plots(generated_functions):
    method = f"plot_{generated_functions}"
    fig = getattr(ntwk1, method)()

def test_plot_passivity():
    ntwk1.plot_passivity()

def test_plot_reciprocity():
    ntwk1.plot_reciprocity()

def test_plot_reciprocity2():
    ntwk1.plot_reciprocity2()

def test_plot_s_db_time():
    ntwk1.plot_s_db_time()

def test_plot_s_smith():
    ntwk1.plot_s_smith()

def test_z_time_impulse_step_z0():
    se_diff = rf.concat_ports([ntwk1, ntwk1], port_order = 'second')
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

@pytest.mark.parametrize("usetex", [True, False])
def test_plot_it_all(usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1.plot_it_all()

@pytest.mark.parametrize("usetex", [True, False])
def test_plot_polar(usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1.plot_s_polar()

def test_plot_uncertainty_bounds_s():
    ntwk_set.plot_uncertainty_bounds_s()

def test_plot_uncertainty_bounds_s_db():
    ntwk_set.plot_uncertainty_bounds_s_db()

def test_plot_uncertainty_bounds_s_time_db():
    ntwk_set.plot_uncertainty_bounds_s_time_db()

def test_plot_uncertainty_decomposition():
    ntwk_set.plot_uncertainty_decomposition()

def test_plot_minmax_bounds_s_db():
    ntwk_set.plot_minmax_bounds_s_db()

def test_plot_minmax_bounds_s_db10():
    ntwk_set.plot_minmax_bounds_s_db10()

def test_plot_minmax_bounds_s_time_db():
    ntwk_set.plot_minmax_bounds_s_time_db()

def test_plot_signature():
    ntwk_set.signature()

def test_generated_violin_plots(generated_functions):
    if "time" not in generated_functions:
        fig = ntwk_set.plot_violin(generated_functions)
    else:
        with pytest.raises(NotImplementedError):
            fig = ntwk_set.plot_violin(generated_functions)
