import sys

try:
    import matplotlib as mpl
except ImportError:
    pass

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
    method = f"plot_violin_{generated_functions}"
    
    if "time" not in method:
        fig = getattr(ntwk_set, method)()
    else:
        with pytest.raises(NotImplementedError):
            fig = getattr(ntwk_set, method)()
