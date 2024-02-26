import sys

import pytest

import skrf as rf

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)

ntwk1 = rf.Network("skrf/tests/ntwk1.s2p")
ntwk1 = ntwk1.extrapolate_to_dc()

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

def test_plot_it_all():
    ntwk1.plot_it_all()
