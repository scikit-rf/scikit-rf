import pytest

import skrf as rf

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
    fig = getattr(rf.data.ring_slot, method)()

def test_generated_function_plots(generated_functions):
    method = f"plot_{generated_functions}"
    fig = getattr(rf.data.ring_slot, method)()

def test_plot_passivity():
    return rf.data.ring_slot.plot_passivity()

def test_plot_reciprocity():
    return rf.data.ring_slot.plot_reciprocity()

def test_plot_reciprocity2():
    return rf.data.ring_slot.plot_reciprocity2()

def test_plot_s_db_time():
    return rf.data.ring_slot.plot_s_db_time()

def test_plot_s_smith():
    return rf.data.ring_slot.plot_s_smith()

def test_plot_it_all(self, *args, **kwargs):
    return rf.data.ring_slot.plot_it_all(self, *args, **kwargs)