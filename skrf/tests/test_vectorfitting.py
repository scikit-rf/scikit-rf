import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

import skrf


class VectorFittingTestCase(unittest.TestCase):

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ringslot_with_proportional(self):
        # perform the fit
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ringslot_default_log(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log')
        self.assertLess(vf.get_rms_error(), 0.01)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ringslot_without_prop_const(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False)
        self.assertLess(vf.get_rms_error(), 0.01)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ringslot_custompoles(self):
        # perform the fit with custom initial poles
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.poles = 2 * np.pi * np.array([-100e9, -10e9 + 100e9j])
        vf.vector_fit(init_pole_spacing='custom')
        self.assertLess(vf.get_rms_error(), 0.01)

    def test_190ghz_measured(self):
        # perform the fit without proportional term
        s2p_file = Path(__file__).parent.parent.parent / 'doc/source/examples/vectorfitting/190ghz_tx_measured.S2P'
        nw = skrf.network.Network(s2p_file)
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=4, fit_proportional=False, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    def test_no_convergence(self):
        s2p_file = Path(__file__).parent.parent.parent / 'doc/source/examples/vectorfitting/190ghz_tx_measured.S2P'
        # perform a bad fit that does not converge and check if a RuntimeWarning is given
        nw = skrf.network.Network(s2p_file)
        vf = skrf.vectorFitting.VectorFitting(nw)

        with pytest.warns(RuntimeWarning) as record:
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=5, fit_proportional=False, fit_constant=True)

        assert len(record) == 1

    def test_dc(self):
        # perform the fit on data including a dc sample (0 Hz)
        s4p_file = Path(__file__).parent / 'cst_example_4ports.s4p'
        nw = skrf.Network(s4p_file)
        vf = skrf.VectorFitting(nw)
        vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)
        # quality of the fit is not important in this test; it only needs to finish
        self.assertLess(vf.get_rms_error(), 0.3)

    def test_read_write_npz(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)

        with pytest.warns(UserWarning) as record:
            vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)

        self.assertTrue(len(record) == 1)

        # export (write) fitted parameters to .npz file in tmp directory
        with  tempfile.TemporaryDirectory() as name:
            vf.write_npz(name)

            # create a new vector fitting instance and import (read) those fitted parameters
            vf2 = skrf.vectorFitting.VectorFitting(nw)
            vf2.read_npz(os.path.join(name, f'coefficients_{nw.name}.npz'))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, vf2.poles))
        self.assertTrue(np.allclose(vf.residues, vf2.residues))
        self.assertTrue(np.allclose(vf.proportional_coeff, vf2.proportional_coeff))
        self.assertTrue(np.allclose(vf.constant_coeff, vf2.constant_coeff))

    @pytest.mark.skipif("matplotlib" in sys.modules, reason="Raise Error only if matplotlib is not installed.")
    def test_matplotlib_missing(self):
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        with self.assertRaises(RuntimeError):
            vf.plot_convergence()

    def test_passivity_enforcement(self):
        vf = skrf.VectorFitting(None)

        # non-passive example parameters from Gustavsen's passivity assessment paper:
        vf.poles = np.array([-1, -5 + 6j])
        vf.residues = np.array([[0.3, 4 + 5j], [0.1, 2 + 3j], [0.1, 2 + 3j], [0.4, 3 + 4j]])
        vf.constant_coeff = np.array([0.2, 0.1, 0.1, 0.3])
        vf.proportional_coeff = np.array([0.0, 0.0, 0.0, 0.0])

        # test if model is not passive
        violation_bands = vf.passivity_test()
        self.assertTrue(np.allclose(violation_bands, np.array([4.2472, 16.434]) / 2 / np.pi, rtol=1e-3, atol=1e-3))
        self.assertFalse(vf.is_passive())

        # enforce passivity with default settings
        vf.passivity_enforce(f_max=2)

        # check if model is now passive
        self.assertTrue(vf.is_passive())

    def test_autofit(self):
        vf = skrf.VectorFitting(skrf.data.ring_slot)
        vf.auto_fit()

        self.assertTrue(vf.get_model_order(vf.poles) == 6)
        self.assertTrue(np.sum(vf.poles.imag == 0.0) == 0)
        self.assertTrue(np.sum(vf.poles.imag > 0.0) == 3)
        self.assertLess(vf.get_rms_error(), 1e-05)


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
