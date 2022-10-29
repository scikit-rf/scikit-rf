import unittest

import pytest
import skrf
import numpy as np
import tempfile
import os
import warnings


class VectorFittingTestCase(unittest.TestCase):

    def test_ringslot_with_proportional(self):
        # perform the fit
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    def test_ringslot_default_log(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log')
        self.assertLess(vf.get_rms_error(), 0.01)

    def test_ringslot_without_prop_const(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        with pytest.warns(UserWarning, match="The fitted network is passive, but the vector fit is not passive") as record:
            vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False)

        assert len(record) == 1
        self.assertLess(vf.get_rms_error(), 0.01)

    def test_190ghz_measured(self):
        # perform the fit without proportional term
        nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=4, fit_proportional=False, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    def test_no_convergence(self):
        # perform a bad fit that does not converge and check if a RuntimeWarning is given
        nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
        vf = skrf.vectorFitting.VectorFitting(nw)
        
        with pytest.warns(RuntimeWarning) as record:
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=5, fit_proportional=False, fit_constant=True)
        
        assert len(record) == 1

    def test_dc(self):
        # perform the fit on data including a dc sample (0 Hz)
        nw = skrf.Network('./skrf/tests/cst_example_4ports.s4p')
        vf = skrf.VectorFitting(nw)
        vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)
        # quality of the fit is not important in this test; it only needs to finish
        self.assertLess(vf.get_rms_error(), 0.2)

    def test_spice_subcircuit(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # write equivalent SPICE subcircuit to tmp file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.sp', delete=False)
        name = tmp_file.name
        tmp_file.close()
        vf.write_spice_subcircuit_s(name)

        # written tmp file should contain 69 lines
        with open(name) as f:
            n_lines = len(f.readlines())
        self.assertEqual(n_lines, 69)
        os.remove(name)

    def test_read_write_npz(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)

        with pytest.warns(UserWarning) as record:
            vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)

        assert len(record) == 1

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

    def test_matplotlib_missing(self):
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        skrf.vectorFitting.mplt = None
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


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
