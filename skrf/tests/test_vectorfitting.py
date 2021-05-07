import unittest
import skrf
import numpy as np
import tempfile
import os


class VectorFittingTestCase(unittest.TestCase):

    def test_vectorfitting_ring_slot(self):
        # expected fitting parameters for skrf.data.ring_slot with 2 initial real poles
        expected_poles = np.array([-7.80605445e+10+5.32645184e+11j])
        expected_zeros = np.array([[7.01837934e+10+1.14737278e+10j],
                                   [7.93470695e+10-4.54467471e+09j],
                                   [7.93470695e+10-4.54467471e+09j],
                                   [8.19724835e+10-2.11876421e+10j]])
        expected_props = np.array([-2.06451610e-15,
                                   -2.45016478e-14,
                                   -2.45016478e-14,
                                   7.79744644e-13])
        expected_const = np.array([-0.9871906,
                                   -0.06043898,
                                   -0.06043898,
                                   -0.99401152])

        # perform the fit
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # relax relative and absolute tolerances, as results from Python 2.7 are slightly different from Python 3.x
        # basically, this disables the absolute tolerance criterion
        rtol = 0.01
        atol = rtol * np.amax(np.abs(expected_poles))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, expected_poles, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.zeros, expected_zeros, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.proportional_coeff, expected_props, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.constant_coeff, expected_const, rtol=rtol, atol=atol))

    def test_model_response(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # compare fitted model responses to original network responses (should match with less than 1% error)
        # s11
        nw_s11 = nw.s[:, 0, 0]
        fit_s11 = vf.get_model_response(0, 0, freqs=nw.f)
        delta_s11_maxabs = np.amax(np.abs((fit_s11 - nw_s11) / nw_s11))
        self.assertLess(delta_s11_maxabs, 0.05)

        # s12
        nw_s12 = nw.s[:, 0, 1]
        fit_s12 = vf.get_model_response(0, 1, freqs=nw.f)
        delta_s12_maxabs = np.amax(np.abs((fit_s12 - nw_s12) / nw_s12))
        self.assertLess(delta_s12_maxabs, 0.05)

        # s21
        nw_s21 = nw.s[:, 1, 0]
        fit_s21 = vf.get_model_response(1, 0, freqs=nw.f)
        delta_s21_maxabs = np.amax(np.abs((fit_s21 - nw_s21) / nw_s21))
        self.assertLess(delta_s21_maxabs, 0.05)

        # s22
        nw_s22 = nw.s[:, 1, 1]
        fit_s22 = vf.get_model_response(1, 1, freqs=nw.f)
        delta_s22_maxabs = np.amax(np.abs((fit_s22 - nw_s22) / nw_s22))
        print(delta_s22_maxabs)
        self.assertLess(delta_s22_maxabs, 0.05)

    def test_spice_subcircuit(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # write equivalent SPICE subcircuit to tmp file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.sp')
        vf.write_spice_subcircuit_s(tmp_file.name)

        # written tmp file should contain 69 lines
        n_lines = len(open(tmp_file.name, 'r').readlines())
        self.assertEqual(n_lines, 69)

    def test_read_write_npz(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # export (write) fitted parameters to .npz file in tmp directory
        tmp_dir = tempfile.TemporaryDirectory()
        vf.write_npz(tmp_dir.name)

        # create a new vector fitting instance and import (read) those fitted parameters
        vf2 = skrf.vectorFitting.VectorFitting(nw)
        vf2.read_npz(os.path.join(tmp_dir.name, 'coefficients_{}.npz'.format(nw.name)))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, vf2.poles))
        self.assertTrue(np.allclose(vf.zeros, vf2.zeros))
        self.assertTrue(np.allclose(vf.proportional_coeff, vf2.proportional_coeff))
        self.assertTrue(np.allclose(vf.constant_coeff, vf2.constant_coeff))

    def test_matplotlib_missing(self):
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        skrf.vectorFitting.mplt = None
        with self.assertRaises(RuntimeError):
            vf.plot_convergence()



suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
