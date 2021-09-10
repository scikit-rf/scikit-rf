import unittest
import skrf
import numpy as np
import tempfile
import os


class VectorFittingTestCase(unittest.TestCase):

    def test_ringslot_with_proportional(self):
        # perform the fit
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)

        # expected fitting parameters for skrf.data.ring_slot with 2 initial real poles
        expected_poles = np.array([-7.82570461e+10+5.32874416e+11j])
        expected_zeros = np.array([[7.04342166e+10+1.12221242e+10j],
                                   [7.95530409e+10-4.87232072e+09j],
                                   [7.95530409e+10-4.87232072e+09j],
                                   [8.21087613e+10-2.15675492e+10j]])
        expected_props = np.array([9.71465239e-16,
                                   -2.10871364e-14,
                                   -2.10871364e-14,
                                   7.83239517e-13])
        expected_const = np.array([-0.98831027,
                                   -0.06128905,
                                   -0.06128905,
                                   -0.99446547])

        # relax relative and absolute tolerances, as results from Python 2.7 are slightly different from Python 3.x
        # basically, this disables the absolute tolerance criterion
        rtol = 0.01
        atol = rtol * np.amax(np.abs(expected_poles))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, expected_poles, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.zeros, expected_zeros, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.proportional_coeff, expected_props, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.constant_coeff, expected_const, rtol=rtol, atol=atol))

    def test_ringslot_default_log(self):
        # perform the fit without proportional term
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log')

        # expected parameters:
        expected_poles = np.array([-8.93571544e+11+1.84502286e+12j, -7.96637282e+10+5.33065163e+11j])
        expected_zeros = np.array([[4.71811349e+09+2.20844260e+10j, 7.22130093e+10+1.10150314e+10j],
                                   [-8.10130339e+10+2.69553730e+10j, 8.12743265e+10-5.28434213e+09j],
                                   [-8.10130339e+10+2.69553730e+10j, 8.12743265e+10-5.28434213e+09j],
                                   [1.50741990e+12+1.03733308e+12j, 8.72835493e+10-2.51863622e+10j]])
        expected_props = np.array([0.0, 0.0, 0.0, 0.0])
        expected_const = np.array([-0.97995369, -0.00434421, -0.00434421, -0.86793172])

        # relax relative and absolute tolerances, as results from Python 2.7 are slightly different from Python 3.x
        # basically, this disables the absolute tolerance criterion
        rtol = 0.01
        atol = rtol * np.amax(np.abs(expected_poles))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, expected_poles, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.zeros, expected_zeros, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.proportional_coeff, expected_props, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.constant_coeff, expected_const, rtol=rtol, atol=atol))

    def test_ringslot_without_prop_const(self):
        # perform the fit without proportional term
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False)

        # expected parameters:
        expected_poles = np.array([-2.68320606e+13+0.0j, -2.39725722e+12+0.0j, -7.94998061e+10+5.33460725e+11j])
        expected_zeros = np.array([[-2.85512308e+13+0.0j, 1.72810467e+11+0.0j, 7.19396489e+10+1.06616739e+10j],
                                   [-2.51468427e+12+0.0j, 6.44668782e+10+0.0j, 8.10600423e+10-5.87840573e+09j],
                                   [-2.51468427e+12+0.0j, 6.44668782e+10+0.0j, 8.10600423e+10-5.87840573e+09j],
                                   [3.10896516e+13+0.0j, -5.51444142e+12+0.0j, 8.57614332e+10-2.64489550e+10j]])
        expected_props = np.array([0.0, 0.0, 0.0, 0.0])
        expected_const = np.array([0.0, 0.0, 0.0, 0.0])

        # relax relative and absolute tolerances, as results from Python 2.7 are slightly different from Python 3.x
        # basically, this disables the absolute tolerance criterion
        rtol = 0.01
        atol = rtol * np.amax(np.abs(expected_poles))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, expected_poles, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.zeros, expected_zeros, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.proportional_coeff, expected_props, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(vf.constant_coeff, expected_const, rtol=rtol, atol=atol))

    def test_190ghz_measured(self):
        # perform the fit without proportional term
        nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=4, fit_proportional=False, fit_constant=True)

        # expected parameters:
        expected_poles = np.array([-1.57475745e+14+0.0j,
                                   -1.18834516e+11+9.09437100e+11j,
                                   -9.31737428e+10+1.28087682e+12j,
                                   -8.62928730e+10+1.17731581e+12j,
                                   -8.40722337e+10+1.06025397e+12j,
                                   -6.78498212e+10+1.40989041e+12j,
                                   -2.14689157e+10+0.0j])
        expected_zeros = np.array([[-2.65287319e+15+0.0j,
                                    -1.93216730e+10-5.07224595e+10j,
                                    -2.21817763e+10-4.06200216e+10j,
                                    1.36015316e+10-1.17553256e+10j,
                                    5.56282943e+09+2.05636021e+10j,
                                    1.67186787e+10+8.76036057e+09j,
                                    -1.45040972e+11+0.0j],
                                   [7.58775308e+14+0.0j,
                                    -2.38135317e+09+7.47127977e+08j,
                                    -4.86143241e+09+1.80283555e+09j,
                                    -6.10784454e+08+1.03682367e+09j,
                                    -7.16135706e+08+6.77740453e+08j,
                                    -3.33102819e+09-7.59790686e+08j,
                                    -3.30724201e+10+0.0j],
                                   [-1.15809727e+15+0.0j,
                                    -1.22716214e+11-9.54130617e+09j,
                                    -9.21636351e+10+1.24415054e+11j,
                                    3.58950981e+11+2.00228548e+11j,
                                    4.99726589e+10-3.38515501e+11j,
                                    -5.72387494e+10+3.80171040e+10j,
                                    4.08499919e+10+0.0j],
                                   [-4.20227295e+15+0.0j,
                                    2.71669215e+10+1.16365804e+10j,
                                    -1.26160359e+10+3.15388544e+09j,
                                    -3.02169283e+10-3.65666828e+10j,
                                    4.32209635e+09-2.19063498e+10j,
                                    2.41311866e+10-3.75303886e+10j,
                                    -2.67088753e+11+0.0j]])
        expected_props = np.array([0.0, 0.0, 0.0, 0.0])
        expected_const = np.array([16.86700896, -4.77902696, 7.19034104, 26.87203581])

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
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0)

        # compare fitted model responses to original network responses (should match with less than 1% error)
        # s11
        nw_s11 = nw.s[:, 0, 0]
        fit_s11 = vf.get_model_response(0, 0, freqs=nw.f)
        delta_s11_maxabs = np.amax(np.abs((fit_s11 - nw_s11) / nw_s11))
        self.assertLess(delta_s11_maxabs, 0.01)

        # s12
        nw_s12 = nw.s[:, 0, 1]
        fit_s12 = vf.get_model_response(0, 1, freqs=nw.f)
        delta_s12_maxabs = np.amax(np.abs((fit_s12 - nw_s12) / nw_s12))
        self.assertLess(delta_s12_maxabs, 0.01)

        # s21
        nw_s21 = nw.s[:, 1, 0]
        fit_s21 = vf.get_model_response(1, 0, freqs=nw.f)
        delta_s21_maxabs = np.amax(np.abs((fit_s21 - nw_s21) / nw_s21))
        self.assertLess(delta_s21_maxabs, 0.01)

        # s22
        nw_s22 = nw.s[:, 1, 1]
        fit_s22 = vf.get_model_response(1, 1, freqs=nw.f)
        delta_s22_maxabs = np.amax(np.abs((fit_s22 - nw_s22) / nw_s22))
        print(delta_s22_maxabs)
        self.assertLess(delta_s22_maxabs, 0.01)

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
        vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)

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

    def test_passivity_enforcement(self):
        vf = skrf.VectorFitting(None)

        # non-passive example parameters from Gustavsen's passivity assessment paper:
        vf.poles = np.array([-1, -5 + 6j])
        vf.zeros = np.array([[0.3, 4 + 5j], [0.1, 2 + 3j], [0.1, 2 + 3j], [0.4, 3 + 4j]])
        vf.constant_coeff = np.array([0.2, 0.1, 0.1, 0.3])
        vf.proportional_coeff = np.array([0.0, 0.0, 0.0, 0.0])

        # testing is_passive() implicitly also tests passivity_test()
        self.assertFalse(vf.is_passive())

        # enforce passivity with default settings
        vf.passivity_enforce()

        # check if model is now passive
        self.assertTrue(vf.is_passive())

        # verify that perturbed zeros are correct
        passive_zeros = np.array([[0.11758964+0.j, 2.65059197+3.29414469j],
                                  [-0.06802029+0.j, 0.77242142+1.44226975j],
                                  [-0.06802029+0.j, 0.77242142+1.44226975j],
                                  [0.24516918+0.j, 1.88377719+2.57735204j]])
        self.assertTrue(np.allclose(vf.zeros, passive_zeros))


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
