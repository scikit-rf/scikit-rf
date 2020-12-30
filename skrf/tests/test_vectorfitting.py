import unittest
import skrf
import numpy as np
import os.path


class VectorFittingTestCase(unittest.TestCase):

    def test_vectorfitting_ring_slot(self):
        # load expected model parameters
        expected_parameters = np.load(os.path.join(os.path.dirname(os.path.abspath(skrf.__file__)), 'tests',
                                                   'vectorfit_ringslot_2poles.npz'))
        expected_poles = expected_parameters['arr_0']
        expected_zeros = expected_parameters['arr_1']
        expected_props = expected_parameters['arr_2']
        expected_const = expected_parameters['arr_3']

        # perform the fit
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vectorfit(n_poles_real=2, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # compare both sets of parameters
        self.assertTrue(np.all(np.isclose(vf.poles, expected_poles)))
        self.assertTrue(np.all(np.isclose(vf.zeros, expected_zeros)))
        self.assertTrue(np.all(np.isclose(vf.proportional_coeff, expected_props)))
        self.assertTrue(np.all(np.isclose(vf.constant_coeff, expected_const)))


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
