import unittest
import skrf
import numpy as np
import os.path


class VectorFittingTestCase(unittest.TestCase):

    def test_vectorfitting_ring_slot(self):
        # load expected model parameters
        expected_parameters = np.load(os.path.join(os.path.dirname(os.path.abspath(self.__file__)),
                                                   'vectorfit_ring slot_2poles.npz'))
        expected_poles = expected_parameters['poles']
        expected_zeros = expected_parameters['zeros']
        expected_props = expected_parameters['proportionals']
        expected_const = expected_parameters['constants']

        # perform the fit
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        vf.vectorfit(n_poles_real=2, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # compare both sets of parameters
        self.assertTrue(np.isclose(vf.poles, expected_poles))
        self.assertTrue(np.isclose(vf.zeros, expected_zeros))
        self.assertTrue(np.isclose(vf.proportional_coeff, expected_props))
        self.assertTrue(np.isclose(vf.constant_coeff, expected_const))


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
