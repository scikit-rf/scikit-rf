import skrf as rf
import numpy as np
import unittest
import os
from numpy.testing import (
    assert_almost_equal, assert_allclose, 
    assert_array_equal, run_module_suite
    )


class QfactorTests(unittest.TestCase):
    """ Q-factor class tests.

    References
    ----------
    "Q-factor Measurement by using a Vector Network Analyser",
        A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
        https://eprintspublications.npl.co.uk/9304/
    """

    def setUp(self):
        """
        Q-factor tests initalizer.
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.ntwk_2port = rf.data.ring_slot
        self.ntwk_1port = rf.data.ring_slot_meas

    def csv_file_example_to_network(self, file: str) -> rf.Network:
        """Convert the S-parameter txt file to Network.        

        Parameters
        ----------
        file : str
            file path.

        Returns
        -------
        ntwk : rf.Network (1-port)

        """
        # Load frequency and S-parameter data from a file.
        try:
            f, s_re, s_im = np.loadtxt(file, comments="%", unpack=True)
        except ValueError as e:
            f, s_re, s_im, s_abs, s_mag = np.loadtxt(file, comments="%", unpack=True)
        
        s = s_re + 1j * s_im
        freq = rf.Frequency.from_f(f, unit='GHz')
        return rf.Network(s=s, frequency=freq)        

    def test_constructor(self):
        """
        Test the Qfactor() constructor.
        """       
        # constructor tests
        _Q1 = rf.Qfactor(self.ntwk_1port, res_type='reflection')
        _Q2 = rf.Qfactor(self.ntwk_1port, res_type='reflection', Q_L0=3)
        _Q3 = rf.Qfactor(self.ntwk_1port, res_type='reflection', f_L0=85e9)
        _Q3 = rf.Qfactor(self.ntwk_1port, res_type='reflection', Q_L0=3, f_L0=85e9)

    def test_exceptions(self):
        "Test the raised exceptions."
        
        # Passing a 2-port Network raises a ValueError
        self.assertRaises(ValueError, rf.Qfactor, self.ntwk_2port, 'reflection')
        
        # Uncorrect resonance type raises a ValueError
        self.assertRaises(ValueError, rf.Qfactor, self.ntwk_1port, 'dummy')
        
        # Asking for fitted S-param and Network without prior fit raises a ValueError
        _Q = rf.Qfactor(self.ntwk_1port, res_type='reflection')
        self.assertRaises(ValueError, _Q.fitted_s)
        self.assertRaises(ValueError, _Q.fitted_network)        

    def test_NLQFIT6(self):
        """
        Fit FL and QL to transmission (S21) data by using the NLQFIT6 algorithm.

        References
        ----------
        Test data is read from file Figure6b.txt (as used in Figure 6(b) of MAT 58).

        """
        # File 'Figure6b.txt' contains S21 data for Fig. 6(b) in MAT 58
        ntwk = self.csv_file_example_to_network(self.test_dir + "qfactor_data/Figure6b.txt")
        
        Q = rf.Qfactor(ntwk, res_type='transmission', verbose=True)

        # Test against expected solutions
        assert_almost_equal(Q._a, 0.8104 - 1.6928j, decimal=4)
        assert_almost_equal(Q._b, 0.0077 - 0.0071j, decimal=4)
        assert_almost_equal(Q.Q_L, 7440.848, decimal=3)

        # Optimised weighted fit --> result vector
        res = Q.fit(method="NLQFIT6")
        
        # Test against expected solutions
        assert_allclose(res.f_L, 3.987848e9)
        assert_almost_equal(res.Q_L, 7454, decimal=0)
        assert_almost_equal(res.RMS_Error, 0.00001216)
        assert_array_equal(res.f_L, Q.f_L)
        assert_array_equal(res.Q_L, Q.Q_L)

        # Now calculate unloaded Q-factor and some other useful quantities.
        # Reciprocal of |S21| of a thru in place of resonator
        scaling_factor_A = 1 / 0.874  # 1/|S21_thru|
        Q0 = Q.Q_unloaded(res, scaling_factor_A)
        cal_diam, cal_gamma_V, cal_gamma_T = Q.Q_circle(res, scaling_factor_A)

        # Q-factor of uncoupled two-port resonator (unloaded Q-factor)
        assert_almost_equal(Q0, 7546, decimal=0)
        assert_almost_equal(cal_diam, 0.0121, decimal=4)
        # S21 detuned = leakage vector
        assert_almost_equal(cal_gamma_V, -0.00008895 + 0.00003852j)
        # S21 tuned
        assert_almost_equal(cal_gamma_T, 0.00849357 - 0.00845349j)

    def test_NLQFIT6_2(self):
        """
        Fit FL and QL to transmission (S21) data by using the NLQFIT6 algorithm.

        From a two-port superconducting absorption resonator.

        References
        ----------
        Test data is read from file Figure27.txt (as used in Figure 27 of MAT 58).

        """
        ntwk = self.csv_file_example_to_network(self.test_dir + "qfactor_data/Figure27.txt")
        
        Q = rf.Qfactor(ntwk, res_type='absorption', verbose=True)

        # Test against expected solutions
        assert_almost_equal(Q._a, -17072.3098 + 9047.0761j, decimal=4)
        assert_almost_equal(Q._b, 0.0063 + 0.0168j, decimal=4)
        
        # Q.tol = 1.0e-5 * np.argmax(np.abs(Q.s))
       
        # Step 2: Optimised weighted fit --> result vector
        res = Q.fit(method="NLQFIT6")
        
        assert_allclose(res.f_L, 6.07225567e9)
        assert_almost_equal(res.Q_L, 56019.85, decimal=0)
        assert_almost_equal(res.weighting_ratio, 4.714, decimal=3)
        assert_almost_equal(res.RMS_Error, 0.01394722)
        assert_allclose(Q.Q_L, res.Q_L)
        assert_allclose(Q.f_L, res.f_L)
        
        Q0 = Q.Q_unloaded(res)
        cal_diam, cal_gamma_V, cal_gamma_T = Q.Q_circle(res)

        assert_allclose(Q0, 1846782, rtol=1/100)
        assert_allclose(cal_diam, 0.970, rtol=1/100)
        assert_almost_equal(2.0 * res.f_L / res.Q_L/1e9, 0.00021678942580071302, decimal=10)



    def test_NLQFIT7(self):
        """
        Fit FL and QL to reflection (S11) data by using the NLQFIT7 algorithm.

        References
        ----------
        Test data is read from file Table6c27.txt (as used in Figure 16 and
        Table 6(c) of MAT 58).

        """
        ntwk = self.csv_file_example_to_network(self.test_dir + "qfactor_data/Table6c27.txt")
        
        Q = rf.Qfactor(ntwk, res_type='reflection', verbose=True)
        # Expected results after initial fit
        assert_almost_equal(Q._a, 760.9731 + 67.7804j, decimal=4)
        assert_almost_equal(Q._b, 0.0609 - 0.6432j, decimal=4)
        assert_almost_equal(Q.Q_L, 779.068, decimal=3)
        
        # Expected results after fit
        res = Q.fit(method='NLQFIT7')
        # Fitted length of uncalibrated line [m]
        assert_almost_equal(-res.m7a*rf.c/(4.0*np.pi*1.3), 57.47056249053462e-3)
        assert_allclose(Q.f_L, 3.65293800e9)
        assert_almost_equal(Q.Q_L, 708, decimal=0)

        # Unloaded Q-factor and some other useful quantities.
        print("Q-factor of unloaded one-port resonator by Method 1:")
        print("Assumes attenuating uncalibrated line")
        Q0 = Q.Q_unloaded(res)
        cal_diam, cal_gamma_V, cal_gamma_T = Q.Q_circle(res)
        
        # Test against expected solutions
        assert_almost_equal(Q0, 862, decimal=0)
        assert_almost_equal(cal_diam, 0.3573, decimal=4)
        assert_almost_equal(cal_gamma_V, 0.09084890 - 0.99586469j)  # S11 detuned
        assert_almost_equal(cal_gamma_T, 0.05878773 - 0.64003179j)  # S11 tuned

        print("Q-factor of unloaded one-port resonator by Method 2:")
        print("Scaling factor A = 1.0 (assume no attenuation in uncalibrated line)")
        Q2 = rf.Qfactor(ntwk, res_type='reflection_method2', verbose=True)
        res2 = Q2.fit(method='NLQFIT7')

        Q0_2 = Q.Q_unloaded(res2, A=1)
        cal_diam2, cal_gamma_V2, cal_gamma_T2 = Q.Q_circle(res2, A=1)

        # Test against expected solutions
        assert_almost_equal(Q0_2, 862, decimal=0)
        assert_almost_equal(res2.weighting_ratio, 28.317, decimal=3)
        assert_almost_equal(res2.RMS_Error, 0.00145957)
        assert_almost_equal(cal_diam2, 0.3573, decimal=2)
        assert_almost_equal(cal_gamma_V2, 0.08996562 - 0.98618239j)
        assert_almost_equal(cal_gamma_T2, 0.05821616 - 0.63380908j)

    def test_NLQFIT8(self):
        """
        Fits to transmission (S21) data by using the NLQFIT8 algorithm.

        frequency-dependent leakage

        References
        ----------
        Test data is read from file Figure23.txt (shown in Figure 23 of MAT 58)

        """
        ntwk = self.csv_file_example_to_network(self.test_dir + "qfactor_data/Figure23.txt")
        
        Q = rf.Qfactor(ntwk, res_type='transmission')
        
        # # De-embed cables
        # N = len(Q.f)
        ncablelen = 1.2  # root_eps * cable length in metres
        # D =  ntwk.s[:,0,0]* np.exp(-1j * 2*np.pi *Q.f * ncablelen / rf.c)
    
        # ntwk2 = ntwk.copy()
        # ntwk2.s[:,0,0] = D
        
        # piece of transmission line to deembbed
        gamma = 0 - 1j*ntwk.frequency.w * ncablelen / rf.c
        coax = rf.media.DefinedGammaZ0(frequency=ntwk.frequency, gamma=gamma)
        line = coax.line(1/2, unit='m')
        ntwk2 = line.inv ** ntwk
        
        # Find peak in |S21| - this is used to give initial value of freq.
        # Tol is 1.0E-5 * |S21| at peak.
        Mg = np.abs(ntwk2.s).squeeze()
        index_max = np.argmax(Mg)
        Tol = Mg[index_max] * 1.0e-5
        Fseed = Q.f[index_max]

        # Set Qseed: An order-of-magnitude estimate for Q-factor
        mult = 5.0  # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
        Qseed = mult * Fseed / (Q.f[-1] - Q.f[0])

        Q = rf.Qfactor(ntwk2, res_type='transmission', Q_L0=Qseed, f_L0=Fseed)
        assert_almost_equal(Q._a, 8.9408 + 2.1298j, decimal=4)
        assert_almost_equal(Q._b, 0.0054 - 0.0045j, decimal=4)
        assert_almost_equal(Q.Q_L, 4664.2418, decimal=3)

        # Step 2: Optimised weighted fit
        res = Q.fit(method="NLQFIT8", loop_plan="fwfwfwc")

        # Unloaded Q-factor and some other useful quantities.
        # Reciprocal of |S21| of a thru in place of resonator
        scaling_factor_A = 1 / 0.949  # 1/|S21_thru|
        Q0 = Q.Q_unloaded(res, scaling_factor_A)
        cal_diam, cal_gamma_V, cal_gamma_T = Q.Q_circle(res, scaling_factor_A)

        # Test against expected solutions
        assert_allclose(Q.f_L, 9.76015571e9)
        assert_almost_equal(Q.Q_L, 4760.04, decimal=0)
        assert_almost_equal(Q0, 4789.49, decimal=0)
        assert_almost_equal(res.weighting_ratio, 5.078, decimal=3)
        assert_almost_equal(res.RMS_Error, 0.00000910)
        assert_almost_equal(cal_diam, 0.0061, decimal=2)

    def test_Q_unloaded(self):
        """Test unloaded Q factor method."""
        Q = rf.Qfactor(self.ntwk_1port, res_type='reflection')
        res = Q.fit()
        self.assertRaises(ValueError, Q.Q_unloaded, A='dummy')
        self.assertRaises(ValueError, Q.Q_unloaded, A=1j)
        self.assertRaises(ValueError, Q.Q_unloaded, res, A='dummy')
        self.assertRaises(ValueError, Q.Q_unloaded, res, A=1j)

        # passing of not the fitted results after fit should be the same
        self.assertEqual(Q.Q_unloaded(res), Q.Q_unloaded())
        # passing a different solution should lead to different values
        res2 = Q.fit(method="NLQFIT7")
        self.assertNotEqual(Q.Q_unloaded(res), Q.Q_unloaded(res2))

    def test_Q_circle(self):
        """Test Q-circle method."""
        Q = rf.Qfactor(self.ntwk_1port, res_type='reflection')
        res = Q.fit(method="NLQFIT6")
        self.assertRaises(ValueError, Q.Q_circle, A='dummy')
        self.assertRaises(ValueError, Q.Q_circle, A=1j)   
        self.assertRaises(ValueError, Q.Q_circle, res, A='dummy')
        self.assertRaises(ValueError, Q.Q_circle, res, A=1j)        
        
        # passing of not the fitted results after fit should be the same
        self.assertEqual(Q.Q_circle(res), Q.Q_circle())
        # passing a different solution should lead to different values
        res2 = Q.fit(method="NLQFIT7")
        self.assertNotEqual(Q.Q_circle(res), Q.Q_circle(res2))

    def test_f_L(self):
        "Test resonant frequency values."
        # expected values
        f_L_expected = self.ntwk_2port.f[np.argmin(self.ntwk_2port.s11.s_mag)]
        f_L_expected_scaled = f_L_expected/self.ntwk_2port.frequency.multiplier
        # fitted values
        Q = rf.Qfactor(self.ntwk_2port.s11, res_type='reflection')
        # before the fit, warnings should be raised
        with self.assertWarns(Warning):
            # the resonance frequency corresponds to min value before fitting
            assert_almost_equal(Q.f_L, f_L_expected)
            assert_almost_equal(Q.f_L_scaled, f_L_expected_scaled)
        # NB: after the fit this should not be the case anymore (slight deviation)      
            
    def test_BW(self):
        "Test bandwidth values."
        Q = rf.Qfactor(self.ntwk_2port.s11, res_type='reflection')
        # before the fit, warnings should be raised
        with self.assertWarns(Warning):
            BW = Q.BW
            BW_scaled = Q.BW_scaled
    

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
