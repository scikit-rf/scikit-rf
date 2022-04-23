import skrf as rf
import numpy as np
import unittest
import os
from numpy.testing import assert_almost_equal, run_module_suite


class QfactorTests(unittest.TestCase):
    """
    Q-factor class tests.

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

    def test_NLQFIT6(self):
        """
        Fit FL and QL to transmission (S21) data by using the NLQFIT6 algorithm.

        References
        ----------
        Test data is read from file Figure6b.txt (as used in Figure 6(b) of MAT 58).

        """
        # Load frequency and S-parameter data from a file.
        # File 'Figure6b.txt' contains S21 data for Fig. 6(b) in MAT 58
        f, s21re, s21im = np.loadtxt(
            self.test_dir + "qfactor_data/Figure6b.txt", comments="%", unpack=True
        )
        s21 = s21re + 1j * s21im

        # The convergence algorithm uses a number of steps set by loop_plan, a string of characters as follows:
        #   f - fit once without testing for convergence
        #   c - repeated fit, iterating until convergence is obtained
        #   w - re-calculate weighting factors on basis of previous fit
        #       Initially the weighting factors are all unity.
        #   The first character in loop_plan must not be w.
        loop_plan = "fwfwc"

        # Find peak in |S21| - this is used to give initial value of freq.
        # Tol is 1.0E-5 * |S21| at peak.
        Mg = np.absolute(s21)
        index_max = np.argmax(Mg)

        Tol = Mg[index_max] * 1.0e-5
        Fseed = f[index_max]

        # Set Qseed: An order-of-magnitude estimate for Q-factor
        mult = 5.0  # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
        Qseed = mult * Fseed / (f[-1] - f[0])

        quiet = False
        if not quiet:
            print("Initial values for iteration:  Freq=", Fseed, " QL=", Qseed)
        # Step 1: Initial unweighted fit --> solution vector
        N = len(f)
        Q = rf.Qfactor()

        sv = Q.initial_fit(f, s21, N, Fseed, Qseed)
        a_re, a_im, b_re, b_im, QL = sv
        
        # Test against expected solutions
        assert_almost_equal(a_re + 1j*a_im, 0.8104 - 1.6928j, decimal=4)
        assert_almost_equal(b_re + 1j*b_im, 0.0077 - 0.0071j, decimal=4)
        assert_almost_equal(QL, 7440.848, decimal=3)

        # Step 2: Optimised weighted fit --> result vector
        mv, weighting_ratio, number_iterations, RMS_Error = Q.optimise_fit6(
            f, s21, N, Fseed, sv, loop_plan, Tol, quiet
        )
        m1, m2, m3, m4, QL, FL = mv

        # Test against expected solutions
        assert_almost_equal(FL, 3.987848, decimal=6)
        assert_almost_equal(QL, 7454, decimal=0)
        assert_almost_equal(weighting_ratio, 5.149, decimal=3)
        assert_almost_equal(RMS_Error, 0.00001216)

        # Now calculate unloaded Q-factor and some other useful quantities.
        # Reciprocal of |S21| of a thru in place of resonator
        scaling_factor_A = 1 / 0.874  # 1/|S21_thru|
        trmode = "transmission"
        p = Q.Q_unloaded(mv, scaling_factor_A, trmode, quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p

        # Q-factor of uncoupled two-port resonator (unloaded Q-factor)
        assert_almost_equal(Qo, 7546, decimal=0)
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
        quiet = True
        f, s21re, s21im = np.loadtxt(
            self.test_dir + "qfactor_data/Figure27.txt", comments="%", unpack=True
        )
        s21 = s21re + 1j * s21im

        loop_plan = "fwfwc"

        # Find notch in |S21| - this is used to give initial value of freq.
        Mg = np.abs(s21)
        index_min = np.argmin(Mg)
        Fseed = f[index_min]

        # argmax(Mg) could be much less than 1.0 as cables
        # through cryostat wall attenuate the signal
        Tol = 1.0e-5 * np.argmax(Mg)

        # Set Qseed: An order-of-magnitude estimate for Q-factor
        mult = 5.0  # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
        Qseed = mult * Fseed / (f[-1] - f[0])

        # Step 1: Initial unweighted fit --> solution vector
        N = len(f)
        Q = rf.Qfactor()
        sv = Q.initial_fit(f, s21, N, Fseed, Qseed)
        a_re, a_im, b_re, b_im, QL = sv

        # Step 2: Optimised weighted fit --> result vector
        mv, weighting_ratio, number_iterations, RMS_Error = Q.optimise_fit6(
            f, s21, N, Fseed, sv, loop_plan, Tol, quiet
        )
        m1, m2, m3, m4, QL, FL = mv

        p = Q.Q_unloaded(mv, "AUTO", "absorption", quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p

        y = [1.0 / complex(1.0, 2.0 * (QL / FL * _f - QL)) for _f in f]
        DFIT = [complex(m1, m2) + complex(m3, m4) * yi for yi in y]

        # Test against expected solutions
        assert_almost_equal(a_re + 1j * a_im, -17072.3098 + 9047.0761j, decimal=4)
        assert_almost_equal(b_re + 1j * b_im, 0.0063 + 0.0168j, decimal=4)
        assert_almost_equal(FL, 6.07225567, decimal=6)
        assert_almost_equal(QL, 56019.85, decimal=0)
        assert_almost_equal(Qo, 1846782.51, decimal=0)
        assert_almost_equal(weighting_ratio, 4.714, decimal=3)
        assert_almost_equal(RMS_Error, 0.01394722)
        assert_almost_equal(cal_diam, 0.970, decimal=3)
        assert_almost_equal(2.0 * FL / QL, 0.00021678942580071302, decimal=10)

    def test_NLQFIT7(self):
        """
        Fit FL and QL to reflection (S11) data by using the NLQFIT7 algorithm.

        References
        ----------
        Test data is read from file Table6c27.txt (as used in Figure 16 and
        Table 6(c) of MAT 58).

        """
        f, s11re, s11im, s11abs, s11phase = np.loadtxt(
            self.test_dir + "qfactor_data/Table6c27.txt", comments="%", unpack=True
        )
        s11 = s11re + 1j * s11im

        loop_plan = "fwfwc"

        # Find minimum in |S11| - this is used to give initial value of freq.
        Mg = np.absolute(s11)
        index_min = np.argmin(Mg)

        Tol = 1.0e-5
        Fseed = f[index_min]

        # Set Qseed: An order-of-magnitude estimate for Q-factor
        mult = 5.0  # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
        Qseed = mult * Fseed / (f[-1] - f[0])

        quiet = True
        if not quiet:
            print("Initial values for iteration:  Freq=", Fseed, " QL=", Qseed)
        # Step 1: Initial unweighted fit --> solution vector
        N = len(f)
        Q = rf.Qfactor()
        sv = Q.initial_fit(f, s11, N, Fseed, Qseed)
        a_re, a_im, b_re, b_im, QL = sv

        assert_almost_equal(a_re + 1j * a_im, 760.9731 + 67.7804j, decimal=4)
        assert_almost_equal(b_re + 1j * b_im, 0.0609 - 0.6432j, decimal=4)
        assert_almost_equal(QL, 779.068, decimal=3)

        # Step 2: Optimised weighted fit --> result vector
        mv, weighting_ratio, number_iterations, RMS_Error = Q.optimise_fit7(
            f, s11, N, Fseed, sv, loop_plan, Tol, quiet
        )
        m1, m2, m3, m4, QL, FL, m7a = mv

        # Fitted length of uncalibrated line [mm]
        assert_almost_equal(
            -m7a * 2.99792458e2 / (4.0 * np.pi * 1.3), 57.47056249053462
        )

        # Test against expected solutions
        assert_almost_equal(FL, 3.65293800, decimal=6)
        assert_almost_equal(QL, 708, decimal=0)

        # Now calculate unloaded Q-factor and some other useful quantities.
        print("Q-factor of unloaded one-port resonator by Method 1:")
        print("Assumes attenuating uncalibrated line")
        scaling_factor_A = "Auto"
        trmode = "reflection_method1"
        p = Q.Q_unloaded(mv, scaling_factor_A, trmode, quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p

        # Test against expected solutions
        assert_almost_equal(Qo, 862, decimal=0)
        assert_almost_equal(cal_diam, 0.3573, decimal=4)
        assert_almost_equal(cal_gamma_V, 0.09084890 - 0.99586469j)  # S11 detuned
        assert_almost_equal(cal_gamma_T, 0.05878773 - 0.64003179j)  # S11 tuned

        print("Q-factor of unloaded one-port resonator by Method 2:")
        print("Scaling factor A = 1.0 (assume no attenuation in uncalibrated line)")
        scaling_factor_A = 1.0
        trmode = "reflection_method2"
        p = Q.Q_unloaded(mv, scaling_factor_A, trmode, quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p

        # Test against expected solutions
        assert_almost_equal(Qo, 862, decimal=0)
        assert_almost_equal(weighting_ratio, 28.317, decimal=3)
        assert_almost_equal(RMS_Error, 0.00145957)
        assert_almost_equal(cal_diam, 0.3573, decimal=2)
        assert_almost_equal(cal_gamma_V, 0.08996562 - 0.98618239j)
        assert_almost_equal(cal_gamma_T, 0.05821616 - 0.63380908j)

    def test_NLQFIT8(self):
        """
        Fits to transmission (S21) data by using the NLQFIT8 algorithm.

        frequency-dependent leakage

        References
        ----------
        Test data is read from file Figure23.txt (shown in Figure 23 of MAT 58)

        """
        f, s21re, s21im, s21db, s21phase = np.loadtxt(
            self.test_dir + "qfactor_data/Figure23.txt", comments="%", unpack=True
        )
        s21 = s21re + 1j * s21im

        # De-embed cables
        N = len(f)
        ncablelen = 1.2  # root_eps * cable length in metres
        D = [
            s21[i]
            * np.exp(complex(0.0, -2e9 * np.pi * f[i] * ncablelen / 2.99792458e8))
            for i in range(N)
        ]

        loop_plan = "fwfwfwc"

        # Find peak in |S21| - this is used to give initial value of freq.
        # Tol is 1.0E-5 * |S21| at peak.
        Mg = np.abs(D)
        index_max = np.argmax(Mg)
        Tol = Mg[index_max] * 1.0e-5
        Fseed = f[index_max]

        # Set Qseed: An order-of-magnitude estimate for Q-factor
        mult = 5.0  # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
        Qseed = mult * Fseed / (f[-1] - f[0])

        quiet = False
        if not quiet:
            print("Initial values for iteration:  Freq=", Fseed, " QL=", Qseed)

        # Step 1: Initial unweighted fit --> solution vector
        Q = rf.Qfactor()
        sv = Q.initial_fit(f, D, N, Fseed, Qseed)
        a_re, a_im, b_re, b_im, QL = sv

        assert_almost_equal(a_re + 1j * a_im, 8.9408 + 2.1298j, decimal=4)
        assert_almost_equal(b_re + 1j * b_im, 0.0054 - 0.0045j, decimal=4)
        assert_almost_equal(QL, 4664.2418, decimal=3)

        # Step 2: Optimised weighted fit --> result vector
        mv, weighting_ratio, number_iterations, RMS_Error = Q.optimise_fit8(
            f, D, N, Fseed, sv, loop_plan, Tol, quiet
        )
        m1, m2, m3, m4, m8, m9, QL, FL = mv

        # Now calculate unloaded Q-factor and some other useful quantities.
        # Reciprocal of |S21| of a thru in place of resonator
        scaling_factor_A = 1 / 0.949  # 1/|S21_thru|
        trmode = "transmission"
        p = Q.Q_unloaded(mv, scaling_factor_A, trmode, quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p

        # Test against expected solutions
        assert_almost_equal(FL, 9.76015571, decimal=6)
        assert_almost_equal(QL, 4760.04, decimal=0)
        assert_almost_equal(Qo, 4789.49, decimal=0)
        assert_almost_equal(weighting_ratio, 5.078, decimal=3)
        assert_almost_equal(RMS_Error, 0.00000910)
        assert_almost_equal(cal_diam, 0.0061, decimal=2)


if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
