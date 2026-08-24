import os
import unittest

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0

import skrf as rf
from skrf.constants import mil
from skrf.media.rectangularWaveguide import RectangularWaveguide
from skrf.network import Network

# WR-12, and a resistivity of brass
A, B = 3.0988e-3, 1.5494e-3
RHO = 1/(0.28*58e6)

class MediaTestCase(unittest.TestCase):
    """

    """
    def setUp(self):
        """

        """
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
        self.pwd = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))

    def test_line(self):
        """
        Cross-check against a qucs export of the same 200 mil length of WR-10.
        """
        qucs_ntwk = Network(os.path.join(self.files_dir,
                                         'rectangularWaveguideWR10,200mil.s2p'))
        wg = RectangularWaveguide(frequency=qucs_ntwk.frequency, a=100*mil, z0_port=50.)
        skrf_ntwk = wg.line(200*mil, 'm')
        self.assertTrue(np.all(np.abs(qucs_ntwk.s - skrf_ntwk.s) < 0.02))

    def test_conductor_loss(self):
        """
        This only compares the magnitude of the generated line, because
        the loss approximation doesn't account for reactance of field on
        sidewalls.
        """
        ntwk = Network(os.path.join(self.pwd, 'wr1p5_1in_swg_Al_0rough.s2p'))
        wg = RectangularWaveguide(
            frequency = ntwk.frequency,
            a=15*mil,
            z0_override=50,
            rho = 1/(3.8e7),
            )
        self.assertTrue(
            max(abs(wg.line(1,'in').s_mag[:,1,0] - ntwk.s_mag[:,1,0]))<1e-3 )

    def test_roughness(self):
        """
        Roughness now goes through the gradient model of surface_impedance rather
        than the correction factor HFSS applies, so this no longer reproduces the
        HFSS export kept here. It has to stay in its neighbourhood while predicting
        somewhat more loss, as that factor capped at twice the smooth surface
        resistance regardless of the roughness of the wall.
        """
        rough_ref = Network(os.path.join(self.pwd, 'wr1p5_1in_swg_Al_100nm_rough.s2p'))
        smooth_ref = Network(os.path.join(self.pwd, 'wr1p5_1in_swg_Al_0rough.s2p'))
        wg = RectangularWaveguide(
            rough_ref.frequency,
            a=15*mil,
            z0_override=50,
            rho = 1/(3.8e7),
            roughness = 100e-9,
            )
        s21 = wg.line(1,'in').s_mag[:,1,0]
        # lossier than the same guide left smooth
        self.assertTrue(np.all(s21 < smooth_ref.s_mag[:,1,0]))
        # lossier than HFSS, and in the same neighbourhood
        self.assertTrue(np.all(s21 < rough_ref.s_mag[:,1,0]))
        self.assertTrue(np.all(np.abs(s21 - rough_ref.s_mag[:,1,0]) < 0.05))

    def test_lossless_walls_match_field_theory(self):
        """
        With lossless walls both models reduce to sqrt(kc**2 - k0**2), a complex
        ep_r attenuating the mode through that square root alone.
        """
        freq = rf.Frequency(60, 90, 21, unit='GHz')
        for ep_r in (1, 2.1 - 0.02j):
            kw = {'a': A, 'b': B, 'ep_r': ep_r, 'rho': 0}
            wg = RectangularWaveguide(freq, **kw)
            g = np.sqrt(np.asarray(wg.kc**2 - wg.k0**2, dtype=complex))
            np.testing.assert_allclose(wg.gamma, g, rtol=1e-12)
            np.testing.assert_allclose(wg.z0, 1j*freq.w*wg.mu/wg.gamma, rtol=1e-12)
            np.testing.assert_allclose(
                RectangularWaveguide(freq, **kw, model='marcuvitz').gamma, g, rtol=1e-12)

        # only the lossy filling is left to attenuate the mode
        self.assertTrue(np.all(RectangularWaveguide(freq, a=A, b=B, ep_r=2.1 - 0.02j,
                                                    rho=0).gamma.real > 0))

    def test_material_dicts(self):
        """
        The wall and dielectric dicts reproduce rho and ep_r, the walls default to
        copper, and roughness only adds loss.
        """
        freq = rf.Frequency(60, 90, 11, unit='GHz')
        ref = RectangularWaveguide(freq, a=A, b=B, rho=RHO)

        wall = {'sigma': 1/RHO}
        np.testing.assert_allclose(
            RectangularWaveguide(freq, a=A, b=B, wall_a=wall, wall_b=wall).gamma,
            ref.gamma, rtol=1e-12)

        # and the dielectric dict reproduces ep_r
        np.testing.assert_allclose(
            RectangularWaveguide(freq, a=A, b=B, rho=RHO,
                                 dielectric={'ep_r': 2.1}).gamma,
            RectangularWaveguide(freq, a=A, b=B, rho=RHO, ep_r=2.1).gamma, rtol=1e-12)

        # a stack of identical layers has to degenerate to the same bulk conductor
        stack = [wall, wall]
        np.testing.assert_allclose(
            RectangularWaveguide(freq, a=A, b=B, wall_a=stack, wall_b=stack).gamma,
            ref.gamma, rtol=1e-8)

        # the walls default to copper, and rho of 0 is a perfect conductor
        np.testing.assert_allclose(RectangularWaveguide(freq, a=A, b=B).gamma,
                                   RectangularWaveguide(freq, a=A, b=B, rho=1/58e6).gamma,
                                   rtol=1e-15)
        np.testing.assert_allclose(RectangularWaveguide(freq, a=A, b=B, rho=0).alpha_c,
                                   0, atol=0)

        # roughening the walls can only add loss
        rough = {'sigma': 1/RHO, 'rms_roughness': 1e-6}
        rougher = RectangularWaveguide(freq, a=A, b=B, wall_a=rough, wall_b=rough)
        self.assertTrue(np.all(rougher.gamma.real > ref.gamma.real))

        # describing one pair and not the other falls back, and says so
        with pytest.warns(UserWarning, match='wall_b stays a smooth bulk conductor'):
            RectangularWaveguide(freq, a=A, b=B, wall_a={'sigma': 41.1e6})

    def test_marcuvitz_model(self):
        """
        The power loss method reproduces ch. 2, eq. (14a) of Marcuvitz as published,
        and leaves beta at its lossless value where the two-wire model shifts it.
        """
        freq = rf.Frequency(60, 90, 21, unit='GHz')
        f = freq.f
        r = (1/(2*A*np.sqrt(mu_0*epsilon_0))/f)**2
        ch2_eq14a = (np.sqrt(np.pi*f*mu_0*RHO)/(np.sqrt(mu_0/epsilon_0)*B)
                     * (1 + 2*B/A*r)/np.sqrt(1 - r))

        lossless = RectangularWaveguide(freq, a=A, b=B, rho=0)
        iec = RectangularWaveguide(freq, a=A, b=B, rho=RHO, model='marcuvitz')
        default = RectangularWaveguide(freq, a=A, b=B, rho=RHO)

        np.testing.assert_allclose(iec.alpha_c, ch2_eq14a, rtol=1e-12)
        np.testing.assert_allclose(iec.gamma.imag, lossless.gamma.imag, rtol=1e-12)
        np.testing.assert_allclose(iec.gamma.real, iec.alpha_c, rtol=1e-12)
        # the two-wire model shifts beta instead, by parts in ten thousand here,
        # while the two still agree on alpha
        shift = np.abs(default.gamma.imag/lossless.gamma.imag - 1)
        self.assertTrue(np.all(shift > 1e-5))
        np.testing.assert_allclose(default.gamma.real, iec.gamma.real, rtol=1e-2)

    def test_model_falls_back_for_uncovered_modes(self):
        """
        The two-wire model covers TE_m0 and TE_0n only. A mode it does not cover has
        to come out of the power loss method instead, and to say so.
        """
        freq = rf.Frequency(160, 200, 11, unit='GHz')

        def both_models(**mode):
            """the mode as it comes, and the same mode asked for the power loss method"""
            return (RectangularWaveguide(freq, a=A, b=B, rho=RHO, **mode).gamma,
                    RectangularWaveguide(freq, a=A, b=B, rho=RHO, model='marcuvitz',
                                         **mode).gamma)

        # a covered mode is the two-wire model, parting from the other by parts in
        # ten thousand here
        for m, n in [(1, 0), (0, 1)]:
            default, power_loss = both_models(mode_type='te', m=m, n=n)
            self.assertTrue(np.all(np.abs(default/power_loss - 1) > 1e-5), f'TE{m}{n}')

        # one it does not cover falls back to the power loss method exactly
        for mode_type, m, n in [('te', 1, 1), ('tm', 1, 1)]:
            label = f'{mode_type.upper()}{m}{n}'
            with pytest.warns(UserWarning, match=f'does not cover the {label} mode'):
                default, power_loss = both_models(mode_type=mode_type, m=m, n=n)
            # assert_allclose takes NaN for NaN, so rule it out separately
            self.assertTrue(np.all(np.isfinite(default)), label)
            np.testing.assert_allclose(default, power_loss, rtol=1e-12)

    def test_invalid_arguments_raise(self):
        """
        A mode with no field is refused, a TE mode needing one nonzero index and a
        TM mode both, as are an unknown mode type and an unknown model.
        """
        freq = rf.Frequency(60, 90, 11, unit='GHz')
        for mode_type, m, n in [('te', 0, 0), ('tm', 1, 0), ('tm', 0, 2)]:
            with self.assertRaises(ValueError, msg=f'{mode_type}{m}{n}'):
                RectangularWaveguide(freq, a=A, b=B, mode_type=mode_type, m=m, n=n)
        with self.assertRaises(ValueError):
            RectangularWaveguide(freq, a=A, b=B, mode_type='not_a_mode_type')
        with self.assertRaises(ValueError):
            RectangularWaveguide(freq, a=A, b=B, model='not_a_model')

    def test_te01_is_the_rotated_te10(self):
        """A TE01 mode is the TE10 mode of the guide turned on its side."""
        freq = rf.Frequency(160, 200, 11, unit='GHz')
        # two tellable-apart materials, so a wall pair landing on the wrong side of
        # the rotated guide could not pass unnoticed
        rough = {'sigma': 0.28*58e6, 'rms_roughness': 1e-6}
        smooth = {'sigma': 0.50*58e6}
        te01 = RectangularWaveguide(freq, a=A, b=B, mode_type='te', m=0, n=1,
                                    wall_a=rough, wall_b=smooth)
        te10 = RectangularWaveguide(freq, a=B, b=A, mode_type='te', m=1, n=0,
                                    wall_a=smooth, wall_b=rough)
        np.testing.assert_allclose(te01.gamma, te10.gamma, rtol=1e-12)
        np.testing.assert_allclose(te01.z0, te10.z0, rtol=1e-12)

    def test_evanescent_and_dc_stay_finite(self):
        """Below cutoff and at dc the mode is evanescent, and must not go NaN."""
        # TE20 cuts off at 96.7 GHz, so it is evanescent over this band
        below = RectangularWaveguide(rf.Frequency(60, 90, 11, unit='GHz'), a=A, b=B,
                                     mode_type='te', m=2, n=0, rho=RHO)
        self.assertTrue(np.all(np.isfinite(below.gamma)))

        # at dc the lossless mode decays at exactly kc, and lossy walls stay finite
        freq = rf.Frequency(0, 90, 4, unit='GHz')
        lossless = RectangularWaveguide(freq, a=A, b=B, rho=0)
        np.testing.assert_allclose(lossless.gamma[0].real, lossless.kc, rtol=1e-9)
        self.assertTrue(np.all(np.isfinite(
            RectangularWaveguide(freq, a=A, b=B, rho=RHO).gamma)))
