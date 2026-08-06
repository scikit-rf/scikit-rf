import os
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal
from scipy.constants import mu_0
from scipy.special import ive, kve

import skrf as rf
from skrf.mathFunctions import db_2_np, meter_2_feet
from skrf.media import Coaxial


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



    def test_line(self):
        """
        """
        fname = os.path.join(self.files_dir,\
                'coaxial.s2p')
        qucs_ntwk = rf.Network(fname)

        a_media = Coaxial(
            frequency = qucs_ntwk.frequency,
            Dint=1e-3, Dout=3e-3, epsilon_r=2.29,
            tan_delta=4e-4, sigma=1./1.68e-8,
            z0_port = 50.
            )
        skrf_ntwk = a_media.line(200e-3,'m')
        # Equal assertion fails if tan_delta or resistivity are non-zero
        #self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertTrue(
            max(abs(skrf_ntwk.s_mag[:,1,0] - qucs_ntwk.s_mag[:,1,0])) < 1e-3
            )


    def test_init_from_attenuation_VF_units(self):
        """
        Test the attenuation unit conversions in the Coaxial classmethod
        `from_attenuation_VF_units`.
        """
        # create a dummy Coaxial media for various attenuation and test the
        # resulting alpha values (real part of gamma)
        rng = np.random.default_rng()
        frequency = rf.Frequency(rng.random(), unit='GHz', npoints=1)
        _att = rng.random()
        # dB/m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/m')
        assert_almost_equal(coax.gamma.real,  db_2_np(_att))
        # dB/100m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/100m')
        assert_almost_equal(coax.gamma.real,  db_2_np(_att)/100)
        # dB/feet
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/feet')
        assert_almost_equal(coax.gamma.real,  db_2_np(_att)*meter_2_feet())
        # dB/100m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/100feet')
        assert_almost_equal(coax.gamma.real,  db_2_np(_att)/100*meter_2_feet())
        # Neper/m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='Np/m')
        assert_almost_equal(coax.gamma.real,  _att)
        # Neper/feet
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='Np/feet')
        assert_almost_equal(coax.gamma.real,  _att*meter_2_feet())

        with self.assertRaises(ValueError):
            coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=np.array([.1, .2]), unit='Np/feet')
        frequency = rf.Frequency(1., 1.1, unit='GHz', npoints=2)
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=np.array([.1, .2]), unit='Np/feet')
        self.assertEqual( coax.frequency.f.shape, (2,))


    def test_init_from_attenuation_VF_array_att(self):
        """
        Test passing array as attenuation in the Coaxial classmethod
        `from_attenuation_VF_units`.
        """
        # create a Coaxial media for frequency-dependent attenuation and
        # test the resulting alpha values (real part of gamma)
        frequency = rf.Frequency(start=1, stop=2, unit='GHz', npoints=101)
        # k0k1k2 attenuation model
        # values taken for HUBER+SUHNER DATA SHEET Coaxial Cable S_10172_B-1
        # attenuation in dB/m for frequency in GHz
        att = 0 + 0.0826*np.sqrt(frequency.f_scaled) + 0.0129*frequency.f_scaled

        coax = Coaxial.from_attenuation_VF(frequency=frequency, att=att, unit='dB/m')
        # check alpha in gamma
        assert_array_almost_equal(db_2_np(att), coax.gamma.real)

        # if the attenuation array length does not match the frequency,
        # should raise a ValueError
        frequency2 = rf.Frequency(start=1, stop=2, unit='GHz', npoints=10)
        with self.assertRaises(ValueError):
            coax = Coaxial.from_attenuation_VF(frequency=frequency2, att=att)

    def test_R(self):
        freq = rf.Frequency(0, 100, 2, unit='GHz')

        rho = 1e-7
        dint = 0.44e-3
        coax = Coaxial(freq, z0_port = 50, Dint = dint, Dout = 1.0e-3,
                       sigma = 1/rho)

        dc_res = rho / (np.pi * (dint/2)**2)

        # Old R calculation valid only when skin depth is much smaller
        # then inner conductor radius
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            R_simple = coax.Rs/(2*np.pi)*(1/coax.a + 1/coax.b)

            self.assertTrue(abs(1 - coax.R[0]/dc_res) < 1e-2)
            self.assertTrue(abs(1 - coax.R[1]/R_simple[1]) < 1e-2)

    def test_LC(self):
        """Assert that LC = mu*eps_prime."""
        coax = Coaxial(
            frequency = rf.Frequency(1, 10, npoints=100, unit='GHz'),
            Dint=1e-3, Dout=3e-3, epsilon_r=2.29,
            tan_delta=4e-4, sigma=1./1.68e-8,
            z0_port = 50.
            )
        assert_array_almost_equal(coax.L*coax.C, mu_0*coax.epsilon_prime)

    def test_material_dicts_match_scalar_parameters(self):
        """The material dicts reproduce epsilon_r, tan_delta and sigma."""
        freq = rf.Frequency(1, 40, 21, unit='GHz')
        geometry = {'Dint': 1e-3, 'Dout': 3e-3}
        ref = Coaxial(freq, **geometry, epsilon_r=2.1, tan_delta=2e-4, sigma=58e6)
        media = Coaxial(freq, **geometry,
                        dielectric={'ep_r': 2.1*(1 - 2e-4j)},
                        inner_conductor={'sigma': 58e6},
                        outer_conductor={'sigma': 58e6})
        assert_allclose(media.gamma, ref.gamma, rtol=1e-12)
        assert_allclose(media.z0, ref.z0, rtol=1e-12)

    def test_zero_roughness_matches_smooth_conductor(self):
        """A boundary of zero rms roughness leaves the smooth conductor."""
        freq = rf.Frequency(1, 40, 21, unit='GHz')
        kw = {'Dint': 1e-3, 'Dout': 3e-3, 'epsilon_r': 2.1, 'tan_delta': 2e-4}
        ref = Coaxial(freq, **kw, sigma=58e6)

        # a single smooth layer, where the surface impedance has a closed form
        smooth = {'sigma': 58e6, 'rms_roughness': 0}
        media = Coaxial(freq, **kw, inner_conductor=smooth, outer_conductor=smooth)
        assert_allclose(media.gamma, ref.gamma, rtol=1e-12)

        # two identical layers instead go through surface_impedance, which has to
        # degenerate to the same bulk conductor
        stack = [{'sigma': 58e6}, {'sigma': 58e6}]
        media = Coaxial(freq, **kw, inner_conductor=stack, outer_conductor=stack)
        assert_allclose(media.gamma, ref.gamma, rtol=1e-8)

        # roughening the conductors can only add loss
        rough = {'sigma': 58e6, 'rms_roughness': 1e-6}
        media = Coaxial(freq, **kw, inner_conductor=rough, outer_conductor=rough)
        self.assertTrue(np.all(media.gamma.real > ref.gamma.real))

    def test_R_dc_limit(self):
        """At dc the current fills the conductors uniformly."""
        rho, a, b, t = 1.68e-8, 0.5e-3, 1.5e-3, 0.2e-3
        freq = rf.Frequency(0, 10, 3, unit='GHz')
        for model in ('schelkunoff', 'tesche'):
            # a wall of finite thickness carries a dc resistance of its own
            media = Coaxial(freq, Dint=2*a, Dout=2*b, sigma=1/rho, tout=t,
                            model=model)
            assert_allclose(media.R[0],
                            rho/(np.pi*a**2) + rho/(2*np.pi*b*t), rtol=1e-9)

            # an infinitely thick one does not
            media = Coaxial(freq, Dint=2*a, Dout=2*b, sigma=1/rho,
                            model=model)
            assert_allclose(media.R[0], rho/(np.pi*a**2), rtol=1e-9)

            # the internal inductance is dropped at dc, leaving the external one
            assert_allclose(media.L[0], mu_0/(2*np.pi)*np.log(b/a), rtol=1e-12)

    def test_thick_outer_conductor(self):
        """A wall many skin depths thick is the infinitely thick one."""
        freq = rf.Frequency(1, 40, 21, unit='GHz')
        kw = {'Dint': 1e-3, 'Dout': 3e-3, 'epsilon_r': 2.1, 'sigma': 58e6}
        ref = Coaxial(freq, **kw)  # tout defaults to None, an infinitely thick wall

        # the field dies out inside the wall, so the Bessel solution sits at its
        # limit already for a wall of a few skin depths
        assert_allclose(Coaxial(freq, **kw, tout=0.1e-3).gamma, ref.gamma, rtol=1e-12)

        # the equivalent circuit only approaches the limit as the log of the wall
        # thickness, through the internal inductance of the tube
        ref_simple = Coaxial(freq, **kw, model='tesche')
        deviation = [
            np.max(np.abs(Coaxial(freq, **kw, tout=t, model='tesche').gamma
                          - ref_simple.gamma)/np.abs(ref_simple.gamma))
            for t in (0.1e-3, 1e-3, 1e-2)]
        self.assertTrue(deviation[0] < 1e-4)
        self.assertTrue(np.all(np.diff(deviation) < 0))

        # numpy's infinity is the same infinitely thick wall as the default None,
        # and so is a wall whose thickness is merely far beyond any skin depth
        assert_allclose(Coaxial(freq, **kw, tout=np.inf).gamma, ref.gamma, rtol=1e-15)
        for t in (1e10, 1e99, 1e300):
            assert_allclose(Coaxial(freq, **kw, tout=t).gamma, ref.gamma, rtol=1e-15)
            # the equivalent circuit converges to it from below, without overflowing
            assert_allclose(Coaxial(freq, **kw, tout=t, model='tesche').gamma,
                            ref_simple.gamma, rtol=1e-7)

        # the limit itself is eq. (74) of Schelkunoff with the outer radius sent to
        # infinity, a solid rod inside and a half space outside
        a, b, w = kw['Dint']/2, kw['Dout']/2, freq.w
        g, Zs = np.sqrt(1j*w*mu_0*kw['sigma']), np.sqrt(1j*w*mu_0/kw['sigma'])
        assert_allclose(ref.R,
                        (Zs/(2*np.pi*a)*ive(0, g*a)/ive(1, g*a)
                         + Zs/(2*np.pi*b)*kve(0, g*b)/kve(1, g*b)).real, rtol=1e-12)

    def test_perfect_conductor(self):
        """An infinite conductivity is lossless."""
        freq = rf.Frequency(1, 40, 21, unit='GHz')
        kw = {'Dint': 1e-3, 'Dout': 3e-3, 'epsilon_r': 2.1}
        for model in ('schelkunoff', 'tesche'):
            media = Coaxial(freq, **kw, sigma=np.inf, model=model)
            assert_allclose(media.R, 0, atol=0)
            assert_allclose(media.gamma.real, 0, atol=1e-12)
            # without loss the inductance is purely external
            assert_allclose(media.L, mu_0/(2*np.pi)*np.log(3.), rtol=1e-12)

            # a large but finite conductivity approaches the same limit
            media = Coaxial(freq, **kw, sigma=1e30, model=model)
            assert_allclose(media.R, 0, atol=1e-9)
            assert_allclose(media.L, mu_0/(2*np.pi)*np.log(3.), rtol=1e-12)
