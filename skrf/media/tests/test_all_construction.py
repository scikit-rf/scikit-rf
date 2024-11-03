"""
this test on tests ability for all media class to pass construction
of all general circuit components

"""
import unittest

import numpy as np
from scipy.constants import mil

import skrf as rf
from skrf.media import CPW, CircularWaveguide, Coaxial, DistributedCircuit, Freespace, MLine, RectangularWaveguide


class MediaTestCase:
    """Base class, contains tests for all media."""
    def test_gamma(self):
        self.media.gamma

    def test_z0_value(self):
        self.media.z0

    def test_match(self):
        self.media.match()

    def test_load(self):
        self.media.load(1)

    def test_short(self):
        self.media.short()

    def test_open(self):
        self.media.open()

    def test_capacitor(self):
        self.media.capacitor(1)

    def test_capacitor_q(self):
        self.media.capacitor_q(1, 2, 3)

    def test_inductor(self):
        self.media.inductor(1)

    def test_inductor_q(self):
        self.media.capacitor_q(1, 2, 3)

    def test_impedance_mismatch(self):
        self.media.impedance_mismatch(1, 2)

    def test_tee(self):
        self.media.tee()

    def test_splitter(self):
        self.media.splitter(4)

    def test_thru(self):
        self.media.thru()

    def test_line(self):
        self.media.line(1)

    def test_line_floating(self):
        self.media.line_floating(1)

    def test_delay_load(self):
        self.media.delay_load(1,2)

    def test_delay_short(self):
        self.media.delay_short(1)

    def test_delay_open(self):
        self.media.delay_open(1)

    def test_shunt_delay_load(self):
        self.media.shunt_delay_load(1,1)

    def test_shunt_delay_short(self):
        self.media.shunt_delay_short(1)

    def test_shunt_delay_open(self):
        self.media.shunt_delay_open(1)

    def test_shunt_capacitor(self):
        self.media.shunt_capacitor(1)

    def test_shunt_inductor(self):
        self.media.shunt_inductor(1)

    def test_Z0_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning) as context:
            self.media.Z0

    def test_embed_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning) as context:
            self.media.line(1, unit = 'deg', embed = True)

class Z0InitDeprecationTestCase(unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(1,2,2,'GHz')

    def testZ0InitDeprecation(self):
        # 1-conductor waveguide media
        with self.assertWarns(DeprecationWarning) as context:
            cw = CircularWaveguide(self.frequency, z0 = 50)
            self.assertTrue(np.all(cw.z0 == 50))
        with self.assertWarns(DeprecationWarning) as context:
            rw = RectangularWaveguide(self.frequency, z0 = 50)
            self.assertTrue(np.all(rw.z0 == 50))
        # 2-conductor other media
        with self.assertWarns(DeprecationWarning) as context:
            coax = Coaxial(self.frequency, z0 = 50)
            self.assertTrue(np.all(coax.z0 == 50))
        with self.assertWarns(DeprecationWarning) as context:
            cpw = CPW(self.frequency, z0 = 50)
            self.assertTrue(np.all(cpw.z0 == 50))
        with self.assertWarns(DeprecationWarning) as context:
            air = Freespace(self.frequency, z0 = 50)
            self.assertTrue(np.all(air.z0 == 50))
        with self.assertWarns(DeprecationWarning) as context:
            mlin = MLine(self.frequency, z0 = 50)
            self.assertTrue(np.all(mlin.z0 == 50))


class FreespaceTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'GHz')
        self.media = Freespace(self.frequency)

    def test_z0_value(self):
        self.assertEqual(round(\
            self.media.z0[0].real), 377)


class CPWTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'GHz')
        self.media = CPW(\
            frequency=self.frequency,
            w=10e-6,
            s=5e-6,
            ep_r=11.7,
            t=1e-6,
            rho=22e-9)


class RectangularWaveguideTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'GHz')
        self.media = RectangularWaveguide(\
            frequency=self.frequency,
            a=100*mil,
            )


class DistributedCircuitTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'GHz')
        self.media = DistributedCircuit(\
            frequency=self.frequency,
            L=1,C=1,R=0,G=0
            )


suite = unittest.TestSuite()
loader = unittest.TestLoader()


suite.addTests([\
    loader.loadTestsFromTestCase(FreespaceTestCase),
    loader.loadTestsFromTestCase(CPWTestCase),
    loader.loadTestsFromTestCase(RectangularWaveguideTestCase),
    loader.loadTestsFromTestCase(DistributedCircuitTestCase),
    loader.loadTestsFromTestCase(Z0InitDeprecationTestCase),
    ])

#suite = unittest.TestLoader().loadTestsFromTestCase(FreespaceTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
