import unittest
import skrf as rf
from scipy.constants import *


class MediaTestCase():
    """Base class, contains tests for all media."""
    def test_propagation_constant(self):
        self.media.propagation_constant

    def test_characterisitc_impedance_value(self):
        self.media.characteristic_impedance

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

    def test_inductor(self):
        self.media.inductor(1)

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


class FreespaceTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'ghz')
        self.media = rf.media.Freespace(self.frequency)

    def test_characterisitc_impedance_value(self):
        self.assertEqual(round(\
            self.media.characteristic_impedance[0].real), 377)


class CPWTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'ghz')
        self.media = rf.media.CPW(\
            frequency=self.frequency,
            w=10e-6,
            s=5e-6,
            ep_r=11.7,
            t=1e-6,
            rho=22e-9)


class RectangularWaveguideTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'ghz')
        self.media = rf.media.RectangularWaveguide(\
            frequency=self.frequency,
            a=100*mil,
            )


class DistributedCircuitTestCase(MediaTestCase, unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(75,110,101,'ghz')
        self.media = rf.media.DistributedCircuit(\
            frequency=self.frequency,
            I=1,C=1,R=0,G=0
            )


suite = unittest.TestSuite()
loader = unittest.TestLoader()


suite.addTests([\
    loader.loadTestsFromTestCase(FreespaceTestCase),
    loader.loadTestsFromTestCase(CPWTestCase),
    loader.loadTestsFromTestCase(RectangularWaveguideTestCase),
    loader.loadTestsFromTestCase(DistributedCircuitTestCase),
    ])

#suite = unittest.TestLoader().loadTestsFromTestCase(FreespaceTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
