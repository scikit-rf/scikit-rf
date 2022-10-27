import unittest
import os
import numpy as npy
from numpy.testing import run_module_suite


from skrf.media import DefinedGammaZ0, Media
from skrf.network import Network
from skrf.frequency import Frequency
import skrf


class DefinedGammaZ0TestCase(unittest.TestCase):
    def setUp(self):
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
        self.dummy_media = DefinedGammaZ0(
            frequency = Frequency(1,100,21,'ghz'),
            gamma=1j,
            z0 = 50 ,
            )

    def test_impedance_mismatch(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'impedanceMismatch,50to25'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.thru(z0=50, name = name)**\
            self.dummy_media.thru(z0=25)

        self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertEqual(qucs_ntwk.name, skrf_ntwk.name)
        
    def test_tee(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'tee'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.tee(name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_splitter(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'splitter'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.splitter(3, name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_thru(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'thru'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.thru(name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_line(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'line'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.line(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_delay_load(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_load'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.delay_load(1j, 90, 'deg',
                                                      name = name)
        self.assertEqual(name, skrf_ntwk.name)
    
    def test_shunt_delay_load(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_load'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.shunt_delay_load(1j, 90, 'deg',
                                                      name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_delay_open(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_open'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.delay_open(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_shunt_delay_open(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_open'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.shunt_delay_open(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_delay_short(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_short'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.delay_short(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_shunt_delay_short(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_short'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.shunt_delay_short(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_resistor(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'resistor,1ohm'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.resistor(1, name = name)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertEqual(qucs_ntwk.name, skrf_ntwk.name)

    def test_capacitor(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'capacitor,p01pF'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.capacitor(.01e-12, name = name)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertEqual(qucs_ntwk.name, skrf_ntwk.name)
        
    def test_shunt_capacitor(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_capacitor,p01pF'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.capacitor(.01e-12, name = name)
        self.assertEqual(name, skrf_ntwk.name)


    def test_inductor(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'inductor,p1nH'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.inductor(.1e-9, name = name)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertEqual(qucs_ntwk.name, skrf_ntwk.name)
        
    def test_shunt_inductor(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_inductor,p1nH'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.inductor(.1e-9, name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_attenuator(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'attenuator,-10dB'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.attenuator(-10, d = 90, unit = 'deg',
                                                name = name)
        self.assertEqual(name, skrf_ntwk.name)
        
    def test_lossless_mismatch(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'lossless_mismatch,-10dB'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.lossless_mismatch(-10, name = name)
        self.assertEqual(name, skrf_ntwk.name)
    
    def test_isolator(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'isolator'
        self.dummy_media.frequency = Frequency(1, 1, 1, 'GHz')
        skrf_ntwk = self.dummy_media.isolator(name = name)
        self.assertEqual(name, skrf_ntwk.name)


    def test_scalar_gamma_z0_media(self):
        """
        test ability to create a Media from scalar quantities for gamma/z0
        """
        freq = Frequency(1,10,101)
        a = DefinedGammaZ0(freq, gamma=1j, z0=50)
        self.assertEqual(len(freq), len(a))
        self.assertEqual(len(freq), len(a.gamma))
        self.assertEqual(len(freq), len(a.z0))


    def test_vector_gamma_z0_media(self):
        """
        test ability to create a Media from vector quantities for gamma/z0
        """
        freq = Frequency(1,10,101)
        a = DefinedGammaZ0(freq,
                           gamma = 1j*npy.ones(len(freq)) ,
                           z0 =  50*npy.ones(len(freq)),
                            )
        self.assertEqual(len(freq), len(a))
        self.assertEqual(len(freq), len(a.gamma))
        self.assertEqual(len(freq), len(a.z0))
        
    def test_write_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        os.remove(fname)


    def test_from_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        a_media = DefinedGammaZ0.from_csv(fname)
        self.assertEqual(a_media,self.dummy_media)
        os.remove(fname)


class STwoPortsNetworkTestCase(unittest.TestCase):
    """
    Check that S parameters of media base elements versus theoretical results.
    """
    def setUp(self):
        self.dummy_media = DefinedGammaZ0(
            frequency=Frequency(1, 100, 21, 'GHz'),
            gamma=1j,
            z0=50,
            )

    def test_s_series_element(self):
        """
        Series elements of impedance Z:

        ○---[Z]---○

        ○---------○

        have S matrix of the form:
        [ Z/Z0 / (Z/Z0 + 2)     2/(Z/Z0 + 2) ]
        [ 2/(Z/Z0 + 2)          Z/Z0 / (Z/Z0 + 2) ]
        """
        R = 1.0  # Ohm
        ntw = self.dummy_media.resistor(R)
        Z0 = self.dummy_media.z0
        S11 = (R/Z0) / (R/Z0 + 2)
        S21 = 2 / (R/Z0 + 2)
        npy.testing.assert_array_almost_equal(ntw.s[:,0,0], S11)
        npy.testing.assert_array_almost_equal(ntw.s[:,0,1], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,0], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,1], S11)

    def test_s_shunt_element(self):
        """
        Shunt elements of admittance Y:

        ○---------○
            |
           [Y]
            |
        ○---------○

        have S matrix of the form:
        [ -Y Z0 / (Y Z0 + 2)     2/(Y Z0 + 2) ]
        [ 2/(Y Z0 + 2)          Z/Z0 / (Y Z0 + 2) ]
        """
        R = 1.0  # Ohm
        ntw = self.dummy_media.shunt(self.dummy_media.resistor(R)**self.dummy_media.short())
        Z0 = self.dummy_media.z0
        S11 = -(1/R*Z0) / (1/R*Z0 + 2)
        S21 = 2 / (1/R*Z0 + 2)
        npy.testing.assert_array_almost_equal(ntw.s[:,0,0], S11)
        npy.testing.assert_array_almost_equal(ntw.s[:,0,1], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,0], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,1], S11)

    def test_s_lossless_line(self):
        """
        Lossless transmission line of characteristic impedance z1, length l
        and wavenumber beta
              _______
        ○-----       -----○
          z0     z1    z0
        ○-----_______-----○

        """
        l = 5.0
        z1 = 30.0
        z0 = self.dummy_media.z0

        ntw = self.dummy_media.line(d=0, unit='m', z0=z0) \
            ** self.dummy_media.line(d=l, unit='m', z0=z1) \
            ** self.dummy_media.line(d=0, unit='m', z0=z0)

        beta = self.dummy_media.beta
        _z1 = z1/z0
        S11 = 1j*(_z1**2 - 1)*npy.sin(beta*l) / \
            (2*_z1*npy.cos(beta*l) + 1j*(_z1**2 + 1)*npy.sin(beta*l))
        S21 = 2*_z1 / \
            (2*_z1*npy.cos(beta*l) + 1j*(_z1**2 + 1)*npy.sin(beta*l))
        npy.testing.assert_array_almost_equal(ntw.s[:,0,0], S11)
        npy.testing.assert_array_almost_equal(ntw.s[:,0,1], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,0], S21)
        npy.testing.assert_array_almost_equal(ntw.s[:,1,1], S11)

    def test_s_lossy_line(self):
        """
        Lossy transmission line of characteristic impedance Z0, length l
        and propagation constant gamma = alpha + j beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cosh(gamma l)       Z0 sinh(gamma l) ]
        [ 1/Z0 sinh(gamma l)  cosh(gamma l) ]
        """


class ABCDTwoPortsNetworkTestCase(unittest.TestCase):
    """
    Check that ABCD parameters of media base elements (such as lumped elements)
    versus theoretical results.
    """
    def setUp(self):
        self.dummy_media = DefinedGammaZ0(
            frequency=Frequency(1, 100, 21,'GHz'),
            gamma=1j,
            z0=50 ,
            )

    def test_abcd_series_element(self):
        """
        Series elements of impedance Z:

        ○---[Z]---○

        ○---------○

        have ABCD matrix of the form:
        [ 1  Z ]
        [ 0  1 ]
        """
        R = 1.0  # Ohm
        ntw = self.dummy_media.resistor(R)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], R)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_shunt_element(self):
        """
        Shunt elements of admittance Y:

        ○---------○
            |
           [Y]
            |
        ○---------○

        have ABCD matrix of the form:
        [ 1  0 ]
        [ Y  1 ]
        """
        R = 1.0  # Ohm
        ntw = self.dummy_media.shunt(self.dummy_media.resistor(R)**self.dummy_media.short())
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1.0/R)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_series_shunt_elements(self):
        """
        Series and Shunt elements of impedance Zs and Zp:

        ○---[Zs]--------○
                 |
                [Zp]
                 |
        ○--------------○
        have ABCD matrix of the form:
        [ 1 + Zs/Zp    Zs ]
        [ 1/Zp          1 ]
        """
        Rs = 2.0
        Rp = 3.0
        serie_resistor = self.dummy_media.resistor(Rs)
        shunt_resistor = self.dummy_media.shunt(self.dummy_media.resistor(Rp) ** self.dummy_media.short())

        ntw = serie_resistor ** shunt_resistor

        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0+Rs/Rp)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], Rs)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1.0/Rp)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_thru(self):
        """
        Thru has ABCD matrix of the form:
        [ 1  0 ]
        [ 0  1 ]
        """
        ntw = self.dummy_media.thru()
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_lossless_line(self):
        """
        Lossless transmission line of characteristic impedance Z0, length l
        and wavenumber beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cos(beta l)       j Z0 sin(beta l) ]
        [ j/Z0 sin(beta l)  cos(beta l) ]
        """
        l = 5
        z0 = 80
        ntw = self.dummy_media.line(d=l, unit='m', z0=z0)
        beta = self.dummy_media.beta
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], npy.cos(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 1j*z0*npy.sin(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1j/z0*npy.sin(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], npy.cos(beta*l))

    def test_abcd_lossy_line(self):
        """
        Lossy transmission line of characteristic impedance Z0, length l
        and propagation constant gamma = alpha + j beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cosh(gamma l)       Z0 sinh(gamma l) ]
        [ 1/Z0 sinh(gamma l)  cosh(gamma l) ]
        """
        l = 5.0
        z0 = 30.0
        alpha = 0.5
        beta = 2.0
        lossy_media = DefinedGammaZ0(
            frequency=Frequency(1, 100, 21, 'GHz'),
            gamma=alpha + 1j*beta,
            z0=z0
            )
        ntw = lossy_media.line(d=l, unit='m', z0=z0)
        gamma = lossy_media.gamma
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], npy.cosh(gamma*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], z0*npy.sinh(gamma*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1.0/z0*npy.sinh(gamma*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], npy.cosh(gamma*l))

class DefinedGammaZ0_s_def(unittest.TestCase):
    """Test various media constructs with complex ports and different s_def"""

    def test_complex_ports(self):
        m = DefinedGammaZ0(
            frequency = Frequency(1,1,1,'ghz'),
            gamma=1j,
            z0 = 50,
            Z0 = 10+20j,
            )
        self.assertTrue(m.Z0.imag != 0)

        # Powerwave short is -Z0.conj()/Z0
        short = m.short(z0=m.Z0, s_def='power')
        self.assertTrue(short.s != -1)
        # Should be -1 when converted to traveling s_def
        npy.testing.assert_allclose(short.s_traveling, -1)

        short = m.short(z0=m.Z0, s_def='traveling')
        npy.testing.assert_allclose(short.s, -1)

        short = m.short(z0=m.Z0, s_def='pseudo')
        npy.testing.assert_allclose(short.s, -1)

        # Mismatches agree with real port impedances
        mismatch_traveling = m.impedance_mismatch(z1=10, z2=50, s_def='traveling')
        mismatch_pseudo = m.impedance_mismatch(z1=10, z2=50, s_def='pseudo')
        mismatch_power = m.impedance_mismatch(z1=10, z2=50, s_def='power')
        npy.testing.assert_allclose(mismatch_traveling.s, mismatch_pseudo.s)
        npy.testing.assert_allclose(mismatch_traveling.s, mismatch_power.s)

        mismatch_traveling = m.impedance_mismatch(z1=10+10j, z2=50-20j, s_def='traveling')
        mismatch_pseudo = m.impedance_mismatch(z1=10+10j, z2=50-20j, s_def='pseudo')
        mismatch_power = m.impedance_mismatch(z1=10+10j, z2=50-20j, s_def='power')

        # Converting thru to new impedance should give impedance mismatch.
        # The problem is that thru Z-parameters have infinities
        # and renormalization goes through Z-parameters making
        # it very inaccurate.
        thru_traveling = m.thru(s_def='traveling')
        thru_traveling.renormalize(z_new=[10+10j,50-20j])
        thru_pseudo = m.thru(s_def='pseudo')
        thru_pseudo.renormalize(z_new=[10+10j,50-20j])
        thru_power = m.thru(s_def='power')
        thru_power.renormalize(z_new=[10+10j,50-20j])

        npy.testing.assert_allclose(thru_traveling.s, mismatch_traveling.s, rtol=1e-3)
        npy.testing.assert_allclose(thru_pseudo.s, mismatch_pseudo.s, rtol=1e-3)
        npy.testing.assert_allclose(thru_power.s, mismatch_power.s, rtol=1e-3)

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
