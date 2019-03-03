# -*- coding: utf-8 -*-
import unittest
import os
import numpy as npy


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
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'impedanceMismatch,50to25.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.thru(z0=50)**\
            self.dummy_media.thru(z0=25)

        self.assertEqual(qucs_ntwk, skrf_ntwk)

    def test_resistor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'resistor,1ohm.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.resistor(1)
        self.assertEqual(qucs_ntwk, skrf_ntwk)

    def test_capacitor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'capacitor,p01pF.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.capacitor(.01e-12)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


    def test_inductor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'inductor,p1nH.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.inductor(.1e-9)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


    def test_scalar_gamma_z0_media(self):
        '''
        test ability to create a Media from scalar quanties for gamma/z0
        and change frequency resolution
        '''
        a = DefinedGammaZ0 (Frequency(1,10,101),gamma=1j,z0 = 50)
        self.assertEqual(a.line(1),a.line(1))

        # we should be able to re-sample the media
        a.npoints = 21
        self.assertEqual(len(a.gamma), len(a))
        self.assertEqual(len(a.z0), len(a))
        self.assertEqual(len(a.z0), len(a))


    def test_vector_gamma_z0_media(self):
        '''
        test ability to create a Media from vector quanties for gamma/z0
        '''
        freq = Frequency(1,10,101)
        a = DefinedGammaZ0(freq,
                           gamma = 1j*npy.ones(len(freq)) ,
                           z0 =  50*npy.ones(len(freq)),
                            )


        self.assertEqual(a.line(1),a.line(1))
        with self.assertRaises(NotImplementedError):
            a.npoints=4

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
    '''
    Check that S parameters of media base elements versus theoretical results.
    '''
    def setUp(self):
        self.dummy_media = DefinedGammaZ0(
            frequency=Frequency(1, 100, 21, 'GHz'),
            gamma=1j,
            z0=50,
            )

    def test_s_series_element(self):
        '''
        Series elements of impedance Z:

        ○---[Z]---○

        ○---------○

        have S matrix of the form:
        [ Z/Z0 / (Z/Z0 + 2)     2/(Z/Z0 + 2) ]
        [ 2/(Z/Z0 + 2)          Z/Z0 / (Z/Z0 + 2) ]
        '''
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
        '''
        Shunt elements of admittance Y:

        ○---------○
            |
           [Y]
            |
        ○---------○

        have S matrix of the form:
        [ -Y Z0 / (Y Z0 + 2)     2/(Y Z0 + 2) ]
        [ 2/(Y Z0 + 2)          Z/Z0 / (Y Z0 + 2) ]
        '''
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
        '''
        Lossless transmission line of characteristic impedance z1, length l
        and wavenumber beta
              _______
        ○-----       -----○
          z0     z1    z0
        ○-----_______-----○

        '''
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
        '''
        Lossy transmission line of characteristic impedance Z0, length l
        and propagation constant gamma = alpha + j beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cosh(gamma l)       Z0 sinh(gamma l) ]
        [ 1/Z0 sinh(gamma l)  cosh(gamma l) ]
        '''


class ABCDTwoPortsNetworkTestCase(unittest.TestCase):
    '''
    Check that ABCD parameters of media base elements (such as lumped elements)
    versus theoretical results.
    '''
    def setUp(self):
        self.dummy_media = DefinedGammaZ0(
            frequency=Frequency(1, 100, 21,'GHz'),
            gamma=1j,
            z0=50 ,
            )

    def test_abcd_series_element(self):
        '''
        Series elements of impedance Z:

        ○---[Z]---○

        ○---------○

        have ABCD matrix of the form:
        [ 1  Z ]
        [ 0  1 ]
        '''
        R = 1.0  # Ohm
        ntw = self.dummy_media.resistor(R)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], R)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_shunt_element(self):
        '''
        Shunt elements of admittance Y:

        ○---------○
            |
           [Y]
            |
        ○---------○

        have ABCD matrix of the form:
        [ 1  0 ]
        [ Y  1 ]
        '''
        R = 1.0  # Ohm
        ntw = self.dummy_media.shunt(self.dummy_media.resistor(R)**self.dummy_media.short())
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1.0/R)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_series_shunt_elements(self):
        '''
        Series and Shunt elements of impedance Zs and Zp:

        ○---[Zs]--------○
                 |
                [Zp]
                 |
        ○--------------○
        have ABCD matrix of the form:
        [ 1 + Zs/Zp    Zs ]
        [ 1/Zp          1 ]
        '''
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
        '''
        Thru has ABCD matrix of the form:
        [ 1  0 ]
        [ 0  1 ]
        '''
        ntw = self.dummy_media.thru()
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_lossless_line(self):
        '''
        Lossless transmission line of characteristic impedance Z0, length l
        and wavenumber beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cos(beta l)       j Z0 sin(beta l) ]
        [ j/Z0 sin(beta l)  cos(beta l) ]
        '''
        l = 5
        z0 = 80
        ntw = self.dummy_media.line(d=l, unit='m', z0=z0)
        beta = self.dummy_media.beta
        npy.testing.assert_array_almost_equal(ntw.a[:,0,0], npy.cos(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,0,1], 1j*z0*npy.sin(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,0], 1j/z0*npy.sin(beta*l))
        npy.testing.assert_array_almost_equal(ntw.a[:,1,1], npy.cos(beta*l))

    def test_abcd_lossy_line(self):
        '''
        Lossy transmission line of characteristic impedance Z0, length l
        and propagation constant gamma = alpha + j beta

        ○---------○

        ○---------○

        has ABCD matrix of the form:

        [ cosh(gamma l)       Z0 sinh(gamma l) ]
        [ 1/Z0 sinh(gamma l)  cosh(gamma l) ]
        '''
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
