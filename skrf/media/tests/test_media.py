import os
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skrf.constants import S_DEF_DEFAULT
from skrf.frequency import Frequency
from skrf.media import DefinedGammaZ0
from skrf.network import Network, concat_ports, connect


class DefinedGammaZ0TestCase(unittest.TestCase):
    def setUp(self):
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
        self.dummy_media = DefinedGammaZ0(
            frequency = Frequency(1, 100, 21, unit='ghz'),
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
        # sNp-files store the same impedance for every port, so to compare the
        # networks, the port impedance has to be set manually to [50,25]
        qucs_ntwk.z0 = [50,25]
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
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.tee(name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_splitter(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        Test mismatched splitter.
        """
        name = 'splitter'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.splitter(3, name = name)
        self.assertEqual(name, skrf_ntwk.name)

        # Check the s-parameters
        s = np.zeros_like(skrf_ntwk.s, dtype='complex')
        y0s = np.array(1./skrf_ntwk.z0)
        y_k = y0s.sum(axis=1)
        s = 2 *np.sqrt(np.einsum('ki,kj->kij', y0s, y0s)) / y_k[:, None, None]
        np.einsum('kii->ki', s)[:] -= 1  # Sii
        assert_array_almost_equal(skrf_ntwk.s, s)

    def test_mismatch_splitter(self):
        """
        Test mismatched splitter against 50-ohm splitter connected to
        equivalent mismatched thrus.
        """
        freq = Frequency(1, 1, 1, unit='GHz')
        med = DefinedGammaZ0(frequency=freq, z0=50)

        for n in range(2, 5):
            z_port = np.arange(1, n + 1) * 20
            split =med.splitter(n, z0=z_port)
            thrus = []
            for k in range(n):
                thrus.append(med.thru(z0 = z_port[k]))

            # connect
            thru = concat_ports(thrus, port_order='second')
            ref = med.splitter(n)
            ref = connect(ref, 0, thru, 0, n)
            assert_array_almost_equal(split.s, ref.s)

    def test_complex_impedance_mismatch_tee(self):
        """
        Test the complex impedance mismatch.
        """
        z0 = (25+25j, 50, 75-25j)
        ref = self.dummy_media.tee()
        ref.renormalize(z_new=z0)
        tee = self.dummy_media.tee(z0=z0)
        assert_array_almost_equal(ref.s, tee.s)
        assert_array_almost_equal(np.linalg.inv(tee.s), tee.s.conj())

    def test_splitter_is_reciprocal_and_unitary(self):
        """
        Test the splitter's s-parameters is reciprocal and unitary matrix for
        different port impedances.
        """
        # A unitary matrix satisfies the property that its inverse is equal to its conjugate transpose.
        # Additionally, a reciprocal matrix satisfies the property that its transpose is equal to itself.
        # Combining these properties, the inverse of the splitter's S-parameters is equal to its conjugate.
        for z0 in ((50, 50, 50), (25, 50, 75), (25+25j, 50, 75-25j)):
            # Test the reciprocal and unitary property for matched, mismatched and complex impedance Tee/splitters.
            tee = self.dummy_media.tee(z0=z0)
            assert_array_almost_equal(np.linalg.inv(tee.s), tee.s.conj())

    def test_thru(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'thru'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.thru(name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_line(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'line'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.line(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_delay_load(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_load'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.delay_load(1j, 90, 'deg',
                                                      name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_shunt_delay_load(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_load'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.shunt_delay_load(1j, 90, 'deg',
                                                      name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_delay_open(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_open'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.delay_open(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_shunt_delay_open(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_open'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.shunt_delay_open(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_delay_short(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'delay_short'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.delay_short(90, 'deg', name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_shunt_delay_short(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_delay_short'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
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
        # test vs analytical ABCD parameters of a series resistor
        Z = 10 - 4j
        ntwk = self.dummy_media.resistor(Z)
        ABCD = np.full(ntwk.s.shape, [[1, -1+1j],
                                       [0, 1]])
        ABCD[:,0,1] = Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_resistor_mismatch(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        The mismatch QUCS component is generated by renormalize() method.
        """
        # Config the qucs_ntwk and media
        name = 'resistor,1ohm'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        mismatch_z0_tuple = ([25, 50], [25-5j, 50+10j], [25, 50-10j])

        # Test against Y-parameter defination
        def resistor_y_def(media: DefinedGammaZ0, R, **kwargs):
            result = media.match(nports=2, **kwargs)
            y = np.zeros(shape=result.s.shape, dtype=complex)
            R = np.array(R)
            y[:, 0, 0] = 1.0 / R
            y[:, 1, 1] = 1.0 / R
            y[:, 0, 1] = -1.0 / R
            y[:, 1, 0] = -1.0 / R
            result.y = y
            return result

        # Test real and imag port impedance with Y-parameter defination and renormalize()
        for mismatch_z0 in mismatch_z0_tuple:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0)
            skrf_ntwk = self.dummy_media.resistor(1, name = name, z0=mismatch_z0)
            skrf_ntwk_y = resistor_y_def(self.dummy_media, 1, z0=mismatch_z0)
            self.assertEqual(qucs_ntwk_copy, skrf_ntwk)
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s)

        # Config the s_def parameters
        mismatch_z0 = [25, 50-10j]
        s_defs = ("power", "pseudo", "traveling")

        # Test different s_def configurations
        for s_def in s_defs:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0, s_def=s_def)
            skrf_ntwk = self.dummy_media.resistor(1, name=name, z0=mismatch_z0, s_def=s_def)
            skrf_ntwk_y = resistor_y_def(self.dummy_media, 1, z0=mismatch_z0, s_def=s_def)
            self.assertEqual(skrf_ntwk, qucs_ntwk_copy)
            self.assertEqual(skrf_ntwk.s_def, qucs_ntwk_copy.s_def)
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s)
            self.assertEqual(skrf_ntwk.s_def, skrf_ntwk_y.s_def)

    def test_shunt_resistor(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_resistor,1ohm'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.resistor(1, name = name)
        self.assertEqual(name, skrf_ntwk.name)
        # test vs analytical ABCD parameters of a shunt resistor
        Z = 10 - 4j
        ntwk = self.dummy_media.shunt_resistor(Z)
        ABCD = np.asarray([[[1, 0], [1/Z, 1]]])
        assert_array_almost_equal(ABCD, ntwk.a)

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
        # test vs analytical ABCD parameters of a series capacitor
        C = 0.1e-12
        ntwk = self.dummy_media.capacitor(C)
        Z = 1/(1j*C*ntwk.frequency.w)
        ABCD = np.full(ntwk.s.shape, [[1, -1+1j],
                                       [0, 1]])
        ABCD[:,0,1] = Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_capacitor_mismatch(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        The mismatch QUCS component is generated by renormalize() method.
        """
        # Config the qucs_ntwk and media
        name = 'capacitor,p01pF'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        mismatch_z0_tuple = ([25, 50], [25-5j, 50+10j], [25, 50-10j])

        # Test against Y-parameter defination
        def capacitor_y_def(media: DefinedGammaZ0, C, **kwargs):
            result = media.match(nports=2, **kwargs)
            w = media.frequency.w
            y = np.zeros(shape=result.s.shape, dtype=complex)
            C = np.array(C)
            y[:, 0, 0] = 1j * w * C
            y[:, 1, 1] = 1j * w * C
            y[:, 0, 1] = -1j * w * C
            y[:, 1, 0] = -1j * w * C
            result.y = y
            return result

        # Test real and imag port impedance with Y-parameter defination and renormalize()
        for mismatch_z0 in mismatch_z0_tuple:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0)
            skrf_ntwk = self.dummy_media.capacitor(.01e-12, name = name, z0=mismatch_z0)
            skrf_ntwk_y = capacitor_y_def(self.dummy_media, .01e-12, z0=mismatch_z0)
            self.assertEqual(qucs_ntwk_copy, skrf_ntwk)
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s)

        # Config the s_def parameters
        mismatch_z0 = [25, 50-10j]
        s_defs = ("power", "pseudo", "traveling")

        # Test different s_def configurations
        for s_def in s_defs:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0, s_def=s_def)
            skrf_ntwk = self.dummy_media.capacitor(.01e-12, name=name, z0=mismatch_z0, s_def=s_def)
            skrf_ntwk_y = capacitor_y_def(self.dummy_media, .01e-12, z0=mismatch_z0, s_def=s_def)
            self.assertEqual(skrf_ntwk, qucs_ntwk_copy)
            self.assertEqual(skrf_ntwk.s_def, qucs_ntwk_copy.s_def)
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s)
            self.assertEqual(skrf_ntwk.s_def, skrf_ntwk_y.s_def)

    def test_shunt_capacitor(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_capacitor,p01pF'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.capacitor(.01e-12, name = name)
        self.assertEqual(name, skrf_ntwk.name)
        # test vs analytical ABCD parameters of a shunt capacitor
        C = 0.1e-12
        ntwk = self.dummy_media.shunt_capacitor(C)
        Z = 1/(1j*C*ntwk.frequency.w)
        ABCD = np.full(ntwk.s.shape, [[1, 0],
                                       [-1+1j, 1]])
        ABCD[:,1,0] = 1/Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_capacitor_q(self):
        """
        Compare the component values against s-parameters generated by ADS.
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'capacitor_q'
        ads_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ads'
            )
        ads_ntwk = Network(os.path.join(ads_path, name + '.s2p'))
        self.dummy_media.frequency = ads_ntwk.frequency
        skrf_ntwk = self.dummy_media.capacitor_q(C=1.0e-12, f_0=1.0e9, q_factor=30.0, name = name)
        self.assertEqual(ads_ntwk, skrf_ntwk)
        self.assertEqual(ads_ntwk.name, skrf_ntwk.name)
        # test vs analytical ABCD parameters of a series capacitor with q_factor
        C = 0.1e-12
        Q = 30.0
        F = 1.0e9
        ntwk = self.dummy_media.capacitor_q(C=C, f_0=F, q_factor=Q)

        Z = 1/(2*np.pi*F*C/Q+1j*C*ntwk.frequency.w)
        ABCD = np.full(ntwk.s.shape, [[1, -1+1j],
                                       [0, 1]])
        ABCD[:,0,1] = Z
        assert_array_almost_equal(ABCD, ntwk.a)

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
        # test vs analytical ABCD parameters of a series inductor
        L = 0.1e-9
        ntwk = self.dummy_media.inductor(L)
        Z = 1j*L*ntwk.frequency.w
        ABCD = np.full(ntwk.s.shape, [[1, -1+1j],
                                       [0, 1]])
        ABCD[:,0,1] = Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_inductor_mismatch(self):
        """
        Compare the component values against s-parameters generated by QUCS.
        The mismatch QUCS component is generated by renormalize() method.
        """
        # Config the qucs_ntwk and media
        name = 'inductor,p1nH'
        qucs_ntwk = Network(os.path.join(self.files_dir, name + '.s2p'))
        self.dummy_media.frequency = qucs_ntwk.frequency
        mismatch_z0_tuple = ([25, 50], [25-5j, 50+10j], [25, 50-10j])

        # Test against Y-parameter defination
        def inductor_y_def(media: DefinedGammaZ0, L, **kwargs):
            result = media.match(nports=2, **kwargs)
            w = media.frequency.w
            y = np.zeros(shape=result.s.shape, dtype=complex)
            L = np.array(L)
            y[:, 0, 0] = 1.0 / (1j * w * L)
            y[:, 1, 1] = 1.0 / (1j * w * L)
            y[:, 0, 1] = -1.0 / (1j * w * L)
            y[:, 1, 0] = -1.0 / (1j * w * L)
            result.y = y
            return result

        # Test real and imag port impedance with Y-parameter defination and renormalize()
        for mismatch_z0 in mismatch_z0_tuple:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0)
            skrf_ntwk = self.dummy_media.inductor(.1e-9, name = name, z0=mismatch_z0)
            skrf_ntwk_y = inductor_y_def(self.dummy_media, .1e-9, z0=mismatch_z0)
            self.assertEqual(qucs_ntwk_copy, skrf_ntwk)
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s)

        # Config the s_def parameters
        mismatch_z0 = [25, 50-10j]
        s_defs = ("power", "pseudo", "traveling")

        # Test different s_def configurations
        for s_def in s_defs:
            qucs_ntwk_copy = qucs_ntwk.copy()
            qucs_ntwk_copy.renormalize(mismatch_z0, s_def=s_def)
            skrf_ntwk = self.dummy_media.inductor(.1e-9, name=name, z0=mismatch_z0, s_def=s_def)
            skrf_ntwk_y = inductor_y_def(self.dummy_media, .1e-9, z0=mismatch_z0, s_def=s_def)
            self.assertEqual(skrf_ntwk, qucs_ntwk_copy)
            self.assertEqual(skrf_ntwk.s_def, qucs_ntwk_copy.s_def)
            # numerical precision is slightly reduced when using pseudo s_def
            assert_array_almost_equal(skrf_ntwk.s, skrf_ntwk_y.s, decimal={'pseudo': 5}.get(s_def, 6))
            self.assertEqual(skrf_ntwk.s_def, skrf_ntwk_y.s_def)

    def test_shunt_inductor(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'shunt_inductor,p1nH'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.inductor(.1e-9, name = name)
        self.assertEqual(name, skrf_ntwk.name)
        # test vs analytical ABCD parameters of a shunt inductor
        L = 0.1e-9
        ntwk = self.dummy_media.shunt_inductor(L)
        Z = 1j*L*ntwk.frequency.w
        ABCD = np.full(ntwk.s.shape, [[1, 0],
                                       [1-1j, 1]])
        ABCD[:,1,0] = 1/Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_inductor_q(self):
        """
        Compare the component values against s-parameters generated by ADS.
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'inductor_q'
        ads_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ads'
            )
        ads_ntwk = Network(os.path.join(ads_path, name + '.s2p'))
        self.dummy_media.frequency = ads_ntwk.frequency
        skrf_ntwk = self.dummy_media.inductor_q(L=1.0e-9, f_0=1.0e9, q_factor=30.0, name = name)
        self.assertEqual(ads_ntwk, skrf_ntwk)
        self.assertEqual(ads_ntwk.name, skrf_ntwk.name)
        # test vs analytical ABCD parameters of a series capacitor with q_factor
        L = 0.1e-9
        Q = 30.0
        F = 1.0e9
        ntwk = self.dummy_media.inductor_q(L=L, f_0=F, q_factor=Q)
        rdc = F*0.05*(2 * np.pi)*L/Q
        w_q = 2*np.pi*F
        rq1 = w_q*L/Q
        rq2 = np.sqrt(rq1**2-rdc**2)
        qt = w_q*L/rq2
        rac = ntwk.frequency.w*L/qt

        Z = np.sqrt(rdc**2+rac**2)+1j*ntwk.frequency.w*L
        ABCD = np.full(ntwk.s.shape, [[1, -1+1j],
                                       [0, 1]])
        ABCD[:,0,1] = Z
        assert_array_almost_equal(ABCD, ntwk.a)

    def test_attenuator(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'attenuator,-10dB'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.attenuator(-10, d = 90, unit = 'deg',
                                                name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_lossless_mismatch(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'lossless_mismatch,-10dB'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.lossless_mismatch(-10, name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_isolator(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'isolator'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.isolator(name = name)
        self.assertEqual(name, skrf_ntwk.name)

    def test_line_floating(self):
        """
        Test the naming of the network. When circuit is used to connect a
        topology of networks, they should have unique names.
        """
        name = 'floating_line'
        self.dummy_media.frequency = Frequency(1, 1, 1, unit='GHz')
        skrf_ntwk = self.dummy_media.line_floating(90, 'deg', name=name)
        self.assertEqual(name, skrf_ntwk.name)

        # Expected to be a 4-port network
        self.assertEqual(skrf_ntwk.nports, 4)

        # zero length case
        s_zero = 1/2 * np.array([[1, 1, 1, -1],
                                 [1, 1, -1, 1],
                                 [1, -1, 1, 1],
                                 [-1, 1, 1, 1]]
                                ).transpose().reshape(-1,4,4)
        for z0 in [50, 1]:
            zero_length = self.dummy_media.line_floating(d=0, z0=z0)
            assert_array_almost_equal(zero_length.s, s_zero)

    def test_line_floating2(self):
        """
        Test against line
        """
        d = 100
        unit = 'mm'
        z0 = 50
        line = self.dummy_media.line(d, unit, z0)
        line_floating = self.dummy_media.line_floating(d, unit, z0)
        gnd = self.dummy_media.short(n_ports=1)
        line_floating = connect(line_floating, 3, gnd, 0)
        line_floating = connect(line_floating, 2, gnd, 0)
        assert_array_almost_equal(line_floating.s, line.s)

    def test_line_floating3(self):
        """
        Test against Y-parameter definition
        """
        d = 100
        unit = 'mm'

        def yparam_ref(media, d, unit, z0, s_def=S_DEF_DEFAULT):
            result = media.match(nports=4, z0=z0, s_def='traveling')

            theta = media.electrical_length(media.to_meters(d=d, unit=unit))

            # From AWR docs on TLINP4.
            y11 = 1 / (z0 * np.tanh(theta))
            y12 = -1 / (z0 * np.sinh(theta))
            y22 = y11
            y21 = y12

            result.y = \
                    np.array([[ y11,  y12, -y11, -y12],
                              [ y21,  y22, -y21, -y22],
                              [-y11, -y12,  y11,  y12],
                              [-y21, -y22,  y21,  y22]]).transpose(2,0,1)
            if media.z0_port is not None:
                result.renormalize(media.z0_port)
            result.renormalize(result.z0, s_def=s_def)
            return result

        for z0 in [50, 1, 100+10j]:
            line = self.dummy_media.line_floating(d, unit, z0)
            ref_line = yparam_ref(self.dummy_media, d, unit, z0)
            assert_array_almost_equal(ref_line.s, line.s)

    def test_scalar_gamma_z0_media(self):
        """
        test ability to create a Media from scalar quantities for gamma/z0
        """
        freq = Frequency(1, 10, 101, unit='GHz')
        a = DefinedGammaZ0(freq, gamma=1j, z0=50)
        self.assertEqual(len(freq), len(a))
        self.assertEqual(len(freq), len(a.gamma))
        self.assertEqual(len(freq), len(a.z0))


    def test_vector_gamma_z0_media(self):
        """
        test ability to create a Media from vector quantities for gamma/z0
        """
        freq = Frequency(1, 10, 101, unit='GHz')
        a = DefinedGammaZ0(freq,
                           gamma = 1j*np.ones(len(freq)) ,
                           z0 =  50*np.ones(len(freq)),
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
            frequency=Frequency(1, 100, 21, unit='GHz'),
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
        assert_array_almost_equal(ntw.s[:,0,0], S11)
        assert_array_almost_equal(ntw.s[:,0,1], S21)
        assert_array_almost_equal(ntw.s[:,1,0], S21)
        assert_array_almost_equal(ntw.s[:,1,1], S11)

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
        assert_array_almost_equal(ntw.s[:,0,0], S11)
        assert_array_almost_equal(ntw.s[:,0,1], S21)
        assert_array_almost_equal(ntw.s[:,1,0], S21)
        assert_array_almost_equal(ntw.s[:,1,1], S11)

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
        S11 = 1j*(_z1**2 - 1)*np.sin(beta*l) / \
            (2*_z1*np.cos(beta*l) + 1j*(_z1**2 + 1)*np.sin(beta*l))
        S21 = 2*_z1 / \
            (2*_z1*np.cos(beta*l) + 1j*(_z1**2 + 1)*np.sin(beta*l))
        assert_array_almost_equal(ntw.s[:,0,0], S11)
        assert_array_almost_equal(ntw.s[:,0,1], S21)
        assert_array_almost_equal(ntw.s[:,1,0], S21)
        assert_array_almost_equal(ntw.s[:,1,1], S11)

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
            frequency=Frequency(1, 100, 21, unit='GHz'),
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
        assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        assert_array_almost_equal(ntw.a[:,0,1], R)
        assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        assert_array_almost_equal(ntw.a[:,1,1], 1.0)

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
        assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        assert_array_almost_equal(ntw.a[:,1,0], 1.0/R)
        assert_array_almost_equal(ntw.a[:,1,1], 1.0)

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

        assert_array_almost_equal(ntw.a[:,0,0], 1.0+Rs/Rp)
        assert_array_almost_equal(ntw.a[:,0,1], Rs)
        assert_array_almost_equal(ntw.a[:,1,0], 1.0/Rp)
        assert_array_almost_equal(ntw.a[:,1,1], 1.0)

    def test_abcd_thru(self):
        """
        Thru has ABCD matrix of the form:
        [ 1  0 ]
        [ 0  1 ]
        """
        ntw = self.dummy_media.thru()
        assert_array_almost_equal(ntw.a[:,0,0], 1.0)
        assert_array_almost_equal(ntw.a[:,0,1], 0.0)
        assert_array_almost_equal(ntw.a[:,1,0], 0.0)
        assert_array_almost_equal(ntw.a[:,1,1], 1.0)

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
        assert_array_almost_equal(ntw.a[:,0,0], np.cos(beta*l))
        assert_array_almost_equal(ntw.a[:,0,1], 1j*z0*np.sin(beta*l))
        assert_array_almost_equal(ntw.a[:,1,0], 1j/z0*np.sin(beta*l))
        assert_array_almost_equal(ntw.a[:,1,1], np.cos(beta*l))

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
            frequency=Frequency(1, 100, 21, unit='GHz'),
            gamma=alpha + 1j*beta,
            z0=z0
            )
        ntw = lossy_media.line(d=l, unit='m', z0=z0)
        gamma = lossy_media.gamma
        assert_array_almost_equal(ntw.a[:,0,0], np.cosh(gamma*l))
        assert_array_almost_equal(ntw.a[:,0,1], z0*np.sinh(gamma*l))
        assert_array_almost_equal(ntw.a[:,1,0], 1.0/z0*np.sinh(gamma*l))
        assert_array_almost_equal(ntw.a[:,1,1], np.cosh(gamma*l))

class DefinedGammaZ0_s_def(unittest.TestCase):
    """Test various media constructs with complex ports and different s_def"""

    def test_complex_ports(self):
        m = DefinedGammaZ0(
            frequency = Frequency(1, 1, 1, unit='ghz'),
            gamma=1j,
            z0_port = 50,
            z0 = 10+20j,
            )
        self.assertTrue(m.z0.imag != 0)

        # Powerwave short is -Z0.conj()/Z0
        short = m.short(z0=m.z0, s_def='power')
        self.assertTrue(short.s != -1)
        # Should be -1 when converted to traveling s_def
        np.testing.assert_allclose(short.s_traveling, -1)

        short = m.short(z0=m.z0, s_def='traveling')
        np.testing.assert_allclose(short.s, -1)

        short = m.short(z0=m.z0, s_def='pseudo')
        np.testing.assert_allclose(short.s, -1)

        # Mismatches agree with real port impedances
        mismatch_traveling = m.impedance_mismatch(z1=10, z2=50, s_def='traveling')
        mismatch_pseudo = m.impedance_mismatch(z1=10, z2=50, s_def='pseudo')
        mismatch_power = m.impedance_mismatch(z1=10, z2=50, s_def='power')
        np.testing.assert_allclose(mismatch_traveling.s, mismatch_pseudo.s)
        np.testing.assert_allclose(mismatch_traveling.s, mismatch_power.s)

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

        np.testing.assert_allclose(thru_traveling.s, mismatch_traveling.s, rtol=1e-3)
        np.testing.assert_allclose(thru_pseudo.s, mismatch_pseudo.s, rtol=1e-3)
        np.testing.assert_allclose(thru_power.s, mismatch_power.s, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
