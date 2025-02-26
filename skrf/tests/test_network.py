import io
import os
import pickle
import sys
import tempfile
import unittest
import warnings
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import skrf as rf
from skrf import setup_pylab
from skrf.constants import S_DEF_HFSS_DEFAULT, S_DEFINITIONS
from skrf.frequency import Frequency, InvalidFrequencyWarning
from skrf.media import CPW, DefinedGammaZ0, DistributedCircuit
from skrf.networkSet import tuner_constellation

try:
    from skrf.plotting import plot_contour
except ImportError:
    pass

class NetworkTestCase(unittest.TestCase):
    """
    Network class operation test case.
    The following is true, as tested by lihan in ADS,
        test3 == test1 ** test2

    To test for 2N-port deembeding Meas, Fix and DUT are created such as:
    ::
        Meas == Fix ** DUT
            Meas             Fix           DUT
         +---------+     +---------+   +---------+
        -|0       4|-   -|0       4|---|0       4|-
        -|1       5|- = -|1       5|---|1       5|-
        -|2       6|-   -|2       6|---|2       6|-
        -|3       7|-   -|3       7|---|3       7|-
         +---------+     +---------+   +---------+

    Note:
    -----
    due to the complexity of inv computations, there will be an unavoidable
    precision loss. thus Fix.inv ** Meas will show a small difference with DUT.
    """
    def setUp(self):
        """
        this also tests the ability to read touchstone files
        without an error
        """
        setup_pylab()
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.freq = rf.Frequency(75,110,101,'ghz')
        self.cpw =  CPW(self.freq, w=10e-6, s=5e-6, ep_r=10.6)
        l1 = self.cpw.line(0.20, 'm', z0=50)
        l2 = self.cpw.line(0.07, 'm', z0=50)
        l3 = self.cpw.line(0.47, 'm', z0=50)
        self.l2 = l2
        freq = Frequency(0, 9, 10, 'GHz')
        m50 = DefinedGammaZ0(frequency = freq, z0_port = 50, z0 = 50)
        self.o1 = m50.open()
        self.splitter = m50.splitter(nports = 3, z0 = [10, 20, 30])
        self.splitter.port_names = ["a", "b", "c"]
        self.thru = rf.concat_ports([m50.thru()] * 2, port_order='second')
        self.thru.renormalize([1, 2, 3, 4])
        self.Fix = rf.concat_ports([l1, l1, l1, l1])
        self.DUT = rf.concat_ports([l2, l2, l2, l2])
        self.Meas = rf.concat_ports([l3, l3, l3, l3])
        self.Fix2 = rf.concat_ports([l1, l1, l1, l1], port_order='first')
        self.DUT2 = rf.concat_ports([l2, l2, l2, l2], port_order='first')
        self.Meas2 = rf.concat_ports([l3, l3, l3, l3], port_order='first')
        self.fet = rf.Network(os.path.join(self.test_dir, 'fet.s2p'))
        self.rng = np.random.default_rng()
        self.ntwk_noise = rf.Network(os.path.join(self.test_dir,'ntwk_noise.s2p'))

    def test_network_copy(self):
        n = self.ntwk1
        n._ext_attrs['_is_circuit_port'] = True
        n2 = n.copy()
        self.assertEqual( n.frequency, n2.frequency)
        self.assertNotEqual( id(n.frequency), id(n2.frequency))
        self.assertNotEqual( id(n.frequency.f), id(n2.frequency.f))

        n.frequency.f[0] = 0
        self.assertNotEqual(n2.frequency.f[0], 0)

        self.assertEqual(True, n2._ext_attrs.get('_is_circuit_port', False))

    def test_two_port_reflect(self):
        number_of_data_points = 10
        f = rf.Frequency.from_f(np.linspace(2e6, 3e6, number_of_data_points), unit="Hz")
        n=rf.Network(frequency=f, s=np.linspace(0.1, .8, number_of_data_points), name='test')
        n2 = rf.two_port_reflect(n, n)
        self.assertEqual(n2.name, n.name + '-' + n.name )
        self.assertEqual(n2.s.shape, (number_of_data_points, 2, 2))
        np.testing.assert_array_equal(n2.s[:, 0, 1], np.zeros(number_of_data_points))
        np.testing.assert_array_equal(n2.s[:, 0, 0], n.s.flatten())

        n2 = rf.two_port_reflect(n, n, name = 'new_name')
        self.assertEqual(n2.name, 'new_name' )

    def test_network_empty_frequency_range(self):
        number_of_data_points = 10
        f = rf.Frequency.from_f(np.linspace(2e6, 3e6, number_of_data_points), unit="Hz")
        n=rf.Network(
            frequency=f,
            s=np.linspace(0.1, .8, number_of_data_points),
            name='test',
            z0=np.linspace(50, 50.1,number_of_data_points ))
        empty_network = n[n.f < 0]
        self.assertIn('1-Port Network', repr(empty_network))

    def test_network_sequence_frequency_with_f_unit(self):
        n=rf.Network(frequency=self.freq.f, f_unit=self.freq.unit)
        np.allclose(n.f, self.freq.f)
        n=rf.Network(f=self.freq.f, f_unit=self.freq.unit)
        np.allclose(n.f, self.freq.f)

    def test_timedomain(self):
        t = self.ntwk1.s11.s_time
        s = self.ntwk1.s11.s
        self.assertTrue(len(t)== len(s))

    def test_time_gate(self):
        ntwk = self.ntwk1
        gated = self.ntwk1.s11.time_gate(0,.2, t_unit='ns')
        self.assertTrue(len(gated)== len(ntwk))

    def test_time_gate_custom_window(self):
        for window in ["hamming", ('kaiser', 6)]:
            gated1 = self.ntwk1.s11.time_gate(0,.2, t_unit='ns', window=window, fft_window=window)

            get_window = partial(signal.get_window, window)

            gated2 = self.ntwk1.s11.time_gate(0,.2, t_unit='ns', window=get_window, fft_window=get_window)
            assert gated1 == gated2

    def test_time_gate_raises(self):
        ntwk = self.ntwk1
        with pytest.warns(DeprecationWarning, match="Time unit not passed"):
            gated = self.ntwk1.s11.time_gate(0,.2)

        self.assertTrue(len(gated)== len(ntwk))

        with pytest.warns(DeprecationWarning, match="Time unit not passed"):
            gated = self.ntwk1.s11.time_gate(0,.2, t_unit='')
        self.assertTrue(len(gated)== len(ntwk))

    def test_autogate(self):
        l1 = self.cpw.line(0.1, 'm', z0=50)
        l2 = self.cpw.line(0.1, 'm', z0=30)
        l3 = self.cpw.line(0.1, 'm', z0=50)

        ntwk = l1 ** l2 ** l3

        # Auto gate should not raise
        gated = ntwk.s11.time_gate()
        self.assertTrue(len(gated)== len(ntwk))

    def test_lpi(self):
        """Test low pass impulse response against data generated with METAS VNA Tools."""

        path = Path(self.test_dir) / "metas_tdr"

        for fname in ["short_10ps_dc_50g", "short_10ps_dc_40g"]:

            netw = rf.Network(path / f"{fname}.s1p")
            ref = np.loadtxt(path / f"{fname}_low_pass_impulse.csv", skiprows=1, delimiter=";")
            t, y = netw.impulse_response(window="boxcar", pad=0, squeeze=True)

            np.testing.assert_allclose(ref[:,0], t * 1e12, rtol=2e-5)
            np.testing.assert_allclose(ref[:,1], y, rtol=5e-5)

    def test_lps(self):
        """Test low pass step response against data generated with METAS VNA Tools."""
        path = Path(self.test_dir) / "metas_tdr"

        for fname in ["short_10ps_dc_50g", "short_10ps_dc_40g"]:
            netw = rf.Network(path / f"{fname}.s1p")
            ref = np.loadtxt(path / f"{fname}_low_pass_step.csv", skiprows=1, delimiter=";")
            t, y = netw.step_response(window="boxcar", pad=0, squeeze=True)

            np.testing.assert_allclose(ref[:, 0], t * 1e12, rtol=2e-5)
            np.testing.assert_allclose(ref[:, 1], y, rtol=5e-5)

    def test_bpi(self):
        """Test band pass impulse response against data generated with METAS VNA Tools."""
        path = Path(self.test_dir) / "metas_tdr"
        for window in ["boxcar", None]:
            # Check if window=None equals to window="boxcar"
            for fname in ["short_10ps_dc_50g", "short_10ps_dc_40g", "short_10ps_10g_50g", "short_10ps_10g_40g"]:

                netw = rf.Network(path / f"{fname}.s1p")
                ref = np.loadtxt(path / f"{fname}_band_pass_impulse.csv", skiprows=1, delimiter=";")
                t, y = netw.impulse_response(window=window, pad=0, squeeze=True, bandpass=True)

                np.testing.assert_allclose(ref[:,0], t * 1e12, rtol=2e-5)
                np.testing.assert_allclose(ref[:,1], np.abs(y), atol=1e-5)

    def test_auto_use_bandpass(self):
        path = Path(self.test_dir) / "metas_tdr"

        for fname in ["short_10ps_dc_50g", "short_10ps_10g_50g"]:

            netw = rf.Network(path / f"{fname}.s1p")
            t, _y = netw.impulse_response(window="boxcar", pad=0, squeeze=True)
            if netw.frequency.start == 0:
                assert len(t) == 2 * len(netw) - 1
            else:
                assert len(t) == len(netw)



    def test_time_transform_v2(self):
        spb = (4, 5)
        data_rate = 5e9
        num_taps = (100, 101)
        for i in range(2):
            tps = 1. / spb[i] / data_rate
            num_points = spb[i] * num_taps[i]
            # Frequency terms should NOT contain Nyquist frequency if number of points is odd
            inc_nyq = True if num_points % 2 == 0 else False
            freq = np.linspace(0, 1. / 2 / tps, num_points // 2 + 1, endpoint=inc_nyq)

            dut = self.ntwk1.copy()
            freq_valid = freq[np.logical_and(freq >= dut.f[0], freq <= dut.f[-1])]
            dut.interpolate_self(rf.Frequency.from_f(freq_valid, unit='hz'))

            dut_dc = dut.extrapolate_to_dc()
            t, y = dut_dc.s21.impulse_response(n=num_points)
            self.assertEqual(len(t), num_points)
            self.assertEqual(len(y), num_points)
            self.assertTrue(np.isclose(t[1] - t[0], tps))
            t, y = dut_dc.s21.step_response(n=num_points)
            self.assertEqual(len(t), num_points)
            self.assertEqual(len(y), num_points)
            self.assertTrue(np.isclose(t[1] - t[0], tps))

    def test_impulse_response_dirac(self):
        """
        Test if the impulse response of a perfect transmission line is pure Dirac
        """
        f_points = 10
        freq = rf.Frequency.from_f(np.arange(f_points), unit='Hz')
        s = np.ones(10)
        netw = rf.Network(frequency=freq, s=s)

        t, y = netw.impulse_response("boxcar", pad=0)

        y_true = np.zeros_like(y)
        y_true[t == 0] = 1
        np.testing.assert_almost_equal(y, y_true)

    def test_time_transform_nonlinear_f(self):
        netw_nonlinear_f = rf.Network(os.path.join(self.test_dir, 'ntwk_arbitrary_frequency.s2p'))
        with self.assertRaises(NotImplementedError):
            netw_nonlinear_f.s11.step_response()

    def test_time_transform(self):
        with self.assertWarns(RuntimeWarning):
            self.ntwk1.s11.step_response()

    def test_time_transform_multiport(self):
        dut_dc = self.ntwk1.extrapolate_to_dc()

        y1 = np.zeros((1000, dut_dc.nports, dut_dc.nports))

        for (i,j) in dut_dc.port_tuples:
            oneport = getattr(dut_dc, f's{i+1}{j+1}')
            t1, y1[:,i,j] = oneport.step_response(n=1000)

        t2, y2 = dut_dc.step_response(n=1000)

        np.testing.assert_almost_equal(t1, t2)
        np.testing.assert_almost_equal(y1, y2)

    def test_time_transform_squeeze(self):
        dut_dc = self.ntwk1.extrapolate_to_dc()
        assert dut_dc.s11.step_response()[1].ndim == 1
        assert dut_dc.s11.step_response(squeeze=False)[1].ndim == 3
        assert dut_dc.step_response()[1].ndim == 3
        assert dut_dc.step_response(squeeze=False)[1].ndim == 3

    def test_constructor_empty(self):
        rf.Network()

    def test_constructor_from_values(self):
        rf.Network(f=[1,2],s=[1,2],z0=[1,2] )

    def test_constructor_from_touchstone(self):
        rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))

    def test_constructor_from_touchstone_special_encoding(self):
        " Test creating Network from Touchstone file with various file encodings."
        filename_latin1 = os.path.join(self.test_dir, '../io/tests/test_encoding_ISO-8859-1.s2p')
        filename_utf8 = os.path.join(self.test_dir, '../io/tests/test_encoding_UTF-8-SIG.s2p')

        ntwk1 = rf.Network(filename_latin1)
        ntwk1_ = rf.Network(filename_latin1, encoding='latin_1')
        self.assertEqual(ntwk1.comments, ntwk1_.comments)

        ntwk2 = rf.Network(filename_utf8, encoding='utf_8_sig')
        self.assertEqual(ntwk1.comments, ntwk2.comments)

    def test_constructor_from_hfss_touchstone(self):
        # HFSS can provide the port characteristic impedances in its generated touchstone file.
        # Check if reading a HFSS touchstone file with non-50Ohm impedances
        ntwk_hfss = rf.Network(os.path.join(self.test_dir, 'hfss_threeport_DB.s3p'))
        self.assertFalse(np.isclose(ntwk_hfss.z0[0,0], 50))

    def test_constructor_from_pathlib(self):
        rf.Network(Path(self.test_dir) / 'ntwk1.ntwk')

    def test_constructor_from_pickle(self):
        rf.Network(os.path.join(self.test_dir, 'ntwk1.ntwk'))

    def test_constructor_from_fid_touchstone(self):
        filename= os.path.join(self.test_dir, 'ntwk1.s2p')
        with open(filename,'rb') as fid:
            rf.Network(fid)
        with open(filename) as fid:
            rf.Network(fid)

    def test_constructor_from_stringio(self):
        filename= os.path.join(self.test_dir, 'ntwk1.s2p')
        with open(filename) as fid:
            data = fid.read()
            sio = io.StringIO(data)
            sio.name = os.path.basename(filename) # hack a bug to touchstone reader
            rf.Network(sio)

    def test_constructor_from_stringio_hfss(self):
        filename = os.path.join(self.test_dir, 'hfss_oneport.s1p')
        with open(filename) as fid:
            data = fid.read()
            sio = io.StringIO(data)
            sio.name = os.path.basename(filename) # hack a bug to touchstone reader
            rf.Network(sio)

    def test_constructor_from_stringio_name_kwawrg(self):
        filename = os.path.join(self.test_dir, 'ntwk1.s2p')
        with open(filename) as fid:
            data = fid.read()
            sio = io.StringIO(data)
            rf.Network(sio, name=filename)

    def test_different_ext(self):
        filename= os.path.join(self.test_dir, 'ntwk1.s2p')
        for par in ["g", "h", "s", "y", "z"]:
            with open(filename) as fid:
                data = fid.read()
                sio = io.StringIO(data)
                sio.name = f"test.{par}2p"
                rf.Network(sio)

    def test_constructor_from_parameters(self):
        """Test creating Network from all supported parameters
        with default z0 and specified z0.
        """
        random_z0 = self.rng.uniform(0.1, 1, (1, 2)) + \
                1j*self.rng.uniform(0.1, 1, (1, 2))
        for z0 in [None, random_z0]:
            for param in rf.Network.PRIMARY_PROPERTIES:
                params = self.rng.uniform(0.1, 1, (1, 2, 2)) + \
                        1j*self.rng.uniform(0.1, 1, (1, 2, 2))
                kwargs = {param:params}
                if z0 is not None:
                    kwargs['z0'] = z0
                net = rf.Network(**kwargs)
                if z0 is not None:
                    np.testing.assert_allclose(net.z0, z0)
                np.testing.assert_allclose(getattr(net, param), params)

    def test_constructor_from_parameters2(self):
        """Test creating Network from all supported parameters
        with default z0 and specified z0.
        Multiple frequency points and z0 broadcasted
        """
        random_z0 = self.rng.uniform(0.1, 1, 2) + \
                1j*self.rng.uniform(0.1, 1, 2)
        for z0 in [None, random_z0]:
            for param in rf.Network.PRIMARY_PROPERTIES:
                params = self.rng.uniform(0.1, 1, (5, 2, 2)) + \
                        1j*self.rng.uniform(0.1, 1, (5, 2, 2))
                kwargs = {param:params}
                if z0 is not None:
                    kwargs['z0'] = z0
                net = rf.Network(**kwargs)
                if z0 is not None:
                    # Network z0 is broadcasted
                    np.testing.assert_allclose(net.z0[0,:], z0)
                np.testing.assert_allclose(getattr(net, param), params)

    def test_constructor_invalid_networks(self):
        with pytest.raises(Exception) as e_info:
            # z0 size doesn't match
            rf.Network(s=np.zeros((2,2,2)), z0=[1,2,3])
        with pytest.raises(Exception) as e_info:
            # z0 size doesn't match, Z-parameters
            rf.Network(z=np.zeros((2,2,2)), z0=[1,2,3])
        with pytest.raises(Exception) as e_info:
            # Invalid s shape, non-square matrix
            rf.Network(s=np.zeros((2,2,1)))
        with pytest.raises(Exception) as e_info:
            # invalid s shape, too many dimensions
            rf.network(s=np.zeros((1,2,2,2)))
        with pytest.raises(Exception) as e_info:
            # Multiple input parameters
            rf.network(s=1, z=1)

    def test_zipped_touchstone(self):
        zippath = os.path.join(self.test_dir, 'ntwks.zip')
        fname = 'ntwk1.s2p'
        rf.Network.zipped_touchstone(fname, zipfile.ZipFile(zippath))

    def test_open_saved_touchstone(self):
        self.ntwk1.write_touchstone('ntwk1Saved',dir=self.test_dir)
        ntwk1Saved = rf.Network(os.path.join(self.test_dir, 'ntwk1Saved.s2p'))
        self.assertEqual(self.ntwk1, ntwk1Saved)

        # Test that it still works with Pathlib objects
        self.ntwk1.write_touchstone(Path('ntwk1Saved'),dir=Path(self.test_dir))
        ntwk1Saved = rf.Network(Path(os.path.join(self.test_dir, 'ntwk1Saved.s2p')))
        self.assertEqual(self.ntwk1, ntwk1Saved)

        os.remove(os.path.join(self.test_dir, 'ntwk1Saved.s2p'))

    def test_write_touchstone(self):
        ports = 2
        s_random = self.rng.uniform(-1, 1, (self.freq.npoints, ports, ports)) +\
                1j * self.rng.uniform(-1, 1, (self.freq.npoints, ports, ports))
        random_z0 = self.rng.uniform(1, 100, (self.freq.npoints, ports)) +\
                    1j * self.rng.uniform(-100, 100, (self.freq.npoints, ports))
        ntwk = rf.Network(s=s_random, frequency=self.freq, z0=random_z0, name='test_ntwk', s_def='traveling')

        # Writing a network with non-constant z0 should raise
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)

        # Writing Touchstone and reading it back
        snp = ntwk.write_touchstone(return_string=True, write_z0=True)

        strio = io.StringIO(snp)
        # Required for reading touchstone file
        strio.name = f'StringIO.s{ports}p'
        ntwk2 = rf.Network(strio)

        np.testing.assert_allclose(ntwk2.s, s_random)
        np.testing.assert_allclose(ntwk2.z0, random_z0)

        # Renormalize output to 50 ohms
        snp = ntwk.write_touchstone(return_string=True, r_ref=50)

        # Network should not have been modified
        self.assertTrue(np.all(ntwk.s == s_random))
        self.assertTrue(np.all(ntwk.z0 == random_z0))

        # Read back the written touchstone
        strio = io.StringIO(snp)
        strio.name = f'StringIO.s{ports}p'
        ntwk2 = rf.Network(strio)

        # Renormalize original network to match the written one
        ntwk.renormalize(50)

        np.testing.assert_allclose(ntwk.s, ntwk2.s)
        np.testing.assert_allclose(ntwk.z0, ntwk2.z0)

        ntwk.z0[0, 0] = 1
        # Writing network with non-constant real z0 should fail without r_ref
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)
        ntwk.z0[0, 0] = 50

        ntwk.renormalize(50 + 1j)

        # Writing complex characteristic impedance should fail
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)

    def test_write_touchstone_noisy(self):
        ntwk = self.ntwk_noise

        # Test with and without noise data formatting
        for use_formatting in (False, True):
            # Read back the written touchstone
            if use_formatting:
                ntwkstr = ntwk.write_touchstone(
                    return_string=True,
                    format_spec_freq='{:<6.4f}',
                    format_spec_A='\t{:>6.4f}',
                    format_spec_B='\t{:>6.4f}',
                    format_spec_nf_freq='{:<6.4f}',
                    format_spec_nf_min='\t{:<6.4f}',
                    format_spec_g_opt_mag='\t{:<6.4f}',
                    format_spec_g_opt_phase='\t{:<6.4f}',
                    format_spec_rn='\t{:<6.4f}',
                )
            else:
                ntwkstr = ntwk.write_touchstone(return_string=True)
            strio = io.StringIO(ntwkstr)
            strio.name = 'StringIO.s2p'
            new_ntwk = rf.Network(strio)

            # Only compare to original noise data, not interpolated
            ntwk.resample(ntwk.f_noise)
            new_ntwk.resample(new_ntwk.f_noise)

            # Newly written noise properties should match the original
            np.testing.assert_allclose(ntwk.f_noise.f_scaled, new_ntwk.f_noise.f_scaled)
            np.testing.assert_allclose(ntwk.nfmin, new_ntwk.nfmin)
            np.testing.assert_allclose(ntwk.nfmin_db, new_ntwk.nfmin_db)
            np.testing.assert_allclose(ntwk.g_opt, new_ntwk.g_opt)
            np.testing.assert_allclose(ntwk.rn, new_ntwk.rn)
            np.testing.assert_allclose(ntwk.z0, new_ntwk.z0)

    def test_pickling(self):
        original_ntwk = self.ntwk1
        with tempfile.NamedTemporaryFile(dir=self.test_dir, suffix='ntwk') as fid:
            pickle.dump(original_ntwk, fid, protocol=2)  # Default Python2: 0, Python3: 3
            fid.seek(0)
            unpickled = pickle.load(fid)
        self.assertEqual(original_ntwk, unpickled)

    def test_stitch(self):
        tmp = self.ntwk1.copy()
        tmp.frequency = Frequency.from_f(tmp.f + tmp.f[0], 'Hz')
        with pytest.warns(rf.frequency.InvalidFrequencyWarning):
            c = rf.stitch(self.ntwk1, tmp)

    def test_cascade(self):
        self.assertEqual(self.ntwk1 ** self.ntwk2, self.ntwk3)
        self.assertEqual(self.Fix ** self.DUT ** self.Fix.flipped(), self.Meas)

    def test_cascade2(self):
        self.assertEqual(self.ntwk1 >> self.ntwk2, self.ntwk3)
        self.assertEqual(self.Fix2 >> self.DUT2 >> self.Fix2.flipped(), self.Meas2)

    def test_concat_ports(self):
        for idx in range(4):
            i,j = 2*idx, 2*(idx+1)
            self.assertTrue(np.allclose(self.DUT2.s[:, i:j, i:j], self.l2.s)) # check s-parameters
            self.assertTrue(np.allclose(self.DUT2.z0[:, i:j], self.l2.z0)) # check z0
        self.assertTrue(np.all(self.DUT2.port_modes == np.array(['S']*8))) # check port mode

    def test_connect(self):
        self.assertEqual(rf.connect(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        xformer = rf.Network()
        xformer.frequency=rf.Frequency(1, 1, 1, unit='GHz')
        xformer.s = ((0,1),(1,0))  # connects thru
        xformer.z0 = (50,25)  # transforms 50 ohm to 25 ohm
        c = rf.connect(xformer,0,xformer,1)  # connect 50 ohm port to 25 ohm port
        self.assertTrue(np.all(np.abs(c.s-rf.impedance_mismatch(50, 25)) < 1e-6))

    def test_connect_nport_2port(self):
        freq = rf.Frequency(1, 10, npoints=10, unit='GHz')

        # create a line which can be connected to each port
        med = rf.DefinedGammaZ0(freq)
        line = med.line(1, unit='m')
        line.z0 = [10, 20]

        for nport_portnum in [1,2,3,4,5,6,7,8]:

            # create a Nport network with port impedance i at port i
            nport = rf.Network()
            nport.frequency = freq
            nport.s = np.zeros((10, nport_portnum, nport_portnum))
            nport.z0 = np.arange(start=1, stop=nport_portnum + 1)

            # Connect the line to each port and check for port impedance
            for port in range(nport_portnum):
                nport_line = rf.connect(nport, port, line, 0)
                z0_expected = nport.z0
                z0_expected[:,port] = line.z0[:,1]
                np.testing.assert_allclose(
                        nport_line.z0,
                        z0_expected
                    )

    def test_connect_no_frequency(self):
        """ Connecting 2 networks defined without frequency returns Error
        """
        # try to connect two networks defined without their frequency properties
        s = self.rng.random((10, 2, 2))
        ntwk1 = rf.Network(s=s)
        ntwk2 = rf.Network(s=s)

        with self.assertRaises(ValueError):
            ntwk1**ntwk2

    def test_connect_complex_ports(self):
        """ Test that connecting two networks gives the same resulting network
        (same Z-parameters) with all s_def when using complex port impedances.
        """

        # Generate random Z-parameters for two networks
        z1 = self.rng.uniform(1, 100, size=(1, 2, 2)) +\
                1j*self.rng.uniform(-100, 100, size=(1, 2, 2))
        z2 = self.rng.uniform(1, 100, size=(1, 2, 2)) +\
                1j*self.rng.uniform(-100, 100, size=(1, 2, 2))

        # Port impedances
        z0_1 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)
        z0_2 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)
        z0_3 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)

        # Cascade Z-parameters calculated with ABCD parameters
        net3_z = rf.a2z(rf.z2a(z1) @ rf.z2a(z2))

        for s_def in rf.S_DEFINITIONS:
            net3_ref = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_3, s_def=s_def)
            net3_ref.z = net3_z

            net1 = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_1, s_def=s_def)
            net2 = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_2, s_def=s_def)
            net1.z = z1
            net2.z = z2

            # Cascade calculated with S-parameters
            net12 = net1 ** net2
            net12.renormalize(z0_3)
            self.assertTrue(net12.s_def == s_def)
            self.assertTrue(net3_ref.s_def == s_def)
            np.testing.assert_almost_equal(net3_ref.z0, net12.z0)
            np.testing.assert_almost_equal(net12.s, net3_ref.s)
            np.testing.assert_almost_equal(net3_z, net3_ref.z)

    def test_connect_different_s_def(self):
        """ Test that connecting two networks gives the same resulting
        network (same Z-parameters) with all s_def when using complex port
        impedances.
        """

        # Generate random Z-parameters for two networks
        z1 = self.rng.uniform(1, 100, size=(1, 2, 2)) +\
                1j*self.rng.uniform(-100, 100, size=(1, 2, 2))
        z2 = self.rng.uniform(1, 100, size=(1, 2, 2)) +\
                1j*self.rng.uniform(-100, 100, size=(1, 2, 2))

        # Port impedances
        z0_1 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)
        z0_2 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)
        z0_3 = self.rng.uniform(1, 100, size=2) +\
                1j*self.rng.uniform(-100, 100, size=2)

        # Cascade Z-parameters calculated with ABCD parameters
        net3_z = rf.a2z(rf.z2a(z1) @ rf.z2a(z2))

        for s_def1 in rf.S_DEFINITIONS:
            net3_ref = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_3, s_def=s_def1)
            net3_ref.z = net3_z
            net1 = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_1, s_def=s_def1)
            net1.z = z1

            for s_def2 in rf.S_DEFINITIONS:
                net2 = rf.Network(s=[[0,0],[0,0]], f=1, z0=z0_2, s_def=s_def2)
                net2.z = z2

                # Cascade calculated with S-parameters
                with warnings.catch_warnings(record=True) as w:
                    # Trigger all warnings
                    warnings.simplefilter("always")
                    net12 = net1 ** net2
                    # Check that method warns about connecting networks with
                    # different s_def.
                    if net1.s_def != net2.s_def:
                        self.assertTrue(len(w) == 1)
                    else:
                        self.assertTrue(len(w) == 0)
                net12.renormalize(z0_3)
                self.assertTrue(net12.s_def == s_def1)
                self.assertTrue(net3_ref.s_def == s_def1)
                np.testing.assert_almost_equal(net3_ref.z0, net12.z0)
                np.testing.assert_almost_equal(net12.s, net3_ref.s)
                np.testing.assert_almost_equal(net3_z, net3_ref.z)

    def test_connect_drop_ext_attrs(self):
        """Test that connecting a network created using the Circuit's class
        method, which has '_ext_attr' attributes, to another standard network.
        """
        # Create a network with '_ext_attr' attributes
        ntwk_tmp = self.ntwk1.copy()
        ntwk_tmp._ext_attrs['_is_circuit_open'] = True

        # Connect the network to another standard network
        ntwk_connected = rf.connect(ntwk_tmp, 1, self.ntwk2, 0)
        self.assertFalse(ntwk_connected._ext_attrs.get('_is_circuit_open', False))

        # Connect the network to another standard network
        ntwk_connected = rf.connect(self.ntwk2, 0, ntwk_tmp, 1)
        self.assertFalse(ntwk_connected._ext_attrs.get('_is_circuit_open', False))

    def test_connect_port_names(self):
        """Test that connecting a network with port_names to another network
        without port_names in case of mismatch and multiple connections gives
        the propers port_names and port impedances.
        """
        ntwk1 = rf.connect(self.splitter, 1, self.thru, 0, 2)

        # this keeps port_names from splitter and provides port_names for thru
        np.testing.assert_almost_equal(ntwk1.z0[0], [10, 3, 4])
        self.assertTrue(ntwk1.port_names == ["a", "2", "3"])

        # this removes port_names from splitter
        ntwk2 = rf.connect(self.thru, 2, self.splitter, 0, 2)
        np.testing.assert_almost_equal(ntwk2.z0[0], [1, 2, 30])
        self.assertTrue(ntwk2.port_names is None)

    def test_interconnect_complex_ports(self):
        """ Test that connecting two complex ports in a network
        gives the same S-parameters with all s_def when the rest of
        the ports are real
        """
        for p in range(3, 5):
            z = self.rng.uniform(1, 100, size=(1, p, p)) +\
                    1j*self.rng.uniform(-100, 100, size=(1, p, p))
            z0 = self.rng.uniform(1, 100, size=(1, p)) + 1j*0
            z0[:, -2:] += 1j*self.rng.uniform(-100, 100, size=(1, 2))
            nets = []
            for s_def in rf.S_DEFINITIONS:
                net = rf.Network(s=np.zeros((p,p)), f=1, z0=z0, s_def=s_def)
                net.z = z
                # Connect the last two complex ports together
                rf.innerconnect(net, p - 2, 2)
                nets.append(net)
            for net in nets[1:]:
                np.testing.assert_almost_equal(net.s, net[0].s)
                self.assertTrue((net.z0 == net[0].z0).all())
                self.assertTrue(net.s_def != net[0].s_def)

    def test_parallelconnect(self):
        # Create 2 network with 2 ports
        ntwka = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwka')
        ntwkb = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwkb')

        # Connect the 2 networks together by connect
        ntwk_cnt = rf.connect(ntwka, 1, ntwkb, 0)

        # Connect the 2 networks together by parallelconnect
        ntwk_par = rf.parallelconnect([ntwka, ntwkb], [1, 0])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_cnt.s, ntwk_par.s))

    def test_parallelconnect_open(self):
        # Create a network with 4 ports
        s = self.rng.random((1, 4, 4))
        ntwk = rf.Network(s=s, f=1)
        open_port = rf.Network(s=np.ones((1, 1, 1)), f=1)

        # Connect the first 2 ports together by innerconnect
        ntwk_cnt = rf.connect(ntwk, 3, open_port, 0)

        # Connect the first 2 ports together by parallelconnect
        par_ntwk = rf.parallelconnect(ntwk, [3])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_cnt.s, par_ntwk.s))

    def test_parallelconnect_inner(self):
        # Create a network with 4 ports
        s = self.rng.random((1, 4, 4))
        ntwk = rf.Network(s=s, f=1, name='ntwk')

        # Connect the first 2 ports together by innerconnect
        ntwk_inter = rf.innerconnect(ntwk, 0, 1)

        # Connect the first 2 ports together by parallelconnect
        par_ntwka = rf.parallelconnect([ntwk], [[0, 1]])
        par_ntwkb = rf.parallelconnect(ntwk, [[0, 1]])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_inter.s, par_ntwka.s))
        self.assertTrue(np.allclose(ntwk_inter.s, par_ntwkb.s))

        # Connect the last 3 ports together by circuit
        port = rf.Circuit.Port(frequency=ntwk.frequency, name='port')
        cnx = [
            [(port, 0), (ntwk, 0)],
            [(ntwk, 1), (ntwk, 2), (ntwk, 3)]
        ]
        ckt_ntwk = rf.Circuit(cnx, name='ckt_ntwk').network

        # Connect the last 3 ports together by parallelconnect
        par_ntwk = rf.parallelconnect(ntwk, [[1, 2, 3]])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ckt_ntwk.s, par_ntwk.s))

    def test_parallelconnect_mismatch(self):
        # Create 2 network with 2 ports
        ntwka = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwka', z0=25)
        ntwkb = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwkb', z0=75)

        # Connect the 2 networks together by connect
        ntwk_cnt = rf.connect(ntwka, 1, ntwkb, 0)

        # Connect the 2 networks together by parallelconnect
        ntwk_par = rf.parallelconnect([ntwka, ntwkb], [1, 0])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_cnt.s, ntwk_par.s))

        # Create matched network in circuit
        port1 = rf.Circuit.Port(frequency=ntwka.frequency, name='port1', z0=50)
        port2 = rf.Circuit.Port(frequency=ntwka.frequency, name='port2', z0=50)

        cnx = [
            [(port1, 0), (ntwka, 0)],
            [(ntwka, 1), (ntwkb, 0)],
            [(ntwkb, 1), (port2, 0)]
        ]
        ntwk_ckt = rf.Circuit(cnx, name='ckt_ntwk').network

        # Check that the two networks are not equal
        self.assertFalse(np.allclose(ntwk_ckt.s, ntwk_par.s))

        # Renormalize matched network to match circuit
        ntwk_par.renormalize(ntwk_ckt.z0)

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_ckt.s, ntwk_par.s))


    def test_innerconnect_with_T(self):
        # Create 3 network with 2 ports
        ntwka = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwka')
        ntwkb = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwkb')
        ntwkc = rf.Network(s=self.rng.random((1, 2, 2)), f=1, name='ntwkc')

        # Connect the 3 networks together by tee
        media = rf.media.DefinedGammaZ0(frequency=ntwka.frequency)
        tee_ntwk = media.tee()
        tee_ntwk = rf.connect(tee_ntwk, 0, ntwka, 1)
        tee_ntwk = rf.connect(tee_ntwk, 1, ntwkb, 1)
        tee_ntwk = rf.connect(tee_ntwk, 2, ntwkc, 1)

        # Connect the 3 networks together by circuit
        port1 = rf.Circuit.Port(frequency=ntwka.frequency, name='port1')
        port2 = rf.Circuit.Port(frequency=ntwkb.frequency, name='port2')
        port3 = rf.Circuit.Port(frequency=ntwkc.frequency, name='port3')

        cnxs = [
            [(port1, 0), (ntwka, 0)],
            [(port2, 0), (ntwkb, 0)],
            [(port3, 0), (ntwkc, 0)],
            [(ntwka, 1), (ntwkb, 1), (ntwkc, 1)]
        ]
        ckt_ntwk = rf.Circuit(cnxs, name='ckt_ntwk').network

        # Connect the 3 networks together by parallelconnect
        ntwk_par = rf.parallelconnect([ntwka, ntwkb, ntwkc], [1, 1, 1])

        # Check that the two networks are the same
        self.assertTrue(np.allclose(ntwk_par.s, tee_ntwk.s))
        self.assertTrue(np.allclose(ntwk_par.s, ckt_ntwk.s))

    def test_innerconnect_with_singular_case(self):
        # Schematic of the test circuit:
        #   +------+    +------+          +------------+            +------------+
        #  -|0     |    |     1|-      p1-|T1(0)  T2(1)|-p3      p1-|T1(0)  T2(1)|-p2
        #   |     2|----|0     |   =>     |            |     =>  +--|T1(1)  T2(2)|--+
        #  -|1     |    |     2|-      p2-|T1(1)  T2(2)|-p4      |  +------------+  |
        #   +------+    +------+          +------------+         +------------------+
        #      T1          T2                  Temp                      Thru

        # Create media and Tees
        media = rf.media.DefinedGammaZ0()
        T1, T2 = media.tee(name='T1'), media.tee(name='T2')

        # Connect the T1 and T2 together
        temp = rf.connect(T1, 2, T2, 0)

        # Check the s-parameters of the temporary Network
        self.assertTrue(np.allclose(temp.s, np.array([ [ [-0.5,  0.5,  0.5,  0.5],
                                                       [ 0.5, -0.5,  0.5,  0.5],
                                                       [ 0.5,  0.5, -0.5,  0.5],
                                                       [ 0.5,  0.5,  0.5, -0.5], ]
                                                    for _ in range(media.frequency.npoints) ]
                                                    ,dtype=complex)))

        # Innerconnect the temp to ntw and compares with the expected result
        with self.assertWarns(RuntimeWarning):
            ntw = rf.innerconnect(temp, 1, 3)

        self.assertTrue(np.allclose(ntw.s, media.thru().s))

    def test_max_stable_gain(self):
        # Check whether the maximum stable gain agrees with that derived from Y-parameters
        y12 = self.fet.y[:, 0, 1]
        y21 = self.fet.y[:, 1, 0]
        # Maximum stable gain derived from Y-parameters
        gms_y = np.abs(y21) / np.abs(y12)
        self.assertTrue(
            np.all(
                np.abs(self.fet.max_stable_gain - gms_y) < 1e-6
            )
        )

        # Check whether a runtime warning is raised when zero division occurs
        net = rf.Network(f=[1], s=[[0, 0],[0, 0]], z0=50)
        with pytest.raises(RuntimeWarning):
            net.max_stable_gain

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.max_stable_gain

    def test_max_gain(self):
        # Check whether the max gain agrees with that calculated with ADS
        maxgain_ads = np.loadtxt(os.path.join(self.test_dir, 'maxgain_ads.csv'), encoding='utf-8', delimiter=',')
        self.assertTrue(
            np.all(
                np.abs(10 * np.log10(self.fet.max_gain) - maxgain_ads[:,1]) < 1e-6
            )
        )

        # Check whether a runtime warning is raised when zero division occurs
        net = rf.Network(f=[1], s=[[0, 0],[0, 0]], z0=50)
        with pytest.raises(RuntimeWarning):
            net.max_gain

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.max_gain

    def test_unilateral_gain(self):
        # Check whether the unilateral gain agrees with that derived from Y-parameters
        y11 = self.fet.y[:, 0, 0]
        y12 = self.fet.y[:, 0, 1]
        y21 = self.fet.y[:, 1, 0]
        y22 = self.fet.y[:, 1, 1]
        # Unilateral gain derived from Y-parameters
        U_y = (np.abs(y21 - y12) ** 2) \
              / (4 * (np.real(y11) * np.real(y22) - np.real(y12) * np.real(y21)))
        self.assertTrue(
            np.all(
                np.abs(self.fet.unilateral_gain - U_y) < 1e-6
            )
        )

        # Check whether a runtime warning is raised when zero division occurs
        net = rf.Network(f=[1], s=[[0, 0],[0, 0]], z0=50)
        with pytest.raises(RuntimeWarning):
            net.unilateral_gain

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.unilateral_gain

    def test_delay(self):
        ntwk1_delayed = self.ntwk1.delay(1,'ns',port=0)
        self.assertTrue(
            np.all(
                self.ntwk1.group_delay[:,:,:]
                -ntwk1_delayed.group_delay[:,:,:]
                -np.array(
                    [[-1.e-09+0.j, -5.e-10+0.j],
                     [-5.e-10+0.j,  0.e+00+0.j]]
                ) < 1e-9
            )
        )
        ntwk2_delayed = self.ntwk2.delay(1,'ps',port=1)
        self.assertTrue(
            np.all(
                self.ntwk2.group_delay[:,:,:]
                -ntwk2_delayed.group_delay[:,:,:]
                -np.array(
                    [[ 0.e+00+0.j, -5.e-13+0.j],
                     [-5.e-13+0.j, -1.e-12+0.j]]
                ) < 1e-9
            )
        )

    def test_connect_multiports(self):
        a = rf.Network()
        a.frequency = rf.Frequency(1, 1, 1, unit='GHz')
        a.s = np.arange(16).reshape(4,4)
        a.z0 = np.arange(4) + 1 #  Z0 should never be zero

        b = rf.Network()
        b.frequency = rf.Frequency(1, 1, 1, unit='GHz')
        b.s = np.arange(16).reshape(4,4)
        b.z0 = np.arange(4)+10

        c=rf.connect(a,2,b,0,2)
        self.assertTrue((c.z0==[1,2,12,13]).all())

        d=rf.connect(a,0,b,0,3)
        self.assertTrue((d.z0==[4,13]).all())

    @pytest.mark.skip(reason="not supporting this function currently ")
    def test_connect_fast(self):
        self.assertEqual(rf.connect_fast(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        xformer = rf.Network()
        xformer.frequency = rf.Frequency(1, 1, 1, unit='GHz')
        xformer.s = ((0,1),(1,0))  # connects thru
        xformer.z0 = (50,25)  # transforms 50 ohm to 25 ohm
        c = rf.connect_fast(xformer,0,xformer,1)  # connect 50 ohm port to 25 ohm port
        self.assertTrue(np.all(np.abs(c.s-rf.impedance_mismatch(50, 25)) < 1e-6))

    def test_flip(self):
        self.assertEqual(rf.connect(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        gain = rf.Network()
        gain.frequency = rf.Frequency(1, 1, 1, unit='GHz')
        gain.s = ((0,2),(0.5,0))  # connects thru with gain of 2.0
        gain.z0 = (37,82)
        flipped = gain.copy()
        flipped.flip()
        c = rf.connect(gain,1,flipped,0)
        self.assertTrue(np.all(np.abs(c.s - np.array([[0,1],[1,0]])) < 1e-6))

    def test_renumber(self):
        ntwk = self.ntwk1
        from_ports_num = [0,1]
        to_ports_num = [1,0]
        from_ports_name = ["A", "B"]
        to_ports_name = ["B", "A"]

        ntwk.port_names = from_ports_name
        ntwk_renum = ntwk.renumbered(from_ports_num, to_ports_num)

        assert to_ports_name == ntwk_renum.port_names

    def test_de_embed_by_inv(self):
        self.assertEqual(self.ntwk1.inv ** self.ntwk3, self.ntwk2)
        self.assertEqual(self.ntwk3 ** self.ntwk2.inv, self.ntwk1)
        self.assertEqual(self.Fix.inv ** self.Meas ** self.Fix.flipped().inv,
                         self.DUT)

    def test_de_embed_port_impedance(self):
        ntw = self.ntwk1.copy()
        ntw.renormalize((25, 75))
        ntw_inv = ntw.inv
        self.assertTrue(np.allclose(ntw_inv.z0, (75, 25)))
        rst = ntw_inv ** self.ntwk3
        rst.renormalize(50)
        self.assertEqual(rst, self.ntwk2)

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_one_port_db(self):
        self.ntwk1.plot_s_db(0,0)

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_one_port_deg(self):
        self.ntwk1.plot_s_deg(0,0)

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_one_port_smith(self):
        self.ntwk1.plot_s_smith(0,0)

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_two_port_db(self):
        self.ntwk1.plot_s_db()

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_two_port_deg(self):
        self.ntwk1.plot_s_deg()

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_two_port_smith(self):
        self.ntwk1.plot_s_smith()

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_plot_z_responses_singularities(self):
        with np.errstate(divide='raise'):
            self.o1.plot_z_time_impulse(window = None)

    def test_zy_singularities(self):
        networks = [
            rf.N(f=[1], s=[1], z0=[50]),
            rf.N(f=[1], s=[-1], z0=[50]),
            rf.N(f=[1], s=[[0, 1], [1, 0]], z0=[50]),
            rf.N(f=[1], s=[[1, 0], [0, 1]], z0=50),
            rf.N(f=[1], s=[[-1, 0], [0, -1]], z0=50),
            rf.N(f=[1], s=[[0.5, 0.5], [0.5, 0.5]], z0=[50]),
            rf.N(f=[1], s=[[-0.5, -0.5], [-0.5, -0.5]], z0=[50]),
        ]
        # These conversion can be very inaccurate since results are very close
        # to singular.
        # Test that they are close with loose accuracy tolerance.
        for net in networks:
            for s_def in rf.S_DEFINITIONS:
                np.testing.assert_allclose(
                    rf.z2s(rf.s2z(net.s, net.z0, s_def=s_def), net.z0, s_def=s_def), net.s, atol=1e-3
                    )
                np.testing.assert_allclose(
                    rf.y2s(rf.s2y(net.s, net.z0, s_def=s_def), net.z0, s_def=s_def), net.s, atol=1e-3
                    )

    def test_conversions(self):
        #Converting to other format and back to S-parameters should return the original network
        s_random = self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2)) +\
                   1j * self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2))
        ntwk_random = rf.Network(s=s_random, frequency=self.freq)
        for test_z0 in (50, 10, 90+10j, 4-100j):
            for test_ntwk in (self.ntwk1, self.ntwk2, self.ntwk3, ntwk_random):
                ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0)
                np.testing.assert_allclose(rf.a2s(rf.s2a(ntwk.s, test_z0), test_z0), ntwk.s)
                np.testing.assert_allclose(rf.z2s(rf.s2z(ntwk.s, test_z0), test_z0), ntwk.s)
                np.testing.assert_allclose(rf.y2s(rf.s2y(ntwk.s, test_z0), test_z0), ntwk.s)
                np.testing.assert_allclose(rf.h2s(rf.s2h(ntwk.s, test_z0), test_z0), ntwk.s)
                np.testing.assert_allclose(rf.t2s(rf.s2t(ntwk.s)), ntwk.s)
        np.testing.assert_allclose(rf.t2s(rf.s2t(self.Fix.s)), self.Fix.s)

    def test_multiport_conversions(self):
        #Converting to other format and back to S-parameters should return the original network
        for ports in range(3, 6):
            s_random = self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports)) +\
                       1j * self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = self.rng.uniform(1, 100, (self.freq.npoints, ports)) +\
                        1j * self.rng.uniform(-100, 100, (self.freq.npoints, ports))
            for test_z0 in (50, random_z0):
                for s_def in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0, s_def=s_def)
                    np.testing.assert_allclose(
                        rf.z2s(rf.s2z(ntwk.s, test_z0, s_def=s_def), test_z0, s_def=s_def), ntwk.s
                        )
                    np.testing.assert_allclose(
                        rf.y2s(rf.s2y(ntwk.s, test_z0, s_def=s_def), test_z0, s_def=s_def), ntwk.s
                        )

    def test_y_z_compatability(self):
        # Test that np.linalg.inv(Z) == Y
        fpoints = 3
        for p in range(2, 6):
            s = self.rng.uniform(-1, 1, (fpoints, p, p)) + 1j * self.rng.uniform(-1, 1, (fpoints, p, p))
            random_z0 = self.rng.uniform(1, 100, (fpoints, p)) + 1j * self.rng.uniform(-100, 100, (fpoints, p))
            for test_z0 in (50, random_z0):
                for s_def in rf.S_DEFINITIONS:
                    z = rf.s2z(s, test_z0, s_def=s_def)
                    y = np.linalg.inv(z)
                    np.testing.assert_allclose(rf.y2s(y, test_z0, s_def=s_def), s)

    def test_unknown_s_def(self):
        # Test that Exception is raised when given unknown s_def
        s = np.array([0]).reshape(1, 1, 1)
        z0 = np.array([50])
        # These should work
        # These also test that functions work with dtype=float input
        rf.s2z(s, z0)
        rf.z2s(s, z0)
        rf.s2y(s, z0)
        rf.y2s(s, z0)
        with pytest.raises(Exception) as e:
            rf.s2z(s, z0, s_def='error')
        with pytest.raises(Exception) as e:
            rf.z2s(s, z0, s_def='error')
        with pytest.raises(Exception) as e:
            rf.y2s(s, z0, s_def='error')
        with pytest.raises(Exception) as e:
            rf.s2y(s, z0, s_def='error')

    def test_sparam_renormalize(self):
        #Converting to other format and back to S-parameters should return the original network
        for ports in range(2, 6):
            s_random = self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports)) +\
                       1j * self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = self.rng.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*self.rng.uniform(-100, 100, size=(self.freq.npoints, ports))
            for test_z0 in (50, 20+60j, random_z0):
                for method in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=50)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(test_z0, method)
                    ntwk_renorm.renormalize(50, method)
                    np.testing.assert_allclose(ntwk_renorm.s, ntwk.s)

    def test_sparam_renorm_s2s(self):
        """
        Test changing S-parameter definition with complex ports
        """
        for ports in range(2, 6):
            s_random = self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports)) +\
                       1j * self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = self.rng.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*self.rng.uniform(-100, 100, size=(self.freq.npoints, ports))
            for def1 in rf.S_DEFINITIONS:
                for def2 in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=random_z0, s_def=def1)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def2)
                    np.testing.assert_allclose(ntwk_renorm.z, ntwk.z)
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def1)
                    np.testing.assert_allclose(ntwk_renorm.s, ntwk.s)
                    np.testing.assert_allclose(ntwk_renorm.z0, ntwk.z0)

    def test_sparam_renorm_different_z0(self):
        """
        Test changing S-parameter definition with complex ports.
        renormalize handles cases with same z0 and different z0 before and
        after conversion with different method.
        """
        for ports in range(2, 6):
            s_random = self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports)) +\
                       1j * self.rng.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = self.rng.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*self.rng.uniform(-100, 100, size=(self.freq.npoints, ports))
            random_z0_2 = self.rng.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*self.rng.uniform(-100, 100, size=(self.freq.npoints, ports))
            for def1 in rf.S_DEFINITIONS:
                for def2 in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=random_z0, s_def=def1)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(random_z0_2, s_def=def2)
                    np.testing.assert_allclose(ntwk_renorm.z, ntwk.z)
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def1)
                    np.testing.assert_allclose(ntwk_renorm.s, ntwk.s)
                    np.testing.assert_allclose(ntwk_renorm.z0, ntwk.z0)

    def test_setters(self):
        s_random = self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2)) +\
                   1j * self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2))
        ntwk = rf.Network(s=s_random, frequency=self.freq)
        ntwk.z0 = self.rng.uniform(1, 100, len(ntwk.z0)) + 1j*self.rng.uniform(-100, 100, len(ntwk.z0))
        ntwk.s = ntwk.s
        np.testing.assert_allclose(ntwk.s, s_random)
        ntwk.a = ntwk.a
        np.testing.assert_allclose(ntwk.s, s_random)
        ntwk.z = ntwk.z
        np.testing.assert_allclose(ntwk.s, s_random)
        ntwk.y = ntwk.y
        np.testing.assert_allclose(ntwk.s, s_random)
        ntwk.t = ntwk.t
        np.testing.assert_allclose(ntwk.s, s_random)
        ntwk.h = ntwk.h
        np.testing.assert_allclose(ntwk.s, s_random)

    def test_s_def_setters(self):
        s_random = self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2)) +\
            1j * self.rng.uniform(-10, 10, (self.freq.npoints, 2, 2))
        z0_scalar = 50
        z0_array_port = self.rng.uniform(1, 100, 2)
        z0_array_freq = self.rng.uniform(1, 100, self.freq.npoints) +\
            1j*self.rng.uniform(-100, 100, self.freq.npoints)
        z0_array_freq_port = self.rng.uniform(1, 100, (self.freq.npoints,2)) +\
            1j*self.rng.uniform(-100, 100, (self.freq.npoints,2))

        for z0 in [z0_scalar, z0_array_port, z0_array_freq, z0_array_freq_port]:
            ntwk = rf.Network(s=s_random, frequency=self.freq, z0=z0)

            s_traveling = rf.s2s(s_random, z0, 'traveling', ntwk.s_def)
            s_power = rf.s2s(s_random, z0, 'power', ntwk.s_def)
            s_pseudo = rf.s2s(s_random, z0, 'pseudo', ntwk.s_def)

            ntwk.s_traveling = s_traveling
            np.testing.assert_allclose(ntwk.s, s_random)
            ntwk.s_power = s_power
            np.testing.assert_allclose(ntwk.s, s_random)
            ntwk.s_pseudo = s_pseudo
            np.testing.assert_allclose(ntwk.s, s_random)

            np.testing.assert_allclose(ntwk.s_traveling, s_traveling)
            np.testing.assert_allclose(ntwk.s_power, s_power)
            np.testing.assert_allclose(ntwk.s_pseudo, s_pseudo)

    def test_sparam_conversion_with_complex_char_impedance(self):
        """
        Renormalize a 2-port network wrt to complex characteristic impedances
        using power-waves definition of s-param
        Example based on scikit-rf issue #313
        """
        f0 = rf.Frequency(75.8, npoints=1, unit='GHz')
        s0 = np.array([
                [-0.194 - 0.228j, -0.721 + 0.160j],
                [-0.721 + 0.160j, +0.071 - 0.204j]])
        ntw = rf.Network(frequency=f0, s=s0, z0=50, name='dut')

        # complex characteristic impedance to renormalize to
        zdut = 100 + 10j

        # reference solutions obtained from ANSYS Circuit or ADS (same res)
        # case 1: z0=[50, zdut]
        s_ref = np.array([[
            [-0.01629813-0.29764199j, -0.6726785 +0.24747539j],
            [-0.6726785 +0.24747539j, -0.30104687-0.10693578j]]])
        np.testing.assert_allclose(rf.z2s(ntw.z, z0=[50, zdut]), s_ref)
        np.testing.assert_allclose(rf.renormalize_s(ntw.s, [50,50], [50,zdut]), s_ref)

        # case 2: z0=[zdut, zdut]
        s_ref = np.array([[
            [-0.402829859501534 - 0.165007172677339j,-0.586542065592524 + 0.336098534178339j],
            [-0.586542065592524 + 0.336098534178339j,-0.164707376748782 - 0.21617153431756j]]])
        np.testing.assert_allclose(rf.z2s(ntw.z, z0=[zdut, zdut]), s_ref)
        np.testing.assert_allclose(rf.renormalize_s(ntw.s, [50,50], [zdut,zdut]), s_ref)

        # Comparing Z and Y matrices from reference ones (from ADS)
        # Z or Y matrices do not depend of characteristic impedances.
        # Precision is 1e-4 due to rounded results in ADS export files
        z_ref = np.array([[
            [34.1507 -65.6786j, -37.7994 +73.7669j],
            [-37.7994 +73.7669j, 55.2001 -86.8618j]]])
        np.testing.assert_allclose(ntw.z, z_ref, atol=1e-4)

        y_ref = np.array([[
            [0.0926 +0.0368j, 0.0770 +0.0226j],
            [0.0770 +0.0226j, 0.0686 +0.0206j]]])
        np.testing.assert_allclose(ntw.y, y_ref, atol=1e-4)

    def test_sparam_conversion_vs_sdefinition(self):
        """
        Check that power-wave or pseudo-waves scattering parameters definitions
        give same results for real characteristic impedances
        """
        f0 = rf.Frequency(75.8, npoints=1, unit='GHz')
        s_ref = np.array([[  # random values
            [-0.1000 -0.2000j, -0.3000 +0.4000j],
            [-0.3000 +0.4000j, 0.5000 -0.6000j]]])
        ntw = rf.Network(frequency=f0, s=s_ref, z0=50, name='dut')

        # renormalize s parameter according one of the definition.
        # As characteristic impedances are all real, should be all equal
        np.testing.assert_allclose(ntw.s, s_ref)
        np.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='power'), s_ref)
        np.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='pseudo'), s_ref)
        np.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='traveling'), s_ref)

        # also check Z and Y matrices, just in case
        z_ref = np.array([[
            [18.0000 -16.0000j, 20.0000 + 40.0000j],
            [20.0000 +40.0000j, 10.0000 -80.0000j]]])
        np.testing.assert_allclose(ntw.z, z_ref, atol=1e-4)

        y_ref = np.array([[
            [0.0251 +0.0023j, 0.0123 -0.0066j],
            [0.0123 -0.0066j, 0.0052 +0.0055j]]])
        np.testing.assert_allclose(ntw.y, y_ref, atol=1e-4)

        # creating network by specifying s-params definition
        ntw_power = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='power')
        ntw_pseudo = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='pseudo')
        ntw_legacy = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='traveling')
        self.assertTrue(ntw_power == ntw_pseudo)
        self.assertTrue(ntw_power == ntw_legacy)

    def test_sparam_from_hfss_with_power_wave(self):
        """
        Open a Touchstone generated by HFSS with the power-wave definition.
        """
        pwfile = 'hfss_oneport_powerwave.s1p'
        pwfile_skrf = 'tmp_skrf_oneport_powerwave.s1p'

        # s_def must be explicitely passed as 'power',
        # otherwise 'traveling' would have been assumed (being default HFSS setting)
        ntwk_orig = rf.Network(os.path.join(self.test_dir, pwfile))
        self.assertEqual(ntwk_orig.s_def, S_DEF_HFSS_DEFAULT)

        ntwk_orig = rf.Network(os.path.join(self.test_dir, pwfile), s_def='power')
        self.assertEqual(ntwk_orig.s_def, 'power')

        # write Touchstone file and read it. Results should be the same
        ntwk_orig.write_touchstone(os.path.join(self.test_dir, pwfile_skrf), write_z0=True, form='RI')
        ntwk_skrf = rf.Network(os.path.join(self.test_dir, pwfile_skrf))

        # check if the s_def could be correctly recovered from scikit-rf's Touchstone file
        self.assertTrue(ntwk_orig.s_def == ntwk_skrf.s_def)
        self.assertTrue(ntwk_orig == ntwk_skrf)

    def test_network_from_z_or_y(self):
        ' Construct a network from its z or y parameters '
        # test for both real and complex char. impedance
        # and for 2 frequencies
        z0 = [self.rng.random(), self.rng.random()+1j*self.rng.random()]
        freqs = np.array([1, 2])
        # generate arbitrary complex z and y
        z_ref = self.rng.random((2,3,3)) + 1j*self.rng.random((2,3,3))
        y_ref = self.rng.random((2,3,3)) + 1j*self.rng.random((2,3,3))
        # create networks from z or y and compare ntw.z to the reference
        # check that the conversions work for all s-param definitions
        for s_def in S_DEFINITIONS:
            ntwk = rf.Network(s_def=s_def)
            ntwk.z0 = rf.fix_z0_shape(z0, 2, 3)
            ntwk.frequency = Frequency.from_f(freqs, unit='GHz')
            # test #1: define the network directly from z
            ntwk.z = z_ref
            np.testing.assert_allclose(ntwk.z, z_ref)
            # test #2: define the network from s, after z -> s (s_def is important)
            ntwk.s = rf.z2s(z_ref, z0, s_def=s_def)
            np.testing.assert_allclose(ntwk.z, z_ref)
            # test #3: define the network directly from y
            ntwk.y = y_ref
            np.testing.assert_allclose(ntwk.y, y_ref)
            # test #4: define the network from s, after y -> s (s_def is important)
            ntwk.s = rf.y2s(y_ref, z0, s_def=s_def)
            np.testing.assert_allclose(ntwk.y, y_ref)

    def test_z0_pure_imaginary(self):
        ' Test cases where z0 is pure imaginary '
        # test that conversion to Z or Y does not give NaN for pure imag z0
        for s_def in S_DEFINITIONS:
            ntwk = rf.Network(s_def=s_def)
            ntwk.z0 = np.array([50j, -50j])
            ntwk.frequency = Frequency.from_f(np.array([1000]), unit='GHz')
            ntwk.s = self.rng.random((1,2,2)) + self.rng.random((1,2,2))*1j
            self.assertFalse(np.any(np.isnan(ntwk.z)))
            self.assertFalse(np.any(np.isnan(ntwk.y)))

    def test_z0_scalar(self):
        'Test a scalar z0'
        ntwk = rf.Network()
        ntwk.z0 = 1
        # Test setting the z0 before and after setting the s shape
        self.assertEqual(ntwk.z0, 1)
        ntwk.s = self.rng.random((1,2,2))
        ntwk.z0 = 10
        self.assertTrue(np.allclose(ntwk.z0, np.full((1,2), 10)))

    def test_z0_vector(self):
        'Test a 1 dimensional z0'
        ntwk = rf.Network()
        z0 = [1,2]
        # Test setting the z0 before and after setting the s shape
        ntwk.z0 = [1,2] # Passing as List
        self.assertTrue(np.allclose(ntwk.z0, np.array(z0, dtype=complex)))
        ntwk.z0 = np.array(z0[::-1]) # Passing as np.array
        self.assertTrue(np.allclose(ntwk.z0, np.array(z0[::-1], dtype=complex)))

        # If the s-array has been set, the z0 value should broadcast to the required shape
        ntwk.s = self.rng.random((3,2,2))
        ntwk.z0 = z0
        self.assertTrue(np.allclose(ntwk.z0, np.array([z0, z0, z0], dtype=complex)))

        # If the s-array has been set and we want to set z0 along the frequency axis,
        # wer require the frequency vector to be set too.
        # Unfortunately the frequency vector and the s shape can distinguish
        z0 = [1,2,3]
        ntwk.s = self.rng.random((3,2,2))
        ntwk.z0 = z0[::-1]

        ntwk.frequency = Frequency.from_f([1,2,3], unit='GHz')
        self.assertTrue(np.allclose(ntwk.z0, np.array([z0[::-1], z0[::-1]], dtype=complex).T))

    def test_z0_matrix(self):
        ntwk = rf.Network()
        z0 = [[1,2]]
        ntwk.z0 = z0
        self.assertTrue(np.allclose(ntwk.z0, np.array(z0, dtype=complex)))
        ntwk.z0 = np.array(z0) + 1 # Passing as np.array
        self.assertTrue(np.allclose(ntwk.z0, np.array(z0, dtype=complex)+1))

        # Setting the frequency is required to be set, as the matrix size is checked against the
        # frequency vector
        ntwk.s = self.rng.random((1,2,2))
        ntwk.frequency = Frequency.from_f([1], unit='GHz')
        ntwk.z0 = z0
        self.assertTrue(np.allclose(ntwk.z0, np.array(z0, dtype=complex)))

    def test_z0_assign(self):
        """ Test that z0 getter returns a reference to _z0 so that it can
        be assigned to with array indexing"""
        ntwk = rf.Network(s=np.zeros((3,2,2)), f=[1,2,3])
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

        ntwk = rf.Network(s=np.zeros((3,2,2)), f=[1,2,3], z0=np.ones(3))
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

        ntwk = rf.Network(s=np.zeros((3,2,2)), f=[1,2,3], z0=np.ones((3,2)))
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

    def test_yz(self):
        tinyfloat = 1e-12
        ntwk = rf.Network()
        ntwk.z0 = np.array([28,75+3j])
        ntwk.frequency = Frequency.from_f(np.array([1000, 2000]), unit='GHz')
        ntwk.s = rf.z2s(np.array([[[1+1j,5,11],[40,5,3],[16,8,9+8j]],
                                   [[1,20,3],[14,10,16],[27,18,-19-2j]]]))
        self.assertTrue((abs(rf.y2z(ntwk.y)-ntwk.z) < tinyfloat).all())
        self.assertTrue((abs(rf.y2s(ntwk.y, ntwk.z0)-ntwk.s) < tinyfloat).all())
        self.assertTrue((abs(rf.z2y(ntwk.z)-ntwk.y) < tinyfloat).all())
        self.assertTrue((abs(rf.z2s(ntwk.z, ntwk.z0)-ntwk.s) < tinyfloat).all())

    def test_mul(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a*a).s == np.array([[[-3+4j]],[[-7+24j]]])).all())
        # operating on numbers
        self.assertTrue( ((2*a*2).s == np.array([[[4+8j]],[[12+16j]]])).all())
        # operating on list
        self.assertTrue( ((a*[1,2]).s == np.array([[[1+2j]],[[6+8j]]])).all())
        self.assertTrue( (([1,2]*a).s == np.array([[[1+2j]],[[6+8j]]])).all())

    def test_sub(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a-a).s == np.array([[[0+0j]],[[0+0j]]])).all())
        # operating on numbers
        self.assertTrue( ((a-(2+2j)).s == np.array([[[-1+0j]],[[1+2j]]])).all())
        # operating on list
        self.assertTrue( ((a-[1+1j,2+2j]).s == np.array([[[0+1j]],[[1+2j]]])).all())

    def test_div(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a/a).s == np.array([[[1+0j]],[[1+0j]]])).all())
        # operating on numbers
        self.assertTrue( ((a/2.).s == np.array([[[.5+1j]],[[3/2.+2j]]])).all())
        # operating on list
        self.assertTrue( ((a/[1,2]).s == np.array([[[1+2j]],[[3/2.+2j]]])).all())

    def test_add(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a+a).s == np.array([[[2+4j]],[[6+8j]]])).all())
        # operating on numbers
        self.assertTrue( ((a+2+2j).s == np.array([[[3+4j]],[[5+6j]]])).all())
        # operating on list
        self.assertTrue( ((a+[1+1j,2+2j]).s == np.array([[[2+3j]],[[5+6j]]])).all())


    def test_interpolate_linear(self):
        net = rf.Network(f=[0, 1, 3, 4], s=[0,1,9,16], f_unit="Hz")

        interp = net.interpolate(rf.Frequency(0, 4, 5, unit="Hz"), kind="linear")
        assert np.allclose(interp.s[2], 5.0)

    def test_interpolate_cubic(self):
        net = rf.Network(f=[0, 1, 3, 4], s=[0,1,9,16], f_unit="Hz")

        interp = net.interpolate(rf.Frequency(0, 4, 5, unit="Hz"), kind="cubic")
        assert np.allclose(interp.s[2], 4.0)

    def test_interpolate_rational(self):
        a = rf.N(f=np.linspace(1,2,5),s=np.linspace(0,1,5)*(1+1j),z0=1, f_unit="ghz")
        freq = rf.F.from_f(np.linspace(1,2,6,endpoint=True), unit='GHz')
        b = a.interpolate(freq, kind='rational')
        self.assertFalse(any(np.isnan(b.s)))
        # Test that the endpoints are the equal
        self.assertTrue(b.s[0] == a.s[0])
        self.assertTrue(b.s[-1] == a.s[-1])
        # Check that abs(S) is increasing
        self.assertTrue(all(np.diff(np.abs(b.s.flatten())) > 0))
        self.assertTrue(b.z0[0] == a.z0[0])

    def test_interpolate_freq_cropped(self):
        a = rf.N(f=np.arange(20), s=np.arange(20)*(1+1j),z0=1, f_unit="ghz")
        freq = rf.F.from_f(np.linspace(1,2,3,endpoint=True), unit='GHz')
        for method in ('linear', 'cubic', 'quadratic', 'rational'):
            b = a.interpolate(freq, freq_cropped=False, kind=method)
            c = a.interpolate(freq, kind=method)
            self.assertTrue(np.allclose(b.s, c.s))

    def test_interpolate_self(self):
        """Test resample."""
        a = rf.N(f=[1,2], s=[1+2j, 3+4j], z0=1)
        a.interpolate_self(4)
        self.assertEqual(len(a), 4)
        # also test the alias name
        a.resample(6)
        self.assertEqual(len(a), 6)
        # TODO: numerically test for correct interpolation

    def test_slicer(self):
        a = rf.Network(f=[1,2,4,5,6],
                       s=[1,1,1,1,1],
                       z0=50,
                       f_unit="ghz")

        b = a['2-5ghz']
        tinyfloat = 1e-12
        self.assertTrue((abs(b.frequency.f - [2e9,4e9,5e9]) < tinyfloat).all())

    # Network classifiers
    def test_is_reciprocal(self):
        a = rf.Network(f=[1],
                       s=[[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]],
                       z0=50)
        self.assertFalse(a.is_reciprocal(), 'A circulator is not reciprocal.')
        b = rf.Network(f=[1],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertTrue(b.is_reciprocal(), 'This power divider is reciprocal.')
        return

    def test_is_symmetric(self):
        # 2-port
        a = rf.Network(f=[1, 2, 3],
                       s=[[[-1, 0], [0, -1]],
                          [[-2, 0], [0, -2]],
                          [[1, 0], [0, 1]]],
                       z0=50)
        self.assertTrue(a.is_symmetric(), 'A short is symmetric.')
        self.assertRaises(ValueError, a.is_symmetric, port_order={1: 2})  # error raised by renumber()
        a.s[0, 0, 0] = 1
        self.assertFalse(a.is_symmetric(), 'non-symmetrical')

        # another 2-port (transmission line with 201 frequency samples)
        nw_line = rf.data.line
        self.assertTrue(nw_line.is_symmetric())

        # another 2-port (ring slot with 201 frequency samples)
        nw_ringslot = rf.data.ring_slot
        self.assertFalse(nw_ringslot.is_symmetric())

        # 3-port
        b = rf.Network(f=[1, 3],
                       s=[[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
                          [[0, 0.2, 0.2], [0.2, 0, 0.2], [0.2, 0.2, 0]]],
                       z0=50)
        with self.assertRaises(ValueError) as context:
            b.is_symmetric()
        self.assertEqual(str(context.exception),
                         'Using is_symmetric() is only valid for a 2N-port network (N=2,4,6,8,...)')

        # 4-port
        c = rf.Network(f=[1, 3, 5, 6],
                       s=[[[0, 1j, 1, 0], [1j, 0, 0, 1], [1, 0, 0, 1j], [0, 1, 1j, 0]],
                          [[0, 0.8j, 0.7, 0], [0.8j, 0, 0, 0.7], [0.7, 0, 0, 0.8j], [0, 0.7, 0.8j, 0]],
                          [[0, 0.3j, 1, 0], [0.3j, 0, 0, 1], [1, 0, 0, 0.3j], [0, 1, 0.3j, 0]],
                          [[0, -1j, -1, 0], [-1j, 0, 0, -1], [-1, 0, 0, -1j], [0, -1, -1j, 0]]],
                       z0=50)
        self.assertTrue(c.is_symmetric(n=2), 'This quadrature hybrid coupler is symmetric.')
        self.assertTrue(c.is_symmetric(n=2, port_order={0: 1, 1: 2, 2: 3, 3: 0}),
                        'This quadrature hybrid coupler is symmetric even after rotation.')
        with self.assertRaises(ValueError) as context:
            c.is_symmetric(n=3)
        self.assertEqual(str(context.exception), 'specified order n = 3 must be between 1 and N = 2, inclusive')

        d = rf.Network(f=[1],
                       s=[[1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1]],
                       z0=50)
        self.assertTrue(d.is_symmetric(n=1), 'This contrived non-reciprocal device has a line of symmetry.')
        self.assertFalse(d.is_symmetric(n=2), 'This device only has first-order line symmetry.')
        self.assertFalse(d.is_symmetric(port_order={0: 1, 1: 0}),
                         'This device is no longer symmetric after reordering ports 1 and 2.')
        self.assertTrue(d.is_symmetric(port_order={0: 1, 1: 0, 2: 3, 3: 2}),
                        'This device is symmetric after swapping ports 1 with 2 and 3 with 4.')

        # 6-port
        x = rf.Network(f=[1],
                       s=[[0, 0, 0, 0, 0, 0],
                          [0, 1, 9, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0],
                          [0, 0, 0, 9, 1, 0],
                          [0, 0, 0, 0, 0, 0]],
                       z0=50)
        self.assertFalse(x.is_symmetric(n=3))
        self.assertFalse(x.is_symmetric(n=2))
        self.assertTrue(x.is_symmetric(n=1))
        self.assertTrue(x.is_symmetric(n=1, port_order={-3: -1, -1: -3, 0: 2, 2: 0}))

        # 8-port
        s8p_diag = [1j, -1j, -1j, 1j, 1j, -1j, -1j, 1j]
        s8p_mat = np.identity(8, dtype=complex)
        for row in range(8):
            s8p_mat[row, :] *= s8p_diag[row]
        y = rf.Network(f=[1],
                       s=s8p_mat,
                       z0=50)
        self.assertTrue(y.is_symmetric())
        self.assertTrue(y.is_symmetric(n=2))
        self.assertFalse(y.is_symmetric(n=4))
        return

    def test_is_passive(self):
        a = rf.Network(f=[1],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertTrue(a.is_passive(), 'This power divider is passive.')
        b = rf.Network(f=[1],
                       s=[[0, 0],
                          [10, 0]],
                       z0=50)
        self.assertFalse(b.is_passive(), 'A unilateral amplifier is not passive.')
        return

    def test_is_lossless(self):
        a = rf.Network(f=[1],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertFalse(a.is_lossless(), 'A resistive power divider is lossy.')
        b = rf.Network(f=[1],
                       s=[[0, -1j/np.sqrt(2), -1j/np.sqrt(2)],
                          [-1j/np.sqrt(2), 1./2, -1./2],
                          [-1j/np.sqrt(2), -1./2, 1./2]],
                       z0=50)
        self.assertTrue(b.is_lossless(), 'This unmatched power divider is lossless.')
        return

    def test_noise(self):
        a = self.ntwk_noise

        nf = 10**(0.05)
        self.assertTrue(a.noisy)
        self.assertTrue(abs(a.nfmin[0] - nf) < 1.e-6, 'noise figure does not match original spec')
        self.assertTrue(abs(a.z_opt[0] - 50.) < 1.e-6, 'optimal resistance does not match original spec')
        self.assertTrue(abs(a.rn[0] - 0.1159*50.) < 1.e-6, 'equivalent resistance does not match original spec')
        self.assertTrue(np.all(abs(a.g_opt) < 1.e-6),
                        'calculated optimal reflection coefficient does not match original coefficients')

        b = rf.Network(f=[1, 2],
                       s=[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                       z0=50, f_unit="ghz").interpolate(a.frequency)
        with self.assertRaises(ValueError) as context:
            b.n
        with self.assertRaises(ValueError) as context:
            b.f_noise
        self.assertEqual(str(context.exception), 'network does not have noise')

        c = a ** b
        self.assertTrue(a.noisy)
        self.assertTrue(abs(c.nfmin[0] - nf) < 1.e-6, 'noise figure does not match original spec')
        self.assertTrue(abs(c.z_opt[0] - 50.) < 1.e-6, 'optimal resistance does not match original spec')
        self.assertTrue(abs(c.rn[0] - 0.1159*50.) < 1.e-6, 'equivalent resistance does not match original spec')

        d = b ** a
        self.assertTrue(d.noisy)
        self.assertTrue(abs(d.nfmin[0] - nf) < 1.e-6, 'noise figure does not match original spec')
        self.assertTrue(abs(d.z_opt[0] - 50.) < 1.e-6, 'optimal resistance does not match original spec')
        self.assertTrue(abs(d.rn[0] - 0.1159*50.) < 1.e-6, 'equivalent resistance does not match original spec')

        e = a ** a
        self.assertTrue(abs(e.nfmin[0] - (nf + (nf-1)/(10**2))) < 1.e-6, 'noise figure does not match Friis formula')

        self.assertTrue(a.noisy)
        self.assertTrue(abs(a.nfmin[0] - nf) < 1.e-6, 'noise figure was altered')
        self.assertTrue(abs(a.z_opt[0] - 50.) < 1.e-6, 'optimal resistance was altered')
        self.assertTrue(abs(a.rn[0] - 0.1159*50.) < 1.e-6, 'equivalent resistance was altered')

        tem = DistributedCircuit(z0_port = 50)
        inductor = tem.inductor(1e-9).interpolate(a.frequency)

        f = inductor ** a
        expected_zopt = 50 - 2j*np.pi*1e+9*1e-9
        self.assertTrue(abs(f.z_opt[0] - expected_zopt) < 1.e-6, 'optimal resistance was not 50 ohms - inductor')


        return

    def test_noise_dc_extrapolation(self):
        ntwk = self.ntwk_noise
        ntwk = ntwk["0-1.5GHz"] # using only the first samples, as ntwk_noise has duplicate x value
        s11 = ntwk.s11
        s11_dc = s11.extrapolate_to_dc(kind='cubic')

    def test_dc_extrapolation_dc_sparam(self):
        zeros = np.zeros((self.ntwk1.nports, self.ntwk1.nports))
        net_dc = self.ntwk1.extrapolate_to_dc(dc_sparam=zeros)
        net_dc = self.ntwk1.extrapolate_to_dc(dc_sparam=zeros.tolist())

    @pytest.mark.skipif("matplotlib" not in sys.modules, reason="Requires matplotlib in sys.modules.")
    def test_noise_deembed(self):


        f1_ =[75.5, 75.5]
        f2_=[75.5, 75.6]
        npt_ = [1,2]  # single freq and multifreq
        for f1,f2,npt in zip (f1_,f2_,npt_) :
          freq=rf.Frequency(f1,f2,npt,'ghz')
          ntwk4_n = rf.Network(os.path.join(self.test_dir,'ntwk4_n.s2p'), f_unit='GHz').interpolate(freq)
          ntwk4 = rf.Network(os.path.join(self.test_dir,'ntwk4.s2p'),f_unit='GHz').interpolate(freq)
          thru = rf.Network(os.path.join(self.test_dir,'thru.s2p'),f_unit='GHz').interpolate(freq)

          ntwk4_thru = ntwk4 ** thru
          ntwk4_thru.name ='ntwk4_thru'
          retrieve_thru =  ntwk4.inv ** ntwk4_thru
          retrieve_thru.name ='retrieve_thru'
          self.assertEqual(retrieve_thru, thru)
          self.assertTrue(ntwk4_thru.noisy)
          self.assertTrue(retrieve_thru.noisy)
          self.assertTrue((abs(thru.nfmin - retrieve_thru.nfmin)        < 1.e-6).all(),
                          'nf not retrieved by noise deembed')
          self.assertTrue((abs(thru.rn    - retrieve_thru.rn)           < 1.e-6).all(),
                          'rn not retrieved by noise deembed')
          self.assertTrue((abs(thru.z_opt - retrieve_thru.z_opt)        < 1.e-6).all(),
                          'noise figure does not match original spec')

          ntwk4_n_thru = ntwk4_n ** thru
          ntwk4_n_thru.name ='ntwk4_n_thru'
          retrieve_n_thru =  ntwk4_n.inv ** ntwk4_n_thru
          retrieve_n_thru.name ='retrieve_n_thru'
          self.assertTrue(ntwk4_n_thru.noisy)
          self.assertEqual(retrieve_n_thru, thru)
          self.assertTrue(ntwk4_n_thru.noisy)
          self.assertTrue(retrieve_n_thru.noisy)
          self.assertTrue((abs(thru.nfmin - retrieve_n_thru.nfmin) < 1.e-6).all(),
                          'nf not retrieved by noise deembed')
          self.assertTrue((abs(thru.rn    - retrieve_n_thru.rn)    < 1.e-6).all(),
                          'rn not retrieved by noise deembed')
          self.assertTrue((abs(thru.z_opt - retrieve_n_thru.z_opt) < 1.e-6).all(),
                          'noise figure does not match original spec')

          tuner, x,y,g = tuner_constellation()
          newnetw = thru.copy()
          nfmin_set=4.5
          gamma_opt_set=complex(.7,-0.2)
          rn_set=1
          newnetw.set_noise_a(thru.noise_freq, nfmin_db=nfmin_set, gamma_opt=gamma_opt_set, rn=rn_set )
          z = newnetw.nfdb_gs(g)[:,0]
          freq = thru.noise_freq.f[0]

          if "matplotlib" in sys.modules:
            gamma_opt_rb, nfmin_rb = plot_contour(freq,x,y,z, min0max1=0, graph=False)
            self.assertTrue(abs(nfmin_set - nfmin_rb) < 1.e-2, 'nf not retrieved by noise deembed')
            self.assertTrue(abs(gamma_opt_rb.s[0,0,0] - gamma_opt_set) < 1.e-1, 'nf not retrieved by noise deembed')

    def test_noise_interpolation(self):

        # Get a handle for the test network. Note that the s-parameter frequency range is beyond that of the NF data
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk_noise_interp.s2p'))

        # Pulling out the noise data should interpolate and fill extrapolated values with the default np.nan
        self.assertIn(True, np.isnan(ntwk.copy().n))

        # Check that a particular fill value is NOT in the noise data
        fill_val = 12345 + 1j * 67890
        new_ntwk = ntwk.copy()
        self.assertNotIn(fill_val, new_ntwk.n)

        # Change the interpolation fill value and check if it filled in properly
        new_ntwk = ntwk.copy()
        new_ntwk.noise_fill_value = fill_val
        self.assertIn(fill_val, new_ntwk.n)

    def test_spar_interpolation(self):

        # Create new frequency vectors beyond the original limits of ntwk1
        new_freqs_low = rf.Frequency.from_f(self.ntwk1.f / 2, unit="Hz")
        new_freqs_high = rf.Frequency.from_f(self.ntwk1.f * 2, unit="Hz")

        # Test that no kwargs results in ValueErrors
        for new_f in (new_freqs_low, new_freqs_high):
            new_ntwk = self.ntwk1.copy()
            with self.assertRaises(ValueError) as context:
                new_ntwk.resample(new_f)

        # Test that kwargs can let the resampling work
        for new_f in (new_freqs_low, new_freqs_high):
            new_ntwk = self.ntwk1.copy()
            new_ntwk.resample(new_f, bounds_error=False)
            self.assertIn(True, np.isnan(new_ntwk.s))

    def test_se2gmm2se(self):
        # Test that se2gmm followed by gmm2se gives back the original network
        for z0 in [None, 45, 75]:
            ntwk4 = rf.Network(os.path.join(self.test_dir, 'cst_example_4ports.s4p'))

            if z0 is not None:
                ntwk4.z0 = z0

            ntwk4t = ntwk4.copy()
            self.assertTrue(np.all(ntwk4t.port_modes == "S"))
            ntwk4t.se2gmm(p=2)
            self.assertTrue(np.all(ntwk4t.port_modes == ["D", "D", "C", "C"]))
            ntwk4t.gmm2se(p=2)
            self.assertTrue(np.all(ntwk4t.port_modes == "S"))

            self.assertTrue(np.allclose(ntwk4.s, ntwk4t.s))
            self.assertTrue(np.allclose(ntwk4.z0, ntwk4t.z0))

    def test_se2gmm(self):
        # Test mixed mode conversion of two parallel thrus
        se = np.zeros((1,4,4), dtype=complex)
        se[:,2,0] = 1
        se[:,0,2] = 1
        se[:,3,1] = 1
        se[:,1,3] = 1
        net = rf.Network(s=se, f=[1], z0=50)
        gmm = np.zeros((1,4,4), dtype=complex)
        gmm[:,0,1] = 1
        gmm[:,1,0] = 1
        gmm[:,2,3] = 1
        gmm[:,3,2] = 1
        net.se2gmm(p=2)
        self.assertTrue(np.allclose(net.z0, np.array([[100,100,25,25]])))
        self.assertTrue(np.allclose(net.s, gmm))

    def test_se2gmm_3port(self):
        # Test mixed mode conversion of ideal balun
        se = np.zeros((1,3,3), dtype=complex)
        se[:,2,0] =  1/2**0.5
        se[:,0,2] =  1/2**0.5
        se[:,2,1] = -1/2**0.5
        se[:,1,2] = -1/2**0.5
        net = rf.Network(s=se, f=[1], z0=50)
        gmm = np.zeros((1,3,3), dtype=complex)
        gmm[:,0,2] = 1
        gmm[:,2,0] = 1
        self.assertTrue(np.all(net.port_modes == "S"))
        net.se2gmm(p=1)
        self.assertTrue(np.all(net.port_modes == ["D", "C", "S"]))
        self.assertTrue(np.allclose(net.z0, np.array([[100,25,50]])))
        self.assertTrue(np.allclose(net.s, gmm))

    def test_se2gmm_renorm(self):
        # Test that se2gmm renormalization is compatible with network renormalization
        freq = rf.Frequency(1, 1, 1, unit='GHz')
        # Single-ended ports
        for s_def in rf.S_DEFINITIONS:
            for ports in range(2, 10):
                # Number of differential pairs to convert
                for p in range(0, ports//2 + 1):
                    # Create a random network, z0=50
                    s_random = self.rng.uniform(-1, 1, (1, ports, ports)) +\
                                1j * self.rng.uniform(-1, 1, (1, ports, ports))
                    net = rf.Network(s=s_random, frequency=freq, z0=50)
                    net_original = net.copy()
                    net_renorm = net.copy()

                    # Random z0 for mixed mode ports
                    z0 = self.rng.uniform(1, 100, 2*p) +\
                            1j * self.rng.uniform(-100, 100, 2*p)

                    # Convert net to mixed mode with random z0
                    net.se2gmm(p=p, z0_mm=z0, s_def=s_def)

                    # Convert net_renorm to mixed mode with different z0
                    z0_mm = np.zeros(2*p, dtype=complex)
                    z0_mm[:p] = 100
                    z0_mm[p:] = 25
                    net_renorm.se2gmm(p=p, z0_mm=z0_mm, s_def=s_def)

                    if p > 0:
                        # S-parameters should be different
                        self.assertFalse(np.allclose(net.s, net_renorm.s))
                    else:
                        # Same if no differential ports
                        self.assertTrue(np.allclose(net.s, net_renorm.s))

                    # Renormalize net_renorm to the random z0
                    # net and net_renorm should match after this
                    # Single-ended ports stay 50 ohms
                    full_z0 = 50 * np.ones(ports, dtype=complex)
                    full_z0[:2*p] = z0
                    net_renorm.renormalize(z_new=full_z0, s_def=s_def)

                    # Nets should match now
                    self.assertTrue(np.allclose(net.z0, net_renorm.z0))
                    self.assertTrue(np.allclose(net.s, net_renorm.s))

                    # Test that we get the original network back
                    net.gmm2se(p=p, z0_se=50, s_def=s_def)
                    self.assertTrue(np.allclose(net.z0, net_original.z0))
                    self.assertTrue(np.allclose(net.s, net_original.s))


    def test_s_active(self):
        """
        Test the active s-parameters of a 2-ports network
        """
        s_ref = self.ntwk1.s
        # s_act should be equal to s11 if a = [1,0]
        np.testing.assert_array_almost_equal(rf.s2s_active(s_ref, [1, 0])[:,0], s_ref[:,0,0])
        # s_act should be equal to s22 if a = [0,1]
        np.testing.assert_array_almost_equal(rf.s2s_active(s_ref, [0, 1])[:,1], s_ref[:,1,1])
        # s_act should be equal to s11 if a = [1,0]
        np.testing.assert_array_almost_equal(self.ntwk1.s_active([1, 0])[:,0], s_ref[:,0,0])
        # s_act should be equal to s22 if a = [0,1]
        np.testing.assert_array_almost_equal(self.ntwk1.s_active([0, 1])[:,1], s_ref[:,1,1])

    def test_vswr_active(self):
        """
        Test the active vswr-parameters of a 2-ports network
        """
        s_ref = self.ntwk1.s
        vswr_ref = self.ntwk1.s_vswr
        # vswr_act should be equal to vswr11 if a = [1,0]
        np.testing.assert_array_almost_equal(rf.s2vswr_active(s_ref, [1, 0])[:,0], vswr_ref[:,0,0])
        # vswr_act should be equal to vswr22 if a = [0,1]
        np.testing.assert_array_almost_equal(rf.s2vswr_active(s_ref, [0, 1])[:,1], vswr_ref[:,1,1])
        # vswr_act should be equal to vswr11 if a = [1,0]
        np.testing.assert_array_almost_equal(self.ntwk1.vswr_active([1, 0])[:,0], vswr_ref[:,0,0])
        # vswr_act should be equal to vswr22 if a = [0,1]
        np.testing.assert_array_almost_equal(self.ntwk1.vswr_active([0, 1])[:,1], vswr_ref[:,1,1])

    def test_twport_to_nport(self):
        fpoints = 2
        nports = 4
        s = np.ones((fpoints, 2, 2), dtype=complex)
        f = rf.F(1, 10, fpoints, unit='GHz')
        twoport = rf.Network(s=s, frequency=f)
        nport = rf.twoport_to_nport(twoport, 0, 1, nports)
        zeros = np.zeros(fpoints, dtype=complex)
        for i in range(nports):
            for j in range(nports):
                if i in [0, 1] and j in [0, 1]:
                    np.testing.assert_array_almost_equal(nport.s[:,i,j], twoport.s[:,i,j])
                else:
                    np.testing.assert_array_almost_equal(nport.s[:,i,j], zeros)


    def test_generate_subnetworks_nportsbelow10(self):
        """
        Testing generation of one-port subnetworks for ports below 10
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))
        np.testing.assert_array_almost_equal(
            ntwk.s[:,4,5],
            ntwk.s5_6.s[:,0,0]
        )

    def test_generate_subnetworks_nportsabove10(self):
        """
        Testing generation of one-port subnetworks for ports above 10
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))
        np.testing.assert_array_almost_equal(
            ntwk.s[:,1,15],
            ntwk.s2_16.s[:,0,0]
        )


    def test_generate_subnetwork_nounderscore(self):
        """
        Testing no underscore alias of one-port subnetworks for ports below 10.
        This is for backward compatibility with old code.
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))

        np.testing.assert_array_almost_equal(
            ntwk.s[:, 8, 8],
            ntwk.s99.s[:,0,0]
        )


    def test_generate_subnetworks_allports(self):
        """
        Testing generation of all one-port subnetworks in case of edge problems.
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))
        for m in range(ntwk.nports):
            for n in range(ntwk.nports):
                np.testing.assert_array_almost_equal(
                    ntwk.s[:,m,n],
                    getattr(ntwk, f's{m+1}_{n+1}').s[:,0,0]
                )


    def test_subnetwork(self):
        """ Test subnetwork creation and recombination """
        tee = rf.data.tee # 3 port Network

        # modify the z0 to dummy values just to check it works for any z0
        tee.z0 =self.rng.random(3) + 1j*self.rng.random(3)

        # Using rf.subnetwork()
        # 2 port Networks as if one measures the tee with a 2 ports VNA
        tee12 = rf.subnetwork(tee, [0, 1])  # 2 port Network from ports 1 & 2, port 3 matched
        tee23 = rf.subnetwork(tee, [1, 2])  # 2 port Network from ports 2 & 3, port 1 matched
        tee13 = rf.subnetwork(tee, [0, 2])  # 2 port Network from ports 1 & 3, port 2 matched
        # recreate the original 3 ports Network from the thee 2-port sub-Networks
        ntw_list = [tee12, tee23, tee13]
        tee2 = rf.n_twoports_2_nport(ntw_list, nports=3)
        self.assertTrue(tee2 == tee)

        # Same from the subnetwork() method.
        tee12 = tee.subnetwork([0, 1])
        tee23 = tee.subnetwork([1, 2])
        tee13 = tee.subnetwork([0, 2])
        ntw_list = [tee12, tee23, tee13]
        tee2 = rf.n_twoports_2_nport(ntw_list, nports=3)
        self.assertTrue(tee2 == tee)

    def test_subnetwork_port_names(self):
        """ Test that subnetwork keeps port_names property. Issue #429 """
        self.ntwk1.port_names = ['A', 'B']
        extract_ports = ['A']  # list of port names to extract
        extract_ports_idx = [self.ntwk1.port_names.index(p) for p in extract_ports]  # get port indices
        sub_nwk1 = self.ntwk1.subnetwork(extract_ports_idx)
        self.assertEqual(sub_nwk1.port_names, extract_ports)

        tee = rf.data.tee
        tee.port_names = ['A', 'B', 'C']
        extract_ports = ['A', 'C']
        extract_ports_idx = [tee.port_names.index(p) for p in extract_ports]
        sub_nwk = tee.subnetwork(extract_ports_idx)
        self.assertEqual(sub_nwk.port_names, extract_ports)

    def test_invalid_freq(self):

        dat = np.arange(5)
        dat[4] = 3

        with self.assertWarns(InvalidFrequencyWarning):
            freq = rf.Frequency.from_f(dat, unit='Hz')

        s = np.tile(dat,4).reshape(2,2,-1).T

        with self.assertWarns(InvalidFrequencyWarning):
            net = rf.Network(s=s, frequency=freq, z0=dat)

        net.drop_non_monotonic_increasing()

        self.assertTrue(np.allclose(net.f, freq.f[:4]))
        self.assertTrue(np.allclose(net.s, s[:4]))
        self.assertFalse(np.allclose(net.s.shape, s.shape))

    def test_stability(self):
        net = rf.Network(f=[1], s=[[0, 1],[0, 0]], z0=50)
        self.assertTrue(net.stability == [np.inf])

        net = rf.Network(f=[1], s=[[0, 1],[1, 0]], z0=50)
        self.assertTrue(net.stability == [1])

        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.stability

    def test_equality(self):
        s = self.rng.standard_normal((10, 2, 2))
        f1 = np.arange(10)
        f2 = np.arange(10)
        n1 = rf.Network(s=s,f=f1)
        n2 = rf.Network(s=s,f=f2)
        self.assertTrue(n1 == n2)

        f2 = np.arange(11)
        n2 = rf.Network(s=s,f=f2)
        self.assertFalse(n1 == n2)

        f2 = np.arange(10)
        n2 = rf.Network(s=s,f=f2)
        n2.s_def = 'pseudo'
        self.assertTrue(n1 == n2)

        n2.z0 = n2.z0 +np.array([0+0.00001j])
        self.assertTrue(n1 == n2)

        n2.z0 = n2.z0 +np.array([1+0j])
        self.assertFalse(n1 == n2)

        f2 = 10 * f1
        n2 = rf.Network(s=s,f=f2)
        self.assertFalse(n1 == n2)

    def test_stability_circle(self):
        # Check whether the load stability circle agrees with that calculated with ADS
        load_stability_circle_ads = np.loadtxt(os.path.join(self.test_dir, 'load_stability_circle_ads.csv'),
                                                encoding='utf-8', delimiter=',')

        assert np.allclose(
            rf.complex_2_magnitude(self.fet['30GHz'].stability_circle(target_port=1, npoints=6)[:,0]),
            load_stability_circle_ads[:,0]
        )

        assert np.allclose(
            rf.complex_2_degree(self.fet['30GHz'].stability_circle(target_port=1, npoints=6)[:,0]),
            load_stability_circle_ads[:,1]
        )

        # Check whether the source stability circle agrees with that calculated with ADS
        source_stability_circle_ads = np.loadtxt(os.path.join(self.test_dir, 'source_stability_circle_ads.csv'),
                                                  encoding='utf-8', delimiter=',')

        assert np.allclose(
            rf.complex_2_magnitude(self.fet['30GHz'].stability_circle(target_port=0, npoints=6)[:,0]),
            source_stability_circle_ads[:,0]
        )

        assert np.allclose(
            rf.complex_2_degree(self.fet['30GHz'].stability_circle(target_port=0, npoints=6)[:,0]),
            source_stability_circle_ads[:,1]
        )

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.stability_circle(target_port=1)

        # Check whether an error is raised when the number of points is not positive.
        with pytest.raises(ValueError):
            net.stability_circle(target_port=1, npoints=0)

        # Check whether an error is raised when an incorrect target_port is specified.
        with pytest.raises(ValueError):
            net.stability_circle(target_port='foobar')

    def test_gain_circle(self):
        # Check whether the load stability circle agrees with that calculated with ADS
        load_gain_circle_ads = np.loadtxt(os.path.join(self.test_dir, 'load_gain_circle_ads.csv'), encoding='utf-8',
                                           delimiter=',')

        assert np.allclose(
            rf.complex_2_magnitude(self.fet['30GHz'].gain_circle(target_port=1, gain=1.0, npoints=6)[:,0]),
            load_gain_circle_ads[:,0],
        )

        assert np.allclose(
            rf.complex_2_degree(self.fet['30GHz'].gain_circle(target_port=1, gain=1.0, npoints=6)[:,0]),
            load_gain_circle_ads[:,1],
        )

        # Check whether the source stability circle agrees with that calculated with ADS
        source_gain_circle_ads = np.loadtxt(os.path.join(self.test_dir, 'source_gain_circle_ads.csv'),
                                             encoding='utf-8', delimiter=',')


        assert np.allclose(
            rf.complex_2_magnitude(self.fet['30GHz'].gain_circle(target_port=0, gain=1.0, npoints=6)[:,0]),
            source_gain_circle_ads[:,0],
        )

        assert np.allclose(
            rf.complex_2_degree(self.fet['30GHz'].gain_circle(target_port=0, gain=1.0, npoints=6)[:,0]),
            source_gain_circle_ads[:,1],
        )

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.gain_circle(target_port=1, gain=2.0)

        # Check whether an error is raised when the number of points is not positive.
        with pytest.raises(ValueError):
            net.gain_circle(target_port=1, gain=2.0, npoints=0)

        # Check whether an error is raised when an incorrect target_port is specified.
        with pytest.raises(ValueError):
            net.gain_circle(target_port='foobar', gain=2.0)

        # Check whether the specified gain is too large.
        with pytest.raises(RuntimeWarning):
            self.fet['30GHz'].gain_circle(target_port=1, gain=100)

    def test_nf_circle(self):
        # Check whether the noise figure circle agrees with that calculated with Microwave Office
        nf_circle_mwo = np.loadtxt(os.path.join(self.test_dir, 'nf_circle_mwo.csv'), encoding='utf-8',
                                           delimiter=',')

        assert np.allclose(
            self.ntwk_noise["1GHz"].nf_circle(nf=1.0, npoints=6).flatten().real,
            nf_circle_mwo[:6,0],
        )
        assert np.allclose(
            self.ntwk_noise["1GHz"].nf_circle(nf=1.0, npoints=6).flatten().imag,
            nf_circle_mwo[:6,1],
        )

        assert np.allclose(
            self.ntwk_noise["2GHz"].nf_circle(nf=2.0, npoints=6).flatten().real,
            nf_circle_mwo[6:12,0],
        )
        assert np.allclose(
            self.ntwk_noise["2GHz"].nf_circle(nf=2.0, npoints=6).flatten().imag,
            nf_circle_mwo[6:12,1],
        )

        # Check whether an error is raised when the network is not 2 port.
        net = rf.Network(f=[1], s=np.eye(3), z0=50)
        with pytest.raises(ValueError):
            net.nf_circle(nf=1.0)

        # Check whether an error is raised when the number of points is not positive.
        with pytest.raises(ValueError):
            net.nf_circle(nf=1.0, npoints=0)

        # Check whether an error is raised when the network is missing noise data.
        with pytest.raises(ValueError):
            self.ntwk1.nf_circle(nf=1.0, npoints=0)

        # Check whether the specified noise figure is too small.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.ntwk_noise['1GHz'].nf_circle(nf=0.1)

            # Check that a warning was raised
            assert len(w) > 0, "Expected a warning to be raised"

            # Check that the warning is a RuntimeWarning
            assert any(item.category is RuntimeWarning for item in w), "Expected RuntimeWarning was not raised"

    def test_de_embed_by_floordiv(self):
        ntwk_result_1 = self.ntwk1 // self.ntwk2
        ntwk_result_2 = self.ntwk1 // (self.ntwk2)
        np.testing.assert_array_almost_equal(ntwk_result_1.s, ntwk_result_2.s)

        # By definition A // B      => B.inv * A
        # By definition A // [B, C] => B.inv * A * C.inv
        # So, A // [B, C] should be equal to (A // B) * C.inv
        ntwk_result_3 = self.ntwk1 // (self.ntwk2, self.ntwk3)
        ntwk_result_4 = (self.ntwk1 // self.ntwk2) ** self.ntwk3.inv
        np.testing.assert_array_almost_equal(ntwk_result_3.s, ntwk_result_4.s)

        # Check weather an error is raised when more than two networks are specified
        with pytest.raises(ValueError):
            ntwk_result_3 = self.ntwk1 // (self.ntwk1, self.ntwk2, self.ntwk3)

    def test_fmt_trace_name(self):
        # Test trace name of differential thru
        s = np.zeros((1,4,4), dtype=complex)
        s[:,2,0] = 1
        s[:,0,2] = 1
        s[:,3,1] = 1
        s[:,1,3] = 1
        # single-ended
        se_thru = rf.Network(s=s, f=[1], z0=50)
        self.assertTrue(np.all(se_thru.port_modes == "S"))
        self.assertTrue(se_thru._fmt_trace_name(0, 0) == "11")
        self.assertTrue(se_thru._fmt_trace_name(1, 0) == "21")
        mm_thru = se_thru.copy()
        mm_thru.se2gmm(p=2)
        self.assertTrue(np.all(mm_thru.port_modes == ["D", "D", "C", "C"]))
        self.assertTrue(mm_thru._fmt_trace_name(0, 0) == "dd11")
        self.assertTrue(mm_thru._fmt_trace_name(1, 0) == "dd21")
        self.assertTrue(mm_thru._fmt_trace_name(2, 2) == "cc33")
        self.assertTrue(mm_thru._fmt_trace_name(3, 2) == "cc43")
        self.assertTrue(mm_thru._fmt_trace_name(2, 0) == "cd31")
        self.assertTrue(mm_thru._fmt_trace_name(1, 3) == "dc24")

suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
