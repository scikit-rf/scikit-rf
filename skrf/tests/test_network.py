from numpy.core.fromnumeric import squeeze
import py
import pytest
from skrf.frequency import Frequency, InvalidFrequencyWarning
import unittest
import os
import io
import tempfile
import zipfile
import sys
import numpy as npy
from pathlib import Path
import pickle
import skrf as rf
import warnings

from skrf import setup_pylab
from skrf.media import CPW
from skrf.media import DistributedCircuit
from skrf.constants import S_DEFINITIONS
from skrf.networkSet import tuner_constellation
from skrf.plotting import plot_contour

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
        self.Fix = rf.concat_ports([l1, l1, l1, l1])
        self.DUT = rf.concat_ports([l2, l2, l2, l2])
        self.Meas = rf.concat_ports([l3, l3, l3, l3])
        self.Fix2 = rf.concat_ports([l1, l1, l1, l1], port_order='first')
        self.DUT2 = rf.concat_ports([l2, l2, l2, l2], port_order='first')
        self.Meas2 = rf.concat_ports([l3, l3, l3, l3], port_order='first')

    def test_timedomain(self):
        t = self.ntwk1.s11.s_time
        s = self.ntwk1.s11.s
        self.assertTrue(len(t)== len(s))
    def test_time_gate(self):
        ntwk = self.ntwk1
        gated = self.ntwk1.s11.time_gate(0,.2)

        self.assertTrue(len(gated)== len(ntwk))
    def test_time_transform(self):
        spb = (4, 5)
        data_rate = 5e9
        num_taps = (100, 101)
        for i in range(2):
            tps = 1. / spb[i] / data_rate
            num_points = spb[i] * num_taps[i]
            # Frequency terms should NOT contain Nyquist frequency if number of points is odd
            inc_nyq = True if num_points % 2 == 0 else False
            freq = npy.linspace(0, 1. / 2 / tps, num_points // 2 + 1, endpoint=inc_nyq)

            dut = self.ntwk1.copy()
            freq_valid = freq[npy.logical_and(freq >= dut.f[0], freq <= dut.f[-1])]
            dut.interpolate_self(rf.Frequency.from_f(freq_valid, unit='hz'))

            dut_dc = dut.extrapolate_to_dc()
            t, y = dut_dc.s21.impulse_response(n=num_points)
            self.assertEqual(len(t), num_points)
            self.assertEqual(len(y), num_points)
            self.assertTrue(npy.isclose(t[1] - t[0], tps))
            t, y = dut_dc.s21.step_response(n=num_points)
            self.assertEqual(len(t), num_points)
            self.assertEqual(len(y), num_points)
            self.assertTrue(npy.isclose(t[1] - t[0], tps))

    def test_impulse_response_dirac(self):
        """
        Test if the impulse response of a perfect transmission line is pure Dirac
        """
        f_points = 10
        freq = rf.Frequency.from_f(npy.arange(f_points), unit='Hz')
        s = npy.ones(10)
        netw = rf.Network(frequency=freq, s=s)

        n_lst = npy.arange(-1,2) + 2 * (f_points) - 2
        for n in n_lst:
            t,y = netw.impulse_response('boxcar', n=n)

            y_true = npy.zeros_like(y)
            y_true[t == 0] = 1
            npy.testing.assert_almost_equal(y, y_true)


    def test_time_transform_nonlinear_f(self):
        netw_nonlinear_f = rf.Network(os.path.join(self.test_dir, 'ntwk_arbitrary_frequency.s2p'))
        with self.assertRaises(NotImplementedError):
            netw_nonlinear_f.s11.step_response()

    def test_time_transform(self):
        with self.assertWarns(RuntimeWarning):
            self.ntwk1.s11.step_response()

    def test_time_transform_multiport(self):
        dut_dc = self.ntwk1.extrapolate_to_dc()

        y1 = npy.zeros((1000, dut_dc.nports, dut_dc.nports))

        for (i,j) in dut_dc.port_tuples:
            oneport = getattr(dut_dc, f's{i+1}{j+1}')
            t1, y1[:,i,j] = oneport.step_response(n=1000)

        t2, y2 = dut_dc.step_response(n=1000)

        npy.testing.assert_almost_equal(t1, t2)
        npy.testing.assert_almost_equal(y1, y2)

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
        self.assertFalse(npy.isclose(ntwk_hfss.z0[0,0], 50))

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

        filename = os.path.join(self.test_dir, 'hfss_oneport.s1p')
        with open(filename) as fid:
            data = fid.read()
            sio = io.StringIO(data)
            sio.name = os.path.basename(filename) # hack a bug to touchstone reader
            rf.Network(sio)

    def test_constructor_from_parameters(self):
        """Test creating Network from all supported parameters
        with default z0 and specified z0.
        """
        random_z0 = npy.random.uniform(0.1, 1, (1, 2)) + \
                1j*npy.random.uniform(0.1, 1, (1, 2))
        for z0 in [None, random_z0]:
            for param in rf.Network.PRIMARY_PROPERTIES:
                params = npy.random.uniform(0.1, 1, (1, 2, 2)) + \
                        1j*npy.random.uniform(0.1, 1, (1, 2, 2))
                kwargs = {param:params}
                if z0 is not None:
                    kwargs['z0'] = z0
                net = rf.Network(**kwargs)
                if z0 is not None:
                    npy.testing.assert_allclose(net.z0, z0)
                npy.testing.assert_allclose(getattr(net, param), params)

    def test_constructor_from_parameters2(self):
        """Test creating Network from all supported parameters
        with default z0 and specified z0.
        Multiple frequency points and z0 broadcasted
        """
        random_z0 = npy.random.uniform(0.1, 1, 2) + \
                1j*npy.random.uniform(0.1, 1, 2)
        for z0 in [None, random_z0]:
            for param in rf.Network.PRIMARY_PROPERTIES:
                params = npy.random.uniform(0.1, 1, (5, 2, 2)) + \
                        1j*npy.random.uniform(0.1, 1, (5, 2, 2))
                kwargs = {param:params}
                if z0 is not None:
                    kwargs['z0'] = z0
                net = rf.Network(**kwargs)
                if z0 is not None:
                    # Network z0 is broadcasted
                    npy.testing.assert_allclose(net.z0[0,:], z0)
                npy.testing.assert_allclose(getattr(net, param), params)

    def test_constructor_invalid_networks(self):
        with pytest.raises(Exception) as e_info:
            # z0 size doesn't match
            rf.Network(s=npy.zeros((2,2,2)), z0=[1,2,3])
        with pytest.raises(Exception) as e_info:
            # z0 size doesn't match, Z-parameters
            rf.Network(z=npy.zeros((2,2,2)), z0=[1,2,3])
        with pytest.raises(Exception) as e_info:
            # Invalid s shape, non-square matrix
            rf.Network(s=npy.zeros((2,2,1)))
        with pytest.raises(Exception) as e_info:
            # invalid s shape, too many dimensions
            rf.network(s=npy.zeros((1,2,2,2)))
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
        os.remove(os.path.join(self.test_dir, 'ntwk1Saved.s2p'))

    def test_write_touchstone(self):
        ports = 2
        s_random = npy.random.uniform(-1, 1, (self.freq.npoints, ports, ports)) +\
                1j * npy.random.uniform(-1, 1, (self.freq.npoints, ports, ports))
        random_z0 = npy.random.uniform(1, 100, (self.freq.npoints, ports)) +\
                    1j * npy.random.uniform(-100, 100, (self.freq.npoints, ports))
        ntwk = rf.Network(s=s_random, frequency=self.freq, z0=random_z0, name='test_ntwk', s_def='traveling')

        # Writing a network with non-constant z0 should raise
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)

        ntwk.s_def = 'power'
        # Should raise warning when write_z0 and s_def != 'traveling'.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            snp = ntwk.write_touchstone(return_string=True, write_z0=True)
            self.assertTrue(len(w) == 1)

        ntwk.s_def = 'traveling'
        # Ok if write_z0 specified
        snp = ntwk.write_touchstone(return_string=True, write_z0=True)

        # Test reading it back
        strio = io.StringIO(snp)
        # Required for reading touchstone file
        strio.name = f'StringIO.s{ports}p'
        ntwk2 = rf.Network(strio)

        npy.testing.assert_allclose(ntwk2.s, s_random)
        npy.testing.assert_allclose(ntwk2.z0, random_z0)

        # Renormalize output to 50 ohms
        snp = ntwk.write_touchstone(return_string=True, r_ref=50)

        # Network should not have been modified
        self.assertTrue(npy.all(ntwk.s == s_random))
        self.assertTrue(npy.all(ntwk.z0 == random_z0))

        # Read back the written touchstone
        strio = io.StringIO(snp)
        strio.name = f'StringIO.s{ports}p'
        ntwk2 = rf.Network(strio)

        # Renormalize original network to match the written one
        ntwk.renormalize(50)

        npy.testing.assert_allclose(ntwk.s, ntwk2.s)
        npy.testing.assert_allclose(ntwk.z0, ntwk2.z0)

        ntwk.z0[0, 0] = 1
        # Writing network with non-constant real z0 should fail without r_ref
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)
        ntwk.z0[0, 0] = 50

        ntwk.renormalize(50 + 1j)

        # Writing complex characteristic impedance should fail
        with pytest.raises(ValueError) as e_info:
            snp = ntwk.write_touchstone(return_string=True)

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

    def test_connect(self):
        self.assertEqual(rf.connect(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        xformer = rf.Network()
        xformer.frequency=(1,)
        xformer.s = ((0,1),(1,0))  # connects thru
        xformer.z0 = (50,25)  # transforms 50 ohm to 25 ohm
        c = rf.connect(xformer,0,xformer,1)  # connect 50 ohm port to 25 ohm port
        self.assertTrue(npy.all(npy.abs(c.s-rf.impedance_mismatch(50, 25)) < 1e-6))

    def test_connect_nport_2port(self):
        freq = rf.Frequency(1, 10, npoints=10, unit='GHz')

        # create a line which can be connected to each port
        med = rf.DefinedGammaZ0(freq)
        line = med.line(1, unit='m')
        line.z0 = [10, 20]

        for nport_portnum in [3,4,5,6,7,8]:

            # create a Nport network with port impedance i at port i
            nport = rf.Network()
            nport.frequency = freq
            nport.s = npy.zeros((10, nport_portnum, nport_portnum))
            nport.z0 = npy.arange(start=1, stop=nport_portnum + 1)

            # Connect the line to each port and check for port impedance
            for port in range(nport_portnum):
                nport_line = rf.connect(nport, port, line, 0)
                z0_expected = nport.z0
                z0_expected[:,port] = line.z0[:,1]
                npy.testing.assert_allclose(
                        nport_line.z0,
                        z0_expected
                    )

    def test_connect_no_frequency(self):
        """ Connecting 2 networks defined without frequency returns Error
        """
        # try to connect two networks defined without their frequency properties
        s = npy.random.rand(10, 2, 2)
        ntwk1 = rf.Network(s=s)
        ntwk2 = rf.Network(s=s)

        with self.assertRaises(ValueError):
            ntwk1**ntwk2

    def test_connect_complex_ports(self):
        """ Test that connecting two networks gives the same resulting network
        (same Z-parameters) with all s_def when using complex port impedances.
        """

        # Generate random Z-parameters for two networks
        z1 = npy.random.uniform(1, 100, size=(1, 2, 2)) +\
                1j*npy.random.uniform(-100, 100, size=(1, 2, 2))
        z2 = npy.random.uniform(1, 100, size=(1, 2, 2)) +\
                1j*npy.random.uniform(-100, 100, size=(1, 2, 2))

        # Port impedances
        z0_1 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)
        z0_2 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)
        z0_3 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)

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
            npy.testing.assert_almost_equal(net3_ref.z0, net12.z0)
            npy.testing.assert_almost_equal(net12.s, net3_ref.s)
            npy.testing.assert_almost_equal(net3_z, net3_ref.z)

    def test_connect_different_s_def(self):
        """ Test that connecting two networks gives the same resulting
        network (same Z-parameters) with all s_def when using complex port
        impedances.
        """

        # Generate random Z-parameters for two networks
        z1 = npy.random.uniform(1, 100, size=(1, 2, 2)) +\
                1j*npy.random.uniform(-100, 100, size=(1, 2, 2))
        z2 = npy.random.uniform(1, 100, size=(1, 2, 2)) +\
                1j*npy.random.uniform(-100, 100, size=(1, 2, 2))

        # Port impedances
        z0_1 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)
        z0_2 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)
        z0_3 = npy.random.uniform(1, 100, size=2) +\
                1j*npy.random.uniform(-100, 100, size=2)

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
                npy.testing.assert_almost_equal(net3_ref.z0, net12.z0)
                npy.testing.assert_almost_equal(net12.s, net3_ref.s)
                npy.testing.assert_almost_equal(net3_z, net3_ref.z)

    def test_interconnect_complex_ports(self):
        """ Test that connecting two complex ports in a network
        gives the same S-parameters with all s_def when the rest of
        the ports are real
        """
        for p in range(3, 5):
            z = npy.random.uniform(1, 100, size=(1, p, p)) +\
                    1j*npy.random.uniform(-100, 100, size=(1, p, p))
            z0 = npy.random.uniform(1, 100, size=(1, p)) + 1j*0
            z0[:, -2:] += 1j*npy.random.uniform(-100, 100, size=(1, 2))
            nets = []
            for s_def in rf.S_DEFINITIONS:
                net = rf.Network(s=npy.zeros((p,p)), f=1, z0=z0, s_def=s_def)
                net.z = z
                # Connect the last two complex ports together
                rf.innerconnect(net, p - 2, 2)
                nets.append(net)
            for net in nets[1:]:
                npy.testing.assert_almost_equal(net.s, net[0].s)
                self.assertTrue((net.z0 == net[0].z0).all())
                self.assertTrue(net.s_def != net[0].s_def)

    def test_delay(self):
        ntwk1_delayed = self.ntwk1.delay(1,'ns',port=0)
        self.assertTrue(
            npy.all(
                self.ntwk1.group_delay[:,:,:]
                -ntwk1_delayed.group_delay[:,:,:]
                -npy.array(
                    [[-1.e-09+0.j, -5.e-10+0.j],
                     [-5.e-10+0.j,  0.e+00+0.j]]
                ) < 1e-9
            )
        )
        ntwk2_delayed = self.ntwk2.delay(1,'ps',port=1)
        self.assertTrue(
            npy.all(
                self.ntwk2.group_delay[:,:,:]
                -ntwk2_delayed.group_delay[:,:,:]
                -npy.array(
                    [[ 0.e+00+0.j, -5.e-13+0.j],
                     [-5.e-13+0.j, -1.e-12+0.j]]
                ) < 1e-9
            )
        )

    def test_connect_multiports(self):
        a = rf.Network()
        a.frequency=(1,)
        a.s = npy.arange(16).reshape(4,4)
        a.z0 = npy.arange(4) + 1 #  Z0 should never be zero

        b = rf.Network()
        b.frequency=(1,)
        b.s = npy.arange(16).reshape(4,4)
        b.z0 = npy.arange(4)+10

        c=rf.connect(a,2,b,0,2)
        self.assertTrue((c.z0==[1,2,12,13]).all())

        d=rf.connect(a,0,b,0,3)
        self.assertTrue((d.z0==[4,13]).all())

    @pytest.mark.skip(reason="not supporting this function currently ")
    def test_connect_fast(self):
        self.assertEqual(rf.connect_fast(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        xformer = rf.Network()
        xformer.frequency=(1,)
        xformer.s = ((0,1),(1,0))  # connects thru
        xformer.z0 = (50,25)  # transforms 50 ohm to 25 ohm
        c = rf.connect_fast(xformer,0,xformer,1)  # connect 50 ohm port to 25 ohm port
        self.assertTrue(npy.all(npy.abs(c.s-rf.impedance_mismatch(50, 25)) < 1e-6))

    def test_flip(self):
        self.assertEqual(rf.connect(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        gain = rf.Network()
        gain.frequency=(1,)
        gain.s = ((0,2),(0.5,0))  # connects thru with gain of 2.0
        gain.z0 = (37,82)
        flipped = gain.copy()
        flipped.flip()
        c = rf.connect(gain,1,flipped,0)
        self.assertTrue(npy.all(npy.abs(c.s - npy.array([[0,1],[1,0]])) < 1e-6))

    def test_de_embed_by_inv(self):
        self.assertEqual(self.ntwk1.inv ** self.ntwk3, self.ntwk2)
        self.assertEqual(self.ntwk3 ** self.ntwk2.inv, self.ntwk1)
        self.assertEqual(self.Fix.inv ** self.Meas ** self.Fix.flipped().inv,
                         self.DUT)

    def test_plot_one_port_db(self):
        self.ntwk1.plot_s_db(0,0)

    def test_plot_one_port_deg(self):
        self.ntwk1.plot_s_deg(0,0)

    def test_plot_one_port_smith(self):
        self.ntwk1.plot_s_smith(0,0)

    def test_plot_two_port_db(self):
        self.ntwk1.plot_s_db()

    def test_plot_two_port_deg(self):
        self.ntwk1.plot_s_deg()

    def test_plot_two_port_smith(self):
        self.ntwk1.plot_s_smith()

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
                npy.testing.assert_allclose(rf.z2s(rf.s2z(net.s, net.z0, s_def=s_def), net.z0, s_def=s_def), net.s, atol=1e-3)
                npy.testing.assert_allclose(rf.y2s(rf.s2y(net.s, net.z0, s_def=s_def), net.z0, s_def=s_def), net.s, atol=1e-3)

    def test_conversions(self):
        #Converting to other format and back to S-parameters should return the original network
        s_random = npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2))
        ntwk_random = rf.Network(s=s_random, frequency=self.freq)
        for test_z0 in (50, 10, 90+10j, 4-100j):
            for test_ntwk in (self.ntwk1, self.ntwk2, self.ntwk3, ntwk_random):
                ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0)
                npy.testing.assert_allclose(rf.a2s(rf.s2a(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.z2s(rf.s2z(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.y2s(rf.s2y(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.h2s(rf.s2h(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.t2s(rf.s2t(ntwk.s)), ntwk.s)
        npy.testing.assert_allclose(rf.t2s(rf.s2t(self.Fix.s)), self.Fix.s)

    def test_multiport_conversions(self):
        #Converting to other format and back to S-parameters should return the original network
        for ports in range(3, 6):
            s_random = npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = npy.random.uniform(1, 100, (self.freq.npoints, ports)) +\
            1j * npy.random.uniform(-100, 100, (self.freq.npoints, ports))
            for test_z0 in (50, random_z0):
                for s_def in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0, s_def=s_def)
                    npy.testing.assert_allclose(rf.z2s(rf.s2z(ntwk.s, test_z0, s_def=s_def), test_z0, s_def=s_def), ntwk.s)
                    npy.testing.assert_allclose(rf.y2s(rf.s2y(ntwk.s, test_z0, s_def=s_def), test_z0, s_def=s_def), ntwk.s)

    def test_y_z_compatability(self):
        # Test that npy.linalg.inv(Z) == Y
        fpoints = 3
        for p in range(2, 6):
            s = npy.random.uniform(-1, 1, (fpoints, p, p)) + 1j * npy.random.uniform(-1, 1, (fpoints, p, p))
            random_z0 = npy.random.uniform(1, 100, (fpoints, p)) + 1j * npy.random.uniform(-100, 100, (fpoints, p))
            for test_z0 in (50, random_z0):
                for s_def in rf.S_DEFINITIONS:
                    z = rf.s2z(s, test_z0, s_def=s_def)
                    y = npy.linalg.inv(z)
                    npy.testing.assert_allclose(rf.y2s(y, test_z0, s_def=s_def), s)

    def test_unknown_s_def(self):
        # Test that Exception is raised when given unknown s_def
        s = npy.array([0]).reshape(1, 1, 1)
        z0 = npy.array([50])
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
            s_random = npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = npy.random.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*npy.random.uniform(-100, 100, size=(self.freq.npoints, ports))
            for test_z0 in (50, 20+60j, random_z0):
                for method in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=50)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(test_z0, method)
                    ntwk_renorm.renormalize(50, method)
                    npy.testing.assert_allclose(ntwk_renorm.s, ntwk.s)

    def test_sparam_renorm_s2s(self):
        """
        Test changing S-parameter definition with complex ports
        """
        for ports in range(2, 6):
            s_random = npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = npy.random.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*npy.random.uniform(-100, 100, size=(self.freq.npoints, ports))
            for def1 in rf.S_DEFINITIONS:
                for def2 in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=random_z0, s_def=def1)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def2)
                    npy.testing.assert_allclose(ntwk_renorm.z, ntwk.z)
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def1)
                    npy.testing.assert_allclose(ntwk_renorm.s, ntwk.s)
                    npy.testing.assert_allclose(ntwk_renorm.z0, ntwk.z0)

    def test_sparam_renorm_different_z0(self):
        """
        Test changing S-parameter definition with complex ports.
        renormalize handles cases with same z0 and different z0 before and
        after conversion with different method.
        """
        for ports in range(2, 6):
            s_random = npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, ports, ports))
            test_ntwk = rf.Network(s=s_random, frequency=self.freq)
            random_z0 = npy.random.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*npy.random.uniform(-100, 100, size=(self.freq.npoints, ports))
            random_z0_2 = npy.random.uniform(1, 100, size=(self.freq.npoints, ports)) +\
                        1j*npy.random.uniform(-100, 100, size=(self.freq.npoints, ports))
            for def1 in rf.S_DEFINITIONS:
                for def2 in rf.S_DEFINITIONS:
                    ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=random_z0, s_def=def1)
                    ntwk_renorm = ntwk.copy()
                    ntwk_renorm.renormalize(random_z0_2, s_def=def2)
                    npy.testing.assert_allclose(ntwk_renorm.z, ntwk.z)
                    ntwk_renorm.renormalize(ntwk.z0, s_def=def1)
                    npy.testing.assert_allclose(ntwk_renorm.s, ntwk.s)
                    npy.testing.assert_allclose(ntwk_renorm.z0, ntwk.z0)

    def test_setters(self):
        s_random = npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2)) + 1j * npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2))
        ntwk = rf.Network(s=s_random, frequency=self.freq)
        ntwk.z0 = npy.random.uniform(1, 100, len(ntwk.z0)) + 1j*npy.random.uniform(-100, 100, len(ntwk.z0))
        ntwk.s = ntwk.s
        npy.testing.assert_allclose(ntwk.s, s_random)
        ntwk.a = ntwk.a
        npy.testing.assert_allclose(ntwk.s, s_random)
        ntwk.z = ntwk.z
        npy.testing.assert_allclose(ntwk.s, s_random)
        ntwk.y = ntwk.y
        npy.testing.assert_allclose(ntwk.s, s_random)
        ntwk.t = ntwk.t
        npy.testing.assert_allclose(ntwk.s, s_random)
        ntwk.h = ntwk.h
        npy.testing.assert_allclose(ntwk.s, s_random)

    def test_s_def_setters(self):
        s_random = npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2)) +\
            1j * npy.random.uniform(-10, 10, (self.freq.npoints, 2, 2))
        z0_scalar = 50
        z0_array_port = npy.random.uniform(1, 100, 2)
        z0_array_freq = npy.random.uniform(1, 100, self.freq.npoints) +\
            1j*npy.random.uniform(-100, 100, self.freq.npoints)
        z0_array_freq_port = npy.random.uniform(1, 100, (self.freq.npoints,2)) +\
            1j*npy.random.uniform(-100, 100, (self.freq.npoints,2))
            
        for z0 in [z0_scalar, z0_array_port, z0_array_freq, z0_array_freq_port]:
            ntwk = rf.Network(s=s_random, frequency=self.freq, z0=z0)

            s_traveling = rf.s2s(s_random, z0, 'traveling', ntwk.s_def)
            s_power = rf.s2s(s_random, z0, 'power', ntwk.s_def)
            s_pseudo = rf.s2s(s_random, z0, 'pseudo', ntwk.s_def)

            ntwk.s_traveling = s_traveling
            npy.testing.assert_allclose(ntwk.s, s_random)
            ntwk.s_power = s_power
            npy.testing.assert_allclose(ntwk.s, s_random)
            ntwk.s_pseudo = s_pseudo
            npy.testing.assert_allclose(ntwk.s, s_random)
    
            npy.testing.assert_allclose(ntwk.s_traveling, s_traveling)
            npy.testing.assert_allclose(ntwk.s_power, s_power)
            npy.testing.assert_allclose(ntwk.s_pseudo, s_pseudo)

    def test_sparam_conversion_with_complex_char_impedance(self):
        """
        Renormalize a 2-port network wrt to complex characteristic impedances
        using power-waves definition of s-param
        Example based on scikit-rf issue #313
        """
        f0 = rf.Frequency(75.8, npoints=1, unit='GHz')
        s0 = npy.array([
                [-0.194 - 0.228j, -0.721 + 0.160j],
                [-0.721 + 0.160j, +0.071 - 0.204j]])
        ntw = rf.Network(frequency=f0, s=s0, z0=50, name='dut')

        # complex characteristic impedance to renormalize to
        zdut = 100 + 10j

        # reference solutions obtained from ANSYS Circuit or ADS (same res)
        # case 1: z0=[50, zdut]
        s_ref = npy.array([[
            [-0.01629813-0.29764199j, -0.6726785 +0.24747539j],
            [-0.6726785 +0.24747539j, -0.30104687-0.10693578j]]])
        npy.testing.assert_allclose(rf.z2s(ntw.z, z0=[50, zdut]), s_ref)
        npy.testing.assert_allclose(rf.renormalize_s(ntw.s, [50,50], [50,zdut]), s_ref)

        # case 2: z0=[zdut, zdut]
        s_ref = npy.array([[
            [-0.402829859501534 - 0.165007172677339j,-0.586542065592524 + 0.336098534178339j],
            [-0.586542065592524 + 0.336098534178339j,-0.164707376748782 - 0.21617153431756j]]])
        npy.testing.assert_allclose(rf.z2s(ntw.z, z0=[zdut, zdut]), s_ref)
        npy.testing.assert_allclose(rf.renormalize_s(ntw.s, [50,50], [zdut,zdut]), s_ref)

        # Comparing Z and Y matrices from reference ones (from ADS)
        # Z or Y matrices do not depend of characteristic impedances.
        # Precision is 1e-4 due to rounded results in ADS export files
        z_ref = npy.array([[
            [34.1507 -65.6786j, -37.7994 +73.7669j],
            [-37.7994 +73.7669j, 55.2001 -86.8618j]]])
        npy.testing.assert_allclose(ntw.z, z_ref, atol=1e-4)

        y_ref = npy.array([[
            [0.0926 +0.0368j, 0.0770 +0.0226j],
            [0.0770 +0.0226j, 0.0686 +0.0206j]]])
        npy.testing.assert_allclose(ntw.y, y_ref, atol=1e-4)

    def test_sparam_conversion_vs_sdefinition(self):
        """
        Check that power-wave or pseudo-waves scattering parameters definitions
        give same results for real characteristic impedances
        """
        f0 = rf.Frequency(75.8, npoints=1, unit='GHz')
        s_ref = npy.array([[  # random values
            [-0.1000 -0.2000j, -0.3000 +0.4000j],
            [-0.3000 +0.4000j, 0.5000 -0.6000j]]])
        ntw = rf.Network(frequency=f0, s=s_ref, z0=50, name='dut')

        # renormalize s parameter according one of the definition.
        # As characteristic impedances are all real, should be all equal
        npy.testing.assert_allclose(ntw.s, s_ref)
        npy.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='power'), s_ref)
        npy.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='pseudo'), s_ref)
        npy.testing.assert_allclose(rf.renormalize_s(ntw.s, 50, 50, s_def='traveling'), s_ref)

        # also check Z and Y matrices, just in case
        z_ref = npy.array([[
            [18.0000 -16.0000j, 20.0000 + 40.0000j],
            [20.0000 +40.0000j, 10.0000 -80.0000j]]])
        npy.testing.assert_allclose(ntw.z, z_ref, atol=1e-4)

        y_ref = npy.array([[
            [0.0251 +0.0023j, 0.0123 -0.0066j],
            [0.0123 -0.0066j, 0.0052 +0.0055j]]])
        npy.testing.assert_allclose(ntw.y, y_ref, atol=1e-4)

        # creating network by specifying s-params definition
        ntw_power = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='power')
        ntw_pseudo = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='pseudo')
        ntw_legacy = rf.Network(frequency=f0, s=s_ref, z0=50, s_def='traveling')
        self.assertTrue(ntw_power == ntw_pseudo)
        self.assertTrue(ntw_power == ntw_legacy)

    def test_network_from_z_or_y(self):
        ' Construct a network from its z or y parameters '
        # test for both real and complex char. impedance
        # and for 2 frequencies
        z0 = [npy.random.rand(), npy.random.rand()+1j*npy.random.rand()]
        freqs = npy.array([1, 2])
        # generate arbitrary complex z and y
        z_ref = npy.random.rand(2,3,3) + 1j*npy.random.rand(2,3,3)
        y_ref = npy.random.rand(2,3,3) + 1j*npy.random.rand(2,3,3)
        # create networks from z or y and compare ntw.z to the reference
        # check that the conversions work for all s-param definitions
        for s_def in S_DEFINITIONS:
            ntwk = rf.Network(s_def=s_def)
            ntwk.z0 = rf.fix_z0_shape(z0, 2, 3)
            ntwk.frequency = Frequency.from_f(freqs)
            # test #1: define the network directly from z
            ntwk.z = z_ref
            npy.testing.assert_allclose(ntwk.z, z_ref)
            # test #2: define the network from s, after z -> s (s_def is important)
            ntwk.s = rf.z2s(z_ref, z0, s_def=s_def)
            npy.testing.assert_allclose(ntwk.z, z_ref)
            # test #3: define the network directly from y
            ntwk.y = y_ref
            npy.testing.assert_allclose(ntwk.y, y_ref)
            # test #4: define the network from s, after y -> s (s_def is important)
            ntwk.s = rf.y2s(y_ref, z0, s_def=s_def)
            npy.testing.assert_allclose(ntwk.y, y_ref)

    def test_z0_pure_imaginary(self):
        ' Test cases where z0 is pure imaginary '
        # test that conversion to Z or Y does not give NaN for pure imag z0
        for s_def in S_DEFINITIONS:
            ntwk = rf.Network(s_def=s_def)
            ntwk.z0 = npy.array([50j, -50j])
            ntwk.frequency = Frequency.from_f(npy.array([1000]))
            ntwk.s = npy.random.rand(1,2,2) + npy.random.rand(1,2,2)*1j
            self.assertFalse(npy.any(npy.isnan(ntwk.z)))
            self.assertFalse(npy.any(npy.isnan(ntwk.y)))

    def test_z0_scalar(self):
        'Test a scalar z0'
        ntwk = rf.Network()
        ntwk.z0 = 1
        # Test setting the z0 before and after setting the s shape
        self.assertEqual(ntwk.z0, 1)
        ntwk.s = npy.random.rand(1,2,2)
        ntwk.z0 = 10
        self.assertTrue(npy.allclose(ntwk.z0, npy.full((1,2), 10)))

    def test_z0_vector(self):
        'Test a 1 dimensional z0'
        ntwk = rf.Network()
        z0 = [1,2]
        # Test setting the z0 before and after setting the s shape
        ntwk.z0 = [1,2] # Passing as List
        self.assertTrue(npy.allclose(ntwk.z0, npy.array(z0, dtype=complex)))
        ntwk.z0 = npy.array(z0[::-1]) # Passing as npy.array
        self.assertTrue(npy.allclose(ntwk.z0, npy.array(z0[::-1], dtype=complex)))

        # If the s-array has been set, the z0 value should broadcast to the required shape
        ntwk.s = npy.random.rand(3,2,2)
        ntwk.z0 = z0
        self.assertTrue(npy.allclose(ntwk.z0, npy.array([z0, z0, z0], dtype=complex)))

        # If the s-array has been set and we want to set z0 along the frequency axis,
        # wer require the frequency vector to be set too.
        # Unfortunately the frequency vector and the s shape can distinguish
        z0 = [1,2,3]
        ntwk.s = npy.random.rand(3,2,2)
        ntwk.z0 = z0[::-1]

        ntwk.frequency = Frequency.from_f([1,2,3])
        self.assertTrue(npy.allclose(ntwk.z0, npy.array([z0[::-1], z0[::-1]], dtype=complex).T))

    def test_z0_matrix(self):
        ntwk = rf.Network()
        z0 = [[1,2]]
        ntwk.z0 = z0
        self.assertTrue(npy.allclose(ntwk.z0, npy.array(z0, dtype=complex)))
        ntwk.z0 = npy.array(z0) + 1 # Passing as npy.array
        self.assertTrue(npy.allclose(ntwk.z0, npy.array(z0, dtype=complex)+1))

        # Setting the frequency is required to be set, as the matrix size is checked against the
        # frequency vector
        ntwk.s = npy.random.rand(1,2,2)
        ntwk.frequency = Frequency.from_f([1])
        ntwk.z0 = z0
        self.assertTrue(npy.allclose(ntwk.z0, npy.array(z0, dtype=complex)))

    def test_z0_assign(self):
        """ Test that z0 getter returns a reference to _z0 so that it can
        be assigned to with array indexing"""
        ntwk = rf.Network(s=npy.zeros((3,2,2)), f=[1,2,3])
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

        ntwk = rf.Network(s=npy.zeros((3,2,2)), f=[1,2,3], z0=npy.ones(3))
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

        ntwk = rf.Network(s=npy.zeros((3,2,2)), f=[1,2,3], z0=npy.ones((3,2)))
        ntwk.z0[0, 0] = 2
        self.assertTrue(ntwk.z0[0, 0] == 2)

    def test_yz(self):
        tinyfloat = 1e-12
        ntwk = rf.Network()
        ntwk.z0 = npy.array([28,75+3j])
        ntwk.frequency = Frequency.from_f(npy.array([1000, 2000]))
        ntwk.s = rf.z2s(npy.array([[[1+1j,5,11],[40,5,3],[16,8,9+8j]],
                                   [[1,20,3],[14,10,16],[27,18,-19-2j]]]))
        self.assertTrue((abs(rf.y2z(ntwk.y)-ntwk.z) < tinyfloat).all())
        self.assertTrue((abs(rf.y2s(ntwk.y, ntwk.z0)-ntwk.s) < tinyfloat).all())
        self.assertTrue((abs(rf.z2y(ntwk.z)-ntwk.y) < tinyfloat).all())
        self.assertTrue((abs(rf.z2s(ntwk.z, ntwk.z0)-ntwk.s) < tinyfloat).all())

    def test_mul(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a*a).s == npy.array([[[-3+4j]],[[-7+24j]]])).all())
        # operating on numbers
        self.assertTrue( ((2*a*2).s == npy.array([[[4+8j]],[[12+16j]]])).all())
        # operating on list
        self.assertTrue( ((a*[1,2]).s == npy.array([[[1+2j]],[[6+8j]]])).all())
        self.assertTrue( (([1,2]*a).s == npy.array([[[1+2j]],[[6+8j]]])).all())

    def test_sub(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a-a).s == npy.array([[[0+0j]],[[0+0j]]])).all())
        # operating on numbers
        self.assertTrue( ((a-(2+2j)).s == npy.array([[[-1+0j]],[[1+2j]]])).all())
        # operating on list
        self.assertTrue( ((a-[1+1j,2+2j]).s == npy.array([[[0+1j]],[[1+2j]]])).all())

    def test_div(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a/a).s == npy.array([[[1+0j]],[[1+0j]]])).all())
        # operating on numbers
        self.assertTrue( ((a/2.).s == npy.array([[[.5+1j]],[[3/2.+2j]]])).all())
        # operating on list
        self.assertTrue( ((a/[1,2]).s == npy.array([[[1+2j]],[[3/2.+2j]]])).all())

    def test_add(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        # operating on  networks
        self.assertTrue( ((a+a).s == npy.array([[[2+4j]],[[6+8j]]])).all())
        # operating on numbers
        self.assertTrue( ((a+2+2j).s == npy.array([[[3+4j]],[[5+6j]]])).all())
        # operating on list
        self.assertTrue( ((a+[1+1j,2+2j]).s == npy.array([[[2+3j]],[[5+6j]]])).all())
     

    def test_interpolate(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        freq = rf.F.from_f(npy.linspace(1,2,4), unit='ghz')
        b = a.interpolate(freq)
        # TODO: numerically test for correct interpolation

    def test_interpolate_rational(self):
        a = rf.N(f=npy.linspace(1,2,5),s=npy.linspace(0,1,5)*(1+1j),z0=1)
        freq = rf.F.from_f(npy.linspace(1,2,6,endpoint=True))
        b = a.interpolate(freq, kind='rational')
        self.assertFalse(any(npy.isnan(b.s)))
        # Test that the endpoints are the equal
        self.assertTrue(b.s[0] == a.s[0])
        self.assertTrue(b.s[-1] == a.s[-1])
        # Check that abs(S) is increasing
        self.assertTrue(all(npy.diff(npy.abs(b.s.flatten())) > 0))
        self.assertTrue(b.z0[0] == a.z0[0])

    def test_interpolate_linear(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=[1,2])
        freq = rf.F.from_f(npy.linspace(1,2,3,endpoint=True))
        b = a.interpolate(freq, kind='linear')
        self.assertFalse(any(npy.isnan(b.s)))
        # Test that the endpoints are the equal
        # Middle point can also be calculated in this case
        self.assertTrue(b.s[0] == a.s[0])
        self.assertTrue(b.s[1] == 0.5*(a.s[0] + a.s[1]))
        self.assertTrue(b.s[-1] == a.s[-1])
        # Check Z0 interpolation
        self.assertTrue(b.z0[0] == a.z0[0])
        self.assertTrue(b.z0[1] == 0.5*(a.z0[0] + a.z0[1]))
        self.assertTrue(b.z0[-1] == a.z0[-1])

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
                       z0=50 )

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
        s8p_mat = npy.identity(8, dtype=complex)
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
                       s=[[0, -1j/npy.sqrt(2), -1j/npy.sqrt(2)],
                          [-1j/npy.sqrt(2), 1./2, -1./2],
                          [-1j/npy.sqrt(2), -1./2, 1./2]],
                       z0=50)
        self.assertTrue(b.is_lossless(), 'This unmatched power divider is lossless.')
        return

    def test_noise(self):
        a = rf.Network(os.path.join(self.test_dir,'ntwk_noise.s2p'))

        nf = 10**(0.05)
        self.assertTrue(a.noisy)
        self.assertTrue(abs(a.nfmin[0] - nf) < 1.e-6, 'noise figure does not match original spec')
        self.assertTrue(abs(a.z_opt[0] - 50.) < 1.e-6, 'optimal resistance does not match original spec')
        self.assertTrue(abs(a.rn[0] - 0.1159*50.) < 1.e-6, 'equivalent resistance does not match original spec')
        self.assertTrue(npy.all(abs(a.g_opt) < 1.e-6), 'calculated optimal reflection coefficient does not match original coefficients')

        b = rf.Network(f=[1, 2],
                       s=[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                       z0=50).interpolate(a.frequency)
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

        tem = DistributedCircuit(z0=50)
        inductor = tem.inductor(1e-9).interpolate(a.frequency)

        f = inductor ** a
        expected_zopt = 50 - 2j*npy.pi*1e+9*1e-9
        self.assertTrue(abs(f.z_opt[0] - expected_zopt) < 1.e-6, 'optimal resistance was not 50 ohms - inductor')


        return

    def test_noise_dc_extrapolation(self):
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk_noise.s2p'))
        ntwk = ntwk["0-1.5GHz"] # using only the first samples, as ntwk_noise has duplicate x value
        s11 = ntwk.s11
        s11_dc = s11.extrapolate_to_dc(kind='cubic')

    def test_dc_extrapolation_dc_sparam(self):
        zeros = npy.zeros((self.ntwk1.nports, self.ntwk1.nports))
        net_dc = self.ntwk1.extrapolate_to_dc(dc_sparam=zeros)
        net_dc = self.ntwk1.extrapolate_to_dc(dc_sparam=zeros.tolist())

    def test_noise_deembed(self):


        f1_ =[75.5, 75.5] ; f2_=[75.5, 75.6] ; npt_ = [1,2]     # single freq and multifreq
        for f1,f2,npt in zip (f1_,f2_,npt_) :
          freq=rf.Frequency(f1,f2,npt,'ghz')
          ntwk4_n = rf.Network(os.path.join(self.test_dir,'ntwk4_n.s2p'), f_unit='GHz').interpolate(freq)
          ntwk4 = rf.Network(os.path.join(self.test_dir,'ntwk4.s2p'),f_unit='GHz').interpolate(freq)
          thru = rf.Network(os.path.join(self.test_dir,'thru.s2p'),f_unit='GHz').interpolate(freq)

          ntwk4_thru = ntwk4 ** thru                  ;ntwk4_thru.name ='ntwk4_thru'
          retrieve_thru =  ntwk4.inv ** ntwk4_thru    ;retrieve_thru.name ='retrieve_thru'
          self.assertEqual(retrieve_thru, thru)
          self.assertTrue(ntwk4_thru.noisy)
          self.assertTrue(retrieve_thru.noisy)
          self.assertTrue((abs(thru.nfmin - retrieve_thru.nfmin)        < 1.e-6).all(), 'nf not retrieved by noise deembed')
          self.assertTrue((abs(thru.rn    - retrieve_thru.rn)           < 1.e-6).all(), 'rn not retrieved by noise deembed')
          self.assertTrue((abs(thru.z_opt - retrieve_thru.z_opt)        < 1.e-6).all(), 'noise figure does not match original spec')

          ntwk4_n_thru = ntwk4_n ** thru                    ;ntwk4_n_thru.name ='ntwk4_n_thru'
          retrieve_n_thru =  ntwk4_n.inv ** ntwk4_n_thru    ;retrieve_n_thru.name ='retrieve_n_thru'
          self.assertTrue(ntwk4_n_thru.noisy)
          self.assertEqual(retrieve_n_thru, thru)
          self.assertTrue(ntwk4_n_thru.noisy)
          self.assertTrue(retrieve_n_thru.noisy)
          self.assertTrue((abs(thru.nfmin - retrieve_n_thru.nfmin) < 1.e-6).all(), 'nf not retrieved by noise deembed')
          self.assertTrue((abs(thru.rn    - retrieve_n_thru.rn)    < 1.e-6).all(), 'rn not retrieved by noise deembed')
          self.assertTrue((abs(thru.z_opt - retrieve_n_thru.z_opt) < 1.e-6).all(), 'noise figure does not match original spec')

          tuner, x,y,g = tuner_constellation()
          newnetw = thru.copy()
          nfmin_set=4.5; gamma_opt_set=complex(.7,-0.2); rn_set=1
          newnetw.set_noise_a(thru.noise_freq, nfmin_db=nfmin_set, gamma_opt=gamma_opt_set, rn=rn_set )
          z = newnetw.nfdb_gs(g)[:,0]
          freq = thru.noise_freq.f[0]
          gamma_opt_rb, nfmin_rb = plot_contour(freq,x,y,z, min0max1=0, graph=False)
          self.assertTrue(abs(nfmin_set - nfmin_rb) < 1.e-2, 'nf not retrieved by noise deembed')
          self.assertTrue(abs(gamma_opt_rb.s[0,0,0] - gamma_opt_set) < 1.e-1, 'nf not retrieved by noise deembed')


    def test_se2gmm2se(self):
        # Test that se2gmm followed by gmm2se gives back the original network
        for z0 in [None, 45, 75]:
            ntwk4 = rf.Network(os.path.join(self.test_dir, 'cst_example_4ports.s4p'))

            if z0 is not None:
                ntwk4.z0 = z0

            ntwk4t = ntwk4.copy()
            ntwk4t.se2gmm(p=2)
            ntwk4t.gmm2se(p=2)

            self.assertTrue(npy.allclose(ntwk4.s, ntwk4t.s))
            self.assertTrue(npy.allclose(ntwk4.z0, ntwk4t.z0))

    def test_se2gmm(self):
        # Test mixed mode conversion of two parallel thrus
        se = npy.zeros((1,4,4), dtype=complex)
        se[:,2,0] = 1
        se[:,0,2] = 1
        se[:,3,1] = 1
        se[:,1,3] = 1
        net = rf.Network(s=se, f=[1], z0=50)
        gmm = npy.zeros((1,4,4), dtype=complex)
        gmm[:,0,1] = 1
        gmm[:,1,0] = 1
        gmm[:,2,3] = 1
        gmm[:,3,2] = 1
        net.se2gmm(p=2)
        self.assertTrue(npy.allclose(net.z0, npy.array([[100,100,25,25]])))
        self.assertTrue(npy.allclose(net.s, gmm))

    def test_se2gmm_3port(self):
        # Test mixed mode conversion of ideal balun
        se = npy.zeros((1,3,3), dtype=complex)
        se[:,2,0] =  1/2**0.5
        se[:,0,2] =  1/2**0.5
        se[:,2,1] = -1/2**0.5
        se[:,1,2] = -1/2**0.5
        net = rf.Network(s=se, f=[1], z0=50)
        gmm = npy.zeros((1,3,3), dtype=complex)
        gmm[:,0,2] = 1
        gmm[:,2,0] = 1
        net.se2gmm(p=1)
        self.assertTrue(npy.allclose(net.z0, npy.array([[100,25,50]])))
        self.assertTrue(npy.allclose(net.s, gmm))

    def test_se2gmm_renorm(self):
        # Test that se2gmm renormalization is compatible with network renormalization

        # Single-ended ports
        for s_def in rf.S_DEFINITIONS:
            for ports in range(2, 10):
                # Number of differential pairs to convert
                for p in range(0, ports//2 + 1):
                    # Create a random network, z0=50
                    s_random = npy.random.uniform(-1, 1, (1, ports, ports)) +\
                                1j * npy.random.uniform(-1, 1, (1, ports, ports))
                    net = rf.Network(s=s_random, frequency=[1], z0=50)
                    net_original = net.copy()
                    net_renorm = net.copy()

                    # Random z0 for mixed mode ports
                    z0 = npy.random.uniform(1, 100, 2*p) +\
                            1j * npy.random.uniform(-100, 100, 2*p)

                    # Convert net to mixed mode with random z0
                    net.se2gmm(p=p, z0_mm=z0, s_def=s_def)

                    # Convert net_renorm to mixed mode with different z0
                    z0_mm = npy.zeros(2*p, dtype=complex)
                    z0_mm[:p] = 100
                    z0_mm[p:] = 25
                    net_renorm.se2gmm(p=p, z0_mm=z0_mm, s_def=s_def)

                    if p > 0:
                        # S-parameters should be different
                        self.assertFalse(npy.allclose(net.s, net_renorm.s))
                    else:
                        # Same if no differential ports
                        self.assertTrue(npy.allclose(net.s, net_renorm.s))

                    # Renormalize net_renorm to the random z0
                    # net and net_renorm should match after this
                    # Single-ended ports stay 50 ohms
                    full_z0 = 50 * npy.ones(ports, dtype=complex)
                    full_z0[:2*p] = z0
                    net_renorm.renormalize(z_new=full_z0, s_def=s_def)

                    # Nets should match now
                    self.assertTrue(npy.allclose(net.z0, net_renorm.z0))
                    self.assertTrue(npy.allclose(net.s, net_renorm.s))

                    # Test that we get the original network back
                    net.gmm2se(p=p, z0_se=50, s_def=s_def)
                    self.assertTrue(npy.allclose(net.z0, net_original.z0))
                    self.assertTrue(npy.allclose(net.s, net_original.s))


    def test_s_active(self):
        """
        Test the active s-parameters of a 2-ports network
        """
        s_ref = self.ntwk1.s
        # s_act should be equal to s11 if a = [1,0]
        npy.testing.assert_array_almost_equal(rf.s2s_active(s_ref, [1, 0])[:,0], s_ref[:,0,0])
        # s_act should be equal to s22 if a = [0,1]
        npy.testing.assert_array_almost_equal(rf.s2s_active(s_ref, [0, 1])[:,1], s_ref[:,1,1])
        # s_act should be equal to s11 if a = [1,0]
        npy.testing.assert_array_almost_equal(self.ntwk1.s_active([1, 0])[:,0], s_ref[:,0,0])
        # s_act should be equal to s22 if a = [0,1]
        npy.testing.assert_array_almost_equal(self.ntwk1.s_active([0, 1])[:,1], s_ref[:,1,1])

    def test_vswr_active(self):
        """
        Test the active vswr-parameters of a 2-ports network
        """
        s_ref = self.ntwk1.s
        vswr_ref = self.ntwk1.s_vswr
        # vswr_act should be equal to vswr11 if a = [1,0]
        npy.testing.assert_array_almost_equal(rf.s2vswr_active(s_ref, [1, 0])[:,0], vswr_ref[:,0,0])
        # vswr_act should be equal to vswr22 if a = [0,1]
        npy.testing.assert_array_almost_equal(rf.s2vswr_active(s_ref, [0, 1])[:,1], vswr_ref[:,1,1])
        # vswr_act should be equal to vswr11 if a = [1,0]
        npy.testing.assert_array_almost_equal(self.ntwk1.vswr_active([1, 0])[:,0], vswr_ref[:,0,0])
        # vswr_act should be equal to vswr22 if a = [0,1]
        npy.testing.assert_array_almost_equal(self.ntwk1.vswr_active([0, 1])[:,1], vswr_ref[:,1,1])


    def test_generate_subnetworks_nportsbelow10(self):
        """
        Testing generation of one-port subnetworks for ports below 10
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))
        npy.testing.assert_array_almost_equal(
            ntwk.s[:,4,5],
            ntwk.s5_6.s[:,0,0]
        )

    def test_generate_subnetworks_nportsabove10(self):
        """
        Testing generation of one-port subnetworks for ports above 10
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))
        npy.testing.assert_array_almost_equal(
            ntwk.s[:,1,15],
            ntwk.s2_16.s[:,0,0]
        )


    def test_generate_subnetwork_nounderscore(self):
        """
        Testing no underscore alias of one-port subnetworks for ports below 10.
        This is for backward compatibility with old code.
        """
        ntwk = rf.Network(os.path.join(self.test_dir,'ntwk.s32p'))

        npy.testing.assert_array_almost_equal(
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
                npy.testing.assert_array_almost_equal(
                    ntwk.s[:,m,n],
                    ntwk.__getattribute__(f's{m+1}_{n+1}').s[:,0,0]
                )


    def test_subnetwork(self):
        """ Test subnetwork creation and recombination """
        tee = rf.data.tee # 3 port Network

        # modify the z0 to dummy values just to check it works for any z0
        tee.z0 = npy.random.rand(3) + 1j*npy.random.rand(3)

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

        dat = npy.arange(5)
        dat[4] = 3

        with self.assertWarns(InvalidFrequencyWarning):
            freq = rf.Frequency.from_f(dat, unit='Hz')

        s = npy.tile(dat,4).reshape(2,2,-1).T

        with self.assertWarns(InvalidFrequencyWarning):
            net = rf.Network(s=s, frequency=freq, z0=dat)

        net.drop_non_monotonic_increasing()

        self.assertTrue(npy.allclose(net.f, freq.f[:4]))
        self.assertTrue(npy.allclose(net.s, s[:4]))
        self.assertFalse(npy.allclose(net.s.shape, s.shape))


suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
