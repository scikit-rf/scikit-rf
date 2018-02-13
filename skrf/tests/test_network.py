import unittest
import os
import tempfile
import six
import numpy as npy
import six.moves.cPickle as pickle
import skrf as rf
from nose.plugins.skip import SkipTest

from skrf import setup_pylab


class NetworkTestCase(unittest.TestCase):
    '''
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
    '''
    def setUp(self):
        '''
        this also tests the ability to read touchstone files
        without an error
        '''
        setup_pylab()
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.freq = rf.Frequency(75,110,101,'ghz')
        self.cpw =  rf.media.CPW(self.freq, w=10e-6, s=5e-6, ep_r=10.6)
        l1 = self.cpw.line(0.20, 'm', z0=50)
        l2 = self.cpw.line(0.07, 'm', z0=50)
        l3 = self.cpw.line(0.47, 'm', z0=50)
        self.Fix = rf.concat_ports([l1, l1, l1, l1])
        self.DUT = rf.concat_ports([l2, l2, l2, l2])
        self.Meas = rf.concat_ports([l3, l3, l3, l3])

    def test_timedomain(self):
        t = self.ntwk1.s11.s_time
        self.assertTrue(npy.sum(npy.abs(t.imag)) == 0)

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

    def test_constructor_empty(self):
        rf.Network()

    def test_constructor_from_values(self):
        rf.Network(f=[1,2],s=[1,2],z0=[1,2] )

    def test_constructor_from_touchstone(self):
        rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))

    def test_constructor_from_hfss_touchstone(self):
        # HFSS can provide the port characteric impedances in its generated touchstone file.
        # Check if reading a HFSS touchstone file with non-50Ohm impedances
        ntwk_hfss = rf.Network(os.path.join(self.test_dir, 'hfss_threeport_DB.s3p'))
        self.assertFalse(npy.isclose(ntwk_hfss.z0[0,0], 50))

    def test_constructor_from_pickle(self):
        rf.Network(os.path.join(self.test_dir, 'ntwk1.ntwk'))

    def test_constructor_from_fid_touchstone(self):
        filename= os.path.join(self.test_dir, 'ntwk1.s2p')
        with open(filename,'rb') as fid:
            rf.Network(fid)

    def test_open_saved_touchstone(self):
        self.ntwk1.write_touchstone('ntwk1Saved',dir=self.test_dir)
        ntwk1Saved = rf.Network(os.path.join(self.test_dir, 'ntwk1Saved.s2p'))
        self.assertEqual(self.ntwk1, ntwk1Saved)
        os.remove(os.path.join(self.test_dir, 'ntwk1Saved.s2p'))

    def test_pickling(self):
        original_ntwk = self.ntwk1
        with tempfile.NamedTemporaryFile(dir=self.test_dir, suffix='ntwk') as fid:
            pickle.dump(original_ntwk, fid, protocol=2)  # Default Python2: 0, Python3: 3
            fid.seek(0)
            unpickled = pickle.load(fid)
        self.assertEqual(original_ntwk, unpickled)

    def test_stitch(self):
        tmp = self.ntwk1.copy()
        tmp.f = tmp.f+ tmp.f[0]
        c = rf.stitch(self.ntwk1, tmp)

    def test_cascade(self):
        self.assertEqual(self.ntwk1 ** self.ntwk2, self.ntwk3)
        self.assertEqual(self.Fix ** self.DUT ** self.Fix.flipped(), self.Meas)

    def test_connect(self):
        self.assertEqual(rf.connect(self.ntwk1, 1, self.ntwk2, 0) , \
            self.ntwk3)

        xformer = rf.Network()
        xformer.frequency=(1,)
        xformer.s = ((0,1),(1,0))  # connects thru
        xformer.z0 = (50,25)  # transforms 50 ohm to 25 ohm
        c = rf.connect(xformer,0,xformer,1)  # connect 50 ohm port to 25 ohm port
        self.assertTrue(npy.all(npy.abs(c.s-rf.impedance_mismatch(50, 25)) < 1e-6))

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

    def test_connect_fast(self):
        raise SkipTest('not supporting this function currently ')
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
        open = rf.N(f=[1], s=[1], z0=[50])
        short = rf.N(f=[1], s=[-1], z0=[50])
        react = rf.N(f=[1],s=[[0,1],[1,0]],z0=50)
        z = open.z
        y = short.y
        a = react.y

    def test_conversions(self):
        #Converting to other format and back to S-parameters should return the original network
        tinyfloat = 1e-12
        for test_z0 in (50, 10, 90+10j, 4-100j):
            for test_ntwk in (self.ntwk1, self.ntwk2, self.ntwk3):
                ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0)

                self.assertTrue((abs(rf.a2s(rf.s2a(ntwk.s, test_z0), test_z0)-ntwk.s) < tinyfloat).all())
                self.assertTrue((abs(rf.z2s(rf.s2z(ntwk.s, test_z0), test_z0)-ntwk.s) < tinyfloat).all())
                self.assertTrue((abs(rf.y2s(rf.s2y(ntwk.s, test_z0), test_z0)-ntwk.s) < tinyfloat).all())
                self.assertTrue((abs(rf.t2s(rf.s2t(ntwk.s))-ntwk.s) < tinyfloat).all())
        self.assertTrue((abs(rf.t2s(rf.s2t(self.Fix.s))-self.Fix.s) < tinyfloat).all())

    def test_yz(self):
        tinyfloat = 1e-12
        ntwk = rf.Network()
        ntwk.z0 = npy.array([28,75+3j])
        ntwk.f = npy.array([1000, 2000])
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
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        freq = rf.F.from_f(npy.linspace(1,2,4), unit='ghz')
        b = a.interpolate(freq, kind='rational')
        # TODO: numerically test for correct interpolation

    def test_interpolate_self_npoints(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        a.interpolate_self_npoints(4)
        # TODO: numerically test for correct interpolation

    def test_interpolate_from_f(self):
        a = rf.N(f=[1,2],s=[1+2j, 3+4j],z0=1)
        a.interpolate_from_f(npy.linspace(1,2,4), unit='ghz')
        # TODO: numerically test for correct interpolation

    def test_slicer(self):
        a = rf.Network(f=[1,2,4,5,6],
                       s=[1,1,1,1,1],
                       z0=50 )

        b = a['2-5ghz']
        tinyfloat = 1e-12
        self.assertTrue((abs(b.frequency.f - [2e9,4e9,5e9]) < tinyfloat).all())

suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
