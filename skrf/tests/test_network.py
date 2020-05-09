import unittest
import os
import tempfile
import six
import numpy as npy
import six.moves.cPickle as pickle
import skrf as rf
from nose.plugins.skip import SkipTest

from skrf import setup_pylab
from skrf.media import CPW
from skrf.media import DistributedCircuit
from skrf.constants import S_DEFINITIONS
from skrf.networkSet import tuner_constellation   
from skrf.plotting import plot_contour   

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
        self.cpw =  CPW(self.freq, w=10e-6, s=5e-6, ep_r=10.6)
        l1 = self.cpw.line(0.20, 'm', z0=50)
        l2 = self.cpw.line(0.07, 'm', z0=50)
        l3 = self.cpw.line(0.47, 'm', z0=50)
        self.Fix = rf.concat_ports([l1, l1, l1, l1])
        self.DUT = rf.concat_ports([l2, l2, l2, l2])
        self.Meas = rf.concat_ports([l3, l3, l3, l3])

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
        for test_z0 in (50, 10, 90+10j, 4-100j):
            for test_ntwk in (self.ntwk1, self.ntwk2, self.ntwk3):
                ntwk = rf.Network(s=test_ntwk.s, f=test_ntwk.f, z0=test_z0)
                npy.testing.assert_allclose(rf.a2s(rf.s2a(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.z2s(rf.s2z(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.y2s(rf.s2y(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.h2s(rf.s2h(ntwk.s, test_z0), test_z0), ntwk.s)
                npy.testing.assert_allclose(rf.t2s(rf.s2t(ntwk.s)), ntwk.s)
        npy.testing.assert_allclose(rf.t2s(rf.s2t(self.Fix.s)), self.Fix.s)        

    def test_sparam_conversion_with_complex_char_impedance(self):
        '''
        Renormalize a 2-port network wrt to complex characteristic impedances
        using power-waves definition of s-param
        Example based on scikit-rf issue #313
        '''
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

        # Compararing Z and Y matrices from reference ones (from ADS)
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
        '''
        Check that power-wave or pseudo-waves scattering parameters definitions 
        give same results for real characteristic impedances
        '''
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
            ntwk.f = freqs
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
            ntwk.f = npy.array([1000])
            ntwk.s = npy.random.rand(1,2,2) + npy.random.rand(1,2,2)*1j
            self.assertFalse(npy.any(npy.isnan(ntwk.z)))
            self.assertFalse(npy.any(npy.isnan(ntwk.y)))
        
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

    # Network classifiers
    def test_is_reciprocal(self):
        a = rf.Network(f=[1, 2],
                       s=[[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]],
                       z0=50)
        self.assertFalse(a.is_reciprocal(), 'A circulator is not reciprocal.')
        b = rf.Network(f=[1, 2],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertTrue(b.is_reciprocal(), 'This power divider is reciprocal.')
        return

    def test_is_symmetric(self):
        # 2-port
        a = rf.Network(f=[1, 2],
                       s=[[-1, 0],
                          [0, -1]],
                       z0=50)
        self.assertTrue(a.is_symmetric(), 'A short is symmetric.')
        self.assertRaises(ValueError, a.is_symmetric, port_order={1: 2})  # error raised by renumber()
        a.s[0, 0, 0] = 1
        self.assertFalse(a.is_symmetric(), 'non-symmetrical')

        # 3-port
        b = rf.Network(f=[1, 2],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        with self.assertRaises(ValueError) as context:
            b.is_symmetric()
        self.assertEqual(str(context.exception), 'test of symmetric is only valid for a 2N-port network')

        # 4-port
        c = rf.Network(f=[1, 2],
                       s=[[0, 1j, 1, 0],
                          [1j, 0, 0, 1],
                          [1, 0, 0, 1j],
                          [0, 1, 1j, 0]],
                       z0=50)
        self.assertTrue(c.is_symmetric(n=2), 'This quadrature hybrid coupler is symmetric.')
        self.assertTrue(c.is_symmetric(n=2, port_order={0: 1, 1: 2, 2: 3, 3: 0}),
                        'This quadrature hybrid coupler is symmetric even after rotation.')
        with self.assertRaises(ValueError) as context:
            c.is_symmetric(n=3)
        self.assertEqual(str(context.exception), 'specified order n = 3 must be between 1 and N = 2, inclusive')

        d = rf.Network(f=[1, 2],
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
        x = rf.Network(f=[1, 2],
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
        y = rf.Network(f=[1, 2],
                       s=s8p_mat,
                       z0=50)
        self.assertTrue(y.is_symmetric())
        self.assertTrue(y.is_symmetric(n=2))
        self.assertFalse(y.is_symmetric(n=4))
        return

    def test_is_passive(self):
        a = rf.Network(f=[1, 2],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertTrue(a.is_passive(), 'This power divider is passive.')
        b = rf.Network(f=[1, 2],
                       s=[[0, 0],
                          [10, 0]],
                       z0=50)
        self.assertFalse(b.is_passive(), 'A unilateral amplifier is not passive.')
        return

    def test_is_lossless(self):
        a = rf.Network(f=[1, 2],
                       s=[[0, 0.5, 0.5],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0]],
                       z0=50)
        self.assertFalse(a.is_lossless(), 'A resistive power divider is lossy.')
        b = rf.Network(f=[1, 2],
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





    def test_s_active(self):
        '''
        Test the active s-parameters of a 2-ports network
        '''
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
        '''
        Test the active vswr-parameters of a 2-ports network
        '''
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

suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
