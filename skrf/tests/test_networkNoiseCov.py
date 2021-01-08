import unittest
import os
import tempfile
import six
import numpy as npy
import six.moves.cPickle as pickle
import skrf as rf
from nose.plugins.skip import SkipTest

from skrf.constants import S_DEFINITIONS



class NetworkNoiseCovTestCase(unittest.TestCase):
    '''
   
    '''
    def setUp(self):
        '''
        this also tests the ability to read touchstone files
        without an error
        '''
        self.f = npy.linspace(0.5, 3, 6)
        self.ovec = npy.ones(len(self.f))
        self.zvec = npy.ones(len(self.f))
        s11 = self.f
        s21 = npy.sin(self.f)
        s12 = npy.cos(self.f)
        s22 = -npy.exp(self.f)

        self.S = rf.network_array([[s11,    s21],
                                   [s12,    s22]])

        self.cc = rf.network_array([[.3*self.ovec,    -.2*self.ovec - .2*1j],
                               [-.2*self.ovec + .2*1j,    .1*self.ovec]])
        

    def test_cs2ct_and_ct2cs(self):
        S = self.S
        T = rf.s2t(S)
        x = rf.NetworkNoiseCov(self.cc, 's', 50, 290)
        nct = x._cs2ct(x.cc, T)
        ncs = nct._ct2cs(nct.cc, S)
        vec_check = npy.isclose(self.cc, ncs.cc, atol=1e-14, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cs2cz_and_cz2cs(self):
        S = self.S
        Z = rf.s2z(S)
        x = rf.NetworkNoiseCov(self.cc, 's', 50, 290)
        ncz = x._cs2cz(x.cc, Z)
        ncs = ncz._cz2cs(ncz.cc, S)
        vec_check = npy.isclose(self.cc, ncs.cc, atol=1e-11, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cs2cy_and_cy2cs(self):
        S = self.S
        Y = rf.s2y(S)
        x = rf.NetworkNoiseCov(self.cc, 's', 50, 290)
        ncy = x._cs2cy(x.cc, Y)
        ncs = ncy._cy2cs(ncy.cc, S)
        vec_check = npy.isclose(self.cc, ncs.cc, atol=1e-12, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cs2ca_and_ca2cs(self):
        S = self.S
        A = rf.s2a(S)
        x = rf.NetworkNoiseCov(self.cc, 's', 50, 290)
        nca = x._cs2ca(x.cc, A)
        ncs = nca._ca2cs(nca.cc, S)
        vec_check = npy.isclose(self.cc, ncs.cc, atol=1e-11, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_ct2cz_cz2ct(self):
        T = self.S
        Z = rf.t2z(T)
        nc1 = rf.NetworkNoiseCov(self.cc, 't', 50, 290)
        nc2 = nc1._ct2cz(nc1.cc, Z)
        ncf = nc2._cz2ct(nc2.cc, T)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_ct2cy_cy2ct(self):
        T = self.S
        Y = rf.t2y(T)
        nc1 = rf.NetworkNoiseCov(self.cc, 't', 50, 290)
        nc2 = nc1._ct2cy(nc1.cc, Y)
        ncf = nc2._cy2ct(nc2.cc, T)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_ct2ca_ca2ct(self):
        T = self.S
        A = rf.t2a(T)
        nc1 = rf.NetworkNoiseCov(self.cc, 't', 50, 290)
        nc2 = nc1._ct2ca(nc1.cc, A)
        ncf = nc2._ca2ct(nc2.cc, T)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cz2cy_cy2cz(self):
        Z = self.S
        Y = rf.z2y(Z)
        nc1 = rf.NetworkNoiseCov(self.cc, 'z', 50, 290)
        nc2 = nc1._cz2cy(nc1.cc, Y)
        ncf = nc2._cy2cz(nc2.cc, Z)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        print(self.cc)
        print(ncf.cc)
        print(vec_check)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cz2ca_ca2cz(self):
        Z = self.S
        A = rf.z2a(Z)
        nc1 = rf.NetworkNoiseCov(self.cc, 'z', 50, 290)
        nc2 = nc1._cz2ca(nc1.cc, A)
        ncf = nc2._ca2cz(nc2.cc, Z)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        print(self.cc)
        print(ncf.cc)
        print(vec_check)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")

    def test_cy2ca_ca2cy(self):
        Y = self.S
        A = rf.y2a(Y)
        nc1 = rf.NetworkNoiseCov(self.cc, 'y', 50, 290)
        nc2 = nc1._cy2ca(nc1.cc, A)
        ncf = nc2._ca2cy(nc2.cc, Y)
        vec_check = npy.isclose(self.cc, ncf.cc, atol=1e-12, rtol=0)
        print(self.cc)
        print(ncf.cc)
        print(vec_check)
        self.assertTrue(vec_check.all(), "covariance transform is not consistent")
