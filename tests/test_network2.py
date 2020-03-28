import unittest
import skrf as rf
import os

from skrf import network2 as n2

def get_abs_file_path(filename):
    test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
    return os.path.join(test_dir, filename)

class Network2TestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.freq = rf.Frequency(1,10,101,'ghz')
        self.s_2port = rf.rand_c(len(self.freq),2,2)
        self.s_1port = rf.rand_c(len(self.freq),1,1)
        self.n = n2.Network(frequency=self.freq, s=self.s_2port, z0=50)


    def get_abs_file_path(filename):
        return os.path.join(self.test_dir, filename)

    ## inits

    #def test_init_empty(self):
    #    n2.Network()

    def test_init_from_s(self):
        n2.Network(frequency=self.freq, s=self.s_2port, z0=50)

    def test_init_from_z(self):
        n2.Network(frequency=self.freq, z=self.s_2port, z0=50)

    def test_init_from_y(self):
        n2.Network(frequency=self.freq, y=self.s_2port, z0=50)

    def test_init_from_ntwkv1(self):
        n2.Network.from_ntwkv1(rf.data.ring_slot)

    ## existence of parameters
    def test_z(self):
        self.n.z

    def test_z(self):
        self.n.y

    def test_z(self):
        self.n.y

    def test_z(self):
        self.n.s_time

    ## existence of parameters
    def test_db(self):
        self.n.s.db

    def test_db10(self):
        self.n.s.db10

    def test_db20(self):
        self.n.s.db20

    def test_mag(self):
        self.n.s.mag

    def test_deg(self):
        self.n.s.deg

    def test_rad(self):
        self.n.s.rad

    def test_re(self):
        self.n.s.re

    def test_im(self):
        self.n.s.im


    def test_call(self):
        for i,j in self.n.port_tuples:
            sij = self.n(i,j)


    def test_windowed(self):
        self.n.windowed()

    def test_string_slicing(self):

        n=self.n['4-7ghz']
