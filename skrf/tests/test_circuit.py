# -*- coding: utf-8 -*-
import skrf as rf
import numpy as np
import unittest
import os
from numpy.testing import assert_array_almost_equal, run_module_suite

class CircuitTestWilkinson(unittest.TestCase):
    '''
    Create a Wilkison power divider Circuit [#]_ and test the results
    against theoretical ones (obtained in [#]_)

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Wilkinson_power_divider
    .. [#] P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).


    '''
    def setUp(self):
        '''
        Circuit setup
        '''
        freq = rf.Frequency(start=1, stop=2, npoints=101)
        # characteristic impedance of the ports
        Z0_ports = 50

        # resistor
        R = 100
        line_resistor = rf.media.DefinedGammaZ0(frequency=freq, Z0=R)
        resistor = line_resistor.resistor(100, name='resistor')

        # branches
        Z0_branches = np.sqrt(2)*Z0_ports
        line_branches = rf.media.DefinedGammaZ0(frequency=freq, Z0=Z0_branches)
        branch1 = line_branches.line(90, unit='deg', name='branch1')
        branch2 = line_branches.line(90, unit='deg', name='branch2')

        # ports
        port1 = rf.Circuit.Port(freq, name='port1')
        port2 = rf.Circuit.Port(freq, name='port2')
        port3 = rf.Circuit.Port(freq, name='port3')

        # Connection setup
        self.connections = [
                   [(port1, 0), (branch1, 0), (branch2, 0)],
                   [(port2, 0), (branch1, 1), (resistor, 0)],
                   [(port3, 0), (branch2, 1), (resistor, 1)]
                ]

        self.C = rf.Circuit(self.connections)

        # theoretical results from ref P.Hallbjörner (2003)
        self.X1_nn = np.array([1 - np.sqrt(2), -1, -1])/(1 + np.sqrt(2))
        self.X2_nn = np.array([1 - np.sqrt(2), -3 + np.sqrt(2), -1 - np.sqrt(2)])/(3 + np.sqrt(2))

        self.X1_m1 = 2 / (1 + np.sqrt(2))
        self.X1_m2 = np.sqrt(2) / (1 + np.sqrt(2))
        self.X1_m3 = np.sqrt(2) / (1 + np.sqrt(2))

        self.X2_m1 = 4 / (3 + np.sqrt(2))
        self.X2_m2 = 2*np.sqrt(2) / (3 + np.sqrt(2))
        self.X2_m3 = 2 / (3 + np.sqrt(2))

        self.X1 = np.array([[0, self.X1_m2, self.X1_m3],
                            [self.X1_m1, 0, self.X1_m3],
                            [self.X1_m1, self.X1_m2, 0]]) + np.diag(self.X1_nn)
        self.X2 = np.array([[0, self.X2_m2, self.X2_m3],
                            [self.X2_m1, 0, self.X2_m3],
                            [self.X2_m1, self.X2_m2, 0]]) + np.diag(self.X2_nn)

    def test_global_admittance(self):
        '''
        Check is Y is correct wrt to ref P.Hallbjörner (2003)
        '''
        Y1 = (1 + np.sqrt(2))/50
        Y2 = (3 + np.sqrt(2))/100

        assert_array_almost_equal(self.C._Y_k(self.connections[0]), Y1)
        assert_array_almost_equal(self.C._Y_k(self.connections[1]), Y2)
        assert_array_almost_equal(self.C._Y_k(self.connections[2]), Y2)

    def test_reflection_coefficients(self):
        '''
        Check if Xnn are correct wrt to ref P.Hallbjörner (2003)
        '''
        assert_array_almost_equal(self.C._Xnn_k(self.connections[0])[0], self.X1_nn)
        assert_array_almost_equal(self.C._Xnn_k(self.connections[1])[0], self.X2_nn)
        assert_array_almost_equal(self.C._Xnn_k(self.connections[2])[0], self.X2_nn)

    def test_transmission_coefficients(self):
        '''
        Check if Xmn are correct wrt to ref P.Hallbjörner (2003)
        '''
        assert_array_almost_equal(self.C._Xmn_k(self.connections[0])[0], np.r_[self.X1_m1, self.X1_m2, self.X1_m2])
        assert_array_almost_equal(self.C._Xmn_k(self.connections[1])[0], np.r_[self.X2_m1, self.X2_m2, self.X2_m3])
        assert_array_almost_equal(self.C._Xmn_k(self.connections[2])[0], np.r_[self.X2_m1, self.X2_m2, self.X2_m3])

    def test_sparam_individual_intersection_matrices(self):
        '''
        Testing the individual intersection scattering matrices X_k
        '''
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[0])[0], self.X1)
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[1])[0], self.X2)
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[2])[0], self.X2)

    def test_sparam_global_intersection_matrix(self):
        '''
        Testing the global intersection scattering matrix
        '''
        from scipy.linalg import block_diag
        assert_array_almost_equal(self.C.X[0], block_diag(self.X1, self.X2, self.X2) )

    def test_sparam_circuit(self):
        '''
        Testing the external scattering matrix
        '''
        S_theoretical = np.array([[0,1,1], [1,0,0], [1,0,0]])*(-1j/np.sqrt(2))

        # extracting the external ports
        S_ext = self.C.S_external

        assert_array_almost_equal(S_ext[0], S_theoretical)


class CircuitTestCascadeNetworks(unittest.TestCase):
    '''
    Build a circuit made of cascading two Network and compare the result
    to usual cascading of two networks.
    '''
    def setUp(self):
        # Importing network examples
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.freq = self.ntwk1.frequency        
        # circuit external ports 
        self.port1 = rf.Circuit.Port(self.freq, name='Port1')
        self.port2 = rf.Circuit.Port(self.freq, name='Port2')
        
    def test_cascade(self):
        # ntwk3 is the cascade of ntwk1 and ntwk2, ie. self.ntwk1 ** self.ntwk2
        connections = [  [(self.port1, 0), (self.ntwk1, 0)], 
                         [(self.ntwk1, 1), (self.ntwk2, 0)], 
                         [(self.ntwk2, 1), (self.port2, 0)] ]
        circuit = rf.Circuit(connections)
        # checl that all scattering parameters are equals
        assert_array_almost_equal(circuit.S_external, self.ntwk3.s)

    def test_cascade2(self):
        # same thing with different ordering of the connections.
        connections = [  [(self.port1, 0), (self.ntwk1, 0)], 
                         [(self.ntwk2, 0), (self.ntwk1, 1)], 
                         [(self.port2, 0), (self.ntwk2, 1)] ]
        circuit = rf.Circuit(connections)
        assert_array_almost_equal(circuit.S_external, self.ntwk3.s)    


if __name__ == "__main__" :
    run_module_suite()
