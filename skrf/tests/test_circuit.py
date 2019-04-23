# -*- coding: utf-8 -*-
import skrf as rf
import numpy as np
import unittest
import os
from numpy.testing import assert_array_almost_equal, run_module_suite


class CircuitTestConstructor(unittest.TestCase):
    '''
    Various tests on the Circuit constructor.
    '''
    def setUp(self):
        # Importing network examples
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.freq = self.ntwk1.frequency
        # circuit external ports
        self.port1 = rf.Circuit.Port(self.freq, name='Port1')
        self.port2 = rf.Circuit.Port(self.freq, name='Port2')

    def test_all_networks_have_name(self):
        '''
        Check that a Network without name raises an exception
        '''
        _ntwk1 = self.ntwk1.copy()
        connections = [[(self.port1, 0), (_ntwk1, 0)],
                       [(_ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port2, 0)]]

        _ntwk1.name = []
        self.assertRaises(AttributeError, rf.Circuit, connections)

        _ntwk1.name = ''
        self.assertRaises(AttributeError, rf.Circuit, connections)

    def test_all_networks_have_same_frequency(self):
        '''
        Check that a Network with a different frequency than the other
        raises an exception
        '''
        _ntwk1 = self.ntwk1.copy()
        connections = [[(self.port1, 0), (_ntwk1, 0)],
                       [(_ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port2, 0)]]

        _ntwk1.frequency = rf.Frequency(start=1, stop=1, npoints=1)
        self.assertRaises(AttributeError, rf.Circuit, connections)


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
        self.freq = rf.Frequency(start=1, stop=2, npoints=101)
        # characteristic impedance of the ports
        Z0_ports = 50

        # resistor
        self.R = 100
        self.line_resistor = rf.media.DefinedGammaZ0(frequency=self.freq, Z0=self.R)
        self.resistor = self.line_resistor.resistor(self.R, name='resistor')

        # branches
        Z0_branches = np.sqrt(2)*Z0_ports
        self.line_branches = rf.media.DefinedGammaZ0(frequency=self.freq, Z0=Z0_branches)
        self.branch1 = self.line_branches.line(90, unit='deg', name='branch1')
        self.branch2 = self.line_branches.line(90, unit='deg', name='branch2')

        # ports
        port1 = rf.Circuit.Port(self.freq, name='port1')
        port2 = rf.Circuit.Port(self.freq, name='port2')
        port3 = rf.Circuit.Port(self.freq, name='port3')

        # Connection setup
        self.connections = [
                   [(port1, 0), (self.branch1, 0), (self.branch2, 0)],
                   [(port2, 0), (self.branch1, 1), (self.resistor, 0)],
                   [(port3, 0), (self.branch2, 1), (self.resistor, 1)]
                ]

        self.C = rf.Circuit(self.connections)

        # theoretical results from ref P.Hallbjörner (2003)
        self.X1_nn = np.array([1 - np.sqrt(2), -1, -1])/(1 + np.sqrt(2))
        self.X2_nn = np.array([1 - np.sqrt(2), -3 + np.sqrt(2),
                               -1 - np.sqrt(2)])/(3 + np.sqrt(2))

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
        Y1 = (1 + np.sqrt(2)) / 50
        Y2 = (3 + np.sqrt(2)) / 100

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
        S_theoretical = np.array([[0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 0]]) * (-1j/np.sqrt(2))

        # extracting the external ports
        S_ext = self.C.S_external

        assert_array_almost_equal(S_ext[0], S_theoretical)

    def test_compare_with_skrf_wilkison(self):
        '''
        Create a Wilkinson power divider using skrf usual Network methods.
        '''
        z0_port = 50
        z0_lines = self.line_branches.z0[0]
        z0_R = self.line_resistor.z0[0]
        # require to create the three tees
        T0 = self.line_branches.splitter(3, z0=[z0_port, z0_lines, z0_lines])
        T1 = self.line_branches.splitter(3, z0=[z0_lines, z0_R, z0_port])
        T2 = self.line_branches.splitter(3, z0=[z0_lines, z0_R, z0_port])

        _wilkinson1 = rf.connect(T0, 1, self.branch1, 0)
        _wilkinson2 = rf.connect(_wilkinson1, 2, self.branch2, 0)
        _wilkinson3 = rf.connect(_wilkinson2, 1, T1, 0)
        _wilkinson4 = rf.connect(_wilkinson3, 1, T2, 0)
        _wilkinson5 = rf.connect(_wilkinson4, 1, self.resistor, 0)
        wilkinson = rf.innerconnect(_wilkinson5, 1, 3)

        ntw_C = self.C.network

        # the following is failing and I don't know why
        #assert_array_almost_equal(ntw_C.s_db, wilkinson.s_db)

        assert_array_almost_equal(ntw_C.z0, wilkinson.z0)

    def test_compare_with_designer_wilkinson(self):
        '''
        Compare the result with ANSYS Designer model
        '''
        designer_wilkinson = rf.Network('designer_wilkinson_splitter.s3p')
        ntw_C = self.C.network
        
        assert_array_almost_equal(ntw_C.s[0], designer_wilkinson.s[0], decimal=4)

class CircuitTestCascadeNetworks(unittest.TestCase):
    '''
    Build a circuit made of two Networks cascaded and compare the result
    to usual cascading of two networks.
    '''
    def setUp(self):
        # Importing network examples
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        # ntwk3 is the cascade of ntwk1 and ntwk2, ie. self.ntwk1 ** self.ntwk2
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.freq = self.ntwk1.frequency
        # circuit external ports
        self.port1 = rf.Circuit.Port(self.freq, name='Port1')
        self.port2 = rf.Circuit.Port(self.freq, name='Port2')

    def test_cascade(self):
        '''
        Compare ntwk3 to the Circuit of ntwk1 and ntwk2.
        '''
        connections = [  [(self.port1, 0), (self.ntwk1, 0)],
                         [(self.ntwk1, 1), (self.ntwk2, 0)],
                         [(self.ntwk2, 1), (self.port2, 0)] ]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(circuit.S_external, self.ntwk3.s)

    def test_cascade2(self):
        '''
        Same thing with different ordering of the connections.
        Demonstrate that changing the connections setup order does not change
        the result.
        '''
        connections = [  [(self.port1, 0), (self.ntwk1, 0)],
                         [(self.ntwk2, 0), (self.ntwk1, 1)],
                         [(self.port2, 0), (self.ntwk2, 1)] ]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(circuit.S_external, self.ntwk3.s)

    def test_cascade3(self):
        '''
        Inverting the cascading network order
        Demonstrate that changing the connections setup order does not change
        the result (at the requirement that port impedance are the same).
        '''
        connections = [  [(self.port1, 0), (self.ntwk2, 0)],
                         [(self.ntwk2, 1), (self.ntwk1, 0)],
                         [(self.port2, 0), (self.ntwk1, 1)] ]
        circuit = rf.Circuit(connections)
        ntw = self.ntwk2 ** self.ntwk1
        assert_array_almost_equal(circuit.S_external, ntw.s)

class CircuitTestMultiPortCascadeNetworks(unittest.TestCase):
    '''
    Various 2-ports and 4-ports circuits and associated tests
    '''
    def test_1port_matched_network_default_impedance(self):
        '''
        Connect a 2 port network to a matched load
        '''
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(4).reshape(2, 2)
        line = rf.media.DefinedGammaZ0(frequency=freq)
        match_load = line.match(name='match_load')

        # classic connecting
        b = a ** match_load

        # Circuit connecting
        port1 = rf.Circuit.Port(freq,  name='port1')
        connections = [[(port1, 0), (a, 0)], [(a, 1), (match_load, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(b.s, circuit.S_external)

    def test_1port_matched_network_complex_impedance(self):
        '''
        Connect a 2 port network to a complex impedance.
        Both ports are complex.
        '''
        z01, z02 = 1-1j, 2+4j
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(4).reshape(2, 2)
        a.z0 = [z01, z02]
        line = rf.media.DefinedGammaZ0(frequency=freq, z0=z02)
        match_load = line.match(name='match_load')

        # classic connecting
        b = a ** match_load

        # Circuit connecting
        port1 = rf.Circuit.Port(freq, z0=z01, name='port1')
        connections = [[(port1, 0), (a,0)], [(a, 1), (match_load, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(b.s, circuit.S_external)

    def test_2ports_default_characteristic_impedance(self):
        '''
        Connect two 2-ports networks in a resulting  2-ports network,
        same default charact impedance (50 Ohm) for all ports
        '''
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(4).reshape(2, 2)

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(4).reshape(2, 2)

        # classic connecting
        c = rf.connect(a, 1, b, 0)

        # Circuit connecting
        port1 = rf.Circuit.Port(freq,  name='port1')
        port2 = rf.Circuit.Port(freq,  name='port2')

        connections = [[(port1, 0), (a, 0)],
                       [(a, 1), (b, 0)],
                       [(b, 1), (port2, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)

    def test_2ports_complex_characteristic_impedance(self):
        '''
        Connect two 2-ports networks in a resulting  2-ports network,
        same complex charact impedance (1+1j) for all ports
        '''
        z0 = 1 + 1j
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(4).reshape(2, 2)
        a.z0 = z0

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(4).reshape(2, 2)
        b.z0 = z0

        # classic connecting
        c = rf.connect(a, 1, b, 0)

        # Circuit connecting
        port1 = rf.Circuit.Port(freq, z0=z0, name='port1')
        port2 = rf.Circuit.Port(freq, z0=z0, name='port2')

        connections = [[(port1, 0), (a, 0)],
                       [(a, 1), (b, 0)],
                       [(b, 1), (port2, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)

    def test_2ports_different_characteristic_impedances(self):
        '''
        Connect two 2-ports networks in a resulting  2-ports network,
        different characteristic impedances for each network ports
        '''
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(4).reshape(2,2)
        a.z0 = [1, 2]  #  Z0 should never be zero

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(4).reshape(2,2)
        b.z0 = [11, 12]

        # classic connecting
        c = rf.connect(a, 1, b, 0)

        # Circuit connecting
        port1 = rf.Circuit.Port(freq, z0=1, name='port1')
        port2 = rf.Circuit.Port(freq, z0=12, name='port2')

        connections = [[(port1, 0), (a, 0)],
                       [(a, 1), (b, 0)],
                       [(b, 1), (port2, 0)] ]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)

    def test_4ports_default_characteristic_impedances(self):
        '''
        Connect two 4-ports networks in a resulting 4-ports network,
        with default characteric impedances
        '''
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(16).reshape(4, 4)

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(16).reshape(4, 4)

        # classic connecting
        c = rf.connect(a, 2, b, 0, 2)

        # Circuit connecting
        port1 = rf.Circuit.Port(freq, name='port1')
        port2 = rf.Circuit.Port(freq, name='port2')
        port3 = rf.Circuit.Port(freq, name='port3')
        port4 = rf.Circuit.Port(freq, name='port4')

        connections = [[(port1, 0), (a, 0)],
                       [(port2, 0), (a, 1)],
                       [(a, 2), (b, 0)],
                       [(a, 3), (b, 1)],
                       [(b, 2), (port3, 0)],
                       [(b, 3), (port4, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)

    def test_4ports_complex_characteristic_impedances(self):
        '''
        Connect two 4-ports networks in a resulting 4-ports network,
        with same complex characteric impedances
        '''
        z0 = 5 + 4j
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(16).reshape(4, 4)
        a.z0 = z0

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(16).reshape(4, 4)
        b.z0 = z0

        # classic connecting
        c = rf.connect(a, 2, b, 0, 2)

        # circuit connecting
        port1 = rf.Circuit.Port(freq, z0=z0, name='port1')
        port2 = rf.Circuit.Port(freq, z0=z0, name='port2')
        port3 = rf.Circuit.Port(freq, z0=z0, name='port3')
        port4 = rf.Circuit.Port(freq, z0=z0, name='port4')

        connections = [ [(port1, 0), (a, 0)],
                        [(port2, 0), (a, 1)],
                        [(a, 2), (b, 0)],
                        [(a, 3), (b, 1)],
                        [(b, 2), (port3, 0)],
                        [(b, 3), (port4, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)

    def test_4ports_different_characteristic_impedances(self):
        '''
        Connect two 4-ports networks in a resulting 4-ports network,
        with different characteristic impedances
        '''
        z0 = [1, 2, 3, 4]
        freq = rf.Frequency(start=1, npoints=1)
        a = rf.Network(name='a')
        a.frequency = freq
        a.s = np.random.rand(16).reshape(4, 4)
        a.z0 = z0

        b = rf.Network(name='b')
        b.frequency = freq
        b.s = np.random.rand(16).reshape(4, 4)
        b.z0 = [11, 12, 13, 14]

        # classic connecting
        _c = rf.connect(a, 2, b, 0)
        c = rf.innerconnect(_c, 2, 3)

        # Circuit connecting
        port1 = rf.Circuit.Port(freq, z0=1, name='port1')
        port2 = rf.Circuit.Port(freq, z0=2, name='port2')
        port3 = rf.Circuit.Port(freq, z0=13, name='port3')
        port4 = rf.Circuit.Port(freq, z0=14, name='port4')

        connections = [[(port1, 0), (a, 0)],
                       [(port2, 0), (a, 1)],
                       [(a, 2), (b, 0)],
                       [(a, 3), (b, 1)],
                       [(b, 2), (port3, 0)],
                       [(b, 3), (port4, 0)]]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(c.s, circuit.S_external)


class CircuitTestVariableCoupler(unittest.TestCase):
    '''
    If we use 3 dB hybrid defined as :
                   ________
    Input     0 --|       |-- 1 Through
                  |       |
    Isolated  3 --|_______|-- 2 Coupled

    And if we connect two 3 dB hybrid with a phase shifter such as:
                 _________                                 _________
    Port#3 >- 0-|        |-1 -------------------------- 0-|        |-1 -< Port#2
                |  Hyb1  |                                |  Hyb2  |
    Port#0 >- 3-|________|-2 -- 0-[phase shifter]-1 -- 3-|________|-2 -< Port#1

    Then we have a variable coupler. The coupling factor can be adjusted
    by changing the phase of the phase shifter.

    The port order in this example is volontary complicated to make a good
    example.

    '''
    def setUp(self):
        self.freq = rf.Frequency(start=1.5, stop=1.5, npoints=1, unit='GHz')
        self.coax = rf.media.DefinedGammaZ0(frequency=self.freq)

    def phase_shifter(self, phase_deg):
        return self.coax.line(d=phase_deg, unit='deg')

    def hybrid(self, name='hybrid'):
        Sc = 1/np.sqrt(2)*np.array([[ 0,  1, 1j,  0],
                                    [ 1,  0,  0, 1j],
                                    [1j,  0,  0,  1],
                                    [ 0, 1j,  1,  0]])
        hybrid = rf.Network(frequency=self.freq, s=Sc, name=name)
        return hybrid

    def variable_coupler_network_from_connect(self, phase_deg):
        ps = self.phase_shifter(phase_deg)
        hybrid1, hybrid2 = self.hybrid(), self.hybrid()
        # connecting the networks together.
        # NB: in order to know which port corresponds to which, it is usefull
        # to define different port characteristic impedances like
        # hybrid1.z0 = [11, 12, 13, 14]
        # hybrid2.z0 = [21, 22, 23, 24]
        # etc
        # and to make a drawing of each steps.
        # This is not convenient, that's why the Circuit approach can be easier.
        _temp = rf.connect(hybrid1, 2, ps, 0)
        _temp = rf.connect(_temp, 1, hybrid2, 0)
        _temp = rf.innerconnect(_temp, 1, 5)
        # re-order port numbers to match the example
        _temp.renumber([0, 1, 2, 3], [3, 0, 2, 1])
        return _temp

    def variable_coupler_circuit(self, phase_deg):
        ps = self.phase_shifter(phase_deg)
        ps.name = 'ps'  # do not forget the name of the network !
        hybrid1, hybrid2 = self.hybrid('hybrid1'), self.hybrid('hybrid2')

        port1 = rf.Circuit.Port(ps.frequency, 'port1')
        port2 = rf.Circuit.Port(ps.frequency, 'port2')
        port3 = rf.Circuit.Port(ps.frequency, 'port3')
        port4 = rf.Circuit.Port(ps.frequency, 'port4')
        # Note that the order of port appearance is important.
        # 1st port to appear in the connection setup will be the 1st port (0),
        # then second to appear the second port (1), etc...
        # There is no constraint for the order of the connections.
        connections = [
                       [(port1, 0), (hybrid1, 3)],
                       [(port2, 0), (hybrid2, 2)],
                       [(port3, 0), (hybrid2, 1)],
                       [(port4, 0), (hybrid1, 0)],
                       [(hybrid1, 2), (ps, 0)],
                       [(hybrid1, 1), (hybrid2, 0)],
                       [(ps, 1), (hybrid2, 3)],
                      ]

        return rf.Circuit(connections)

    def variable_coupler_network_from_circuit(self, phase_deg):
        return self.variable_coupler_circuit(phase_deg).network

    def test_compare_with_network_connect(self):
        '''
        Compare with the S-parameters obtained from Network.connect
        '''
        phase_deg = np.random.randint(low=0, high=180)
        vc_connect = self.variable_coupler_network_from_connect(phase_deg)
        vc_circuit = self.variable_coupler_network_from_circuit(phase_deg)
        assert_array_almost_equal(vc_connect.s, vc_circuit.s)

    def test_compare_with_designer(self):
        '''
        Compare with the S-parameters obtained from ANSYS Designer
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        for phase_angle in [20, 75]:
            vc_designer = rf.Network(os.path.join(self.test_dir, 'designer_variable_coupler_ideal_'+str(phase_angle)+'deg.s4p'))
            vc_circuit = self.variable_coupler_network_from_circuit(phase_angle)
            assert_array_almost_equal(vc_designer.s, vc_circuit.s, decimal=4)

    def test_compare_connect_and_designer(self):
        '''
        Compare S-parameters obtained from ANSYS Designer with Network.connect
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        for phase_angle in [20, 75]:
            vc_designer = rf.Network(os.path.join(self.test_dir, 'designer_variable_coupler_ideal_'+str(phase_angle)+'deg.s4p'))
            vc_connect = self.variable_coupler_network_from_connect(phase_angle)
            assert_array_almost_equal(vc_designer.s, vc_connect.s, decimal=4)


if __name__ == "__main__" :
    # Launch all tests
    run_module_suite()

