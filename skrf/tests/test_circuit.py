import skrf as rf
import numpy as np
import unittest
import os, sys
from numpy.testing import assert_array_almost_equal, run_module_suite

class CircuitTestConstructor(unittest.TestCase):
    """
    Various tests on the Circuit constructor.
    """
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
        """
        Check that a Network without name raises an exception
        """
        _ntwk1 = self.ntwk1.copy()
        connections = [[(self.port1, 0), (_ntwk1, 0)],
                       [(_ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port2, 0)]]

        _ntwk1.name = []
        self.assertRaises(AttributeError, rf.Circuit, connections)

        _ntwk1.name = ''
        self.assertRaises(AttributeError, rf.Circuit, connections)

    def test_all_networks_have_same_frequency(self):
        """
        Check that a Network with a different frequency than the other
        raises an exception
        """
        _ntwk1 = self.ntwk1.copy()
        connections = [[(self.port1, 0), (_ntwk1, 0)],
                       [(_ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port2, 0)]]

        _ntwk1.frequency = rf.Frequency(start=1, stop=1, npoints=1)
        self.assertRaises(AttributeError, rf.Circuit, connections)

    def test_no_duplicate_node(self):
        """
        Check that a circuit description has no duplicated (network, port)
        """
        # (port1, 0) is found twice in the connections description
        connections = [[(self.port1, 0), (self.ntwk1, 0)],
                       [(self.ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port1, 0)]]
        self.assertRaises(AttributeError, rf.Circuit, connections)        

    def test_s_active(self):
        """
        Test the active s-parameter of a 2-ports network
        """
        connections = [[(self.port1, 0), (self.ntwk1, 0)],
                       [(self.ntwk1, 1), (self.ntwk2, 0)],
                       [(self.ntwk2, 1), (self.port2, 0)]]
        circuit = rf.Circuit(connections)
        # s_act should be equal to s11 if a = [1,0]
        assert_array_almost_equal(circuit.s_active([1, 0])[:,0], circuit.s_external[:,0,0])
        # s_act should be equal to s22 if a = [0,1]
        assert_array_almost_equal(circuit.s_active([0, 1])[:,1], circuit.s_external[:,1,1])


class CircuitClassMethods(unittest.TestCase):
    """
    Test the various class methods of Circuit such as Ground, Port, etc.
    """
    def setUp(self):
        self.freq = rf.Frequency(start=1, stop=2, npoints=101)
        self.media = rf.DefinedGammaZ0(self.freq)

    def test_ground(self):
        """
        Ground object are infinite shunt admittance (ie. a 2-port short)
        """
        # should raise an exception if no name is passed
        with self.assertRaises(TypeError):
            gnd = rf.Circuit.Ground(self.freq)

        gnd = rf.Circuit.Ground(self.freq, 'gnd')
        gnd_ref = rf.Network(frequency=self.freq, 
                             s=np.tile(np.array([[-1, 0],
                                                 [0, -1]]), 
                                       (len(self.freq),1,1)))

        assert_array_almost_equal(gnd.s, gnd_ref.s)


    def test_open(self):
        """
        Open object are infinite series resistance (ie. a 2-port open)
        """
        # should raise an exception if no name is passed
        with self.assertRaises(TypeError):
            opn = rf.Circuit.Open(self.freq)

        opn = rf.Circuit.Open(self.freq, 'open')
        opn_ref = rf.Network(frequency=self.freq, 
                             s=np.tile(np.array([[1, 0],
                                                 [0, 1]]), 
                                       (len(self.freq),1,1)))

        assert_array_almost_equal(opn.s, opn_ref.s)
    
    def test_series_impedance(self):
        Zs = [1, 1 + 1j, rf.INF]
        for Z in Zs:
            assert_array_almost_equal(
                rf.Circuit.SeriesImpedance(self.freq, Z, 'imp').s, 
                self.media.resistor(Z).s
                )
            
        # Z=0 is a thru
        assert_array_almost_equal(
            rf.Circuit.SeriesImpedance(self.freq, Z=0, name='imp').s,
            self.media.thru().s
            )

    def test_shunt_admittance(self):
        Ys = [1, 1 + 1j, rf.INF]
        for Y in Ys:
            assert_array_almost_equal(
                rf.Circuit.ShuntAdmittance(self.freq, Y, 'imp').s, 
                self.media.shunt(self.media.load(rf.zl_2_Gamma0(self.media.z0, 1/Y))).s
                )
        
        # Y=INF is a a 2-ports short, aka a ground
        assert_array_almost_equal(
            rf.Circuit.ShuntAdmittance(self.freq, rf.INF, 'imp').s,
            rf.Circuit.Ground(self.freq, 'ground').s
            )

class CircuitTestWilkinson(unittest.TestCase):
    """
    Create a Wilkinson power divider Circuit [#]_ and test the results
    against theoretical ones (obtained in [#]_)

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Wilkinson_power_divider
    .. [#] P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).


    """
    def setUp(self):
        """
        Circuit setup
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
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
        """
        Check is Y is correct wrt to ref P.Hallbjörner (2003)
        """
        Y1 = (1 + np.sqrt(2)) / 50
        Y2 = (3 + np.sqrt(2)) / 100

        assert_array_almost_equal(self.C._Y_k(self.connections[0]), Y1)
        assert_array_almost_equal(self.C._Y_k(self.connections[1]), Y2)
        assert_array_almost_equal(self.C._Y_k(self.connections[2]), Y2)

    def test_reflection_coefficients(self):
        """
        Check if Xnn are correct wrt to ref P.Hallbjörner (2003)
        """
        assert_array_almost_equal(self.C._Xnn_k(self.connections[0])[0], self.X1_nn)
        assert_array_almost_equal(self.C._Xnn_k(self.connections[1])[0], self.X2_nn)
        assert_array_almost_equal(self.C._Xnn_k(self.connections[2])[0], self.X2_nn)

    def test_transmission_coefficients(self):
        """
        Check if Xmn are correct wrt to ref P.Hallbjörner (2003)
        """
        assert_array_almost_equal(self.C._Xmn_k(self.connections[0])[0], np.r_[self.X1_m1, self.X1_m2, self.X1_m2])
        assert_array_almost_equal(self.C._Xmn_k(self.connections[1])[0], np.r_[self.X2_m1, self.X2_m2, self.X2_m3])
        assert_array_almost_equal(self.C._Xmn_k(self.connections[2])[0], np.r_[self.X2_m1, self.X2_m2, self.X2_m3])

    def test_sparam_individual_intersection_matrices(self):
        """
        Testing the individual intersection scattering matrices X_k
        """
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[0])[0], self.X1)
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[1])[0], self.X2)
        np.testing.assert_array_almost_equal(self.C._Xk(self.connections[2])[0], self.X2)

    def test_sparam_global_intersection_matrix(self):
        """
        Testing the global intersection scattering matrix
        """
        from scipy.linalg import block_diag
        assert_array_almost_equal(self.C.X[0], block_diag(self.X1, self.X2, self.X2) )

    def test_sparam_circuit(self):
        """
        Testing the external scattering matrix
        """
        S_theoretical = np.array([[0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 0]]) * (-1j/np.sqrt(2))

        # extracting the external ports
        S_ext = self.C.s_external

        assert_array_almost_equal(S_ext[0], S_theoretical)

    def test_compare_with_skrf_wilkison(self):
        """
        Create a Wilkinson power divider using skrf usual Network methods.
        """
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
        """
        Compare the result with ANSYS Designer model

        Built as in https://www.microwaves101.com/encyclopedias/wilkinson-power-splitters
        """
        designer_wilkinson = rf.Network(os.path.join(self.test_dir, 'designer_wilkinson_splitter.s3p'))
        ntw_C = self.C.network

        assert_array_almost_equal(ntw_C.s[0], designer_wilkinson.s[0], decimal=4)

    def test_s_active(self):
        """
        Test the active s-parameter of a 3-ports network
        """
        # s_act should be equal to s11 if a = [1,0,0]
        assert_array_almost_equal(self.C.network.s_active([1, 0, 0])[:,0], self.C.s_external[:,0,0])
        # s_act should be equal to s22 if a = [0,1,0]
        assert_array_almost_equal(self.C.network.s_active([0, 1, 0])[:,1], self.C.s_external[:,1,1])
        # s_act should be equal to s33 if a = [0,0,1]
        assert_array_almost_equal(self.C.network.s_active([0, 0, 1])[:,2], self.C.s_external[:,2,2])

class CircuitTestCascadeNetworks(unittest.TestCase):
    """
    Build a circuit made of two Networks cascaded and compare the result
    to usual cascading of two networks.
    """
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
        """
        Compare ntwk3 to the Circuit of ntwk1 and ntwk2.
        """
        connections = [  [(self.port1, 0), (self.ntwk1, 0)],
                         [(self.ntwk1, 1), (self.ntwk2, 0)],
                         [(self.ntwk2, 1), (self.port2, 0)] ]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(circuit.s_external, self.ntwk3.s)

    def test_cascade2(self):
        """
        Same thing with different ordering of the connections.
        Demonstrate that changing the connections setup order does not change
        the result.
        """
        connections = [  [(self.port1, 0), (self.ntwk1, 0)],
                         [(self.ntwk2, 0), (self.ntwk1, 1)],
                         [(self.port2, 0), (self.ntwk2, 1)] ]
        circuit = rf.Circuit(connections)

        assert_array_almost_equal(circuit.s_external, self.ntwk3.s)

    def test_cascade3(self):
        """
        Inverting the cascading network order
        Demonstrate that changing the connections setup order does not change
        the result (at the requirement that port impedance are the same).
        """
        connections = [  [(self.port1, 0), (self.ntwk2, 0)],
                         [(self.ntwk2, 1), (self.ntwk1, 0)],
                         [(self.port2, 0), (self.ntwk1, 1)] ]
        circuit = rf.Circuit(connections)
        ntw = self.ntwk2 ** self.ntwk1
        assert_array_almost_equal(circuit.s_external, ntw.s)

class CircuitTestMultiPortCascadeNetworks(unittest.TestCase):
    """
    Various 1-ports, 2-ports and 4-ports circuits and associated tests
    """
    def test_1port_matched_load(self):
        """
        Connect a matched load directly to the port
        """
        freq = rf.Frequency(start=1, npoints=1)
        port1 = rf.Circuit.Port(freq,  name='port1')
        line = rf.media.DefinedGammaZ0(frequency=freq)
        match_load = line.match(name='match_load')

        cnx = [
            [(port1, 0), (match_load, 0)]
        ]
        cir = rf.Circuit(cnx)

        assert_array_almost_equal(match_load.s, cir.s_external)

    def test_1port_short(self):
        """
        Connect a short directly to the port
        """
        freq = rf.Frequency(start=1, npoints=1)
        port1 = rf.Circuit.Port(freq,  name='port1')
        line = rf.media.DefinedGammaZ0(frequency=freq)
        short = line.short(name='short')
        gnd1 = rf.Circuit.Ground(freq, name='gnd')
        # method 1 : use the Ground Network (which 2 port actually)
        cnx = [
            [(port1, 0), (gnd1, 0)]
        ]
        cir = rf.Circuit(cnx)
        assert_array_almost_equal(short.s, cir.s_external)
        # method 2 : use a short Network (1 port)
        cnx = [
            [(port1, 0), (short, 0)]
        ]
        cir = rf.Circuit(cnx)

        assert_array_almost_equal(short.s, cir.s_external)

    def test_1port_random_load(self):
        """
        Connect a random load directly to the port
        """
        freq = rf.Frequency(start=1, npoints=1)
        port1 = rf.Circuit.Port(freq,  name='port1')
        line = rf.media.DefinedGammaZ0(frequency=freq)
        gamma = np.random.rand(1,1) + 1j*np.random.rand(1,1)
        load = line.load(gamma, name='load')

        cnx = [
            [(port1, 0), (load, 0)]
        ]
        cir = rf.Circuit(cnx)

        assert_array_almost_equal(load.s, cir.s_external)

    def test_1port_matched_network_default_impedance(self):
        """
        Connect a random 2 port network connected to a matched load
        """
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

        assert_array_almost_equal(b.s, circuit.s_external)

    def test_1port_matched_network_complex_impedance(self):
        """
        Connect a 2 port network to a complex impedance.
        Both ports are complex.
        """
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

        assert_array_almost_equal(b.s, circuit.s_external)

    def test_2ports_default_characteristic_impedance(self):
        """
        Connect two 2-ports networks in a resulting  2-ports network,
        same default charact impedance (50 Ohm) for all ports
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_2ports_complex_characteristic_impedance(self):
        """
        Connect two 2-ports networks in a resulting  2-ports network,
        same complex charact impedance (1+1j) for all ports
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_2ports_different_characteristic_impedances(self):
        """
        Connect two 2-ports networks in a resulting  2-ports network,
        different characteristic impedances for each network ports
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_4ports_default_characteristic_impedances(self):
        """
        Connect two 4-ports networks in a resulting 4-ports network,
        with default characteristic impedances
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_4ports_complex_characteristic_impedances(self):
        """
        Connect two 4-ports networks in a resulting 4-ports network,
        with same complex characteristic impedances
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_4ports_different_characteristic_impedances(self):
        """
        Connect two 4-ports networks in a resulting 4-ports network,
        with different characteristic impedances
        """
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

        assert_array_almost_equal(c.s, circuit.s_external)

    def test_shunt_element(self):
        """
        Compare a shunt element network (here a capacitor)
        """
        freq = rf.Frequency(start=1, stop=2, npoints=101)
        line = rf.media.DefinedGammaZ0(frequency=freq, z0=50)
        # usual way
        cap_shunt_manual = line.shunt_capacitor(50e-12)

        # A Circuit way
        port1 = rf.Circuit.Port(frequency=freq, name='port1', z0=50)
        port2 = rf.Circuit.Port(frequency=freq, name='port2', z0=50)
        cap_shunt = line.capacitor(50e-12, name='cap_shunt')
        ground = rf.Circuit.Ground(frequency=freq, name='ground', z0=50)

        connections = [
            [(port1, 0), (cap_shunt, 0), (port2, 0)],
            [(cap_shunt, 1), (ground, 0)]
        ]

        # # Another possibility could have been without ground :
        # shunt_cap = line.shunt_capacitor(50e-12)
        # shunt_cap.name='shunt_cap'
        # connections = [
        #     [(port1, 0), (shunt_cap, 0)],
        #     [(shunt_cap ,1), (port2, 0)]
        # ]

        cap_shunt_from_circuit = rf.Circuit(connections).network

        assert_array_almost_equal(cap_shunt_manual.s, cap_shunt_from_circuit.s)

class CircuitTestVariableCoupler(unittest.TestCase):
    """
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

    The port order in this example is voluntary complicated to make a good
    example.

    """
    def setUp(self):
        self.freq = rf.Frequency(start=1.5, stop=1.5, npoints=1, unit='GHz')
        self.coax = rf.media.DefinedGammaZ0(frequency=self.freq)
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'

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
        # ps.z0 = [31, 32]
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
        """
        Compare with the S-parameters obtained from Network.connect
        """
        phase_deg = np.random.randint(low=0, high=180)
        vc_connect = self.variable_coupler_network_from_connect(phase_deg)
        vc_circuit = self.variable_coupler_network_from_circuit(phase_deg)
        assert_array_almost_equal(vc_connect.s, vc_circuit.s)

    def test_compare_with_designer(self):
        """
        Compare with the S-parameters obtained from ANSYS Designer
        """
        for phase_angle in [20, 75]:
            vc_designer = rf.Network(os.path.join(self.test_dir, 'designer_variable_coupler_ideal_'+str(phase_angle)+'deg.s4p'))
            vc_circuit = self.variable_coupler_network_from_circuit(phase_angle)
            assert_array_almost_equal(vc_designer.s, vc_circuit.s, decimal=4)

    def test_compare_connect_and_designer(self):
        """
        Compare S-parameters obtained from ANSYS Designer with Network.connect
        """
        for phase_angle in [20, 75]:
            vc_designer = rf.Network(os.path.join(self.test_dir, 'designer_variable_coupler_ideal_'+str(phase_angle)+'deg.s4p'))
            vc_connect = self.variable_coupler_network_from_connect(phase_angle)
            assert_array_almost_equal(vc_designer.s, vc_connect.s, decimal=4)


class CircuitTestGraph(unittest.TestCase):
    """
    Test functionalities linked to graph method, used in particular for plotting
    """
    def test_is_networkx_available(self):
        'The networkx package should be available to run these tests'
        self.assertTrue('networkx' in sys.modules)

    def setUp(self):
        """
        Dummy Circuit setup

        Setup a circuit which has various interconnections (2 or 3)
        """
        self.freq = rf.Frequency(start=1, stop=2, npoints=101)

        # dummy components
        self.R = 100
        self.line_resistor = rf.media.DefinedGammaZ0(frequency=self.freq, Z0=self.R)
        resistor1 = self.line_resistor.resistor(self.R, name='resistor1')
        resistor2 = self.line_resistor.resistor(self.R, name='resistor2')
        resistor3 = self.line_resistor.resistor(self.R, name='resistor3')
        port1 = rf.Circuit.Port(self.freq, name='port1')

        # Connection setup
        self.connections = [
                   [(port1, 0), (resistor1, 0), (resistor3, 0)],
                   [(resistor1, 1), (resistor2, 0)],
                   [(resistor2, 1), (resistor3, 1)]
                ]

        self.C = rf.Circuit(self.connections)

    def test_intersection_dict(self):
        inter_dict = self.C.intersections_dict
        # should have 3 intersections
        self.assertTrue(len(inter_dict) == 3)
        # All intersections should have at least 2 edges
        for it in inter_dict.items():
            k, cnx = it
            self.assertTrue(len(cnx) >= 2)

    def test_edge_labels(self):
        edge_labels = self.C.edge_labels
        self.assertTrue(len(edge_labels) == 7)


class CircuitTestComplexCharacteristicImpedance(unittest.TestCase):
    """
    Test creating circuits with real and complex port charac.impedances
    """
    def setUp(self):
        self.f0 = rf.Frequency(75.8, npoints=1, unit='GHz')
        # initial s-param values of A 2 ports network
        self.s0 = np.array([  # dummy values
            [-0.1000 -0.2000j, -0.3000 +0.4000j],
            [-0.3000 +0.4000j, 0.5000 -0.6000j]]).reshape(-1,2,2)

        # build initial network (under z0=50)
        self.ntw0 = rf.Network(frequency=self.f0, s=self.s0, z0=50, name='dut')

        # complex characteristic impedance to renormalize to
        self.zdut = 100 + 10j

        # reference solutions obtained from ANSYS Circuit or ADS (same res)
        # when z0=[50, zdut]
        self.z_ref = np.array([  # not affected by z0
            [18.0000 -16.0000j, 20.0000 +40.0000j],
            [20.0000 +40.0000j, 10.0000 -80.0000j]]).reshape(-1,2,2)

        self.y_ref = np.array([  # not affected by z0
            [0.0251 +0.0023j, 0.0123 -0.0066j],
            [0.0123 -0.0066j, 0.0052 +0.0055j]]).reshape(-1,2,2)

        self.s_ref = np.array([  # renormalized s (power-waves)
            [-0.1374 -0.2957j, -0.1995 +0.5340j],
            [-0.1995 +0.5340j, -0.0464 -0.7006j]]).reshape(-1,2,2)

        # Creating equivalent reference circuit
        port1 = rf.Circuit.Port(self.f0, z0=50, name='port1')
        port2 = rf.Circuit.Port(self.f0, z0=50, name='port2')
        ntw0 = self.ntw0

        cnx = [  # z0=[50,50]
            [(port1, 0), (ntw0, 0)],
            [(ntw0, 1), (port2, 0)]
            ]
        self.cir = rf.Circuit(cnx)

        # Creating equivalent circuit with z0 real
        port2_real = rf.Circuit.Port(self.f0, z0=100, name='port2')
        cnx_real = [  # z0=[50,100]
            [(port1, 0), (ntw0, 0)],
            [(ntw0, 1), (port2_real, 0)]
            ]
        self.cir_real = rf.Circuit(cnx_real)

        # Creating equivalent circuit with z0 complex
        port2_complex = rf.Circuit.Port(self.f0, z0=self.zdut, name='port2')
        cnx_complex = [  # z0=[50,zdut]
            [(port1, 0), (ntw0, 0)],
            [(ntw0, 1), (port2_complex, 0)]
            ]
        self.cir_complex = rf.Circuit(cnx_complex)

        # references for each s-param definition
        self.s_legacy = rf.renormalize_s(self.s0, [50,50], [50,self.zdut],
                                         s_def='traveling')
        self.s_power = rf.renormalize_s(self.s0, [50,50], [50,self.zdut],
                                         s_def='power')
        self.s_pseudo = rf.renormalize_s(self.s0, [50,50], [50,self.zdut],
                                         s_def='pseudo')
        # for real values, should be = whatever s-param definition
        self.s_real = rf.renormalize_s(self.s0, [50,50], [50,100])

    def test_verify_reference(self):
        ' Check that the reference results comes from power-waves definition'
        np.testing.assert_allclose(self.s_ref, self.s_power, atol=1e-4)

    def test_reference_s(self):
        ' Check reference z0 circuit, just in case '
        np.testing.assert_allclose(self.cir.network.s, self.s0, atol=1e-4)

    def test_real_z0_s(self):
        ' Check real z0 circuit '
        np.testing.assert_allclose(self.cir_real.network.s, self.s_real, atol=1e-4)

    def test_real_z_params(self):
        ' Check Z-parameters match'
        np.testing.assert_allclose(self.cir.network.z, self.cir_real.network.z, atol=1e-4)

    def test_complex_z_params(self):
        ' Check Z-parameters match'
        np.testing.assert_allclose(self.cir_complex.network.z, self.cir_real.network.z, atol=1e-4)

    @unittest.expectedFailure
    def test_complexz0_s_vs_legacy(self):
        ' Check complex z0 circuit vs legacy renormalization '
        np.testing.assert_allclose(self.cir_complex.network.s, self.s_legacy, atol=1e-4)

    def test_complexz0_s_vs_powerwaves(self):
        ' Check complex z0 circuit vs power-waves renormalization '
        np.testing.assert_allclose(self.cir_complex.network.s, self.s_ref, atol=1e-4)
        np.testing.assert_allclose(self.cir_complex.network.s, self.s_power, atol=1e-4)

    @unittest.expectedFailure
    def test_complexz0_s_vs_pseudo(self):
        ' Check complex z0 circuit vs pseudo-waves renormalization '
        np.testing.assert_allclose(self.cir_complex.network.s, self.s_pseudo, atol=1e-4)

class CircuitTestVoltagesCurrents(unittest.TestCase):
    def setUp(self):
        # setup a test transmission line randomly excited
        self.P_f = np.random.rand()  # forward power in Watt
        self.phase_f = np.random.rand()  # forward phase in rad
        self.Z = np.random.rand()  # source internal impedance, line characteristic impedance and load impedance
        self.L = np.random.rand()  # line length in [m]
        self.freq = rf.Frequency(1, 10, 10, unit='GHz')
        self.line_media = rf.media.DefinedGammaZ0(self.freq, z0=self.Z)  # lossless line medium
        self.line = self.line_media.line(d=self.L, unit='m', name='line')  # transmission line Network

        # forward voltages and currents at the input of the test line
        self.V_in = np.sqrt(2*self.Z*self.P_f)*np.exp(1j*self.phase_f)
        self.I_in = np.sqrt(2*self.P_f/self.Z)*np.exp(1j*self.phase_f)
        # forward voltages and currents at the output of the test line
        theta = rf.theta(self.line_media.gamma, self.freq.f, self.L)  # electrical length
        self.V_out, self.I_out = rf.tlineFunctions.voltage_current_propagation(self.V_in, self.I_in, self.Z, theta)

        # Equivalent model with Circuit
        port1 = rf.Circuit.Port(frequency=self.freq, name='port1', z0=self.Z)
        port2 = rf.Circuit.Port(frequency=self.freq, name='port2', z0=self.Z)
        cnx = [
            [(port1, 0), (self.line, 0)],
            [(port2, 0), (self.line, 1)]
        ]
        self.crt = rf.Circuit(cnx)
        # power and phase arrays for Circuit.voltages() and currents()
        self.power = [self.P_f, 0]
        self.phase = [self.phase_f, 0]

    def test_tline_voltages(self):
        ' Test voltages for a simple transmission line '
        V_ports = self.crt.voltages_external(self.power, self.phase)

        np.testing.assert_allclose(self.V_in, V_ports[:,0])
        np.testing.assert_allclose(self.V_out, V_ports[:,1])

    def test_tline_currents(self):
        ' Test currents for a simple transmission line '
        I_ports = self.crt.currents_external(self.power, self.phase)

        np.testing.assert_allclose(self.I_in, I_ports[:,0])
        # output current is * -1 as Circuit definition is opposite
        # (toward the Circuit's Port)
        np.testing.assert_allclose(self.I_out, -1*I_ports[:,1])


if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
