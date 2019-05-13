# -*- coding: utf-8 -*-
'''
.. module:: skrf.circuit
========================================
circuit (:mod:`skrf.circuit`)
========================================


Provides a class representing a circuit of arbitrary topology,
consisting of an arbitrary number of N-ports networks or impedance elements.

The results are returned in :class:`~skrf.network.Circuit` objects


Circuit Class
================

.. autosummary::
   :toctree: generated/

   Circuit

'''
from . network import Network, a2s
from . media import media
from . constants import INF

import numpy as np

try:
    import networkx as nx
except ImportError as e:
    pass

from itertools import chain, product
from scipy.linalg import block_diag

class Circuit():
    '''
    Notes
    -----
    The algorithm used to calculate the resultant network can be found in [#]_.

    References
    ----------
    .. [#] P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).

    '''
    def __init__(self, connections):
        '''
        Circuit constructor.

        Creates a circuit made of a set of N-ports networks.

        Parameters
        -----------
        connections : list of list
            List of the intersections of the circuit.
            Each intersection is a list of tuples containing the network
            Example of an intersection between two network:
            [ [(network1, network1_port_nb), (network2, network2_port_nb)],
              [(network1, network1_port_nb), (network2, network2_port_nb)]

        ! The external ports indexing is defined by the order of appearance of
        the ports in the connections list.


        '''
        self.connections = connections

        # check if all networks have a name
        for cnx in self.connections:
            for (ntw, _) in cnx:
                if not self._is_named(ntw):
                    raise AttributeError('All Networks must have a name.')

        # list of networks for initial checks
        ntws = self.networks_list()

        # check if all networks have same frequency
        ref_freq = ntws[0].frequency
        for ntw in ntws:
            if ntw.frequency != ref_freq:
                raise AttributeError('All Networks must have same frequencies')
        # All frequencies are the same, Circuit frequency can be any of the ntw
        self.frequency = ntws[0].frequency

    def _is_named(self, ntw):
        '''
        Return True is the network has a name, False otherwise
        '''
        if not ntw.name or ntw.name == '':
            return False
        else:
            return True

    @classmethod
    def Port(cls, frequency, name, z0=50+0*1j):
        '''
        Return a port Network. Passing the frequency and name is mendatory

        Parameters
        -----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        name : string
            Name of the port.
            Must inlude the word 'port' inside. (ex: 'Port1' or 'port_3')
        z0 : complex
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        --------
        port : :class:`~skrf.network.Network` object
            (External) 1-port network

        Examples
        -----------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: port1 = rf.Circuit.Port(freq, name='Port1')
        '''
        _media = media.DefinedGammaZ0(frequency, z0=z0)
        return _media.match(name=name)

    @classmethod
    def Ground(cls, frequency, name, z0=50+0j):
        '''
        Ground network. Passing the frequency and name is mendatory.
        
        Ground is considered as in infinite admittance.
        
        Parameters
        -----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        name : string
            Name of the ground.
        z0 : complex
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        --------
        ground : :class:`~skrf.network.Network` object
            (External) 2-port network

        Examples
        -----------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: ground = rf.Circuit.Ground(freq, name='GND')
        
        '''
        Y = INF
        A = np.zeros(shape=(len(frequency), 2, 2), dtype=complex)
        A[:, 0, 0] = 1
        A[:, 0, 1] = 0
        A[:, 1, 0] = Y
        A[:, 1, 1] = 1
        ntw = Network(frequency=frequency, z0=z0, name=name)
        ntw.s = a2s(A)
        return ntw

    def networks_dict(self, connections=None, min_nports=1):
        '''
        Return the dictionnary of Networks from the connection setup X
        '''
        if not connections:
            connections = self.connections

        ntws = []
        for cnx in connections:
            for (ntw, port) in cnx:
                ntws.append(ntw)
        return {ntw.name: ntw for ntw in ntws  if ntw.nports >= min_nports}

    def networks_list(self, connections=None, min_nports=1):
        '''
        return a list of unique networks (sorted by appearing order in connections)
        '''
        if not connections:
            connections = self.connections

        ntw_dict = self.networks_dict(connections)
        return [ntw for ntw in ntw_dict.values() if ntw.nports >= min_nports]

    @property
    def connections_nb(self):
        '''
        Returns the number of intersections in the circuit.
        '''
        return len(self.connections)

    @property
    def connections_list(self):
        '''
        Return the full list of connections, including intersections.
        The resulting list if of the form:
        [
         [connexion_number, connexion],
         [connexion_number, connexion],
         ...
        ]
        '''
        return [[idx_cnx, cnx] for (idx_cnx, cnx) in enumerate(chain.from_iterable(self.connections))]



    @property
    def networks_nb(self):
        '''
        Return the number of connected networks (port excluded)
        '''
        return len(self.networks_list(self.connections))

    @property
    def nodes_nb(self):
        '''
        Return the number of nodes in the circuit.
        '''
        return self.connections_nb + self.networks_nb

    @property
    def dim(self):
        '''
        Return the dimension of the C, X and global S matrices.

        It correspond to the sum of all connections
        '''
        return np.sum([len(cnx) for cnx in self.connections])

    @property
    def G(self):
        return self.graph()

    def graph(self):
        '''
        Generate the graph of the circuit
        '''
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError('networkx package as not been installed and is required. ')

        G = nx.Graph()
        # Adding network nodes
        G.add_nodes_from([it for it in self.networks_dict(self.connections)])

        # Adding edges in the graph between connections and networks
        for (idx, cnx) in enumerate(self.connections):
            cnx_name = 'X'+str(idx)
            # Adding connection nodes and edges
            G.add_node(cnx_name)
            for (ntw, ntw_port) in cnx:
                ntw_name = ntw.name
                G.add_edge(cnx_name, ntw_name)
        return G

    def is_connected(self):
        '''
        Check if the circuit's graph is connected, that is if
        if every pair of vertices in the graph is connected.
        '''
        # Get the circuit graph. Will raise an error if the networkx package
        # is not installed. 
        G = self.G
    
        return nx.algorithms.components.is_connected(G)

    @property
    def intersections_dict(self):
        '''
        Return a dictionnary of all intersections with associated ports and z0
        { k: [(ntw1_name, ntw1_port), (ntw1_z0, ntw2_name, ntw2_port), ntw2_z0], ... }
        '''
        inter_dict = {}
        # for k in range(self.connections_nb):
        #     # get all edges connected to intersection Xk
        #     inter_dict[k] = list(nx.algorithms.boundary.edge_boundary(self.G, ('X'+str(k),) ))

        for (k, cnx) in enumerate(self.connections):
            inter_dict[k] = [(ntw, ntw_port, ntw.z0[0, ntw_port]) \
                                  for (ntw, ntw_port) in cnx]
        return inter_dict

    @property
    def edges(self):
        '''
        Return the list of all circuit connections
        '''
        return list(self.G.edges)

    @property
    def edge_labels(self):
        '''
        Return a dictionnary describing the port and z0 of all graph edges.

        Dictionnary is in the form:
            {('ntw1_name', 'X0'): '3 (50+0j)',
             ('ntw2_name', 'X0'): '0 (50+0j)',
             ('ntw2_name', 'X1'): '2 (50+0j)', ... }
        which can be used in networkx.draw_networkx_edge_labels
        '''
        # for all intersections, 
        # get the N interconnected networks and associated ports and z0
        # and forge the edge label dictionnary containing labels between
        # two nodes
        edge_labels = {}
        for it in self.intersections_dict.items():
            k, cnx = it
            for idx in range(len(cnx)):
                ntw, ntw_port, ntw_z0 = cnx[idx]
                #ntw_z0 = ntw.z0[0,ntw_port]
                edge_labels[(ntw.name, 'X'+str(k))] = str(ntw_port)+'\n'+\
                                        str(np.round(ntw_z0, decimals=1))

        return edge_labels


    def _Y_k(self, cnx):
        '''
        Return the sum of the system admittances of each intersection
        '''
        y_ns = []
        for (ntw, ntw_port) in cnx:
            # formula (2)
            y_ns.append(1/ntw.z0[:,ntw_port] )

        y_k = np.array(y_ns).sum(axis=0)  # shape: (nb_freq,)
        return y_k

    def _Xnn_k(self, cnx_k):
        '''
        Return the reflection coefficients x_nn of the connection matrix [X]_k
        '''
        X_nn = []
        y_k = self._Y_k(cnx_k)

        for (ntw, ntw_port) in cnx_k:
            # formula (1)
            X_nn.append( 2/(ntw.z0[:,ntw_port]*y_k) - 1)

        return np.array(X_nn).T  # shape: (nb_freq, nb_n)

    def _Xmn_k(self, cnx_k):
        '''
        Return the transmission coefficient x_mn of the mth column of
        intersection scattering matrix matrix [X]_k
        '''
        # get the char.impedance of the n
        X_mn = []
        y_k = self._Y_k(cnx_k)

        # There is a problem in the case of two-ports connexion:
        # the formula (3) in P. Hallbjörner (2003) seems incorrect.
        # Instead of Z_n one should have sqrt(Z_1 x Z_2).
        # The formula works with respect to the example given in the paper
        # (3 port connection), but not with 2-port connections made with skrf
        if len(cnx_k) == 2:
            z0s = []
            for (ntw, ntw_port) in cnx_k:
                z0s.append(ntw.z0[:,ntw_port])

            z0eq = np.array(z0s).prod(axis=0)

            for (ntw, ntw_port) in cnx_k:
                X_mn.append( 2/(np.sqrt(z0eq) *y_k) )
        else:
            # formula (3)
            for (ntw, ntw_port) in cnx_k:
                X_mn.append( 2/(ntw.z0[:,ntw_port]*y_k) )

        return np.array(X_mn).T  # shape: (nb_freq, nb_n)

    def _Xk(self, cnx_k):
        '''
        Return the scattering matrices [X]_k of the individual intersections k
        '''
        Xnn = self._Xnn_k(cnx_k)  # shape: (nb_freq, nb_n)
        Xmn = self._Xmn_k(cnx_k)  # shape: (nb_freq, nb_n)
        # # for loop version
        # Xs = []
        # for (_Xnn, _Xmn) in zip(Xnn, Xmn):  # for all frequencies
        #       # repeat Xmn along the lines
        #     _X = np.tile(_Xmn, (len(_Xmn), 1))
        #     _X[np.diag_indices(len(_Xmn))] = _Xnn
        #     Xs.append(_X)

        # return np.array(Xs) # shape : nb_freq, nb_n, nb_n

        # vectorized version
        nb_n = Xnn.shape[1]
        Xs = np.tile(Xmn, (nb_n, 1, 1)).swapaxes(1, 0)
        Xs[:, np.arange(nb_n), np.arange(nb_n)] = Xnn

        return Xs # shape : nb_freq, nb_n, nb_n

        # TEST : Could we use media.splitter() instead ? -> does not work
        # _media = media.DefinedGammaZ0(frequency=self.frequency)
        # Xs = _media.splitter(len(cnx_k), z0=self._cnx_z0(cnx_k))
        # return Xs.s

    @property
    def X(self):
        '''
        Return the concatenated intersection matrix [X] of the circuit.

        It is composed of the individual intersection matrices [X]_k assembled
        by bloc diagonal.
        '''
        # Xk = []
        # for cnx in self.connections:
        #     Xk.append(self._Xk(cnx))
        # Xk = np.array(Xk)

        # #X = np.zeros(len(C.frequency), )
        # Xf = []
        # for (idx, f) in enumerate(self.frequency):
        #     Xf.append(block_diag(*Xk[:,idx,:]))
        # return np.array(Xf)  # shape: (nb_frequency, nb_inter*nb_n, nb_inter*nb_n)

        # Slightly faster version
        Xks = [self._Xk(cnx) for cnx in self.connections]

        Xf = np.zeros((len(self.frequency), self.dim, self.dim), dtype='complex')
        # TODO: avoid this for loop which is a bottleneck for large frequencies
        for idx in np.nditer(np.arange(len(self.frequency))):
            mat_list = [Xk[idx,:] for Xk in Xks]
            Xf[idx,:] = block_diag(*mat_list)  # bottleneck 
        return Xf

    @property
    def C(self):
        '''
        Return the global scattering matrix of the networks
        '''
        # list all networks with at least two ports
        ntws = self.networks_dict(min_nports=2)

        # generate the port reordering indexes from each connections
        ntws_ports_reordering = {ntw:[] for ntw in ntws}
        for (idx_cnx, cnx) in self.connections_list:
            ntw, ntw_port = cnx
            if ntw.name in ntws.keys():
                ntws_ports_reordering[ntw.name].append([ntw_port, idx_cnx])

        # re-ordering scattering parameters
        S = np.zeros((len(self.frequency), self.dim, self.dim), dtype='complex' )

        for (ntw_name, ntw_ports) in ntws_ports_reordering.items():
            # get the port re-ordering indexes (from -> to)
            ntw_ports = np.array(ntw_ports)
            # create the port permutations
            from_port = list(product(ntw_ports[:,0], repeat=2))
            to_port = list(product(ntw_ports[:,1], repeat=2))

            #print(ntw_name, from_port, to_port)
            for (_from, _to) in zip(from_port, to_port):
                #print(f'{_from} --> {_to}')
                S[:, _to[0], _to[1]] = ntws[ntw_name].s[:, _from[0], _from[1]]

        return S  # shape (nb_frequency, nb_inter*nb_n, nb_inter*nb_n)

    @property
    def s(self):
        '''
        Return the global scattering parameter of the circuit, that is with
        both "inner" and "outer" ports
        '''
        # transpose is necessary to get expected result
        #return np.transpose(self.X @ np.linalg.inv(np.identity(self.dim) - self.C @ self.X), axes=(0,2,1))
        # does not use the @ operator for backward Python version compatibility
        return np.transpose(np.matmul(self.X, np.linalg.inv(np.identity(self.dim) - np.matmul(self.C, self.X))), axes=(0,2,1))


    @property
    def port_indexes(self):
        '''
        Return the indexes of the "external" ports. These must be labelled "port"
        '''
        port_indexes = []
        for (idx_cnx, cnx) in enumerate(chain.from_iterable(self.connections)):
            ntw, ntw_port = cnx
            if 'port' in str.lower(ntw.name):
                port_indexes.append(idx_cnx)
        return port_indexes

    def _cnx_z0(self, cnx_k):
        '''
        Return the characteristic impedances of a specific connection
        '''
        z0s = []
        for (ntw, ntw_port) in cnx_k:
            z0s.append(ntw.z0[:,ntw_port])

        return np.array(z0s).T  # shape (nb_freq, nb_ports_at_cnx)

    @property
    def port_z0(self):
        '''
        Return the external port impedances
        '''
        z0s = []
        for cnx in self.connections:
            for (ntw, ntw_port) in cnx:
                z0s.append(ntw.z0[:,ntw_port])

        return np.array(z0s)[self.port_indexes, :].T  # shape (nb_freq, nb_ports)

    @property
    def s_external(self):
        '''
        Return the scattering parameter for the external ports
        '''
        port_indexes = self.port_indexes
        a, b = np.meshgrid(port_indexes, port_indexes)
        S_ext = self.s[:, a, b]
        return S_ext  # shape (nb_frequency, nb_ports, nb_ports)

    @property
    def network(self):
        '''
        Return the Network associated to external ports
        '''
        ntw = Network()
        ntw.frequency = self.frequency
        ntw.z0 = self.port_z0
        ntw.s = self.s_external
        return ntw


    def s_active(self, a):
        '''
        Returns active s-parameters of the circuit's network for a defined wave excitation a.
        
        The active s-parameter at a port is the reflection coefficients 
        when other ports are excited. It is an important quantity for active
        phased array antennas.
        
        Active s-parameters are defined by [#]_:
        
        .. math::
                    
            \mathrm{active}(s)_{mn} = \sum_i\left( s_{mi} a_i \right) / a_n
        
        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude (pseudowave formulation [#]_)
        
        Returns
        ---------
        s_act : complex array of shape (n_freqs, n_ports)
            active s-parameters for the excitation a
        
        
        References
        ---------- 
        .. [#] D. M. Pozar, IEEE Trans. Antennas Propag. 42, 1176 (1994).
        
        .. [#] D. Williams, IEEE Microw. Mag. 14, 38 (2013).
        
        '''
        return self.network.s_active(a)

    def z_active(self, a):
        '''
        Returns the active Z-parameters of the circuit's network for a defined wave excitation a.
        
        The active Z-parameters are defined by:
            
        .. math::
                    
            \mathrm{active}(z)_{m} = z_{0,m} \frac{1 + \mathrm{active}(s)_m}{1 - \mathrm{active}(s)_m}
            
        where :math:`z_{0,m}` is the characteristic impedance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.
        
        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude
    
        Returns
        ----------
        z_act : complex array of shape (nfreqs, nports)
            active Z-parameters for the excitation a
            
        See Also
        -----------
            s_active : active S-parameters
            y_active : active Y-parameters
            vswr_active : active VSWR        
        '''
        return self.network.z_active(a)

    def y_active(self, a):
        '''
        Returns the active Y-parameters of the circuit's network for a defined wave excitation a.
        
        The active Y-parameters are defined by:
            
        .. math::
                    
            \mathrm{active}(y)_{m} = y_{0,m} \frac{1 - \mathrm{active}(s)_m}{1 + \mathrm{active}(s)_m}
            
        where :math:`y_{0,m}` is the characteristic admittance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.    
        
        Parameters
        ----------            
        a : complex array of shape (n_ports)
            forward wave complex amplitude 
        
        Returns
        ----------
        y_act : complex array of shape (nfreqs, nports)
            active Y-parameters for the excitation a
            
        See Also
        -----------
            s_active : active S-parameters
            z_active : active Z-parameters
            vswr_active : active VSWR       
        '''
        return self.network.y_active(a)

    def vswr_active(self, a):
        '''
        Returns the active VSWR of the circuit's network for a defined wave excitation a.
        
        The active VSWR is defined by :
            
        .. math::
                    
            \mathrm{active}(vswr)_{m} = \frac{1 + |\mathrm{active}(s)_m|}{1 - |\mathrm{active}(s)_m|}
    
        where :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.
        
        Parameters
        ----------       
        a : complex array of shape (n_ports)
            forward wave complex amplitude
    
        Returns
        ----------
        vswr_act : complex array of shape (nfreqs, nports)
            active VSWR for the excitation a
            
        See Also
        -----------
            s_active : active S-parameters
            z_active : active Z-parameters
            y_active : active Y-parameters  
        '''        
        return self.network.vswr_active(a)        