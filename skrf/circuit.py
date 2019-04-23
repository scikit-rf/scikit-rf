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
from . network import Network
from . media import media

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

        # create a list of networks for initial checks
        self.ntws = self.networks_list(self.connections)

        # check if all networks have same frequency
        ref_freq = self.ntws[0].frequency
        for ntw in self.ntws:
            if ntw.frequency != ref_freq:
                raise AttributeError('All Networks must have same frequencies')
        # All frequencies are the same, Circuit frequency can be any of the ntw
        self.frequency = self.ntws[0].frequency

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
        Z0 : complex
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
        return nx.algorithms.components.is_connected(self.G)

    @property
    def intersections_dict(self):
        '''
        Return a dictionnary of all intersections with associated ports and z0
        { k: [ntw1_name, ntw1_port, ntw1_z0, ntw2_name, ntw2_port, ntw2_z0], ... }
        '''
        inter_dict = {}
        # for k in range(self.connections_nb):
        #     # get all edges connected to intersection Xk
        #     inter_dict[k] = list(nx.algorithms.boundary.edge_boundary(self.G, ('X'+str(k),) ))

        for (k, cnx) in enumerate(self.connections):
            (ntw1, ntw1_port), (ntw2, ntw2_port) = cnx
            inter_dict[k] = [ntw1.name, ntw1_port, ntw1.z0[0, ntw1_port],
                             ntw2.name, ntw2_port, ntw1.z0[0, ntw2_port]]

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
        # for all connections Xk, get the two interconnected networks
        # and associated ports and z0
        edge_labels = {}
        for k in range(self.connections_nb):
            (ntw1_name, ntw1_port, ntw1_z0,
             ntw2_name, ntw2_port, ntw2_z0) = self.intersections_dict[k]
            # forge the dictionnary elements
            edge_labels[(ntw1_name, 'X'+str(k))] = str(ntw1_port)+'\n'+str(ntw1_z0)
            edge_labels[(ntw2_name, 'X'+str(k))] = str(ntw2_port)+'\n'+str(ntw2_z0)

        return edge_labels

    def plot(self, **kwargs):
        '''
        Plot the graph of the circuit using networkx drawing capabilities.

        Customisation options:
        'network_shape': 's'
        'network_color': 'gray'
        'network_size', 300
        'network_fontsize': 7
        'inter_shape': 'o'
        'inter_color': 'lightblue'
        'inter_size', 300
        'port_shape': '>'
        'port_color': 'red'
        'port_size', 300
        'edges_fontsize': 5
        'is_network_legend': False
        'is_edge_legend': False
        'is_inter_labels': False
        'is_port_labels': False
        'label_shift_x': 0
        'label_shift_y': 0

        '''
        # default values
        network_shape = kwargs.pop('network_shape', 's')
        network_color = kwargs.pop('network_color', 'gray')
        network_fontsize = kwargs.pop('network_fontsize', 7)
        network_size = kwargs.pop('network_size', 300)
        inter_shape = kwargs.pop('inter_shape', 'o')
        inter_color = kwargs.pop('inter_color', 'lightblue')
        inter_size = kwargs.pop('inter_size', 300)
        port_shape = kwargs.pop('port_shape', '>')
        port_color = kwargs.pop('port_color', 'red')
        port_size = kwargs.pop('port_size', 300)
        edge_fontsize = kwargs.pop('edges_fontsize', 5)
        label_shift_x = kwargs.pop('label_shift_x', 0)
        label_shift_y = kwargs.pop('label_shift_y', 0)
        is_network_labels = kwargs.pop('is_network_labels', False)
        is_edge_labels = kwargs.pop('is_edge_labels', False)
        is_inter_labels = kwargs.pop('is_inter_labels', False)
        is_port_labels = kwargs.pop('is_port_labels', False)

        # sort between network nodes and port nodes
        all_ntw_names = [ntw.name for ntw in self.networks_list()]
        port_names = [ntw_name for ntw_name in all_ntw_names if 'port' in ntw_name]
        ntw_names = [ntw_name for ntw_name in all_ntw_names if 'port' not in ntw_name]
        # generate connectins nodes names
        int_names = ['X'+str(k) for k in range(self.connections_nb)]

        fig, ax = plt.subplots()

        G = self.G
        pos = nx.spring_layout(G)
        edge_labels = self.edge_labels

        # draw Networks
        nx.draw_networkx_nodes(G, pos, port_names, ax=ax,
                               node_size=port_size,
                               node_color=port_color, node_shape=port_shape)
        nx.draw_networkx_nodes(G, pos, ntw_names, ax=ax,
                               node_size=network_size,
                               node_color=network_color, node_shape=network_shape)
        # draw intersections
        nx.draw_networkx_nodes(G, pos, int_names, ax=ax,
                               node_size=inter_size,
                               node_color=inter_color, node_shape=inter_shape)
        # labels shifts
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + label_shift_x, 
                                coords[1] + label_shift_y)
                
        # network labels
        if is_network_labels:
            network_labels = {lab:lab for lab in ntw_names}
            
            nx.draw_networkx_labels(G, pos_labels, labels=network_labels, 
                                    fontsize=network_fontsize, ax=ax)

        # intersection labels
        if is_inter_labels:
            inter_labels = {'X'+str(k):'X'+str(k) for k in range(self.connections_nb)}

            nx.draw_networkx_labels(G, pos_labels, labels=inter_labels, 
                                    fontsize=network_fontsize, ax=ax)

        if is_port_labels:
            port_labels = {lab:lab for lab in port_names}

            nx.draw_networkx_labels(G, pos_labels, labels=port_labels, 
                                    fontsize=network_fontsize, ax=ax)           

        # draw edges
        nx.draw_networkx_edges(G, pos, ax=ax)
        if is_edge_labels:
            nx.draw_networkx_edge_labels(G, pos,
                                          edge_labels=edge_labels, label_pos=0.5,
                                          font_size=edge_fontsize, ax=ax)
        # remove x and y axis and labels
        plt.axis('off')
        plt.tight_layout()

        return fig, ax
    
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
        Xnn = self._Xnn_k(cnx_k)
        Xmn = self._Xmn_k(cnx_k)
        # repeat Xmn along the lines
        # TODO : avoid the loop?
        Xs = []
        for (_Xnn, _Xmn) in zip(Xnn, Xmn):  # for all frequencies
            _X = np.tile(_Xmn, (len(_Xmn), 1))
            _X[np.diag_indices(len(_Xmn))] = _Xnn
            Xs.append(_X)

        return np.array(Xs) # shape : nb_freq, nb_n, nb_n

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
        Xk = []
        for cnx in self.connections:
            Xk.append(self._Xk(cnx))
        Xk = np.array(Xk)

        #X = np.zeros(len(C.frequency), )
        Xf = []
        for (idx, f) in enumerate(self.frequency):
            Xf.append(block_diag(*Xk[:,idx,:]))
        return np.array(Xf)  # shape: (nb_frequency, nb_inter*nb_n, nb_inter*nb_n)

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
    def S(self):
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
    def S_external(self):
        '''
        Return the scattering parameter for the external ports
        '''
        port_indexes = self.port_indexes
        a, b = np.meshgrid(port_indexes, port_indexes)
        S_ext = self.S[:, a, b]
        return S_ext  # shape (nb_frequency, nb_ports, nb_ports)

    @property
    def network(self):
        '''
        Return the Network associated to external ports
        '''
        ntw = Network()
        ntw.frequency = self.frequency
        ntw.z0 = self.port_z0
        ntw.s = self.S_external
        return ntw
