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
                [ [(network1, network1_port_nb), (network2, network2_port_nb) ]


        '''
        self.connections = connections
        # TODO: check all network frequencies are the same    
        
        ntws = self.networks_list(self.connections)
        self.frequency = ntws[0].frequency
        
    @classmethod
    def Port(cls, frequency, name, Z0=50+0*1j):
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
        _media = media.DefinedGammaZ0(frequency, Z0=Z0)
        return _media.match(name=name)
        
    def networks_dict(self, connections=None, min_nports=1):
        '''
        Return the dictionnary of Networks from the connection setup X
        '''
        if not connections:
            connections = self.connections
        
        ntws = []
        for cnx in connections:
            for (ntw,port) in cnx:
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
            cnx_name = f'$X_{idx}$'
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

    def plot(self):
        '''
        Plot the graph of the circuit
        '''
        nx.draw(self.G, with_labels=True)


    def _Y_k(self, cnx):
        '''
        Return the sum of the system admittances of each intersection
        '''
        y_ns = []
        for (ntw, ntw_port) in cnx:

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
        for (idx_cnx, cnx) in enumerate(chain.from_iterable(self.connections)):
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
        return self.X @ np.linalg.inv(np.eye(self.dim) - self.C @ self.X)

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
    
    @property
    def S_external(self):
        '''
        Return the scattering parameter for the external ports
        '''
        port_indexes = self.port_indexes
        a, b = np.meshgrid(port_indexes, port_indexes)
        S_ext = self.S[:, a, b]
        return S_ext  #  shape (nb_frequency, nb_ports, nb_ports)
        