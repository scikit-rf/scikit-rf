"""
circuit (:mod:`skrf.circuit`)
========================================

The Circuit class represents a circuit of arbitrary topology,
consisting of an arbitrary number of N-ports networks.

Like in an electronic circuit simulator, the circuit must have one or more ports
connected to the circuit. The Circuit object allows one retrieving the M-ports network,
where M is the number of ports defined.

The results are returned in :class:`~skrf.circuit.Circuit` object.


Building a Circuit
------------------
.. autosummary::
   :toctree: generated/

   Circuit
   Circuit.Port
   Circuit.SeriesImpedance
   Circuit.ShuntAdmittance
   Circuit.Ground
   Circuit.Open
   Circuit.update_networks

Representing a Circuit
----------------------
.. autosummary::
   :toctree: generated/

   Circuit.plot_graph

Network Representations
-----------------------
.. autosummary::
   :toctree: generated/

   Circuit.network
   Circuit.s
   Circuit.s_external
   Circuit.s_active
   Circuit.z_active
   Circuit.y_active
   Circuit.vswr_active
   Circuit.port_z0

Voltages and Currents
---------------------
.. autosummary::
   :toctree: generated/

   Circuit.voltages
   Circuit.voltages_external
   Circuit.currents
   Circuit.currents_external

Circuit internals
------------------
.. autosummary::
   :toctree: generated/

   Circuit.networks_dict
   Circuit.networks_list
   Circuit.connections_nb
   Circuit.connections_list
   Circuit.nodes_nb
   Circuit.dim
   Circuit.intersections_dict
   Circuit.port_indexes
   Circuit.C
   Circuit.X

Circuit reduction
------------------
.. autosummary::
   :toctree: generated/

   reduce_circuit

Graph representation
--------------------
.. autosummary::
   :toctree: generated/

   Circuit.graph
   Circuit.G
   Circuit.edges
   Circuit.edge_labels

"""
from __future__ import annotations

import warnings
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Sequence, TypedDict

import numpy as np
from typing_extensions import NotRequired, Unpack

from .constants import S_DEF_DEFAULT, MemoryLayoutT, NumberLike
from .media import media
from .network import Network, connect, innerconnect, s2s
from .util import subplots

if TYPE_CHECKING:
    from .frequency import Frequency


class Circuit:
    """
    Creates a circuit made of a set of N-ports networks.

    For instructions on how to create Circuit see  :func:`__init__`.

    A Circuit object is representation a circuit assembly of an arbitrary
    number of N-ports networks connected together via an arbitrary topology.

    The algorithm used to calculate the resultant network can be found in [#]_.

    References
    ----------
    .. [#] P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).

    """
    @staticmethod
    def _get_nx():
        """Returns networkx module if available.

        Raises:
        -------
            ImportError: If networkx module is not installed

        Returns:
        --------
            networkx module
        """
        try:
            import networkx as nx
            return nx
        except ImportError as err:
            raise ImportError('networkx package as not been installed and is required.') from err

    class _REDUCE_OPTIONS(TypedDict):
        check_duplication: NotRequired[bool]
        split_ground: NotRequired[bool]
        split_multi: NotRequired[bool]
        max_nports: NotRequired[int]
        dynamic_networks: NotRequired[Sequence[Network]]

    CACHEDPROPERTIES = ('s', 'X', 'X_F', 'C', 'C_F', 'T')

    def __init__(self,
                 connections: list[list[tuple[Network, int]]],
                 name: str | None = None,
                 *,
                 auto_reduce: bool = False, **kwargs: Unpack[_REDUCE_OPTIONS]) -> None:
        """
        Circuit constructor. Creates a circuit made of a set of N-ports networks.

        Parameters
        ----------
        connections : list of list of tuples
            Description of circuit connections.
            Each connection is a described by a list of tuple.
            Each tuple contains (network, network_port_nb).
            Port number indexing starts from zero.
            Network's port not explicitly listed will be treated as matched.
            Network's port listed individually will be treated as open.
        name : string, optional
            Name assigned to the circuit (Network). Default is None.
        auto_reduce : bool, optional
            If True, the circuit will be automatically reduced using :func:`reduce_circuit`.
            This will change the circuit connections description, affecting inner current and voltage distributions.
            Suitable for cases where only the S-parameters of the final circuit ports are of interest. Default is False.
            If `check_duplication`, `split_ground` or `max_nports` are provided as kwargs, `auto_reduce` will be
            automatically set to True, as this indicates an intent to use the `reduce_circuit` method.
        **kwargs :
            keyword arguments passed to `reduce_circuit` method.

            `check_duplication` kwarg controls whether to check the connections have duplicate names. Default is True.

            `split_ground` kwarg controls whether to split the global ground connection to independant.
            Default is False.

            `max_nports` kwarg controls the maximum number of ports of a Network that can be reduced in circuit. If a
            Network in the circuit has a number of ports (nports), using the Network.connect() method to reduce the
            circuit's dimensions becomes less efficient compared to directly calculating it with Circuit.s_external.
            This value depends on the performance of the computer and the scale of the circuit. Default is 20.

            `dynamic_networks` kwarg is a sequence of Networks to be skipped during the reduction process.
            Default is an empty tuple.


        Examples
        --------
        Example of connections between two 1-port networks:
        ::
            connections = [
                [(network1, 0), (network2, 0)],
            ]

        Example of a connection between three 1-port networks connected
        to a single node:
        ::
            connections = [
                [(network1, 0), (network2, 0), (network3, 0)]
            ]

        Example of a connection between two 1-port networks (port1 and port2)
        and two 2-ports networks (ntw1 and ntw2):
        ::
            connections = [
                [(port1, 0), (ntw1, 0)],
                [(ntw1, 1), (ntw2, 0)],
                [(ntw2, 1), (port2, 0)]
            ]

        Example of a connection between three 1-port networks (port1, port2 and port3)
        and a 3-ports network (ntw):
        ::
            connections = [
                [(port1, 0), (ntw, 0)],
                [(port2, 0), (ntw, 1)],
                [(port3, 0), (ntw, 2)]
            ]

        Example of a connection between three 1-port networks (port1, port2 and open)
        and a 3-ports network (ntw):
        ::
            connections = [
                [(port1, 0), (ntw, 0)],
                [(open, 0), (ntw, 1)],
                [(port2, 0), (ntw, 2)]
            ]
            # or equivalently
            connections = [
                [(port1, 0), (ntw, 0)],
                [(ntw, 1)],
                [(port2, 0), (ntw, 2)]
            ]

        Example of a connection between three 1-port networks (port1, port2 and match)
        and a 3-ports network (ntw):
        ::
            connections = [
                [(port1, 0), (ntw, 0)],
                [(match, 0), (ntw, 1)],
                [(port2, 0), (ntw, 2)]
            ]
            # or equivalently
            connections = [
                [(port1, 0), (ntw, 0)],
                [(port2, 0), (ntw, 2)]
            ]

        NB1: Creating 1-port network to be used as a port should be made with :func:`Port`

        NB2: The external ports indexing is defined by the order of appearance of
        the ports in the connections list. Thus, the first network identified
        as a port will be the 1st port of the resulting network (index 0),
        the second network identified as a port will be the second port (index 1),
        etc.

        NB3: When a port of a network is listed individually in the connections, the circuit
        will treat this port with a reflection coefficient of 1, equivalent to an Open.
        Conversely, if a port is not explicitly included in the connections,the circuit will
        treat it as matched.

        """
        self._connections = connections
        self.name = name

        # check if all networks have a name
        for cnx in self.connections:
            for (ntw, _) in cnx:
                if not self._is_named(ntw):
                    raise AttributeError('All Networks must have a name. Faulty network:', ntw)

        # list of networks for initial checks
        ntws = self.networks_list()

        # check if all networks have same frequency
        ref_freq = ntws[0].frequency
        for ntw in ntws:
            if ntw.frequency != ref_freq:
                raise AttributeError('All Networks must have same frequencies')
        # All frequencies are the same, Circuit frequency can be any of the ntw
        self.frequency = ntws[0].frequency

        # Check that a (ntwk, port) combination appears only once in the connexion map
        Circuit.check_duplicate_names(self.connections_list)

        # Get the keyword arguments for the reduce_circuit method
        kwargs_reduce = {k: kwargs[k] for k in self._REDUCE_OPTIONS.__annotations__.keys() if k in kwargs}

        # Reduce the circuit if directly requested or any relevant kwargs are provided
        if auto_reduce or any(kwargs_reduce):
            self.connections = reduce_circuit(self.connections, **kwargs_reduce)

    @property
    def connections(self) -> list[list[tuple[Network, int]]]:
        """
        Circuit connections.

        Description of circuit connections. Each connection is a described by a
        list of tuple. Each tuple contains (network, network_port_nb). Port number
        indexing starts from zero.

        Returns
        -------
        connections : :class:`list[list[tuple[Network, int]]]`
            The circuit connections.

        """
        return self._connections

    @connections.setter
    def connections(self, connections: list[list[tuple[Network, int]]]) -> None:
        """
        Set the circuit connections.

        Parameters
        ----------
        connections : :class:`list[list[tuple[Network, int]]]`
            The circuit connections.

        """
        self._connections = connections

        # Invalidate cached properties
        for item in self.CACHEDPROPERTIES:
            self.__dict__.pop(item, None)

    def update_networks(
        self, networks: tuple[Network],
        name: str | None = None,
        *,
        inplace: bool = False,
        auto_reduce: bool = False, **kwargs: Unpack[_REDUCE_OPTIONS]) -> Circuit | None:
        """
        Update the circuit connections with a new set of networks.

        Parameters
        ----------
        networks : tuple[Network]
            A tuple of Networks to be updated in the circuit.
        name : string, optional
            Name assigned to the circuit (Network). Default is None.
        inplace : bool, optional
            If True, the circuit connections will be updated inplace. Default is False.
        auto_reduce : bool, optional
            If True, the circuit will be automatically reduced using :func:`reduce_circuit`.
            This will change the circuit connections description, affecting inner current and voltage distributions.
            Suitable for cases where only the S-parameters of the final circuit ports are of interest. Default is False.
            If `check_duplication`, `split_ground` or `max_nports` are provided as kwargs, `auto_reduce` will be
            automatically set to True, as this indicates an intent to use the `reduce_circuit` method.
        **kwargs :
            keyword arguments passed to `reduce_circuit` method.

            `check_duplication` kwarg controls whether to check the connections have duplicate names. Default is True.

            `split_ground` kwarg controls whether to split the global ground connection to independant.
            Default is False.

            `max_nports` kwarg controls the maximum number of ports of a Network that can be reduced in circuit. If a
            Network in the circuit has a number of ports (nports), using the Network.connect() method to reduce the
            circuit's dimensions becomes less efficient compared to directly calculating it with Circuit.s_external.
            This value depends on the performance of the computer and the scale of the circuit. Default is 20.

            `dynamic_networks` kwarg is a sequence of Networks to be skipped during the reduction process.
            Default is an empty tuple.

        Returns
        -------
        Circuit or None if inplace=True
            The updated `Circuit` with the specified networks.


        See Also
        --------
        Circuit.__init__ : Circuit construtor method.


        Examples
        --------
        >>> import skrf as rf
        >>> import numpy as np
        >>>
        >>> # Create a series of networks and build a circuit
        >>> connections = [
        ...     ...
        ...     [(ntwk1, 0), (ntwk2, 0)],
        ...     [(ntwk2, 1), (ntwk3, 0)],
        ...     ...
        ... ]
        >>> circuit = rf.Circuit(connections, dynamic_networks=[ntwk2])
        >>>
        >>> # Update the networks' S-parameters
        >>> ntwk2.s = ntwk2.s @ ntwk2.s
        >>>
        >>> # Update the circuit in traditional way (ntwk2 has been updated)
        >>> connections_updated = [
        ...     ...
        ...     [(ntwk1, 0), (ntwk2, 0)],
        ...     [(ntwk2, 1), (ntwk3, 0)],
        ...     ...
        ... ]
        >>>
        >>> circuit_updated_a = rf.Circuit(connections_updated)
        >>>
        >>> # Update the circuit by dynamic networks
        >>> circuit_updated_b = circuit.update_networks(networks=[ntwk2])
        >>>
        >>> np.allclose(circuit_updated_b.s_external, circuit_updated_b.s_external)
        True
        """
        # Get current connection_dict
        cnx_dict = self.networks_dict()

        # Update the connection dict with the new networks
        for ntw in networks:
            if ntw.name in cnx_dict:
                cnx_dict[ntw.name] = ntw
            else:
                raise ValueError(f"Network {ntw.name} not found in circuit.")

        # Update the circuit connections with the new networks
        connections = [
            [(cnx_dict[n.name], p) for n, p in cnx] for cnx in self.connections
        ]

        # Get the keyword arguments for the reduce_circuit method
        kwargs_reduce = {k: kwargs[k] for k in self._REDUCE_OPTIONS.__annotations__.keys() if k in kwargs}

        # Reduce the circuit if directly requested or any relevant kwargs are provided
        if auto_reduce or any(kwargs_reduce):
            connections = reduce_circuit(connections, **kwargs_reduce)

        if inplace:
            self.connections = connections
            return None

        return Circuit(connections=connections, name=name)

    @classmethod
    def check_duplicate_names(cls, connections_list: list[tuple[int, tuple[Network, int]]]):
        """
        Check that a (ntwk, port) combination appears only once in the connexion map
        """
        nodes = [(ntwk.name, port) for (_, (ntwk, port)) in connections_list]
        if len(nodes) > len(set(nodes)):
            duplicate_nodes = [node for node in nodes if nodes.count(node) > 1]
            raise AttributeError(f'Nodes {duplicate_nodes} appears twice in the connection description.')


    def _is_named(self, ntw: Network):
        """
        Return True is the network has a name, False otherwise
        """
        if not ntw.name or ntw.name == '':
            return False
        else:
            return True

    @classmethod
    def _is_port(cls, ntw: Network):
        """
        Return True is the network is a port, False otherwise
        """
        return ntw._ext_attrs.get("_is_circuit_port", False)

    @classmethod
    def _is_ground(cls, ntw: Network):
        """
        Return True is the network is a ground, False otherwise
        """
        return ntw._ext_attrs.get("_is_circuit_ground", False)

    @classmethod
    def _is_open(cls, ntw: Network):
        """
        Return True is the network is a open, False otherwise
        """
        return ntw._ext_attrs.get("_is_circuit_open", False)

    @classmethod
    def Port(cls, frequency: Frequency, name: str, z0: float = 50) -> Network:
        """
        Return a 1-port Network to be used as a Circuit port.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        name : string
            Name of the port.
        z0 : real, optional
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        -------
        port : :class:`~skrf.network.Network` object
            1-port network

        Examples
        --------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: port1 = rf.Circuit.Port(freq, name='Port1')
        """
        _media = media.DefinedGammaZ0(frequency, z0=z0)
        port = _media.match(name=name)
        port._ext_attrs['_is_circuit_port'] = True
        return port

    @classmethod
    def SeriesImpedance(cls, frequency: Frequency, Z: NumberLike, name: str, z0: float = 50) -> Network:
        """
        Return a 2-port network of a series impedance.

        Passing the frequency and name is mandatory.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        Z : complex array of shape n_freqs or complex
            Impedance
        name : string
            Name of the series impedance
        z0 : real, optional
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        -------
        serie_impedance : :class:`~skrf.network.Network` object
            2-port network

        Examples
        --------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: open = rf.Circuit.SeriesImpedance(freq, rf.INF, name='series_impedance')

        """
        A = np.zeros(shape=(len(frequency), 2, 2), dtype=complex)
        A[:, 0, 0] = 1
        A[:, 0, 1] = Z
        A[:, 1, 0] = 0
        A[:, 1, 1] = 1
        ntw = Network(a=A, frequency=frequency, z0=z0, name=name)
        return ntw

    @classmethod
    def ShuntAdmittance(cls, frequency: Frequency, Y: NumberLike, name: str, z0: float = 50) -> Network:
        """
        Return a 2-port network of a shunt admittance.

        Passing the frequency and name is mandatory.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        Y : complex array of shape n_freqs or complex
            Admittance
        name : string
            Name of the shunt admittance
        z0 : real, optional
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        -------
        shunt_admittance : :class:`~skrf.network.Network` object
            2-port network

        Examples
        --------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: short = rf.Circuit.ShuntAdmittance(freq, rf.INF, name='shunt_admittance')

        """
        A = np.zeros(shape=(len(frequency), 2, 2), dtype=complex)
        A[:, 0, 0] = 1
        A[:, 0, 1] = 0
        A[:, 1, 0] = Y
        A[:, 1, 1] = 1
        ntw = Network(a=A, frequency=frequency, z0=z0, name=name)
        return ntw

    @classmethod
    def Ground(cls, frequency: Frequency, name: str, z0: float = 50) -> Network:
        """
        Return a 1-port network of a grounded link.

        Passing the frequency and a name is mandatory.

        The ground link is implemented by media.short object.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        name : string
            Name of the ground.
        z0 : real, optional
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        -------
        ground : :class:`~skrf.network.Network` object
            1-port network

        Examples
        --------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: ground = rf.Circuit.Ground(freq, name='GND')

        """
        _media = media.DefinedGammaZ0(frequency, z0=z0)
        ground = _media.short(name=name)
        ground._ext_attrs['_is_circuit_ground'] = True
        return ground

    @classmethod
    def Open(cls, frequency: Frequency, name: str, z0: float = 50) -> Network:
        """
        Return a 1-port network of an open link.

        Passing the frequency and name is mandatory.

        The open link is implemented by media.open object.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency`
            Frequency common to all other networks in the circuit
        name : string
            Name of the open.
        z0 : real, optional
            Characteristic impedance of the port. Default is 50 Ohm.

        Returns
        -------
        open : :class:`~skrf.network.Network` object
            1-port network

        Examples
        --------
        .. ipython::

            @suppress
            In [16]: import skrf as rf

            In [17]: freq = rf.Frequency(start=1, stop=2, npoints=101)

            In [18]: open = rf.Circuit.Open(freq, name='open')

        """
        _media = media.DefinedGammaZ0(frequency, z0=z0)
        Open = _media.open(name=name)
        Open._ext_attrs['_is_circuit_open'] = True
        return Open

    def networks_dict(self,
                      connections: list[list[tuple[Network, int]]] | None = None,
                      min_nports: int = 1) -> dict[str, Network]:
        """
        Return the dictionary of Networks from the connection setup X.

        Parameters
        ----------
        connections : List, optional
            connections list, by default None (then uses the `self.connections`)
        min_nports : int, optional
            min number of ports, by default 1

        Returns
        -------
        dict
            Dictionnary of Networks
        """
        if not connections:
            connections = self.connections

        ntws: list[Network] = []
        for cnx in connections:
            for (ntw, _port) in cnx:
                ntws.append(ntw)
        return {ntw.name: ntw for ntw in ntws  if ntw.nports >= min_nports}

    def networks_list(self,
                      connections: list[list[tuple[Network, int]]] | None = None,
                      min_nports: int = 1) -> list[Network]:
        """
        Return a list of unique networks (sorted by appearing order in connections).

        Parameters
        ----------
        connections : List, optional
            connections list, by default None (then uses the `self.connections`)
        min_nports : int, optional
            min number of ports, by default 1

        Returns
        -------
        list
            List of unique networks
        """
        if not connections:
            connections = self.connections

        ntw_dict = self.networks_dict(connections)
        return [ntw for ntw in ntw_dict.values() if ntw.nports >= min_nports]

    @property
    def connections_nb(self) -> int:
        """
        Return the number of intersections in the circuit.
        """
        return len(self.connections)

    @property
    def connections_list(self) -> list[tuple[int, tuple[Network, int]]]:
        """
        Return the full list of connections, including intersections.

        The resulting list if of the form:
        ::
            [
             [connexion_number, connexion],
             [connexion_number, connexion],
             ...
            ]
        """
        return list(enumerate(chain.from_iterable(self.connections)))

    @property
    def networks_nb(self) -> int:
        """
        Return the number of connected networks (port excluded).
        """
        return len(self.networks_list(self.connections))

    @property
    def nodes_nb(self) -> int:
        """
        Return the number of nodes in the circuit.
        """
        return self.connections_nb + self.networks_nb

    @property
    def dim(self) -> int:
        """
        Return the dimension of the C, X and global S matrices.

        It correspond to the sum of all connections.
        """
        return np.sum([len(cnx) for cnx in self.connections])

    @property
    def G(self):
        """
        Generate the graph of the circuit. Convenience shortname for :func:`graph`.
        """
        return self.graph()

    def graph(self):
        """
        Generate the graph of the circuit.

        Returns
        -------
        G: :class:`networkx.Graph`
            graph object [#]_ .

        References
        ----------
        .. [#] https://networkx.github.io/
        """

        nx = self._get_nx()
        G = nx.Graph()
        # Adding network nodes
        G.add_nodes_from([it for it in self.networks_dict(self.connections)])

        # Adding edges in the graph between connections and networks
        for (idx, cnx) in enumerate(self.connections):
            cnx_name = 'X'+str(idx)
            # Adding connection nodes and edges
            G.add_node(cnx_name)
            for (ntw, _ntw_port) in cnx:
                ntw_name = ntw.name
                G.add_edge(cnx_name, ntw_name)
        return G

    def is_connected(self) -> bool:
        """
        Check if the circuit's graph is connected.

        Check if every pair of vertices in the graph is connected.
        """

        nx = self._get_nx()
        return nx.algorithms.components.is_connected(self.G)


    @property
    def intersections_dict(self) -> dict:
        """
        Return a dictionary of all intersections with associated ports and z0:

        ::
            { k: [(ntw1_name, ntw1_port), (ntw1_z0, ntw2_name, ntw2_port), ntw2_z0], ... }
        """
        inter_dict = {}
        # for k in range(self.connections_nb):
        #     # get all edges connected to intersection Xk
        #     inter_dict[k] = list(nx.algorithms.boundary.edge_boundary(self.G, ('X'+str(k),) ))

        for (k, cnx) in enumerate(self.connections):
            inter_dict[k] = [(ntw, ntw_port, ntw.z0[0, ntw_port]) \
                                  for (ntw, ntw_port) in cnx]
        return inter_dict

    @property
    def edges(self) -> list:
        """
        Return the list of all circuit connections
        """
        return list(self.G.edges)

    @property
    def edge_labels(self) -> dict:
        """
        Return a dictionary describing the port and z0 of all graph edges.

        Dictionary is in the form:
        ::
            {('ntw1_name', 'X0'): '3 (50+0j)',
             ('ntw2_name', 'X0'): '0 (50+0j)',
             ('ntw2_name', 'X1'): '2 (50+0j)', ... }

        which can be used in `networkx.draw_networkx_edge_labels`
        """
        # for all intersections,
        # get the N interconnected networks and associated ports and z0
        # and forge the edge label dictionary containing labels between
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


    def _Xk(self,
            cnx_k: list[tuple[Network, int]],
            order: MemoryLayoutT = 'C',
            inverse: bool = False) -> np.ndarray:
        """
        Return the scattering matrices [X]_k of the individual intersections k.
        The results in [#]_ do not agree due to an error in the formula (3)
        for mismatched intersections.

        Due to the ideal power splitter is unitary, that is [X]_k^H * [X]_k = I, where [X]_k^H is
        the conjugate transpose of [X]_k. And considers the reciprocity, the inverse of [X]_k could
        be simplified as the conjugate of [X]_k, that is [X]_k^-1 = [X]_k^*.

        Parameters
        ----------
        cnx_k : list of tuples
            each tuple contains (network, port)
        order: str, optional
            'C' or 'F' for row-major or column-major order of the output array.
            Default is 'C'.
        inverse: bool, optional
            If True, return the inverse of the scattering matrix [X]_k.
            Default is False.

        Returns
        -------
        Xs : :class:`numpy.ndarray`
            shape `f x n x n`

        References
        ----------
        .. [#] P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).
        """

        y0s = np.array([1/ntw.z0[:,ntw_port] for (ntw, ntw_port) in cnx_k]).T
        y_k = y0s.sum(axis=1)

        Xs = np.zeros((len(self.frequency), len(cnx_k), len(cnx_k)), dtype='complex', order=order)

        Xs = 2 *np.sqrt(np.einsum('ij,ik->ijk', y0s, y0s)) / y_k[:, None, None]
        np.einsum('kii->ki', Xs)[:] -= 1  # Sii

        return np.conjugate(Xs) if inverse else Xs


    def _X(self, order: MemoryLayoutT = 'C', inverse: bool = False) -> np.ndarray:
        """
        Return the concatenated intersection matrix [X] of the circuit.

        It is composed of the individual intersection matrices [X]_k assembled
        by block diagonal.

        Due to the concatenated intersection matrix [X] is a block diagonal matrix,
        the inverse of [X] could be simplified as the block diagonal matrix of the
        inverses of [X]_k.

        Parameters
        ----------
            order : str, optional
                'C' or 'F' for row-major or column-major order of the output array.
                Default is 'C'.

            inverse : bool, optional
                If True, return the inverse of the concatenated intersection matrix [X].
                Default is False.

        Returns
        -------
        X : :class:`numpy.ndarray`

        Note
        ----
        The block diagonal matrix [X] has a numerical bottleneck that depends on the
        order specified:
        - When 'order' is 'C', the creation of the block diagonal matrix [X] in row-major
          order is a bottleneck. This is because the creation process is port-by-port,
          which does not align well with the row-major memory layout, leading to
          suboptimal performance.
        - When 'order' is 'F', the computation of the block diagonal matrix [X] in column-major
          order is a bottleneck. Numpy generally optimizes operators for 'C' order, which
          lead to performance issues when using 'F' order.
        """
        Xks = [self._Xk(cnx, order, inverse) for cnx in self.connections]

        Xf = np.zeros((len(self.frequency), self.dim, self.dim), dtype='complex', order=order)
        off = np.array([0, 0])
        for Xk in Xks:
            Xf[:, off[0]:off[0] + Xk.shape[1], off[1]:off[1]+Xk.shape[2]] = Xk
            off += Xk.shape[1:]

        return Xf

    @cached_property
    def X(self) -> np.ndarray:
        """
        Return the concatenated intersection matrix [X] of the circuit in C-order.

        It is composed of the individual intersection matrices [X]_k assembled
        by block diagonal.

        Returns
        -------
        X : :class:`numpy.ndarray`

        Note
        ----
        There is a numerical bottleneck in this function,
        when creating the block diagonal matrice [X] from the [X]_k matrices.
        """
        # Check if X_F is alread computed, if so, convert it to C-order
        if self.__dict__.get('X_F', None) is not None:
            return np.ascontiguousarray(self.X_F)

        return self._X()

    @cached_property
    def X_F(self) -> np.ndarray:
        """
        Return the concatenated intersection matrix [X] of the circuit in F-order.
        It is composed of the individual intersection matrices [X]_k assembled
        by block diagonal. The results of this function are the same as :func:`X`
        but in F-order.

        Returns
        -------
        X : :class:`numpy.ndarray`

        Note
        ----
        F-order has a numerical bottleneck in matrix operations, but the assignment
        is more efficient.
        """
        # Check if X is alread computed, if so, convert it to F-order
        if self.__dict__.get('X', None) is not None:
            return np.asfortranarray(self.X)

        return self._X('F')

    @cached_property
    def C(self) -> np.ndarray:
        """
        Return the global scattering matrix [C] of the networks in C-order.

        Returns
        -------
        S : :class:`numpy.ndarray`
            Global scattering matrix of the networks.
            Shape `f x (nb_inter*nb_n) x (nb_inter*nb_n)`
        """
        # Check if C_F is alread computed, if so, convert it to C-order
        if self.__dict__.get('C_F', None) is not None:
            return np.ascontiguousarray(self.C_F)

        return self._C()

    @cached_property
    def C_F(self) -> np.ndarray:
        """
        Return the global scattering matrix [C] of the networks in F-order. The results
        of this function are the same as :func:`C` but in F-order.

        Returns
        -------
        S : :class:`numpy.ndarray`
            Global scattering matrix of the networks.
            Shape `f x (nb_inter*nb_n) x (nb_inter*nb_n)`

        Note
        ----
        F-order has a numerical bottleneck in matrix operations, but the assignment
        is more efficient.
        """
        # Check if C is alread computed, if so, convert it to F-order
        if self.__dict__.get('C', None) is not None:
            return np.asfortranarray(self.C)

        return self._C('F')

    def _C(self, order: MemoryLayoutT = 'C') -> np.ndarray:
        """
        Return the global scattering matrix [C] of the networks.

        Args:
            order : str, optional
                'C' or 'F' for row-major or column-major order of the output array.
                Default is 'C'.

        Returns
        -------
        S : :class:`numpy.ndarray`
            Global scattering matrix of the networks.
            Shape `f x (nb_inter*nb_n) x (nb_inter*nb_n)`
        """
        # list all networks which are not considered as "ports",
        ntws = {k:v for k,v in self.networks_dict().items() if not Circuit._is_port(v)}

        # generate the port reordering indexes from each connections
        ntws_ports_reordering = {ntw:[] for ntw in ntws}
        for (idx_cnx, (ntw, ntw_port)) in self.connections_list:
            if ntw.name in ntws.keys():
                ntws_ports_reordering[ntw.name].append([ntw_port, idx_cnx])

        # re-ordering scattering parameters
        S = np.zeros((len(self.frequency), self.dim, self.dim), dtype='complex', order=order)

        for (ntw_name, ntw_ports) in ntws_ports_reordering.items():
            # get the port re-ordering indexes (from -> to)
            ntw_ports = np.array(ntw_ports)

            # port permutations
            from_port = ntw_ports[:,0]
            to_port = ntw_ports[:,1]

            for (_from, _to) in zip(from_port, to_port):
                S[:, _to, to_port] = ntws[ntw_name].s_traveling[:, _from, from_port]

        return S  # shape (nb_frequency, nb_inter*nb_n, nb_inter*nb_n)

    @cached_property
    def T(self) -> np.ndarray:
        """
        Return the matrix of multiplication of the global scattering matrix [C] and concatenated
        intersection matrix [X] of the networks, that is [T] = - [C] @ [X].

        Returns
        -------
        T : :class:`numpy.ndarray`
            Multiplication of the global scattering matrix [C] and concatenated intersection matrix
            [X] of the networks. In practice, F-contiguous [C_F] and [X_F] are used.
            Shape `f x (nb_inter*nb_n) x (nb_inter*nb_n)`

        Note
        ----
        This is an auxiliary matrix used to break the numerical bottleneck of [C_F] @ [X_F] using the
        mathematical feature of block diagonal matrice [X].
        """
        X, C = self.X_F, self.C_F

        # T will be fully updated, so `empty_like` is safe and faster
        T = np.empty_like(X, dtype="complex", order='F')

        # Precompute the sizes of connections and slices for each intersection
        cnx_size = [len(cnx) for cnx in self.connections]
        slice_ = np.cumsum([0] + cnx_size)
        slices = [slice(slice_[i], slice_[i+1]) for i in range(len(cnx_size))]

        # Perform the multiplication
        for j_slice in slices:
            # Get the Block diagonal part of X and corresponding C matrix buffer
            X_jj = - X[:, j_slice, j_slice]
            C_j = C[:, :, j_slice]

            # Perform the multiplication
            for i_slice in slices:
                T[:, i_slice, j_slice] = C_j[:, i_slice, :] @ X_jj

        return T

    @cached_property
    def s(self) -> np.ndarray:
        """
        Return the global scattering parameters of the circuit.

        Return the scattering parameters of both "inner" and "outer" ports.

        Note
        ----
        The original implementation of this method calculated the S-parameters [S] using the global
        scattering matrix [C] and the concatenated intersection matrix [X] by solving the equation:
        [S] = [X] @ ([E] - [C] @ [X])^-1

        where [E] is the identity matrix. However, this approach is computationally inefficient for
        large networks due to its numerical bottlenecks.

        The current implementation optimizes the calculation by leveraging matrix algebra to simplify
        the relationship between the incident {a} and reflected {b} power waves. The key steps are as
        follows:

        1. Define the relationship between {b} and {a} using [X] and [C]:
            {b} = [X] @ ([E] - [C] @ [X])^-1 @ {a}

        2. Simplify the relationship between {a} and {b}:
            {a} = ([X] @ ([E] - [C] @ [X])^-1)^-1 @ {b}
                = ([E] - [C] @ [X]) @ [X]^-1 @ {b}
                = ([X]^-1 - [C] @ [X] @ [X]^-1) @ {b}
                = ([X]^-1 - [C]) @ {b}

        3. Derive the S-parameters matrix [S] from the simplified relationship:
            [S] = ([X]^-1 - [C])^-1

        4. Leverage the property of the concatenated intersection matrix [X]:
            - [X] is a block diagonal matrix composed of individual intersection matrices [X]_k:
                [X] = diag([X_1], [X_2], ..., [X_n]).
            - Each [X]_k represents an ideal lossless power splitter, which is unitary:
                [X]_k^H @ [X]_k = I,
                where [X]_k^H is the conjugate transpose of [X]_k.
            - Due to reciprocity, the inverse of [X]_k is its conjugate:
                [X]_k^-1 = [X]_k^*.

        5. Substitute the property into the S-parameters equation:
            [S] = ([X^T]^* - [C])^-1

        This approach avoids the explicit computation of [X]^-1, significantly improving computational
        efficiency for large-scale networks.

        Returns
        -------
        S : :class:`numpy.ndarray`
            global scattering parameters of the circuit.
        """
        return np.linalg.inv(self._X(inverse=True) - self.C)

    @property
    def port_indexes(self) -> list[int]:
        """
        Return the indexes of the "external" ports.

        Returns
        -------
        port_indexes : list
        """
        port_indexes = []
        for (idx_cnx, (ntw, _)) in enumerate(chain.from_iterable(self.connections)):
            if Circuit._is_port(ntw):
                port_indexes.append(idx_cnx)
        return port_indexes

    def _cnx_z0(self, cnx_k: list[tuple]) -> np.ndarray:
        """
        Return the characteristic impedances of a specific connections.

        Parameters
        ----------
        cnx_k : list of tuples
            each tuple contains (network, port)

        Returns
        -------
        z0s : :class:`numpy.ndarray`
            shape `f x nb_ports_at_cnx`
        """
        z0s = []
        for (ntw, ntw_port) in cnx_k:
            z0s.append(ntw.z0[:,ntw_port])

        return np.array(z0s).T  # shape (nb_freq, nb_ports_at_cnx)

    @property
    def port_z0(self) -> np.ndarray:
        """
        Return the external port impedances.

        Returns
        -------
        z0s : :class:`numpy.ndarray`
            shape `f x nb_ports`
        """
        z0s = []
        for cnx in self.connections:
            for (ntw, ntw_port) in cnx:
                z0s.append(ntw.z0[:,ntw_port])

        return np.array(z0s)[self.port_indexes, :].T  # shape (nb_freq, nb_ports)

    @property
    def s_external(self) -> np.ndarray:
        """
        Return the scattering parameters for the external ports.

        Returns
        -------
        S : :class:`numpy.ndarray`
            Scattering parameters of the circuit for the external ports.
            Shape `f x nb_ports x nb_ports`
        """
        # The external S-matrix is the submatrix corresponding to external ports:
        # port_indexes = self.port_indexes
        # a, b = np.meshgrid(port_indexes, port_indexes, indexing='ij')
        # S_ext = self.s[:, a, b]

        # Instead of calculating all S-parameters and taking a submatrix,
        # the following faster approach only calculates external the S-parameters
        # from block-matrix operations.
        # generate index lists of internal and external ports
        port_indexes = self.port_indexes
        in_idxs = [(i,) for i in range(self.dim) if i not in port_indexes]
        ext_idxs = [(i,) for i in port_indexes]
        ext_l, in_l = len(ext_idxs), len(in_idxs)

        # generate index slices for each sub-matrices
        idx_a, idx_b, idx_c, idx_d = (
            np.repeat(i, l, axis=1)
            for i, l in (
                (ext_idxs, ext_l),
                (ext_idxs, in_l),
                (in_idxs, ext_l),
                (in_idxs, in_l),
            )
        )

        # sub-matrices index, Matrix = [[A, B], [C, D]]]
        A_idx = (slice(None), idx_a, idx_a.T)
        B_idx = (slice(None), idx_b, idx_c.T)
        C_idx = (slice(None), idx_c, idx_b.T)
        D_idx = (slice(None), idx_d, idx_d.T)

        # Get the buffer of global matrix in f-order [X_T] and intermediate temporary matrix [T]
        # [T] = - [C] @ [X]
        x, t = self.X_F, np.array(self.T)
        np.einsum('...ii->...i', t)[:] += 1

        # Get the sub-matrices of inverse of intermediate temporary matrix t
        # The method np.linalg.solve(A, B) is equivalent to np.inv(A) @ B, but more efficient
        try:
            tmp_mat = np.linalg.solve(t[D_idx], t[C_idx])
        except np.linalg.LinAlgError:
            warnings.warn('Singular matrix detected, using numpy.linalg.lstsq instead.', RuntimeWarning, stacklevel=2)
            # numpy.linalg.lstsq only works for 2D arrays, so we need to loop over frequencies
            tmp_mat = np.zeros((self.frequency.npoints, len(in_idxs), len(ext_idxs)), dtype='complex')
            for i in range(self.frequency.npoints):
                tmp_mat[i, :, :] = np.linalg.lstsq(t[i, D_idx[1], D_idx[2]], t[i, C_idx[1], C_idx[2]], rcond=None)[0]

        # Get the external S-parameters for the external ports
        # Calculated by multiplying the sub-matrices of x and t
        S_ext = (x[A_idx] - x[B_idx] @ tmp_mat) @ np.linalg.inv(
            t[A_idx] - t[B_idx] @ tmp_mat
        )

        S_ext = s2s(S_ext, self.port_z0, S_DEF_DEFAULT, 'traveling')
        return S_ext  # shape (nb_frequency, nb_ports, nb_ports)

    @property
    def network(self) -> Network:
        """
        Return the Network associated to external ports.

        Returns
        -------
        ntw : :class:`~skrf.network.Network`
            Network associated to external ports
        """
        return Network(frequency = self.frequency, z0 = self.port_z0,
                      s = self.s_external, name = self.name)

    def s_active(self, a: NumberLike) -> np.ndarray:
        r"""
        Return "active" s-parameters of the circuit's network for a defined wave excitation `a`.

        The "active" s-parameter at a port is the reflection coefficients
        when other ports are excited. It is an important quantity for active
        phased array antennas.

        Active s-parameters are defined by [#]_:

        .. math::

            \mathrm{active}(s)_{mn} = \sum_i s_{mi} \frac{a_i}{a_n}

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude (power-wave formulation [#]_)

        Returns
        -------
        s_act : complex array of shape (n_freqs, n_ports)
            active s-parameters for the excitation a


        References
        ----------
        .. [#] D. M. Pozar, IEEE Trans. Antennas Propag. 42, 1176 (1994).

        .. [#] D. Williams, IEEE Microw. Mag. 14, 38 (2013).

        """
        return self.network.s_active(a)

    def z_active(self, a: NumberLike) -> np.ndarray:
        r"""
        Return the "active" Z-parameters of the circuit's network for a defined wave excitation a.

        The "active" Z-parameters are defined by:

        .. math::

            \mathrm{active}(z)_{m} = z_{0,m} \frac{1 + \mathrm{active}(s)_m}{1 - \mathrm{active}(s)_m}

        where :math:`z_{0,m}` is the characteristic impedance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        z_act : complex array of shape (nfreqs, nports)
            active Z-parameters for the excitation a

        See Also
        --------
            s_active : active S-parameters
            y_active : active Y-parameters
            vswr_active : active VSWR
        """
        return self.network.z_active(a)

    def y_active(self, a: NumberLike) -> np.ndarray:
        r"""
        Return the "active" Y-parameters of the circuit's network for a defined wave excitation a.

        The "active" Y-parameters are defined by:

        .. math::

            \mathrm{active}(y)_{m} = y_{0,m} \frac{1 - \mathrm{active}(s)_m}{1 + \mathrm{active}(s)_m}

        where :math:`y_{0,m}` is the characteristic admittance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        y_act : complex array of shape (nfreqs, nports)
            active Y-parameters for the excitation a

        See Also
        --------
            s_active : active S-parameters
            z_active : active Z-parameters
            vswr_active : active VSWR
        """
        return self.network.y_active(a)

    def vswr_active(self, a: NumberLike) -> np.ndarray:
        r"""
        Return the "active" VSWR of the circuit's network for a defined wave excitation a.

        The "active" VSWR is defined by :

        .. math::

            \mathrm{active}(vswr)_{m} = \frac{1 + |\mathrm{active}(s)_m|}{1 - |\mathrm{active}(s)_m|}

        where :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

        Parameters
        ----------
        a : complex array of shape (n_ports)
            forward wave complex amplitude

        Returns
        -------
        vswr_act : complex array of shape (nfreqs, nports)
            active VSWR for the excitation a

        See Also
        --------
            s_active : active S-parameters
            z_active : active Z-parameters
            y_active : active Y-parameters
        """
        return self.network.vswr_active(a)

    @property
    def z0(self) -> np.ndarray:
        """
        Characteristic impedances of "internal" ports.

        Returns
        -------
        z0 : complex array of shape (nfreqs, nports)
            Characteristic impedances of both "inner" and "outer" ports

        """
        z0s = []
        for _cnx_idx, (ntw, ntw_port) in self.connections_list:
            z0s.append(ntw.z0[:,ntw_port])
        return np.array(z0s).T

    @property
    def connections_pair(self) -> list:
        """
        List the connections by pair.

        Each connection in the circuit is between a specific pair of two
        (networks, port, z0).

        Returns
        -------
        connections_pair : list
            list of pair of connections

        """
        return [self.connections_list[i:i+2] for i in range(0, len(self.connections_list), 2)]


    @property
    def _currents_directions(self) -> np.ndarray:
        """
        Create a array of indices to define the sign of the current.

        The currents are defined positive when entering an internal network.

        Returns
        -------
        directions : array of int (nports, 2)

        Note
        ----
        This function is used in internal currents and voltages calculations.

        """
        directions = np.zeros((self.dim,2), dtype='int')
        for cnx_pair in self.connections_pair:
            (cnx_idx_A, cnx_A), (cnx_idx_B, cnx_B) = cnx_pair
            directions[cnx_idx_A,:] = cnx_idx_A, cnx_idx_B
            directions[cnx_idx_B,:] = cnx_idx_B, cnx_idx_A
        return directions

    def _a(self, a_external: NumberLike) -> np.ndarray:
        """
        Wave input array at "internal" ports.

        Parameters
        ----------
        a_external : array
            power-wave input vector at ports

        Returns
        -------
        a_internal : array
            Wave input array at internal ports

        """
        # create a zero array and fill the values corresponding to ports
        a_internal = np.zeros(self.dim, dtype='complex')
        a_internal[self.port_indexes] = a_external
        return a_internal

    def _a_external(self, power: NumberLike, phase: NumberLike) -> np.ndarray:
        r"""
        Wave input array at Circuit's ports ("external" ports).

        The array is defined from power and phase by:

        .. math::

            a = \sqrt(2 P_{in} ) e^{j \phi}

        The factor 2 is in order to deal with peak values.

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]

        NB: the size of the power and phase array should match the number of ports

        Returns
        -------
        a_external: array
            Wave input array  at Circuit's ports

        """
        if len(power) != len(self.port_indexes):
            raise ValueError('Length of power array does not match the number of ports of the circuit.')
        if len(phase) != len(self.port_indexes):
            raise ValueError('Length of phase array does not match the number of ports of the circuit.')
        return np.sqrt(2*np.array(power))*np.exp(1j*np.array(phase))

    def _b(self, a_internal: NumberLike) -> np.ndarray:
        """
        Wave output array at "internal" ports

        Parameters
        ----------
        a_internal : array
            Wave input array at internal ports

        Returns
        -------
        b_internal : array
            Wave output array at internal ports

        Note
        ----
        Wave input array at internal ports can be derived from power and
        phase excitation at "external" ports using `_a(power, phase)` method.

        """
        return self.s @ a_internal

    def currents(self, power: NumberLike, phase: NumberLike) -> np.ndarray:
        """
        Currents at internal ports.

        NB: current direction is defined as positive when entering a node.

        NB: external current sign are opposite than corresponding internal ones,
            as the internal currents are actually flowing into the "port" networks

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]

        Returns
        -------
        I : complex array of shape (nfreqs, nports)
            Currents in Amperes [A] (peak) at internal ports.

        """
        a = self._a(self._a_external(power, phase))
        b = self._b(a)
        z0s = self.z0
        i, Is = 0, np.zeros_like(z0s)

        for cnx in self.connections:
            cnx_len = len(cnx)
            z0_segment = z0s[:, i : i + cnx_len]
            tot_shunt_z0 = (1 / z0_segment).sum(axis=1)
            Ij = np.zeros_like(z0_segment)

            # Calculate the ports' output current through the output wave
            for j in range(cnx_len):
                in_z0 = z0_segment[:, j]
                out_z0 = 1 / (tot_shunt_z0 - 1 / in_z0)
                tau = (2 * out_z0) / (out_z0 + in_z0)
                Ij[:, j] = (b[:, i + j] / np.sqrt(in_z0)) * tau

            # The current of each port is different in the same node
            # The ports' current should take into account the output current of each port in the node
            for j in range(cnx_len):
                in_z0 = z0_segment[:, j]
                out_z0 = 1 / (tot_shunt_z0 - 1 / in_z0)
                Itmp = np.zeros_like(Is[:, i + j])
                for k in range(cnx_len):
                    tmp_z0 = z0_segment[:, k]
                    if j == k:
                        Itmp += Ij[:, k] * (tmp_z0 / out_z0)
                    else:
                        Itmp -= Ij[:, k] * (tmp_z0 / in_z0)

                Is[:, i + j] = Itmp

            i += cnx_len

        return Is


    def voltages(self, power: NumberLike, phase: NumberLike) -> np.ndarray:
        """
        Voltages at internal ports.

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]

        Returns
        -------
        V : complex array of shape (nfreqs, nports)
            Voltages in Volt [V] (peak) at internal ports.

        """
        a = self._a(self._a_external(power, phase))
        b = self._b(a)
        z0s = self.z0
        i, Vs = 0, np.zeros_like(z0s)

        for cnx in self.connections:
            cnx_len = len(cnx)
            z0_segment = z0s[:, i : i + cnx_len]
            tot_shunt_z0 = (1 / z0_segment).sum(axis=1)
            Vk = np.zeros(shape=z0s.shape[0], dtype="complex128")

            # Node voltage is the summation of each ports' outwave voltage
            # The voltage of each port in the same node is consistent
            for j in range(cnx_len):
                in_z0 = z0_segment[:, j]
                out_z0 = 1 / (tot_shunt_z0 - 1 / in_z0)
                tau = (2 * out_z0) / (out_z0 + in_z0)
                Vk += (b[:, i + j] * np.sqrt(in_z0)) * tau

            Vs[:, i : i + cnx_len] = Vk[:, None]
            i += cnx_len

        return Vs

    def currents_external(self, power: NumberLike, phase: NumberLike) -> np.ndarray:
        """
        Currents at external ports.

        NB: current direction is defined positive when "entering" into port.

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]

        Returns
        -------
        I : complex array of shape (nfreqs, nports)
            Currents in Amperes [A] (peak) at external ports.

        """
        a = self._a(self._a_external(power, phase))
        b = self._b(a)
        z0s = self.z0
        Is = []
        for port_idx in self.port_indexes:
            Is.append((a[port_idx] - b[:,port_idx])/np.sqrt(z0s[:,port_idx]))
        return np.array(Is).T

    def voltages_external(self, power: NumberLike, phase: NumberLike) -> np.ndarray:
        """
        Voltages at external ports

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]

        Returns
        -------
        V : complex array of shape (nfreqs, nports)
            Voltages in Volt [V] (peak)  at ports

        """
        a = self._a(self._a_external(power, phase))
        b = self._b(a)
        z0s = self.z0
        Vs = []
        for port_idx in self.port_indexes:
            Vs.append((a[port_idx] + b[:,port_idx])*np.sqrt(z0s[:,port_idx]))
        return np.array(Vs).T

    def plot_graph(self, **kwargs):
        """
        Plot the graph of the circuit using networkx drawing capabilities.

        Customisation options with default values:
        ::
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
            'port_fontsize': 7
            'edges_fontsize': 5
            'network_labels': False
            'edge_labels': False
            'inter_labels': False
            'port_labels': False
            'label_shift_x': 0
            'label_shift_y': 0

        """

        nx = self._get_nx()
        G = self.G

        # default values
        network_labels = kwargs.pop('network_labels', False)
        network_shape = kwargs.pop('network_shape', 's')
        network_color = kwargs.pop('network_color', 'gray')
        network_fontsize = kwargs.pop('network_fontsize', 7)
        network_size = kwargs.pop('network_size', 300)
        inter_labels = kwargs.pop('inter_labels', False)
        inter_shape = kwargs.pop('inter_shape', 'o')
        inter_color = kwargs.pop('inter_color', 'lightblue')
        inter_size = kwargs.pop('inter_size', 300)
        port_labels = kwargs.pop('port_labels', False)
        port_shape = kwargs.pop('port_shape', '>')
        port_color = kwargs.pop('port_color', 'red')
        port_size = kwargs.pop('port_size', 300)
        port_fontsize = kwargs.pop('port_fontsize', 7)
        edge_labels = kwargs.pop('edge_labels', False)
        edge_fontsize = kwargs.pop('edge_fontsize', 5)
        label_shift_x = kwargs.pop('label_shift_x', 0)
        label_shift_y = kwargs.pop('label_shift_y', 0)


        # sort between network nodes and port nodes
        all_ntw_names = [ntw.name for ntw in self.networks_list()]
        port_names = [ntw_name for ntw_name in all_ntw_names if 'port' in ntw_name]
        ntw_names = [ntw_name for ntw_name in all_ntw_names if 'port' not in ntw_name]
        # generate connecting nodes names
        int_names = ['X'+str(k) for k in range(self.connections_nb)]

        fig, ax = subplots(figsize=(10,8))

        pos = nx.spring_layout(G)

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
        if network_labels:
            network_labels = {lab:lab for lab in ntw_names}

            nx.draw_networkx_labels(G, pos_labels, labels=network_labels,
                                    font_size=network_fontsize, ax=ax)

        # intersection labels
        if inter_labels:
            inter_labels = {'X'+str(k):'X'+str(k) for k in range(self.connections_nb)}

            nx.draw_networkx_labels(G, pos_labels, labels=inter_labels,
                                    font_size=network_fontsize, ax=ax)

        # port labels
        if port_labels:
            port_labels = {lab:lab for lab in port_names}

            nx.draw_networkx_labels(G, pos_labels, labels=port_labels,
                                    font_size=port_fontsize, ax=ax)

        # draw edges
        nx.draw_networkx_edges(G, pos, ax=ax)
        if edge_labels:
            edge_labels = self.edge_labels
            nx.draw_networkx_edge_labels(G, pos,
                                        edge_labels=edge_labels, label_pos=0.5,
                                        font_size=edge_fontsize, ax=ax)
        # remove x and y axis and labels
        ax.axis('off')
        fig.tight_layout()

## Functions operating on Circuit
def reduce_circuit(connections: list[list[tuple[Network, int]]],
                   check_duplication: bool = True,
                   split_ground: bool = True,
                   split_multi: bool = False,
                   max_nports: int = 20,
                   dynamic_networks: Sequence[Network] = tuple()) -> list[list[tuple[Network, int]]]:
    """
    Return a reduced equivalent circuit connections with fewer components.

    The reduced equivalent circuit connections allows faster calculation of the circuit network.


    Parameters
    ----------
    connections : list[list[tuple[Network, int]]].
            The connection list to reduce.
    check_duplication : bool, optional.
            If True, check if the connections have duplicate names. Default is True.
    split_ground : bool, optional.
            If True, split the global ground connection to independant ground connections. Default is True.
    split_multi : bool, optional.
            If True, use a splitter to handle connections involving more than two components. This approach
            increases the computational load for individual computations. However, it proves advantageous for
            batch processing by enabling a more comprehensive reduction of circuits, leading to more efficiency
            in batch computations. Default is False.
    max_nports : int, optional.
            The maximum number of ports of a Network that can be reduced in circuit. If a Network in the
            circuit has a number of ports (nports), using the Network.connect() method to reduce the circuit's
            dimensions becomes less efficient compared to directly calculating it with Circuit.s_external.
            This value depends on the performance of the computer and the scale of the circuit. Default is 20.
    dynamic_networks : Sequence[Network], optional.
            A sequence of Networks to ignore in the reduction process. Default is an empty tuple.


    Returns
    -------
    reduced_cnxs : list.
            The reduced connections.


    Examples
    --------
    >>> import skrf as rf
    >>> import numpy as np
    >>> circuit = rf.Circuit(connections)
    >>> reduced_cnxs = rf.reduce_circuit(connections)
    >>> reduced_circuit = rf.Circuit(reduced_cnxs)
    >>> ntwkA = circuit.network
    >>> ntwkB = reduced_circuit.network
    >>> np.allclose(ntwkA.s, ntwkB.s)
    True
    """
    # Pre-processing the dynamic_networks tuple
    ignore_ntwk_names: set[str] = set(ntw.name for ntw in dynamic_networks)

    def invalide_to_reduce(cnx: list[tuple[Network, int]]) -> bool:
        return (
            any(
                (
                    Circuit._is_port(ntwk)
                    or ntwk.nports > max_nports
                    or ntwk.name in ignore_ntwk_names
                )
                for ntwk, _ in cnx
            )
            or len(cnx) != 2
        )

    if split_ground:
        tmp_cnxs = []
        for cnx in connections:
            ground_ntwk = next((ntwk for ntwk, _ in cnx if Circuit._is_ground(ntwk)), None)

            # If there is no ground network or if the connection has exactly 2 elements, append it as is
            if not ground_ntwk or len(cnx) == 2:
                tmp_cnxs.append(cnx)
                continue

            # Otherwise, create new ground connections
            for ntwk, port in cnx:
                if Circuit._is_ground(ntwk):
                    continue
                tmp_gnd = Circuit.Ground(frequency=ground_ntwk.frequency,
                                         name=f'G_{ntwk.name}_{port}')
                tmp_gnd.z0 = ground_ntwk.z0
                tmp_cnxs.append([(ntwk, port), (tmp_gnd, 0)])

        connections = tmp_cnxs

    if split_multi:
        tmp_cnxs = []

        for cnx in connections:
            # Check if the connection has more than 2 components
            if len(cnx) <= 2:
                tmp_cnxs.append(cnx)
                continue

            # Create a splitter for the connection
            _media = media.DefinedGammaZ0(cnx[0][0].frequency)
            splitter = _media.splitter(
                name=f"Splt_{'&'.join(ntwk.name for ntwk, _ in cnx)}",
                nports=len(cnx),
                z0=np.array([ntwk.z0[:, p] for ntwk, p in cnx]).T
            )

            # Connect the splitter to the connection
            for (idx, (ntwk, port)) in enumerate(cnx):
                tmp_cnxs.append([(splitter, idx), (ntwk, port)])

        connections = tmp_cnxs

    # Cache the connections that have been processed to avoid loop
    processed_network_names: set[str] = set()

    # Calculate the total number of Network ports in the specified connection
    def calculate_ports(cnx: list[tuple[Network, int]]) -> int:
        if invalide_to_reduce(cnx):
            return -1

        name_list = sorted(ntwk.name for ntwk, _ in cnx)
        ntwks_str: str = ''.join(name_list)

        unique_networks = len(set(name_list))
        total_ports = sum(ntwk.nports for ntwk, _ in cnx) - 2

        # Return the number of ports if the connections performed
        ports = total_ports if unique_networks == 2 else total_ports // 2 - 1

        # If tuples of Networks in 'connections' have the same Networks, they form a loop.
        # Prioritize processing loops in the circuit by reducing the number of ports by 1.
        # This reduces the computational load during the circuit reduction process.
        if ntwks_str in processed_network_names:
            ports -= 1
        else:
            processed_network_names.add(ntwks_str)

        return ports

    # List of tuples containing connection indices and their calculated ports
    cnx_ports_list = [(idx, calculate_ports(cnx)) for idx, cnx in enumerate(connections)]
    reorder_indices = [idx for idx, _ in sorted(cnx_ports_list, key=lambda x: x[1])]

    # Reorder connections
    connections = [connections[i] for i in reorder_indices]

    # check if the connections are valid
    if check_duplication:
        connections_list = [conn for conn in enumerate(chain.from_iterable(connections))]
        Circuit.check_duplicate_names(connections_list)

    # Use list comprehension to find the connection need to be reduced
    gen = (
        (idx, cnx)
        for idx, cnx in enumerate(connections)
        if not invalide_to_reduce(cnx)
    )

    # Get the first connection need to be reduced
    skip_idx, cnx_to_reduce = next(gen, (-1, [(Network(), -1)] * 2))

    # If there is no connection need to reduce, return the original circuit
    if skip_idx == -1:
        return connections

    # Connect the connections that need to be reduced
    (ntwkA, k), (ntwkB, l) = cnx_to_reduce
    ntwks_name = (ntwkA.name, ntwkB.name)

    # Get the Networks' names
    name_cnt, name_a, name_b = "", str(ntwkA.name), str(ntwkB.name)

    # Generate the connected network and the original port index
    if ntwkA.name == ntwkB.name:
        ntwk_cnt = innerconnect(ntwkA=ntwkA, k=k, l=l)
        name_cnt = f"<{name_a}>"
    else:
        ntwk_cnt = connect(ntwkA=ntwkA, k=k, ntwkB=ntwkB, l=l)
        name_cnt = f"({name_a}**{name_b})"

    # Update the name of the connected network
    ntwk_cnt.name = name_cnt

    # Generate the port index, the index is the original port index
    # and the value is the new port index, -1 means the port is removed.
    port_idx = tuple()
    if ntwkA.name == ntwkB.name:
        port_cnt = list(range(ntwk_cnt.nports))
        port_cnt.insert(min(k, l), -1)
        port_cnt.insert(max(k, l), -1)
        port_idx = (tuple(port_cnt), tuple(port_cnt))
    elif ntwkB.nports == 2 and ntwkA.nports > 2:
        # if ntwkB is a 2port, then keep port indices where you expect.
        port_idx = (
            tuple([(i if i != k else -1) for i in range(ntwkA.nports)]),
            ((-1, k) if l == 0 else (k, -1)),
        )
    else:
        portA = list(range(ntwkA.nports - 1))
        portA.insert(k, -1)
        portB = [i + ntwkA.nports - 1 for i in range(ntwkB.nports - 1)]
        portB.insert(l, -1)

        port_idx = (tuple(portA), tuple(portB))

    # Perform the reduction to get the reduced circuit connections
    # Skip the connection that connected and replace the network and port index
    reduced_cnxs = []
    for idx, cnx in enumerate(connections):
        # Skip the connection reduced
        if idx == skip_idx:
            continue

        tmp_cnx = []
        for ntwk, port in cnx:
            name = ntwk.name
            ntwk_changed = name in ntwks_name

            ntwk_tmp = ntwk
            port_tmp = port

            # Update the connected network and port index
            if ntwk_changed:
                ntwk_tmp = ntwk_cnt
                port_tmp = port_idx[ntwks_name.index(name)][port]

            tmp_cnx.append((ntwk_tmp, port_tmp))

        reduced_cnxs.append(tmp_cnx)

    return reduce_circuit(
        connections=reduced_cnxs,
        check_duplication=False,
        split_ground=False,
        split_multi=False,
        max_nports=max_nports,
        dynamic_networks=dynamic_networks,
    )
