# -*- coding: utf-8 -*-
"""
.. module:: skrf.multiNetworkSystem
========================================
multiNetworkSystem (:mod:`skrf.multiNetworkSystem`)
========================================

Tools for combining multiple networks into a single network that also calculates noise covariance matrices. Each of the individual
multiport networks are added to a :class:`MultiNetworkSystem` object via the :func:`MultiNetworkSystem.add`
method. These networks are then connected together via the :func:`MultiNetworkSystem.connect` method. A new
:class:`.Network` is then created out of the collection of networks by calling :func:`MultiNetworkSystem.reduce`

See scikit-rf examples on MultiNetworkSystems

[I think there is overlap with this and the circuit.py module, needs investigation.]


MultiNetworkSystem Class
========================

.. autosummary::
    :toctree: generated/

    MultiNetworkSystem

Building a Network from Multiple Networks
=========================================

.. autosummary::
    :toctree: generated/

    MultiNetworkSystem.add
    MultiNetworkSystem.connect
    MultiNetworkSystem.external_port
    MultiNetworkSystem.verify
    MultiNetworkSystem.reduce

Extras
============================

.. autosummary::
    :toctree: generated/

    MultiNetworkSystem.num_networks

"""

import numpy as npy
from scipy.linalg import block_diag
from .noisyNetwork import NoisyNetwork
from .networkNoiseCov import NetworkNoiseCov

class MultiNoisyNetworkSystem(object):
    """
    An object for calculating a resulting :class:`.NoisyNetwork` from a set of connected NoisyNetworks. An 
    arbitrary number of networks can be connected together within this object, the function :func:`reduce` is
    called on this object, which returns a new :class:`.NoisyNetwork`. The reduced network contains the correctly 
    calculated covariance matrix object :class:`.NetworksNoiseCov` assembled from the covariance matricies
    of the individual networks.

    For instructions on how to create MultiNoisyNetworkSystem see :func:`__init__`.

    See scikit-rf Examples for several examples on how to use this object.

    Properties:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`num_networks`   The number of networks that have been added

    =====================  =============================================

    Here are the five methods used for constructing a reduced Network from 
    a connected set of Networks.

    =====================  =============================================
    Methods                Meaning
    =====================  =============================================
    :func:`add`            Add a :class:`.NoisyNetwork` to the object
    :func:`connect`        Connect ports between Networks
    :func:`external_port`  Label external ports
    :func:`verify`         Verify that all ports have been connected correctly
    :func:`reduce`         Produce a new :class:`.NoisyNetwork` from the individual networks
    =====================  =============================================

   
    """

    def __init__(self):
        '''
        MultiNoisyNetworkSystem constructor.

        Holds the set of networks to be connected and reduced into a final :class:`.NoisyNetwork`.

        No parameters need to be passed to this constructor.
        '''

        self.ntwk_dict = {}
        self.n_ports_total  = 0
        self.n_freqs = 0

        self.sdiag = None
        self.cdiag = None
        self.gamma = None
        self.re = None
        self.ri = None
        self.s = None
        self.cs = None

        self.con_map = None # used to generate gamma
        self.con_list = None # used to generate gamma
        self.port_ext_idx = None # used to generate re and ri

    @property
    def num_networks(self):
        ''' Returns the number of networks that have been added to :class:`MultiNetworkSystem`
        '''
        return len(self.ntwk_list)

    def add(self, ntwk, name=None):
        """
        Add a :class:`.NoisyNetwork` to the object.

        To add a :class:`.NoisyNetwork` to a :class:`MultiNetworkSystem`, you must provide
        a unique name for the added network. You can do this either by adding a network
        that has a name (i.e., ntwk.name != None), or you can add the name within this 
        add function itself. 

        Parameters
        -----------
        ntwk : :class:`.NoisyNetwork`
            Any multiport network the user would like to combine within a MultiNetworkSystem
        name : optional string
            The unique name for the added network. If left blank, the function will use the name within
            the NoisyNetwork object (i.e., ntwk.name). If no name is provided, the function will raise a 
            ValueError.

        Examples
        -----------
        >>> import skrf as rf
        >>> ntwk = rf.NoisyNetwork('sometouchstonefile.s2p')
        >>> mns = rf.MultiNoisyNetworkSystem()
        >>> mns.add(ntwk, 'splitter1')
      
        """
        if not name:
            if ntwk.name:
                name = ntwk.name
            else:
                raise ValueError("Network must have a name")

        ntwk.name = name
        self.ntwk_dict[name] = {'ntwk': ntwk , 'connections': ntwk.nports * [None]}
        self.n_ports_total += ntwk.nports
        self.n_freqs = len(ntwk)
        self.frequency = ntwk.frequency

    def connect(self, ntwk_name1, port_num1, ntwk_name2, port_num2):
        """
        Connect two Networks together that have been added to the :class:`MultiNoisyNetworkSystem` object.

        Once a set of networks have been added to the MultiNoisyNetworkSystem object, they can be connected
        using this method. The user provides the unique string names (`ntwk_name1`, `ntwk_name2`) of the 
        two networks that are to be connected, as well as the individual port numbers of the associated networks that 
        are to be connected together.

        Parameters
        -----------
        ntwk_name1 : string
            The name of the first NoisyNetwork that is to be connected to the second NoisyNetwork.
        port_num1 : integer
            The port number (enumerated as 1, 2, 3, etc.) associated with the first NoisyNetwork that will be
            connected to a port of the second NoisyNetwork.
        ntwk_name2 : string
            The name of the second NoisyNetwork that is to be connected to the first NoisyNetwork.
        port_num2 : integer
            The port number (enumerated as 1, 2, 3, etc.) associated with the second NoisyNetwork that will be
            connected to a port of the first NoisyNetwork.

        Examples
        -----------
        Create two attenuators and connect port two of the first attenuator to port one of the second

        >>> import skrf as rf
        >>> ntwk1 = rf.NoisyNetwork('attenuator1.s2p')
        >>> ntwk2 = rf.NoisyNetwork('attenuator2.s2p')
        >>> mns = rf.MultiNoisyNetworkSystem()
        >>> mns.add(ntwk, 'attn1')
        >>> mns.add(ntwk, 'attn2')
        >>> mns.connect('attn1', 2, 'attn2', 1)
      
        """

        if ntwk_name1 in self.ntwk_dict and ntwk_name2 in self.ntwk_dict:
            # everything is assumed to be reciprocal 
            self.ntwk_dict[ntwk_name1]['connections'][port_num1 - 1] = (ntwk_name2, port_num2)
            self.ntwk_dict[ntwk_name2]['connections'][port_num2 - 1] = (ntwk_name1, port_num1)
        else:
            raise ValueError("Network name(s) have not been added to system")

    def external_port(self, ntwk_name, port_num, external_port_number):
        """
        Label external ports of the reduced network.

        The output ports of the reduced network are labeled using this method. If the resulting NoisyNetwork
        has three ports, the labels must be 1, 2, and 3 (enumerated as 1, 2, 3, etc.). 

        Parameters
        -----------
        ntwk_name : string
            The name of the NoisyNetwork that has a port exposed to the output of the reduced network.
        port_num : integer
            The port number of the named Nework that will be exposed as an output of the reduced network.
        external_port_number : integer
            The desired external port number of the reduced network. The external port numbers must be 
            enumerated starting with 1 and counting up to the total number of external ports.

        Examples
        -----------
        Create two attenuators and connect port two of the first attenuator to port one of the second. The final
        reduced network will also be a two-port network; therefore, we must label the two output ports as 1 and 2.

        >>> import skrf as rf
        >>> ntwk1 = rf.NoisyNetwork('attenuator1.s2p')
        >>> ntwk2 = rf.NoisyNetwork('attenuator2.s2p')
        >>> mns = rf.MultiNoisyNetworkSystem()
        >>> mns.add(ntwk, 'attn1')
        >>> mns.add(ntwk, 'attn2')
        >>> mns.connect('attn1', 2, 'attn2', 1)
        >>> mns.external_port('attn1', 1, 1)
        >>> mns.external_port('attn2', 2, 2)

        As a second example, we could have labeled the output ports of the previous example in opposite order, i.e.

        >>> mns.connect('attn1', 2, 'attn2', 1)
        >>> mns.external_port('attn1', 1, 2)
        >>> mns.external_port('attn2', 2, 1)

      
        """
        if ntwk_name in self.ntwk_dict:
            self.ntwk_dict[ntwk_name]['connections'][port_num - 1] = ('EXT', external_port_number)

    def verify(self):
        """
        Verifies if all the ports within the network have been connected correctly for the `reduce` method.

        Although it is not necessary to call this method, this method will provide the user with any errors
        they may have made in the connection and port labeling proceedures. 

        The function will return a dictionary of all the errors and warnings that should be addressed before 
        calling `reduce`.

        Returns
        -----------
        output : dictionary
            A dictionary containing errors and warnings about the verification proceedure

        Examples
        -----------
        Before calling `reduce` the method `verify` can be called to determine whether or not any errors have
        been made in the connection and labeling proceedures.

        >>> import skrf as rf
        >>> ntwk1 = rf.NoisyNetwork('attenuator1.s2p')
        >>> ntwk2 = rf.NoisyNetwork('attenuator2.s2p')
        >>> mns = rf.MultiNoisyNetworkSystem()
        >>> mns.add(ntwk, 'attn1')
        >>> mns.add(ntwk, 'attn2')
        >>> mns.connect('attn1', 2, 'attn2', 1)
        >>> mns.external_port('attn1', 1, 1)
        >>> mns.external_port('attn2', 2, 2)
        >>> er = mns.verify()
        >>> print(er)
        """

        errors = {}
        enumb = 0
        exts = []
        lens = []
        first = True
        ln = 0
        same_length = True
        for ntwk in self.ntwk_dict:
            if first:
                first = False
                ln = len(self.ntwk_dict[ntwk]['ntwk'].frequency)
            if ln != len(self.ntwk_dict[ntwk]['ntwk'].frequency):
                enumb += 1
                errors[enumb] = ('Number of frequencies for each network must be the same'  )

            for p, connection in enumerate(self.ntwk_dict[ntwk]['connections']):
                if connection == None:
                    enumb += 1
                    errors[enumb] = ('Network [' + ntwk + '] port [' + str(p) + '] not connected to anything'  )
                elif connection[0] == 'EXT':
                    exts.append(connection)

        num_exts = len(exts)
        ext_expected_ports = list(range(1, num_exts + 1))
        for ext in exts:
            if ext[1] in ext_expected_ports:
                ext_expected_ports.remove(ext[1])
            else:
                enumb += 1
                errors[enumb] = ('External ports must have correct assignments, e.g. 1, 2, 3, ... ')
                break
        
        return enumb, errors

    def reduce(self):
        """
        Calculates the final :class:`.NoisyNetwork` from the interconnected Networks.

        This method will return the final :class:`.NoisyNetwork` that is the result of connecting all the Networks
        that have been added to the :class:`MultiNoisyNetworkSystem` object.

        Returns
        -----------
        ntwk_r : :class:`.NoisyNetwork`
            The reduced network.

        Examples
        -----------
        Connect two attenuators together to produce a final two port. Print the noise figure.

        >>> import skrf as rf
        >>> ntwk1 = rf.NoisyNetwork('attenuator1.s2p')
        >>> ntwk2 = rf.NoisyNetwork('attenuator2.s2p')
        >>> mns = rf.MultiNoisyNetworkSystem()
        >>> mns.add(ntwk, 'attn1')
        >>> mns.add(ntwk, 'attn2')
        >>> mns.connect('attn1', 2, 'attn2', 1)
        >>> mns.external_port('attn1', 1, 1)
        >>> mns.external_port('attn2', 2, 2)
        >>> er = mns.verify()
        >>> print(er)
        >>> ntwk_r = mns.reduce()
        >>> print(ntwk_r.nfmin_db)
        """
        self._create_block_diagonal_matrices_and_gamma()
        self._create_re_ri_matrix()

        gii = npy.matmul(self.ri, npy.matmul(self.gamma, npy.conjugate(self.ri.swapaxes(1, 2))))
        sii = npy.matmul(self.ri, npy.matmul(self.sdiag, npy.conjugate(self.ri.swapaxes(1, 2))))
        sie = npy.matmul(self.ri, npy.matmul(self.sdiag, npy.conjugate(self.re.swapaxes(1, 2))))
        sei = npy.matmul(self.re, npy.matmul(self.sdiag, npy.conjugate(self.ri.swapaxes(1, 2))))
        see = npy.matmul(self.re, npy.matmul(self.sdiag, npy.conjugate(self.re.swapaxes(1, 2))))
        
        w = npy.linalg.inv(gii - sii)
        TGD = npy.matmul(sei, npy.matmul(w, self.ri)) + self.re
        TGD_H = npy.conjugate(TGD.swapaxes(1, 2))

        self.ct = npy.matmul(TGD, npy.matmul(self.cdiag, TGD_H))
        self.st = npy.matmul(sei, npy.matmul(w, sie)) + see
        ntwk = NoisyNetwork(s=self.st, frequency = self.frequency)
        ncov = NetworkNoiseCov(self.ct)
        ntwk.noise_source(ncov)

        return ntwk

    def _create_block_diagonal_matrices_and_gamma(self):

        self.sdiag = npy.zeros((self.n_freqs, self.n_ports_total, self.n_ports_total), dtype=npy.complex)
        self.cdiag = npy.zeros((self.n_freqs, self.n_ports_total, self.n_ports_total), dtype=npy.complex)
        self.gamma = npy.zeros((self.n_freqs, self.n_ports_total, self.n_ports_total), dtype=npy.complex)
        ovec = npy.ones(self.n_freqs)

        self.con_map = {}

        idx = 0
        for p, ntwk in enumerate(self.ntwk_dict):
            nports = self.ntwk_dict[ntwk]['ntwk'].nports
            cs = self.ntwk_dict[ntwk]['ntwk'].cs
            for m in range(nports):
                self.con_map[(ntwk, m + 1)] = idx + m  # part of the process for makeing Gamma 
                for n in range(nports):
                    self.sdiag[:, idx + m, idx + n] = self.ntwk_dict[ntwk]['ntwk'].s[:, m, n]
                    self.cdiag[:, idx + m, idx + n] = cs[:, m, n]

            idx += nports

        # create gamma from the list of connections
        idx = 0
        self.con_list = []
        self.port_ext_idx = []
        self.port_ext_unordered = []
        self.port_ext_reorder= []
        for p, ntwk in enumerate(self.ntwk_dict):
            nports = self.ntwk_dict[ntwk]['ntwk'].nports
            for m in range(nports):
                if self.ntwk_dict[ntwk]['connections'][m][0] != 'EXT':
                    row = self.con_map[self.ntwk_dict[ntwk]['connections'][m]]
                    col = idx + m
                    self.con_list.append((row, col))
                    self.gamma[:, row, col] = ovec
                else:
                    self.port_ext_reorder.append(self.ntwk_dict[ntwk]['connections'][m][1])
                    self.port_ext_unordered.append(idx + m) # used for generating re and ri

            idx += nports

        self.port_ext_idx = [self.port_ext_unordered for _,self.port_ext_unordered in sorted(zip(self.port_ext_reorder,self.port_ext_unordered))]

    def _create_re_ri_matrix(self):

        Identity = npy.identity(self.n_ports_total)

        re_single = npy.take(Identity, self.port_ext_idx, axis=0)

        ri_single = npy.delete(Identity, self.port_ext_idx, axis=0)

        self.re = npy.array([re_single]*self.n_freqs, dtype=npy.complex)

        #print(self.port_ext_idx)
        #print(npy.real(self.re[0,:,:]))

        self.ri = npy.array([ri_single]*self.n_freqs, dtype=npy.complex)




