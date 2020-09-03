import numpy as npy
from scipy.linalg import block_diag
from .network import Network
from .networkNoiseCov import NetworkNoiseCov

class MultiNetworkSystem(object):

    def __init__(self):
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
        return len(self.ntwk_list)

    def add(self, ntwk, name=None):
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
        if ntwk_name1 in self.ntwk_dict and ntwk_name2 in self.ntwk_dict:
            # everything is assumed to be reciprocal 
            self.ntwk_dict[ntwk_name1]['connections'][port_num1 - 1] = (ntwk_name2, port_num2)
            self.ntwk_dict[ntwk_name2]['connections'][port_num2 - 1] = (ntwk_name1, port_num1)
        else:
            raise ValueError("Network name(s) have not been added to system")

    def external_port(self, ntwk_name, port_num, external_port_number=None):
        if ntwk_name in self.ntwk_dict:
            self.ntwk_dict[ntwk_name]['connections'][port_num - 1] = ('EXT', external_port_number)

    def verify(self):
        """Verifies that all of the ports are connected correctly

        Returns:
            [dict]: A dictionary of error messages
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
        """ Where the magic happens """

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
        ntwk = Network(s=self.st, frequency = self.frequency)
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
        for p, ntwk in enumerate(self.ntwk_dict):
            nports = self.ntwk_dict[ntwk]['ntwk'].nports
            for m in range(nports):
                if self.ntwk_dict[ntwk]['connections'][m][0] != 'EXT':
                    row = self.con_map[self.ntwk_dict[ntwk]['connections'][m]]
                    col = idx + m
                    self.con_list.append((row, col))
                    self.gamma[:, row, col] = ovec
                else:
                    self.port_ext_idx.append(idx + m) # used for generating re and ri

            idx += nports

    def _create_re_ri_matrix(self):

        Identity = npy.identity(self.n_ports_total)

        re_single = npy.take(Identity, self.port_ext_idx, axis=0)
        ri_single = npy.delete(Identity, self.port_ext_idx, axis=0)

        self.re = npy.array([re_single]*self.n_freqs, dtype=npy.complex)
        self.ri = npy.array([ri_single]*self.n_freqs, dtype=npy.complex)




