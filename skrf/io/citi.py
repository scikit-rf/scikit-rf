"""
.. module:: skrf.io.citi

========================================
citi (:mod:`skrf.io.citi`)
========================================

Citi class and utilities

.. autosummary::
   :toctree: generated/

   Citi

"""
import numpy as np

import numpy as np
import typing
from ..util import get_fid
from ..frequency import Frequency
from ..network import Network, z2s, y2s
from ..networkSet import NetworkSet
from ..mathFunctions import magdeg_2_reim

class Citi():
    """
    Class to read CITI N-port files.

    CITI file (.cti) is a standardized data format, used for exchanging data
    between different computers and instruments. CITI file is an abbreviation
    for "Common Instrumentation Transfer and Interchange file" [#]_ .


    Parameters
    ----------
    file : str or file-object
        mdif file to load

    References
    ----------
    .. [#] https://na.support.keysight.com/plts/help/WebHelp/FilePrint/CITIfile_Format.htm
    .. [#] Handbook of Microwave Component Measurements: with Advanced VNA Techniques, Joel P. Dunsmore, 2020, Section 6.1.6.1

    Examples
    --------
    From filename

    >>> m = rf.Citi('network.cti')

    From file-object

    >>> file = open('network.cti')
    >>> m = rf.Citi(file)
    """
    def __init__(self, file: typing.Union[str, typing.TextIO]):
        """
        Constructor

        Parameters
        ----------
        file : str or file-object
            mdif file to load
        """
        self._comments = []
        self._name = ''
        self._params = dict()
        self._data = dict()

        with get_fid(file) as fid:
            self.filename = fid.name
            self._parse_citi(fid.readlines())

    @property
    def comments(self) -> list:
        """
        Comments defined in the CITI file.

        Returns
        -------
        list : list of strings

        """
        return self._comments

    @property
    def name(self) -> str:
        """
        Name of the CITI package.

        Returns
        -------
        name : str
            Name of the CITI package

        """
        return self._name

    @property
    def params(self) -> list:
        """
        Named parameters list.

        This list excludes the "FREQ" parameter, which is always defined.

        Returns
        -------
        list : list of string
            List of the named parameters defined in the CITI file

        """
        # does not return the 'freq' parameter
        params = self._params.keys()
        return [param for param in params if param.lower() != 'freq']

    def _parse_citi(self, lines):
        """
        Parse the CITI file.

        Parameters
        ----------
        lines : list of str
            Lines of the file

        Raises
        ------
        NotImplementedError
            If the number formatting is not recognized

        """
        # The CITI file is parsed from top to bottom in a single time.
        # The named parameters and data are parsed in their order of appearance
        # then formatted properly after
        params_list = []
        data_list = []
        order = 0

        while lines:
            line = lines.pop(0)

            if line.strip().startswith(('#','!')):
                self._comments.append(line)

            if line.strip().upper().startswith('NAME'):
                self._name = line.strip()[5:]

            if line.strip().upper().startswith('VAR '):
                # Example:
                # VAR param_name MAG 4
                _, name, formt, occ = line.strip().split(' ')
                # be sure frequency is lowered char
                if name.lower() == 'freq':
                    name = 'freq'
                self._params[name] = {'format': formt, 'occurences': int(occ),
                                      'order': order}
                order += 1
                params_list.append(name)  # to be popped out after

            if line.strip().upper().startswith('DATA'):
                # Example:
                # DATA S[1,1] MAGANGLE
                _, name, formt = line.strip().split(' ')
                self._data[name] = {'format': formt}
                data_list.append(name)  # to be popped out after

            if line.strip().upper().startswith('VAR_LIST_BEGIN'):
                # read the number of occurence lines for a param (FIFO param)
                _param_values = []
                cur_name = params_list.pop(0)
                for idx in range(self._params[cur_name]['occurences']):
                    line = lines.pop(0)  # goes next line
                    # reads the nb of occurences
                    _param_values.append(line.strip())

                self._params[cur_name]['values'] = np.array(_param_values, dtype=float)

            if line.upper().startswith('BEGIN'):
                _data_values = []
                cur_name = data_list.pop(0)
                # data are ordered for each param(s), then for each frequency
                # so number of lines to read is the product of the occurences of each param
                nb_lines = np.prod([self._params[name]['occurences'] for name in self._params.keys()])
                for idx in range(nb_lines):
                    line = lines.pop(0)  # goes next line
                    # Expect:
                    #    val1, val2
                    _data_values.append([el.strip() for el in line.split(',')])

                # convert into complex valued array
                _data = np.array(_data_values, dtype=float)

                if self._data[cur_name]['format'].upper() == 'RI':
                    values = _data[:,0] + 1j*_data[:,1]
                elif self._data[cur_name]['format'].upper() == 'MAGANGLE':
                    values = magdeg_2_reim(_data[:,0], _data[:,1])
                elif self._data[cur_name]['format'].upper() ==  'DBANGLE':
                    values = ((10**(_data[:,0]/20.0)) * np.exp(1j*np.pi/180 * _data[:,1]))
                else:
                    raise NotImplementedError('Not implemented format case')

                self._data[cur_name]['values'] = values

    @property
    def networks(self) -> list:
        """
        Return the list of Networks read from the CITI file.

        Returns
        -------
        list : list of `skrf.network.Network`
            List of Networks described in the CITI file

        """
        networks = []

        # should find the frequency parameter
        if not 'freq' in [it.lower() for it in self._params.keys()]:
            raise ValueError('Frequency points not found')

        freq = Frequency.from_f(self._params['freq']['values'])

        # Network parameters: search for S, then Z or Y
        if any([it.startswith('S') for it in self._data.keys()]):
            ntwkprm = 'S'
        elif any([it.startswith('Z') for it in self._data.keys()]):
            ntwkprm = 'Z'
        else:
            raise NotImplementedError('No network parameter found in this file')

        # deduce the rank of the Network
        rank = int(np.sqrt(len([it for it in self._data.keys() if it.startswith(ntwkprm)])))

        # Network parameter generic array
        p = np.zeros((len(freq), rank, rank), dtype=complex)
        
        # occurences of each parameter and total number of frequency sets
        occurences = [self._params[name]['occurences'] for name in self.params]
        occ = np.prod(occurences)
       
        # create a 2D array of all parameters sets
        if self.params:
            params_sets = np.array(np.meshgrid(*[self._params[name]['values'] 
                                           for name in self.params])).reshape(-1,len(self.params))
        else:
            params_sets = []

        z0 = 50  # TODO extract from PortZ[port]

        # create list of Networks assuming the following ordering:
        # val_param1_f1
        # ...
        # val_param1_fN
        # val_param2_f1
        # ...
        # val_param2_fN
        # etc
        for m in range(rank):
            for n in range(rank):
                # network param (m,n) for all params and all frequencies
                if f'{ntwkprm}[{m+1},{n+1}]' in self._data.keys():
                    pp = self._data[f'{ntwkprm}[{m+1},{n+1}]']['values'].reshape((len(freq), int(occ)))
                else:
                    # special case some CITI files for 1port
                    pp = self._data['S']['values'].reshape((len(freq), int(occ)))
                    ntwkprm = 'S'
                    
                for (idx_set, params_set) in enumerate(params_sets):
                    # network param (m,n) for the current set of params 
                    p[:,m,n] = pp[:,idx_set]
                    # params dict {param1: val1, param2: val, etc}
                    params = dict(zip(self.params, params_set))
                    
                    if ntwkprm == 'S':
                        ntwk = Network(frequency=freq, s=p, params=params, z0=z0)
                    elif ntwkprm == 'Z':
                        ntwk = Network(frequency=freq, s=z2s(p, z0), params=params, z0=z0)
                    else:
                        raise NotImplementedError('Unknown Network Parameter')
                    networks.append(ntwk)
        
        return networks


    def to_networkset(self) -> NetworkSet:
        """
        Convert the CITI file data into a NetworkSet.

        Returns
        -------
        ns : `skrf.networkset.NetworkSet`
    
        """
        return NetworkSet(self.networks)