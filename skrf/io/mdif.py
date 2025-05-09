"""
.. module:: skrf.io.mdif

========================================
mdif (:mod:`skrf.io.mdif`)
========================================

Mdif class and utilities

.. autosummary::
   :toctree: generated/

   Mdif

"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TextIO

import numpy as np

from ..frequency import Frequency
from ..mathFunctions import magdeg_2_reim
from ..network import Network, y2s, z2s
from ..networkSet import NetworkSet
from ..util import get_fid


class Mdif:
    """
    Class to read Generalized MDIF N-port files.

    Used to read the Generalized MDIF (GMDIF) file format [#]_ (.mdf extension).
    These files are used store Network parameters which vary with frequency
    and with one or more named parameters.

    Parameters
    ----------
    file : str or file-object
        mdif file to load

    References
    ----------
    .. [#] https://awrcorp.com/download/faq/english/docs/Users_Guide/data_file_formats.html#generalized_mdif

    Examples
    --------
    From filename

    >>> m = rf.Mdif('network.mdf')

    From file-object

    >>> file = open('network.mdf')
    >>> m = rf.Mdif(file)

    List of named parameters defined in the MDIF file

    >>> m.params

    Convert the MDIF to a NetworkSet

    >>> m.to_networkset()

    Using the data as a `NetworkSet` allows you to select a set of Networks
    from their named parameter values or interpolating between Networks.
    See `skrf.networkset.NetworkSet`.

    """

    def __init__(self, file: str | Path | TextIO):
        """
        Constructor

        Parameters
        ----------
        file : str, Path, or file-object
            mdif file to load
        """
        with get_fid(file) as fid:
            self.filename = fid.name
            self._comments = self._parse_comments(fid)
            self._params = []
            self._networks = self._parse_mdif(fid)

    @property
    def params(self) -> list:
        """
        Named parameters list.

        Returns
        -------
        list : list of string
            List of the named parameters defined in the MDIF file

        """
        return self._params

    @property
    def comments(self) -> list:
        """
        Comments defined in MDIF file.

        Returns
        -------
        list : list of string
            MDIF file Comments

        """
        return self._comments

    @property
    def networks(self) -> list:
        """
        List of Networks.

        Returns
        -------
        list : list of :class:`~skrf.network.Network`
            List of the Network stored in the MDIF file.

        """
        return self._networks

    def _parse_comments(self, fid) -> list:
        """
        Parse the comments stored in the MDIF file.

        Parameters
        ----------
        fid : file object

        Returns
        -------
        list : list of strings.

        """
        fid.seek(0)
        comments = []
        for line in fid:
            if line.strip().startswith('!'):
                comments.append(line[1:].strip())

            # Break out after the first network starts, thereby only
            # pulling out the comments for the entire NetworkSet, not
            # everything for each network. That's pulled out later.
            if line.strip().lower().startswith('begin'):
                break

        return comments


    def _parse_data(self, block_data: list) -> Network:
        """
        Parse the block of data corresponding to a set of parameters to a Network.

        Parameters
        ----------
        block_data : list of string
            block of data to parse

        Raises
        ------
        NotImplementedError
            if the data are not recognized

        Returns
        -------
        ntwk : rf.Network
            Network

        """
        kinds = []
        data_lines = []
        comments = []

        # Assumes default unit and format if not found during parsing
        frequency_unit = 'hz'
        parameter = None
        formt = 'ri'
        z0 = 50
        nb_lines_per_freq = 1
        ntwk_name = ''

        # Flag for adding comments to the network
        no_more_comments = False

        # Append tabbed lines to previous line so it reads data block correctly
        cleaned_lines = []
        current_line = block_data[0]
        for line in block_data[1:]:
            if line.strip() == '\n':
                cleaned_lines.append(line)
            elif line.startswith('\t'):
                current_line = current_line.rstrip(' \n')
                stripped_line = line.lstrip('\t').rstrip(' \n')
                current_line += ' '+ stripped_line + ' \n'
            else:
                cleaned_lines.append(current_line)
                current_line = line
        cleaned_lines.append(current_line)

        # Extract and group parameter informations and values
        for line in cleaned_lines:
            # Parse the option line (as in Touchstone)
            if line.startswith('#'):
                # Flag that no more comments are coming as we're now after the option line
                no_more_comments = True

                toks = line[1:].strip().split()
                # fill the option line with the missing defaults
                toks.extend(['ghz', 's', 'ma', 'r', '50'][len(toks):])
                frequency_unit = toks[0].lower()
                parameter = toks[1].lower()
                formt = toks[2].lower()
                z0 = toks[4]
                if frequency_unit not in ['hz', 'khz', 'mhz', 'ghz']:
                    raise NotImplementedError(f'ERROR: illegal frequency_unit {frequency_unit}',  )
                if parameter not in 'syzgh':
                    raise NotImplementedError(f'ERROR: illegal parameter value {parameter}')
                if formt not in ['ma', 'db', 'ri']:
                    raise NotImplementedError(f'ERROR: illegal format value {formt}')

            elif line.strip().startswith('!'):
                if line[1:].startswith(' network name:'):
                    ntwk_name = line.split(':')[-1].strip()

                # Append the comments to the list of comments for that data block
                if no_more_comments is not False:
                    comments.append(line.strip()[1:])

            # Parameter kinds (s11, z21, ...) are described as
            #
            # % kind1 kind2 ...
            # % ...
            # % ... kindN
            elif line.strip().lower().startswith('%'):
                kinds.append(line[1:].split())

            # Data are described as
            #
            # v_kind1 v_kind2 ...
            # ...
            # ... v_kindN
            elif line.strip():  # ignore lines with only whitespace
                data_lines.append(np.array(line.split(), dtype='float'))

        # grouping the data_lines in a single line for each frequency
        nb_lines_per_freq = len(kinds)
        data_lines = [np.concatenate(data_lines[idx*nb_lines_per_freq:idx*nb_lines_per_freq + nb_lines_per_freq])
                      for idx in range(int(len(data_lines)/nb_lines_per_freq))]
        kinds = [item for sublist in kinds for item in sublist]  # flattening
        kinds = [k.lower().replace('(complex)', '') for k in kinds]  # cleaning
        data = np.array(data_lines)
        f = data[:,0]

        # grouping two by two the columns convert data to complex arrays
        if formt == 'ri':
            values = data[:,1::2] + 1j*data[:,2::2]
        elif formt == 'ma':
            values = magdeg_2_reim(data[:,1::2], data[:,2::2])
        elif formt == 'db':
            values = ((10**(data[:,1::2]/20.0)) * np.exp(1j*np.pi/180 * data[:,2::2]))
        else:
            raise NotImplementedError('not implemented case')

        # Extracting s-parameters, depending of the file formatting
        # Nport in MDIF files exported by ADS
        if ('s[1,1]' in kinds):
            # deduce the rank (nb of ports) of the data
            rank = round(np.sqrt(sum('s[' in s for s in kinds)))
            s = np.zeros((len(f), rank, rank), dtype=complex)
            for m in range(rank):
                for n in range(rank):
                    s[:,m,n] = values[:, kinds.index(f's[{m+1},{n+1}]') - 1]

        # Nport as in AWR MDIF file-format description
        elif (parameter == 's') and all(k in kinds for k in ['n11x', 'n11y']):
            rank = round(np.sqrt(sum('n' in s for s in kinds)/2))
            s = values[:,:rank**2].reshape(len(f), rank, rank)
            # if rank is 2 and S21 before S12, swap S21 and S12
            if rank == 2 and (kinds.index('n21x') < kinds.index('n12x')):
                s[:, 1, 0], s[:, 0, 1] =  s[:, 0, 1].copy(), s[:, 1, 0].copy()

        # no S-parameter are found. Maybe Z-param instead?
        elif ('z[1,1]') in kinds:
            # deduce the rank (nb of ports) of the data
            rank = round(np.sqrt(sum('z[' in s for s in kinds)))
            z = np.zeros((len(f), rank, rank), dtype=complex)
            for m in range(rank):
                for n in range(rank):
                    z[:,m,n] = values[:, kinds.index(f'z[{m+1},{n+1}]') - 1]
            s = z2s(z, z0=z0)

        # no S nor Z-parameter are found. Maybe Y-param instead?
        elif ('y[1,1]') in kinds:
            # deduce the rank (nb of ports) of the data
            rank = round(np.sqrt(sum('y[' in s for s in kinds)))
            y = np.zeros((len(f), rank, rank), dtype=complex)
            for m in range(rank):
                for n in range(rank):
                    y[:,m,n] = values[:, kinds.index(f'y[{m+1},{n+1}]') - 1]
            s = y2s(z, z0=z0)

        else:
            raise NotImplementedError('Unrecognized case, probably not implemented')

        # building the Network
        freq = Frequency.from_f(f, unit=frequency_unit)
        ntwk = Network(frequency=freq, s=s, z0=z0, name=ntwk_name, comments="\n".join(comments))

        return ntwk

    def _parse_mdif(self, fid) -> list:
        """
        MDIF parser.

        Parameters
        ----------
        fid : file object

        Returns
        -------
        list: list of Networks
        """
        fid.seek(0)

        block_data = []
        ntwks = []
        params = dict()

        in_data_block = False
        in_noise_block = False

        for line in fid:

            # parse parameters:
            #
            # VAR param_name_1 = A
            # ...
            # VAR param_name_N = Z
            if line.lower().startswith('var'):
                # current parameter
                param_name, param_value = (s.strip() for s in line[3:].split('='))
                # remove the datatype "(blah)" in "varname(blah)" if any
                param_name = param_name.split('(')[0]
                # try to convert the value as a number
                self._params.append(param_name) if param_name not in self.params else self.params
                try:
                    params[param_name] = float(param_value)
                except ValueError:
                    params[param_name] = param_value.replace('"', '')

            # parse numerical data:
            #
            # begin XXX
            # ....
            # end
            if line.lower().startswith('end'):
                if in_data_block:
                    ntwk = self._parse_data(block_data)
                    ntwk.params = params
                    ntwks.append(ntwk)
                    # reset parsed values
                    in_data_block = False
                    params = dict()
                if in_noise_block:
                    in_noise_block = False
                    noise_arr = np.array(
                        [e.split() for e in block_data if not e.startswith(("!", "#", "%"))]
                        ).astype(float)
                    freq, nfmin, gamma_opt_mag, gamma_opt_angle, rn = noise_arr.T
                    nfreq = Frequency.from_f(freq, unit=ntwk.frequency.unit)
                    gamma = gamma_opt_mag * np.exp(1j*np.deg2rad(gamma_opt_angle))
                    ntwk.set_noise_a(nfreq, nfmin, gamma, rn * ntwk.z0[0,0])

                block_data = []

            if in_data_block or in_noise_block:
                block_data.append(line)

            if line.lower().startswith("begin ndata"):
                in_noise_block = True
            elif line.lower().startswith('begin'):
                in_data_block = True

        return ntwks

    def to_networkset(self) -> NetworkSet:
        """
        Return the MDIF data as a NetworkSet.

        Returns
        -------
        ns : :class:`~skrf.networkSet.NetworkSet`

        See Also
        --------
        from_networkset : Write a MDIF file from a NetworkSet.

        """
        ns = NetworkSet(self.networks)
        ns.comments = self.comments
        return ns

    @staticmethod
    def write(ns : NetworkSet,
              filename : str,
              values: dict | None = None,
              data_types: dict | None = None,
              comments: str | None = None,
              *,
              ads_compatible: bool = True,
              **kwargs):
        """
        Write a MDIF file from a NetworkSet.

        Parameters
        ----------
        ns : :class:`~skrf.networkSet.NetworkSet`
            NetworkSet to get the data from.
        filename : string
            Output MDIF file name.
        values : dictionary or None. Default is None.
            The keys of the dictionnary are MDIF variables and its values are
            a list of the parameter values.
            If None, then the values will be set to the NetworkSet names
            and the datatypes will be set to "string".
        data_types: dictionary or None. Default is None.
            The keys are MDIF variables and the value are datatypes
            specified by the following strings: "int", "double", and "string"
        comments: list of strings
            Comments to add to output_file.
            Each list items is a separate comment line
        ads_compatible: bool. Default is True.
            Indicates whether to write the file in a format that
            ADS will read properly.
        **kwargs: dictionary with extra arguments to pass through to the
            underlying write_touchstone() method in the Network class

        See Also
        --------
        io.mdif.Mdif : MDIF file class
        to_networkset : Return the MDIF data as a NetworkSet

        """

        if comments is None:
            comments = []
        if values is None:
            if ns.has_params():
                values = ns.params_values
            else:
                # using Network names as values
                v = list()
                for ntwk in ns:
                    v.append(ntwk.name)

                values = {"name": v}

        if data_types is None:
            if ns.has_params():
                data_types = ns.params_types
            else:
                # using Network names (->string)
                data_types = {"name": "string"}

        # Remove the return_string argument, as it's a required argument for this method
        kwargs.pop('return_string', None)

        # VAR datatypes
        dict_types = dict({"int": "0", "double": "1", "string": "2"})

        # open output_file
        with open(filename, "w") as mdif:

            # write comments
            for c in comments:
                mdif.write(f"! {c}\n")

            nports = ns[0].nports
            is_noisy = ns[0].noisy

            optionstring = Mdif.__create_optionstring(nports)

            for filenumber, ntwk in enumerate(ns):

                mdif.write("!" + "-" * 79 + "\n")

                for p in values:
                    # assign double as the datatype if none is specified
                    if p not in data_types:
                        data_types[p] = "double"

                    var_type = "" if ads_compatible else f"({dict_types[data_types[p]]})"
                    if data_types[p] == "string":
                        var_def_str = f'VAR {p}{var_type} = "{values[p][filenumber]}"'
                    else:
                        var_def_str = f"VAR {p}{var_type} = {values[p][filenumber]}"
                    mdif.write(var_def_str + "\n")

                mdif.write("\nBEGIN ACDATA\n")
                mdif.write(optionstring + "\n")
                mdif.write("! network name: " + ntwk.name + "\n")

                data = ntwk.write_touchstone(return_string=True, write_noise=False, **kwargs)
                mdif.write(data)

                if is_noisy:
                    # this "END" terminates "ACDATA" (s-parameters) and begins noise ("NDATA")
                    mdif.write("END\n\nBEGIN NDATA\n")
                    mdif.write("%F nfmin n11x n11y rn\n")
                    mdif.write(f"# {ntwk.frequency.unit} S MA R {ntwk.z0[0, 0].real}\n")
                    ntwk._write_noisedata(output=mdif)

                mdif.write("END\n\n")

    def __eq__(self, other) -> bool:
        """
        Test if two Mdif objects are equals.

        Test the equality between the NetworkSet under the hood.

        Parameters
        ----------
        other : :class:`~skrf.io.mdif.Mdif` object
            Mdif object to compare with.

        Returns
        -------
        bool

        """
        return self.to_networkset() == other.to_networkset()

    @staticmethod
    def __create_optionstring(nports):
        """create the options string based on the number of ports. Used in the Touchstone and MDIF formats"""

        if nports > 9:
            corestring = "n{}_{}x n{}_{}y "
        else:
            corestring = "n{}{}x n{}{}y "

        optionstring = "% F "

        if nports == 2:
            optionstring += "n11x n11y n21x n21y n12x n12y n22x n22y"

        else:
            # parse the option string for nports not equal to 2

            for i in product(list(range(1, nports + 1)), list(range(1, nports + 1))):

                optionstring += corestring.format(i[0], i[1], i[0], i[1])

                # special case for nports = 3
                if nports == 3:
                    # 3 ports
                    if not (np.remainder(i[1], 3)):
                        optionstring += "\n\t"

                # touchstone spec allows only 4 data pairs per line
                if nports >= 4:
                    if np.remainder(i[1], 4) == 0:
                        optionstring += "\n\t"
                    # NOTE: not sure if this is needed. Doesn't seem to be required by Microwave Office
                    if i[1] == nports:
                       optionstring += "\n\t"

        return optionstring
