"""
.. module:: skrf.io.touchstone

========================================
touchstone (:mod:`skrf.io.touchstone`)
========================================

Touchstone class and utilities

.. autosummary::
   :toctree: generated/

   Touchstone

Functions related to reading/writing touchstones.
-------------------------------------------------

.. autosummary::
   :toctree: generated/

   hfss_touchstone_2_gamma_z0
   hfss_touchstone_2_media
   hfss_touchstone_2_network
   read_zipped_touchstones

"""
from __future__ import annotations

import os
import re
import typing
import warnings
import zipfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from ..constants import FREQ_UNITS, S_DEF_HFSS_DEFAULT, S_DEFINITIONS, SparamFormatT
from ..media import DefinedGammaZ0
from ..network import Network
from ..util import get_fid


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


@dataclass
class ParserState:
    """Class to hold dynamic variables while parsing the touchstone file.
    """
    rank: int | None = None
    option_line_parsed: bool = False
    hfss_gamma: list[list[float]] = field(default_factory=list)
    hfss_impedance: list[list[float]] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    comments_after_option_line: list[str] = field(default_factory=list)
    matrix_format: str = "full"
    parse_network: bool = True
    _parse_noise: bool = False
    f: list[float] = field(default_factory=list)
    f_noise: list[float] = field(default_factory=list)
    s: list[float] = field(default_factory=list)
    noise: list[float] = field(default_factory=list)
    two_port_order_legacy: bool = True
    number_noise_freq: int = 0
    port_names: dict[int, str] = field(default_factory=dict)
    ansys_data_type: str | None = None
    mixed_mode_order: list[str] | None = None
    frequency_unit: str = "ghz"
    parameter: str = "s"
    format: str = "ma"
    resistance: complex = complex(50)

    @property
    def n_ansys_impedance_values(self) -> int:
        """Returns the number of port impedances returned by Ansys HFSS.

        Currently this function returns rank * 2.

        Returns:
            int: number of impedance values.
        """
        # See https://github.com/scikit-rf/scikit-rf/issues/354 for details.
        #if self.ansys_data_type == "terminal":
        #    return self.rank**2 * 2

        return self.rank * 2

    @cached_property
    def numbers_per_line(self) -> int:
        """Returns data points per frequency point.

        Returns:
            int: Number of data points per frequency point.
        """
        if self.matrix_format == "full":
            return self.rank**2 * 2
        return self.rank*(self.rank+1)

    @property
    def parse_noise(self) -> bool:
        """Returns true if the parser expects noise data.

        Returns:
            bool: True, if noise data is expected.
        """
        return self._parse_noise

    @parse_noise.setter
    def parse_noise(self, x: bool) -> None:
        self.parse_network = False
        self._parse_noise = x

    @property
    def frequency_mult(self) -> float:
        _units = {k.lower(): v for k,v in FREQ_UNITS.items()}
        return _units[self.frequency_unit]

    def parse_port(self, line: str):
        """Regex parser for port names.

        Args:
            line (str): Touchstone line.
        """
        m = re.match(r"! Port\[(\d+)\]\s*=\s*(.*)$", line)
        if m:
            self.port_names[int(m.group(1)) - 1] = m.group(2)

    def append_comment(self, line: str) -> None:
        """Append comment, and append appropriate comment list.

        Args:
            line (str): Line to parse
        """
        if self.option_line_parsed:
            self.comments_after_option_line.append(line)
        else:
            self.comments.append(line)

    def parse_option_line(self, line: str) -> None:
        """Parse the option line starting with #

        Args:
            line (str): Line to parse

        Raises:
            ValueError: If option line contains invalid options.

        """
        if self.option_line_parsed:
            return
        toks = line.lower()[1:].strip().split()
        # fill the option line with the missing defaults
        toks.extend(["ghz", "s", "ma", "r", "50"][len(toks) :])
        self.frequency_unit = toks[0]
        self.parameter = toks[1]
        self.format = toks[2]
        self.resistance = complex(toks[4])
        err_msg = ""
        if self.frequency_unit not in ["hz", "khz", "mhz", "ghz"]:
            err_msg = f"ERROR: illegal frequency_unit {self.frequency_unit}\n"
        if self.parameter not in "syzgh":
            err_msg = f"ERROR: illegal parameter value {self.parameter}\n"
        if self.format not in typing.get_args(SparamFormatT):
            err_msg = f"ERROR: illegal format value {self.format}\n"

        if err_msg:
            raise ValueError(err_msg)

        self.option_line_parsed = True


class Touchstone:
    """
    Class to read touchstone s-parameter files.

    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0 [#]_ and
    Touchstone(R) File Format Specification Version 2.0 [#]_

    References
    ----------
    .. [#] https://ibis.org/interconnect_wip/touchstone_spec2_draft.pdf
    .. [#] https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf
    """

    def __init__(self, file: str | Path | typing.TextIO, encoding: str | None = None):
        """
        constructor

        Parameters
        ----------
        file : str, Path, or file-object
            touchstone file to load
        encoding : str, optional
            define the file encoding to use. Default value is None,
            meaning the encoding is guessed (ANSI, UTF-8 or Latin-1).

        Examples
        --------
        From filename

        >>> t = rf.Touchstone('network.s2p')

        File encoding can be specified to help parsing the special characters:

        >>> t = rf.Touchstone('network.s2p', encoding='ISO-8859-1')

        From file-object

        >>> file = open('network.s2p')
        >>> t = rf.Touchstone(file)

        From a io.StringIO object

        >>> link = 'https://raw.githubusercontent.com/scikit-rf/scikit-rf/master/examples/
            basic_touchstone_plotting/horn antenna.s1p'
        >>> r = requests.get(link)
        >>> stringio = io.StringIO(r.text)
        >>> stringio.name = 'horn.s1p'  # must be provided for the Touchstone parser
        >>> horn = rf.Touchstone(stringio)

        """
        ## file format version.
        # Defined by default to 1.0, since version number can be omitted in V1.0 format
        self._version = "1.0"
        ## comments in the file header
        self.comments = ""
        ## unit of the frequency (Hz, kHz, MHz, GHz)
        self.frequency_unit = None
        ## number of frequency points
        self.frequency_nb = None
        ## s-parameter type (S,Y,Z,G,H)
        self.parameter = None
        ## s-parameter format (MA, DB, RI)
        self.format = None
        ## reference resistance, global setup
        self.resistance = None
        ## reference impedance for each s-parameter
        self.reference = None

        ## numpy array of original noise data
        self.noise = None

        ## kind of s-parameter data (s1p, s2p, s3p, s4p)
        self.rank = None
        ## Store port names in a list if they exist in the file
        self.port_names = None

        self.comment_variables = None
        # Does the input file has HFSS per frequency port impedances
        self.has_hfss_port_impedances = False
        self.gamma = None
        self.z0 = None
        self.s_def = None
        self.port_modes = np.array([])

        # open the file depending on encoding
        # Guessing the encoding by trial-and-error, unless specified encoding
        try:
            try:
                if encoding is not None:
                    fid = get_fid(file, encoding=encoding)
                    self.filename = fid.name
                    self.load_file(fid)
                else:
                    # Assume default encoding
                    fid = get_fid(file)
                    self.filename = fid.name
                    self.load_file(fid)
            except Exception as e:
                fid.close()
                raise e

        except UnicodeDecodeError:
            # Unicode fails -> Force Latin-1
            fid = get_fid(file, encoding="ISO-8859-1")
            self.filename = fid.name
            self.load_file(fid)

        except ValueError:
            # Assume Microsoft UTF-8 variant encoding with BOM
            fid = get_fid(file, encoding="utf-8-sig")
            self.filename = fid.name
            self.load_file(fid)

        except Exception as e:
            raise ValueError("Something went wrong by the file opening") from e

        finally:
            fid.close()

    @staticmethod
    def _parse_n_floats(*, line: str, fid: typing.TextIO, n: int, before_comment: bool) -> list[float]:
        """Parse a specified number of floats either in our outside a comment.

        Args:
            line (str): Actual line to parse.
            fid (typing.TextIO): File descriptor to get new lines if necessary.
            n (int): Number of floats to parse.
            before_comment (bool): True, if the floats should get parsed in or outside a comment.

        Returns:
            list[float]: The parsed float values
        """
        def get_part_of_line(line: str) -> str:
            """Return either the part after or before a exclamation mark.

            Args:
                line (str): Line to parse.

            Returns:
                str: string subset.
            """
            if before_comment:
                return line.partition("!")[0]
            else:
                return line.rpartition("!")[2]

        values = get_part_of_line(line).split()
        ret = []
        while True:
            if len(ret) == n:
                break

            if not values:
                values = get_part_of_line(fid.readline()).split()
            try:
                ret.append(float(values.pop(0)))
            except ValueError:
                pass

        return ret

    @property
    def version(self) -> str:
        """The version string.

        Returns:
            str: Version
        """
        return self._version

    @version.setter
    def version(self, x: str) -> None:
        self._version = x
        if x == "2.0":
            self._parse_dict.update(self._parse_dict_v2)

    def _parse_file(self, fid: typing.TextIO) -> ParserState:
        """
        Parse the raw file and generate an structured view.

        Parameters
        ----------
        fid : file object

        Returns
        -------
        state: File content as ParserState

        """
        state = ParserState()

        # Check the filename extension.
        # Should be .sNp for Touchstone format V1.0, and .ts for V2
        extension = self.filename.split(".")[-1].lower()

        m = re.match(r"[ghsyz](\d+)p", extension)
        if m:
            state.rank = int(m.group(1))
        elif extension != "ts":
            msg = (f"{self.filename} does not have a s-parameter extension ({extension})."
                    "Please, correct the extension to of form: 'sNp', where N is any integer for Touchstone v1,"
                    "or ts for Touchstone v2.")
            raise ValueError(msg)

        # Lookup dictionary for parser
        # Dictionary has string keys and values contains functions which
        # need the current line as string argument. The type hints allow
        # the IDE to provide full typing support
        # Take care of the order of the elements when inserting new key words.
        self._parse_dict: dict[str, Callable[[str], None]] = {
            "[version]": lambda x: setattr(self, "version", x.split()[1]),
            "#": lambda x: state.parse_option_line(x),
            "! gamma": lambda x: state.hfss_gamma.append(
                self._parse_n_floats(line=x, fid=fid, n=state.rank * 2, before_comment=False)
            ),
            "! port impedance": lambda x: state.hfss_impedance.append(
                self._parse_n_floats(
                    line=remove_prefix(x.lower(), "! port impedance"),
                    fid=fid,
                    n=state.n_ansys_impedance_values,
                    before_comment=False,
                )
            ),
            "! port": state.parse_port,
            "! terminal data exported": lambda _: setattr(state, "ansys_data_type", "terminal"),
            "! modal data exported": lambda _: setattr(state, "ansys_data_type", "modal"),
            "!": state.append_comment,
        }

        self._parse_dict_v2: dict[str, Callable[[str], None]] = {
            "[number of ports]": lambda x: setattr(state, "rank", int(x.split()[3])),
            "[reference]": lambda x: setattr(
                state, "resistance", self._parse_n_floats(line=x, fid=fid, n=state.rank, before_comment=True)
            ),
            "[number of frequencies]": lambda x: setattr(self, "frequency_nb", int(x.split()[3])),
            "[matrix format]": lambda x: setattr(state, "matrix_format", x.split()[2].lower()),
            "[network data]": lambda _: setattr(state, "parse_network", True),
            "[noise data]": lambda _: setattr(state, "parse_noise", True),
            "[two-port data order]": lambda x: setattr(state, "two_port_order_legacy", "21_12" in x),
            "[number of noise frequencies]": lambda x: setattr(
                state, "number_noise_freq", int(x.partition("]")[2].strip())
            ),
            "[mixed-mode order]": lambda line: setattr(state, "mixed_mode_order", line.lower().split()[2:]),
            "[end]": lambda x: None,
        }

        while True:
            line = fid.readline()
            if not line:
                break

            line_l = line.lower()

            is_data_line = True
            # Avoid traversing the self._parse_dict for each line by checking the first letter
            # {"!", "#", "["} covers all the first letters of the key of the current self._parse_dict
            if line_l[0] in {"!", "#", "["}:
                for k, v in self._parse_dict.items():
                    if line_l.startswith(k):
                        v(line)
                        is_data_line = False
                        break
            if is_data_line:
                if "!" in line:
                    line = line.partition("!")[0]
                values = list(map(float, line.split()))
                if not values:
                    continue

                if (
                    state.f
                    and len(state.s) % state.numbers_per_line == 0
                    and values[0] < state.f[-1]
                    and state.rank == 2
                    and self.version == "1.0"
                ):
                    state.parse_noise = True

                if state.parse_network:
                    if len(state.s) % state.numbers_per_line == 0:
                        state.f.append(values.pop(0))
                    state.s.extend(values)

                elif state.parse_noise:
                    state.noise.append(values)

        return state

    def load_file(self, fid: typing.TextIO):
        """
        Load the touchstone file into the internal data structures.

        Parameters
        ----------
        fid : file object

        """

        state = self._parse_file(fid=fid)

        self.comments = "\n".join([line.strip()[1:] for line in state.comments])
        self.comments_after_option_line = "\n".join([line.strip()[1:] for line in state.comments_after_option_line])
        self.rank = state.rank
        self.frequency_unit = state.frequency_unit
        self.parameter = state.parameter
        self.format = state.format
        self.resistance = state.resistance

        if state.port_names:
            self.port_names = [""] * state.rank
            for k, v in state.port_names.items():
                self.port_names[k] = v

        if state.hfss_gamma:
            self.gamma = np.array(state.hfss_gamma).view(np.complex128)


        # Impedance is parsed in the following order:
        # - HFSS comments for each frequency point and each port.
        # - TS v2 Reference keyword for each port.
        # - Reference impedance from option line.
        if state.hfss_impedance:
            self.z0 = np.array(state.hfss_impedance).view(np.complex128)
            # Comment the line in, when we need when to expect port impedances in NxN format.
            # See https://github.com/scikit-rf/scikit-rf/issues/354 for details.
            #if state.ansys_data_type == "terminal":
            #    self.z0 = np.diagonal(self.z0.reshape(-1, self.rank, self.rank), axis1=1, axis2=2)

            self.s_def = S_DEF_HFSS_DEFAULT
            self.has_hfss_port_impedances = True
            # Load the reference impedance convention from the comments
            for s_def in S_DEFINITIONS:
                if f'S-parameter uses the {s_def} definition' in self.comments:
                    self.s_def = s_def
        elif self.reference is None:
            self.z0 = np.broadcast_to(self.resistance, (len(state.f), state.rank)).copy()
        else:
            self.z0 = np.empty((len(state.f), state.rank), dtype=complex).fill(self.reference)

        self.f = np.array(state.f)
        if not len(self.f):
            self.s = np.empty((0, state.rank, state.rank))
            return

        raw = np.array(state.s).reshape(len(self.f), -1)

        if self.format == "db":
            raw[:, 0::2] = 10 ** (raw[:, 0::2] / 20.0)

        if self.format in (["ma", "db"]):
            s_flat = raw[:, 0::2] * np.exp(1j * raw[:, 1::2] * np.pi / 180)
        elif self.format == "ri":
            s_flat = raw.view(np.complex128)

        self.s_flat = s_flat

        self.s = np.empty((len(self.f), state.rank * state.rank), dtype=complex)
        if state.matrix_format == "full":
            self.s[:] = s_flat
        else:
            index = np.tril_indices(state.rank) if state.matrix_format == "lower" else np.triu_indices(state.rank)
            index_flat = np.ravel_multi_index(index, (state.rank, state.rank))
            self.s[:, index_flat] = s_flat

        if state.rank == 2 and state.two_port_order_legacy:
            self.s = np.transpose(self.s.reshape((-1, state.rank, state.rank)), axes=(0, 2, 1))
        else:
            self.s = self.s.reshape((-1, state.rank, state.rank))

        if state.matrix_format == "upper":
            index_lower = np.tril_indices(state.rank)
            self.s[(...,*index_lower)] = self.s.transpose(0, 2, 1)[(...,*index_lower)]
        elif state.matrix_format == "lower":
            index_upper = np.triu_indices(state.rank)
            self.s[(...,*index_upper)] = self.s.transpose(0, 2, 1)[(...,*index_upper)]

        self.port_modes = np.array(["S"] * state.rank)
        if state.mixed_mode_order:
            new_order = [None] * state.rank
            for i, mm in enumerate(state.mixed_mode_order):
                if mm.startswith("s"):
                    new_order[i] = int(mm[1:]) - 1
                else:
                    p1, p2 = sorted([int(e) - 1 for e in mm[1:].split(",")])

                    if mm.startswith("d"):
                        new_order[i] = p1
                    else:
                        new_order[i] = p2
                self.port_modes[new_order[i]] = mm[0].upper()

            order = np.arange(self.rank, dtype=int)
            self.s[:, new_order, :] = self.s[:, order, :]
            self.s[:, :, new_order] = self.s[:, :, order]
            self.z0[:, self.port_modes == "D"] *= 2
            self.z0[:, self.port_modes == "C"] /= 2

        if self.parameter in ["g", "h", "y", "z"]:
            if self.version == "1.0":
                self.s = self.s * self.z0[:,:, None]

            func_name = f"{self.parameter}2s"
            from .. import network
            self.s: np.ndarray = getattr(network, func_name)(self.s, self.z0)


        # multiplier from the frequency unit
        self.frequency_mult = state.frequency_mult

        if state.noise:
            self.noise = np.array(state.noise)
            self.noise[:, 0] *= self.frequency_mult

        self.f *= self.frequency_mult

    @property
    def sparameters(self) -> np.ndarray:
        """Touchstone data in tabular format.

        Returns:
            np.ndarray: Frequency and data array.
        """
        warnings.warn("This method is deprecated and will be removed.", DeprecationWarning, stacklevel=2)
        return np.hstack((self.f[:, None], self.s_flat.view(np.float64).reshape(len(self.f), -1)))

    def get_comments(self, ignored_comments: list[str]=None) -> str:
        """
        Returns the comments which appear anywhere in the file.

        Comment lines containing ignored comments are removed.
        By default these are comments which contain special meaning withing
        skrf and are not user comments.

        Returns
        -------
        processed_comments : string

        """
        if ignored_comments is None:
            ignored_comments = ["Created with skrf"]
        processed_comments = ""
        if self.comments is None:
            self.comments = ""
        for comment_line in self.comments.split("\n"):
            for ignored_comment in ignored_comments:
                if ignored_comment in comment_line:
                    comment_line = None
            if comment_line:
                processed_comments = processed_comments + comment_line + "\n"
        return processed_comments

    def get_comment_variables(self) -> dict[str, str]:
        """
        Convert hfss variable comments to a dict of vars.

        Returns
        -------
        var_dict : dict (numbers, units)
            Dictionnary containing the comments
        """
        comments = self.comments
        p1 = re.compile(r"\w* = \w*.*")
        p2 = re.compile(r"\s*(\d*\.?\d*)\s*(\w*)")
        var_dict = {}
        for k in re.findall(p1, comments):
            try:
                var, value = k.split("=")
                var = var.rstrip()
                var_dict[var] = p2.match(value).groups()
            except ValueError:
                pass
        return var_dict

    def get_format(self, format: Literal[SparamFormatT, Literal["orig"]]="ri") -> str:
        """
        Returns the file format string used for the given format.

        This is useful to get some information.

        Returns
        -------
        format : string

        """
        if format == "orig":
            frequency = self.frequency_unit
            format = self.format
        else:
            frequency = "hz"
        return f"{frequency} {self.parameter} {format.upper()} r {self.resistance}"

    def get_sparameter_names(self, format: SparamFormatT = "ri") -> list[str]:
        """
        Generate a list of column names for the s-parameter data.
        The names are different for each format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db

        Returns
        -------
        names : list
            list of strings

        """
        warnings.warn("This method is deprecated and will be removed.", DeprecationWarning, stacklevel=2)
        return self.get_sparameter_data(format).keys()

    def get_sparameter_data(self, format: SparamFormatT = "ri") -> dict[str, np.ndarray]:
        """
        Get the data of the s-parameter with the given format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db

        supported formats are:
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitude and angle (degree)

        Returns
        -------
        ret: list
            list of numpy.arrays

        """
        warnings.warn("This method is deprecated and will be removed.", DeprecationWarning, stacklevel=2)
        ret = {"frequency": self.f}
        for j in range(self.rank):
            for k in range(self.rank):
                prefix = f"S{j+1}{k+1}"
                val = self.s[:, j, k]
                if self.rank == 2 and self.filename.split(".")[-1].lower() == "s2p":
                    prefix = f"S{k+1}{j+1}"
                    val = self.s[:, k, j]

                if format == "ri":
                    ret[f"{prefix}R"] = val.real
                    ret[f"{prefix}I"] = val.imag
                elif format == "ma":
                    ret[f"{prefix}M"] = np.abs(val)
                    ret[f"{prefix}A"] = np.angle(val, deg=True)
                elif format == "db":
                    ret[f"{prefix}DB"] = 20 * np.log10(np.abs(val))
                    ret[f"{prefix}A"] = np.angle(val, deg=True)

        return ret

    def get_sparameter_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the s-parameters as a tuple of arrays.

        The first element is the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the s-parameters are complex number.

        Returns
        -------
        param : tuple of arrays

        Examples
        --------
        >>> f, a = self.sgetparameter_arrays()
        >>> s11 = a[:, 0, 0]

        """
        return self.f, self.s

    def get_noise_names(self):
        raise NotImplementedError("not yet implemented")

    def get_noise_data(self):
        # TBD = 1
        # noise_frequency = noise_values[:,0]
        # noise_minimum_figure = noise_values[:,1]
        # noise_source_reflection = noise_values[:,2]
        # noise_source_phase = noise_values[:,3]
        # noise_normalized_resistance = noise_values[:,4]
        raise NotImplementedError("not yet implemented")

    def get_gamma_z0(self):
        """
        Extracts Z0 and Gamma comments from touchstone file (if provided).

        Returns
        -------
        gamma : complex np.ndarray
            complex  propagation constant
        z0 : np.ndarray
            complex port impedance
        """
        return self.gamma, self.z0


def hfss_touchstone_2_gamma_z0(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts Z0 and Gamma comments from touchstone file.

    Takes a HFSS-style Touchstone file with Gamma and Z0 comments and
    extracts a triplet of arrays being: (frequency, Gamma, Z0)

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file


    Returns
    -------
    f : np.ndarray
        frequency vector (in Hz)
    gamma : complex np.ndarray
        complex  propagation constant
    z0 : np.ndarray
        complex port impedance

    Examples
    --------
    >>> f,gamm,z0 = rf.hfss_touchstone_2_gamma_z0('line.s2p')
    """
    ntwk = Network(filename)

    return ntwk.frequency.f, ntwk.gamma, ntwk.z0


def hfss_touchstone_2_media(filename: str) -> list[DefinedGammaZ0]:
    """
    Creates a :class:`~skrf.media.Media` object from a a HFSS-style Touchstone file with Gamma and Z0 comments.

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file

    Returns
    -------
    my_media : :class:`~skrf.media.media.Media` object
        the transmission line model defined by the gamma, and z0
        comments in the HFSS file.

    Examples
    --------
    >>> port1_media, port2_media = rf.hfss_touchstone_2_media('line.s2p')

    See Also
    --------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0
    """
    ntwk = Network(filename)

    freq = ntwk.frequency
    gamma = ntwk.gamma
    z0 = ntwk.z0

    media_list = []

    for port_n in range(gamma.shape[1]):
        media_list.append(DefinedGammaZ0(frequency=freq, gamma=gamma[:, port_n], z0=z0[:, port_n]))

    return media_list


def hfss_touchstone_2_network(filename: str) -> Network:
    """
    Creates a :class:`~skrf.Network` object from a a HFSS-style Touchstone file.

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file

    Returns
    -------
    my_network : :class:`~skrf.network.Network` object
        the n-port network model

    Examples
    --------
    >>> my_network = rf.hfss_touchstone_2_network('DUT.s2p')

    See Also
    --------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0
    """
    my_network = Network(file=filename)
    return my_network


def read_zipped_touchstones(ziparchive: zipfile.ZipFile, dir: str = "") -> dict[str, Network]:
    """
    similar to skrf.io.read_all_networks, which works for directories but only for Touchstones in ziparchives.

    Parameters
    ----------
    ziparchive : :class:`zipfile.ZipFile`
        an zip archive file, containing Touchstone files and open for reading
    dir : str
        the directory of the ziparchive to read networks from, default is "" which reads only the root directory

    Returns
    -------
    dict
        keys are touchstone filenames without extensions
        values are network objects created from the touchstone files
    """
    networks = dict()
    for fname in ziparchive.namelist():  # type: str
        directory = os.path.split(fname)[0]
        if dir == directory and  re.search(r"s\d+p$", fname.lower()):
            network = Network.zipped_touchstone(fname, ziparchive)
            networks[network.name] = network
    return networks
