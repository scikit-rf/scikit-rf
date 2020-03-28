"""
This is a model module.  It will not function correctly to pull data, but needs
to be subclassed.
"""
import copy
import warnings
from typing import Iterable

import numpy as np
import pyvisa

from ...network import Network
from ...frequency import Frequency


class VNA(object):
    """
    class defining a base analyzer for using with scikit-rf

    This class defines the base functionality expected for all Network
    Analyzers. To keep this manageable, it defines only the most commonly used
    applications, which currently are:
    * grabbing data (SNP networks, and traces)
    * grabbing a catalogue of available traces
    * instantiating and grabbing switch terms for calibration
    * setting the frequency parameters (start, stop, number of points, type)

    Attributes
    ----------
    idn
    nports

    Methods
    -------
    get_oneport(ports=(1, 2), **kwargs)
        get an return a 1-port Network object
    get_twoport(ports=1, **kwargs)
        get an return a 2-port Network object
    enter/exit - for using python's with statement
    >>> with Analyzer("GPIB::16::ISNTR") as nwa:
    >>>     ntwk = nwa.measure_twoport_ntwk()

    Methods that must be subclassed
    -------------------------------
    get_traces
    get_list_of_traces
    get_snp_network
    get_switch_terms
    set_frequency_sweep
    """

    def __init__(self, address, **kwargs):
        """
        Initialize a network analyzer object

        Parameters
        ----------
        address : str
            a visa resource string, or an ip address
        kwargs : dict
            visa_library (str), timemout in milliseconds (int), card_number
            (int), interface (str)

        Notes
        -----
        General initialization of a skrf vna object.  Defines the class methods
        that all subclasses should provide, and therefore defines the basic
        functionality, currently focused on grabbing data more than on
        controlling the state of the vna

        Keyword Arguments
        -----------------
        visa_library : str
            allows pyvisa to use different visa_library backends, including the
            python-based pyvisa-py.  backend which can handle SOCKET and Serial
            (though not GPIB) connections.  It should be possible to use this
            library without NI-VISA libraries installed if the analyzer is so
            configured.
        timeout : int
            milliseconds
        interface : str
            one of "SOCKET", "GPIB"
        card_number : int
            for GPIB, default is usually 0
        """

        rm = kwargs.get("resource_manager", None)
        if not rm:
            rm = pyvisa.ResourceManager(visa_library=kwargs.get("visa_library", ""))

        interface = str(kwargs.get("interface", None)).upper()  # GPIB, SOCKET
        if interface == "GPIB":
            board = str(kwargs.get("card_number", "")).upper()
            resource_string = "GPIB{:}::{:}::INSTR".format(board, address)
        elif interface == "SOCKET":
            port = str(kwargs.get("port", 5025))
            resource_string = "TCPIP0::{:}::{:}::SOCKET".format(address, port)
        else:
            resource_string = address
        self.resource = rm.open_resource(resource_string)  # type: pyvisa.resources.messagebased.MessageBasedResource
        self.resource.timeout = kwargs.get("timeout", 3000)

        self.resource.read_termination = "\n"  # most queries are terminated with a newline
        self.resource.write_termination = "\n"
        if "instr" in resource_string.lower():
            self.resource.control_ren(2)

        # convenience pyvisa functions
        self.write = self.resource.write
        self.read = self.resource.read
        self.query = self.resource.query
        self.query_values = self.resource.query_values

    def __enter__(self):
        """
        context manager entry point

        Returns
        -------
        VNA
            the Analyzer Driver Object
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        context manager exit point

        Parameters
        ----------
        exc_type : type
        exc_val : type
        exc_tb : traceback
        """
        self.resource.close()

    def close(self):
        self.__exit__(None, None, None)

    @property
    def idn(self):
        return self.query("*IDN?")

    def reset(self):
        self.write("*RST")

    def wait_until_finished(self):
        self.query("*OPC?")

    def get_list_of_traces(self, **kwargs):
        """
        a catalogue of the available data traces

        Parameters
        ----------
        kwargs : dict
            to allow for optional parameters in the sub-class

        Returns
        -------
        list
            catalogue of the available traces with description and all
            information necessary to index the trace and grab it from the
            analyzer (i.e. channel, name, parameter etc.

        Notes
        -----
        the purpose of this function is to query the analyzer for the available
        traces, and then it then returns a list where each list-item represents
        one available trace.  How this is achieved is completely up to the user
        with the only requirement that the items from this list must be passed
        to the self.get_traces function, which will return one network item for
        each item in the list that is passed.

        Typically the user will get this list of all available traces, and then
        by some functionality in the widgets (or whatever) will down-select the
        list and then return that down-selected list to the get_traces function
        to retrieve the desired networks.

        Each list item then must be a python object (str, list, dict, etc.) with
        all necessary information to retrieve the trace as an Network object.
        For example, each item could be a python dict with the following keys:
        * "name": the name of the measurement e.g. "CH1_S11_1"
        * "channel": the channel number the measurement is on
        * "measurement": the measurement number (MNUM in SCPI)
        * "parameter": the parameter of the measurement, e.g. "S11", "S22" or
          "a1b1,2"
        * "label": the text the item will use to identify itself to the user e.g
          "Measurement 1 on Channel 1"
        """
        raise NotImplementedError("must implement with subclass")

    def get_traces(self, traces, **kwargs):
        """
        retrieve traces as 1-port networks from a list returned by
        get_list_of_traces

        Parameters
        ----------
        traces : list
            list of type that is exported by self.get_list_of_traces
        kwargs : dict
            to provide for optional parameters

        Returns
        -------
        list
            a list of 1-port networks representing the desired traces

        Notes
        -----
        There is no current way to distinguish between traces and 1-port
        networks within skrf
        """
        raise NotImplementedError("must implement with subclass")

    def get_snp_network(self, ports, **kwargs):
        """
        return n-port network as an Network object

        Parameters
        ----------
        ports : Iterable
            a iterable of integers designating the ports to query
        kwargs : dict
            optional arguments for subclassing

        Returns
        -------
        Network

        general function to take in a list of ports and return the full snp
        network as a Network object
        """
        raise NotImplementedError("must implement with subclass")

    def get_twoport(self, ports=(1, 2), **kwargs):
        """
        convenience wrapper for get_snp_network enforcing 2-ports only

        Parameters
        ----------
        ports : Iterable
            a 2-length iterable of integers specifying the ports
        kwargs : dict
            option parameters, i.e. channel, sweep, etc.

        Returns
        -------
        Network
        """
        if len(ports) != 2:
            raise ValueError("Must provide a 2-length list of integers for the ports")
        return self.get_snp_network(ports, **kwargs)

    def get_oneport(self, port=1, **kwargs):
        """
        convenience wrapper for get_snp_network enforcing 1-port only

        Parameters
        ----------
        ports : int
            integer specifying the port number
        kwargs : dict
            option parameters, i.e. channel, sweep, etc.

        Returns
        -------
        Network
        """
        if type(port) in (list, tuple):
            if len(port) > 1:
                raise ValueError("specify the port as an integer")
        else:
            if type(port) is int:
                port = (port,)
            else:
                raise ValueError("specify the port as an integer")

        return self.get_snp_network(port, **kwargs)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        """
        create new traces for the switch terms and return as a 2-length list of
        forward and reverse terms

        Parameters
        ----------
        ports : Iterable
            a 2-length iterable of integers specifying the ports

        kwargs : dict
            optional parameters

        Returns
        -------
        list
            a 2-length list of 1-port networks [forward_switch_terms,
            reverse_switch_terms]
        """
        raise NotImplementedError("must implement with subclass")

    def set_frequency_sweep(self, start_freq, stop_freq, num_points, **kwargs):
        """
        Set the frequency sweep parameters on the specified or active channel

        Parameters
        ----------
        start_freq : float
            the starting frequency in f_units (Hz, GHz, etc.)
        stop_freq : float
            the ending frequency in f_units (Hz, GHz, etc.)
        num_points : int
            the number of points in the frequency sweep
        kwargs : dict
            channel (int), f_units (str), sweep_type (str -> lin, log), other
            optional parameters
        """
        raise NotImplementedError("must implement with subclass")

    @staticmethod
    def to_hz(freq, f_unit):
        """
        A simple convenience function to create frequency in Hz if it is in a
        different unit

        Parameters
        ----------
        freq : float or np.ndarray
            a float or numpy.ndarray of floats of the frequency in f_units
        f_unit : str
            the units of frequency (Hz, kHz, MHz, GHz, THz)

        Returns
        -------
        float or np.ndarray
            the converted frequency sweep in Hz
        """
        return freq * Frequency.multiplier_dict[f_unit.lower()]
