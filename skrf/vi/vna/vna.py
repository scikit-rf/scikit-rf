"""
This is a model module.  It will not function correctly to pull data.
"""
import pyvisa


class VNA(object):
    '''
    class defining a base analyzer for using with scikit-rf

    ***OPTIONAL METHODS***
    init - setup the instrument resource (i.e., pyvisa)
    get_twoport_ntwk  * must implement get_snp_network
    get_oneport_ntwk  * must implement get_snp_network
    enter/exit - for using python's with statement
    >>> with Analyzer("GPIB::16::ISNTR") as nwa:
    >>>     ntwk = nwa.measure_twoport_ntwk()

    ***METHODS THAT MUST BE IMPLEMENTED***
    get_traces
    get_list_of_traces
    get_snp_network
    get_switch_terms
    set_frequency_sweep

    ***OPTIONAL CLASS PROPERTIES TO OVERRIDE***
    DEFAULT_VISA_ADDRESS
    NAME
    NPORTS
    NCHANNELS

    The init method of this base class is fairly generic and can be used with super, or overwritten completely.
    The same is true for the enter/exit methods
    '''
    DEFAULT_VISA_ADDRESS = "GPIB::16::INSTR"
    NAME = "Two Port Analyzer"
    NPORTS = 2
    NCHANNELS = 32

    FCONVERSION = {"hz": 1, "khz": 1000, "mhz": 1e6, "ghz": 1e9, "thz": 1e12, "phz": 1e15}

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        """
        :param address: a visa resource string
        :param kwargs: visa_library, timeout

        general and recommended way of initializing the visa resource.

        visa_library: pyvisa is a frontend that can use different visa_library backends, including the python-based
        pyvisa-py backend which can handle SOCKET (though not GPIB) connections.  It should be possible to use this
        library without NI-VISA libraries installed if the analyzer is so configured.
        """

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

        self.nports = self.NPORTS  # store an instance variable in case it needs to be modified later

    def __enter__(self):
        """
        :return: the Analyzer Driver Object
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        minimum action of closing the visa resource when we are done with it
        """
        self.resource.close()

    def get_list_of_traces(self, **kwargs):
        """
        :return: list
         the purpose of this function is to query the analyzer for the available traces, and then
         it then returns a list where each list-item represents one available trace.  How this is
         achieved is completely up to the user with the only requirement that the items from this list
         must be passed to the self.get_traces function, which will return one network item for each item
         in the list that is passed.

         Typically the user will get this list of all available traces, and then by some functionality
         in the widgets (or whatever) will down-select the list and then return that down-selected list
         to the get_traces function to retrieve the desired networks.

         Each list item then must be a python object (str, list, dict, etc.) with all necessary information to
         retrieve the trace as an skrf.Network object.  For example, each item could be a python dict with the
         following keys:
         * "name": the name of the measurement e.g. "CH1_S11_1"
         * "channel": the channel number the measurement is on
         * "measurement": the measurement number (MNUM in SCPI)
         * "parameter": the parameter of the measurement, e.g. "S11", "S22" or "a1b1,2"
         * "label": the text the item will use to identify itself to the user e.g "Measurement 1 on Channel 1"
        """
        pass

    def get_traces(self, traces, **kwargs):
        """
        :param traces: list of type that is exported by self.get_list_of_traces
        :param kwargs:
        :return: a list of 1-port networks representing the desired traces

        *** I wish there was some way to distinquish between traces and 1-port networks,
            but with the current structure of skrf, I don't know what that is
        """

    def get_snp_network(self, ports, **kwargs):
        """
        :param ports: list of ports
        :return: an n-port skrf.Network object

        general function to take in a list of ports and return the full snp network as an skrf.Network for those ports
        """
        raise AttributeError("get_snp_network not implemented")

    def get_twoport_ntwk(self, ports=(1, 2), **kwargs):
        """
        :param ports: an interable of the ports to measure
        :return: skrf.Network
        """
        if len(ports) != 2:
            raise ValueError("Must provide a 2-length list of integers for the ports")
        return self.get_snp_network(ports, **kwargs)

    def get_oneport_ntwk(self, port, **kwargs):
        """
        :param port: which port to measure
        :return: skrf.Network
        """
        if type(port) in (list, tuple):
            if len(port) > 1:
                raise ValueError("specify the port as an integer")
        else:
            if type(port) is int:
                port = (port)
            else:
                raise ValueError("specify the port as an integer")

        return self.get_snp_network(port, **kwargs)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        """
        :param ports: port to get S1P from
        :return: a lenth-2 iterable of 1-port networks
        """
        raise AttributeError("get_switch_terms not implemented")

    def set_frequency_sweep(self, start_freq, stop_freq, num_points, channel=None):
        raise AttributeError("set_frequency_sweep not implemented")

    def to_hz(self, freq, f_unit):
        """
        :param freq: float
        :param f_unit: str
        :return:

        A simple convenience function to create frequency in Hz if it is in a different unit
        """
        return freq * self.FCONVERSION[f_unit.lower()]
