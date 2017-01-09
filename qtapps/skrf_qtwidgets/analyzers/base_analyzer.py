"""
This is a model module.  It will not function correctly to pull data.
"""

class Analyzer(object):
    '''
    class defining an analyzer for using with skrf_qtwidgets.  The base class needs only 6 methods:
    init - setup the instrument resource (i.e., pyvisa)
    measure_twoport_ntwk
    measure_oneport_ntwk
    measure_switch_terms
    enter/exit - for using python's with statement
    >>> with Analyzer("GPIB0::16::ISNTR") as nwa:
    >>>     ntwk = nwa.measure_twoport_ntwk()

    all 6 methods can easily be overwritten to match the idiosynchrasies of someone else's analyzer in lieu of a
    master set of proper python drivers for these.
    '''
    DEFAULT_VISA_ADDRESS = "GPIB0::16::INSTR"
    NAME = "Two Port Analyzer"
    NPORTS = 2
    NCHANNELS = 256

    def __init__(self, visa_address):
        self.resource = None

    def __enter__(self):
        """
        set things up

        :return: the Analyzer Driver Object
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        tear things down

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return: nothing
        """
        self.resource.close()

    def measure_twoport_ntwk(self, ports=(1, 2), channel=None, sweep=False):
        """
        :param ports: an interable of the ports to measure
        :param sweep: whether or not to trigger a fresh measurement
        :return: skrf.Network
        """
        pass

    def measure_oneport_ntwk(self, port=1, channel=None, sweep=False):
        """
        :param ports: which port to measure
        :param sweep: whether or not to trigger a fresh measurement
        :return: skrf.Network
        """
        pass

    def measure_switch_terms(self, ports=(1, 2), channel=None, sweep=False):
        """
        :param ports: an interable of the ports to measure
        :param sweep: whether or not to trigger a fresh measurement
        :return: a lenth-2 iterable of 1-port networks
        """
        pass

    def measure_traces(self, traces=[], channel=None, sweep=False):
        pass

    def get_list_of_traces(self, channel=None):
        pass

    def get_frequency_sweep(self, channel=None):
        pass

    def set_frequency_sweep(self, start_freq, stop_freq, num_points, channel=None):
        pass
