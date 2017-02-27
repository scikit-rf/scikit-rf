"""
This is a model module.  It will not function correctly to pull data.
"""
from skrf.vi.vna.abcvna import VNA


class Analyzer(VNA):
    """
    class defining an analyzer for using with skrf_qtwidgets.

    The base methods for this class are all defined in the skrf.vi.vna.abcvna.VNA class

    ***OPTIONAL METHODS TO OVERRIDE FOR SKRF_QTWIDGETS***
    init - setup the instrument resource (i.e., pyvisa)
    get_twoport  * must implement get_snp_network
    get_oneport  * must implement get_snp_network
    enter/exit - for using python's with statement
    >>> with Analyzer("GPIB::16::ISNTR") as nwa:
    >>>     ntwk = nwa.measure_twoport_ntwk()

    ***METHODS THAT MUST BE IMPLEMENTED FOR SKRF_QTWIDGETS***
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

    The Required methods provide no functionality through this base class and are merely here for reference for those
    wishing to develop drivers for their own instruments.  These are the methods required to operate with the
    skrf_qtwidgets are provide a very basic VNA API in order to use the widgets.  Depending upon the application,
    it may not be necessary to implement each of the above methods.  For example, get_switch_terms is only required
    currently to use the multiline trl application, and even then, only if switch terms are required for the desired
    calibration.

    ***DRIVER TESTING***
    Although this driver template is designed for use with the widgets, it can be used with any program desired.
    An ipython notebook can be found in the driver development folder that provides a template for how to test the
    functionality of the driver.
    """
