import warnings
from collections import OrderedDict, Iterable

import numpy as np
import skrf
import pyvisa

from . import abcvna
from . import rs_zva_scpi


class ZVA(abcvna.VNA):
    """
    Class for modern Rohde&Schwarz ZVA Vector Network Analyzers
    """

    DEFAULT_VISA_ADDRESS = "GPIB::16::INSTR"
    NAME = "R&S ZVA"
    NPORTS = 4
    NCHANNELS = 32
    SCPI_VERSION_TESTED = 'A.09.20.08'  # taken from PNA class

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        """
        initialization of ZVA Class

        Parameters
        ----------
        address : str
            visa resource string (full string or ip address)
        kwargs : dict
            interface (str), port (int), timeout (int),
        :param address:
        :param kwargs:
        """
        super(ZVA, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi = rs_zva_scpi.SCPI(self.resource)
        # self.use_binary()
        #self.use_ascii()
        print(self.idn)

    def use_binary(self):
        """setup the analyzer to transfer in binary which is faster, especially for large datasets"""
        self.resource.write(':FORM:BORD SWAP')
        self.resource.write(':FORM:DATA REAL,64')
        self.resource.values_format.use_binary(datatype='d', is_big_endian=False, container=np.array)

    def use_ascii(self):
        self.resource.write(':FORM:DATA ASCII')
        self.resource.values_format.use_ascii(converter='f', separator=',', container=np.array)

    @property
    def echo(self):
        return self.scpi.echo

    @echo.setter
    def echo(self, onoff):
        if onoff in (1, True):
            self.scpi.echo = True
        elif onoff in (0, False):
            self.scpi.echo = False
        else:
            raise warnings.warn("echo must be a boolean")

    @property
    def active_channel(self):
        old_timeout = self.resource.timeout
        self.resource.timeout = 500
        try:
            channel = self.scpi.query_active_channel()
        except pyvisa.VisaIOError:
            print("No channel active, using 1")
            channel = 1
        finally:
            self.resource.timeout = old_timeout
        return channel

    @active_channel.setter
    def active_channel(self, channel):
        """
        Set the active channel on the analyzer

        Parameters
        ----------
        channel : int

        Notes
        -----
        There is no specific command to activate a channel, so we ask which channel we want and then activate the first
        trace on that channel.  We do this because if the analyzer gets into a state where it doesn't recognize
        any activated measurements, the get_snp_network method will fail, and possibly others as well.  That is why in
        some methods you will see the fillowing line:
        self.active_channel = channel = kwargs.get("channel", self.active_channel)
        this way we force this property to be set, even if it just resets itself to the same value, but then a trace
        will become active and our get_snp_network method will succeed.
        """
        # TODO: Good chance this will fail if no measurement is on the set channel, need to think about that...
        mnum = self.scpi.query_meas_number_list(channel)[0]
        self.scpi.set_selected_meas_by_number(channel, mnum)
        return

    def sweep(self, **kwargs):
        """
        Initialize a fresh sweep of data on the specified channels

        Parameters
        ----------
        kwargs : dict
            channel ("all", int or list of channels), timeout (milliseconds)

        trigger a fresh sweep on the specified channels (default is "all" if no channel specified)
        Autoset timeout and sweep mode based upon the analyzers current averaging setting,
        and then return to the prior state of continuous trigger or hold.
        """
        #self.resource.clear()
        self.scpi.set_trigger_source("IMM")
        original_timeout = self.resource.timeout

        # expecting either an int or a list of ints for the channel(s)
        channels_to_sweep = kwargs.get("channels", None)
        if not channels_to_sweep:
            channels_to_sweep = kwargs.get("channel", "all")
        if not type(channels_to_sweep) in (list, tuple):
            channels_to_sweep = [channels_to_sweep]
        channels = self.scpi.query_available_channels()

        for i, channel in enumerate(channels):
            sweep_mode = self.scpi.query_sweep_mode(channel)
            was_continuous = "CONT" in sweep_mode.upper()
            sweep_time = self.scpi.query_sweep_time(channel)
            averaging_on = self.scpi.query_averaging_state(channel)
            averaging_mode = self.scpi.query_averaging_mode(channel)

            if averaging_on and "SWE" in averaging_mode.upper():
                sweep_mode = "GROUPS"
                number_of_sweeps = self.scpi.query_averaging_count(channel)
                self.scpi.set_groups_count(channel, number_of_sweeps)
                number_of_sweeps *= self.nports
            else:
                sweep_mode = "SINGLE"
                number_of_sweeps = self.nports
            channels[i] = {
                "cnum": channel,
                "sweep_time": sweep_time,
                "number_of_sweeps": number_of_sweeps,
                "sweep_mode": sweep_mode,
                "was_continuous": was_continuous
            }
            self.scpi.set_sweep_mode(channel, "HOLD")
        timeout = kwargs.get("timeout", None)  # recommend not setting this variable, as autosetting is preferred

        try:
            for channel in channels:
                import time
                if "all" not in channels_to_sweep and channel["cnum"] not in channels_to_sweep:
                    continue  # default for sweep is all, else if we specify, then sweep
                if not timeout:  # autoset timeout based on sweep time
                    sweep_time = channel["sweep_time"] * channel[
                        "number_of_sweeps"] * 1000  # convert to milliseconds, and double for buffer
                    self.resource.timeout = max(sweep_time * 2, 5000)  # give ourselves a minimum 5 seconds for a sweep
                else:
                    self.resource.timeout = timeout
                self.scpi.set_sweep_mode(channel["cnum"], channel["sweep_mode"])
                self.wait_until_finished()
        finally:
            self.resource.clear()
            for channel in channels:
                if channel["was_continuous"]:
                    self.scpi.set_sweep_mode(channel["cnum"], "CONT")
            self.resource.timeout = original_timeout
        return

    def get_frequency(self, **kwargs):
        """
        get an skrf.Frequency object for the current channel

        Parameters
        ----------
        kwargs : dict
            channel (int), f_unit (str)

        Returns
        -------
        skrf.Frequency
        """
        #self.resource.clear()
        channel = kwargs.get("channel", self.active_channel)
        use_log = "LOG" in self.scpi.query_sweep_type(channel).upper()
        f_start = self.scpi.query_f_start(channel)
        f_stop = self.scpi.query_f_stop(channel)
        f_npoints = self.scpi.query_sweep_n_points(channel)
        if use_log:
            freq = np.logspace(np.log10(f_start), np.log10(f_stop), f_npoints)
        else:
            freq = np.linspace(f_start, f_stop, f_npoints)

        frequency = skrf.Frequency.from_f(freq, unit="Hz")
        frequency.unit = kwargs.get("f_unit", "Hz")
        return frequency

    def set_frequency_sweep(self, f_start, f_stop, f_npoints, **kwargs):
        f_unit = kwargs.get("f_unit", "hz").lower()
        if f_unit != "hz":
            f_start = self.to_hz(f_start, f_unit)
            f_stop = self.to_hz(f_stop, f_unit)
        channel = kwargs.get("channel", self.active_channel)
        self.scpi.set_f_start(channel, f_start)
        self.scpi.set_f_stop(channel, f_stop)
        self.scpi.set_sweep_n_points(f_npoints)

    def get_measurement(self, mname=None, mnum=None, **kwargs):
        """
        get a measurement trace from the analyzer, specified by either name or number

        Parameters
        ----------
        mname : str
            the name of the measurement, e.g. 'CH1_S11_1'
        mnum : int
            the number of number of the measurement
        kwargs : dict
            channel (int), sweep (bool)

        Returns
        -------
        skrf.Network
        """
        if mname is None and mnum is None:
            raise ValueError("must provide either a measurement mname or a mnum")

        channel = kwargs.get("channel", self.active_channel)

        if type(mname) is str:
            self.scpi.set_selected_meas(channel, mname)
        else:
            self.scpi.set_selected_meas_by_number(channel, mnum)
        return self.get_active_trace_as_network(**kwargs)

