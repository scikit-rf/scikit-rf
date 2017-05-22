import warnings
from collections import OrderedDict, Iterable

import numpy as np
import skrf
import pyvisa

from . import abcvna
from . import keysight_fieldfox_scpi


class FieldFox(abcvna.VNA):
    """
    Class for Keysight FieldFox Signal Analyzers
    """

    DEFAULT_VISA_ADDRESS = "USB0::0x2A8D::0x5C18::MY56071097::INSTR"
    NAME = "Keysight FieldFox"
    NPORTS = 2
    NCHANNELS = 1
    SCPI_VERSION_TESTED = 'A.06.23'

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        """
        initialization of FieldFox Class

        Parameters
        ----------
        address : str
            visa resource string (full string or ip address)
        kwargs : dict
            interface (str), port (int), timeout (int),
        :param address:
        :param kwargs:
        """
        super(FieldFox, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi = keysight_fieldfox_scpi.SCPI(self.resource)
        self.use_ascii()

    def use_binary(self):
        raise Exception("binary working correctly")
        """setup the analyzer to transfer in binary which is faster, especially for large datasets"""
        self.resource.write(':FORM:BORD SWAP')
        self.resource.write(':FORM:DATA REAL,64')
        # self.resource.write(':FORM:DATA REAL')
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
    def n_traces(self):
        return self.scpi.query_trace_count()

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
        self.resource.clear()
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

    def upload_twoport_calibration(self, cal, port1=1, port2=2, **kwargs):
        """
        upload a calibration to the vna, and set correction on all measurements

        Parameters
        ----------
        cal : skrf.Calibration
        port1: int
        port2: int

        calibration error terms reference
            # forward = (1, 1), reverse = (2, 2)
            "directivity": "EDIR",
            "source match": "ESRM",
            "reflection tracking": "ERFT",

            # forward = (2, 1), reverse = (1, 2)
            "load match": "ELDM",
            "transmission tracking": "ETRT"
            "isolation": "EXTLK"

        """
        self.active_channel = channel = kwargs.get("channel", self.active_channel)

        calname = kwargs.get("calname", "skrf_12term_cal")
        calibrations = self.scpi.query_calset_catalog(cnum=channel, form="NAME")
        if calname in calibrations:
            self.scpi.set_delete_calset(cnum=channel, calset_name=calname)
        self.scpi.set_create_calset(cnum=channel, calset_name=calname)

        cfs = dict()
        for coef, data in cal.coefs_12term.items():
            cfs[coef] = skrf.mf.complex2Scalar(data)

        for eterm, coef in zip(("EDIR", "ESRM", "ERFT"), ("directivity", "source match", "reflection tracking")):
            self.scpi.set_calset_data(channel, eterm, port1, port1, eterm_data=cfs["forward " + coef])
            self.scpi.set_calset_data(channel, eterm, port2, port2, eterm_data=cfs["reverse " + coef])
        for eterm, coef in zip(("ELDM", "ETRT", "EXTLK"), ("load match", "transmission tracking", "isolation")):
            self.scpi.set_calset_data(channel, eterm, port2, port1, eterm_data=cfs["forward " + coef])
            self.scpi.set_calset_data(channel, eterm, port1, port2, eterm_data=cfs["reverse " + coef])

        self.scpi.set_active_calset(1, calname, True)

    def initiate_oneport(self, port=1, **kwargs):
        self.scpi.set_instrument("NA")
        self.scpi.set_trace_display_config("D1")
        if port == 1:
            self.scpi.set_trace_measurement(1, "S11")
        elif port == 2:
            self.scpi.set_trace_measurement(1, "S22")
        else:
            raise Exception("wrong port, must be 1 or 2")
        self.wait_until_finished()
        self.autoscale_all()

    def initiate_twoport(self):
        self.scpi.set_instrument("NA")
        self.scpi.set_trace_display_config("D12_34")
        self.scpi.set_trace_measurement(1, "S11")
        self.scpi.set_trace_measurement(2, "S21")
        self.scpi.set_trace_measurement(3, "S12")
        self.scpi.set_trace_measurement(4, "S22")
        self.wait_until_finished()
        self.autoscale_all()

    def autoscale_all(self):
        for tr in range(1, self.n_traces + 1):
            self.scpi.set_trace_autoscale(tr)

    def get_list_of_traces(self, **kwargs):
        return

        # self.resource.clear()
        # traces = []
        # channels = self.scpi.query_available_channels()
        # for channel in channels:
        #     meas_list = self.scpi.query_meas_name_list(channel)
        #     if len(meas_list) == 1:
        #         continue  # if there isnt a single comma, then there aren't any measurments
        #     parameters = dict([(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)])
        #
        #     meas_numbers = self.scpi.query_meas_number_list()
        #     for mnum in meas_numbers:
        #         name = self.scpi.query_meas_name_from_number(mnum)
        #         item = {"name": name, "channel": channel, "measurement number": mnum,
        #                 "parameter": parameters.get(name, name)}
        #         item["label"] = "{:s} - Chan{:},Meas{:}".format(
        #             item["parameter"], item["channel"], item["measurement number"])
        #         traces.append(item)
        # return traces

    def get_traces(self, traces, **kwargs):
        """
        retrieve traces as 1-port networks from a list returned by get_list_of_traces

        Parameters
        ----------
        traces : list
            list of type that is exported by self.get_list_of_traces
        kwargs : dict
            sweep (bool), name_prefix (str)

        Returns
        -------
        list
            a list of 1-port networks representing the desired traces

        Notes
        -----
        There is no current way to distinguish between traces and 1-port networks within skrf
        """
        self.resource.clear()
        sweep = kwargs.get("sweep", False)

        name_prefix = kwargs.get("name_prefix", "")
        if name_prefix:
            name_prefix += " - "

        channels = OrderedDict()
        for trace in traces:
            ch = trace["channel"]
            if ch not in channels.keys():
                channels[ch] = {
                    "frequency": None,
                    "traces": list()}
            channels[ch]["traces"].append(trace)

        if sweep is True:
            self.sweep(channels=list(channels.keys()))

        traces = []
        for ch, ch_data in channels.items():
            frequency = ch_data["frequency"] = self.get_frequency()
            for trace in ch_data["traces"]:
                self.scpi.set_selected_meas_by_number(trace["channel"], trace["measurement number"])
                sdata = self.scpi.query_data(trace["channel"], "SDATA")
                s = sdata[::2] + 1j * sdata[1::2]
                ntwk = skrf.Network()
                ntwk.s = s
                ntwk.frequency = frequency
                ntwk.name = name_prefix + trace.get("parameter", "trace")
                traces.append(ntwk)
        return traces

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
        self.resource.clear()
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
        self.scpi.set_f_start(f_start)
        self.scpi.set_f_stop(f_stop)
        self.scpi.set_sweep_n_points(f_npoints)

    def get_twoport(self, ports=(1, 2), **kwargs):
        if not self.scpi.query_trace_display_config() == "D12_34":
            raise Exception("display does not show all for SParameters")

    def get_switch_terms(self, **kwargs):
        # TODO: implement switch terms
        pass

        self.resource.clear()
        p1, p2 = 1, 2

        self.active_channel = channel = kwargs.get("channel", self.active_channel)

        measurements = self.get_meas_list()
        max_trace = len(measurements)
        for meas in measurements:  # type: str
            try:
                trace_num = int(meas[0][-2:].replace("_", ""))
                if trace_num > max_trace:
                    max_trace = trace_num
            except ValueError:
                pass

        forward_name = "CH{:}_FS_P{:d}_{:d}".format(channel, p1, max_trace + 1)
        reverse_name = "CH{:}_RS_P{:d}_{:d}".format(channel, p2, max_trace + 2)

        self.create_meas(forward_name, 'a{:}b{:},{:}'.format(p2, p2, p1))
        self.create_meas(reverse_name, 'a{:}b{:},{:}'.format(p1, p1, p2))

        self.sweep(channel=channel)

        forward = self.get_measurement(mname=forward_name, sweep=False)  # type: skrf.Network
        forward.name = "forward switch terms"
        reverse = self.get_measurement(mname=reverse_name, sweep=False)  # type: skrf.Network
        reverse.name = "reverse switch terms"

        self.scpi.set_delete_meas(channel, forward_name)
        self.scpi.set_delete_meas(channel, reverse_name)
        return forward, reverse

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

    def get_active_trace_as_network(self, **kwargs):
        """
        get the active trace as a network object

        Parameters
        ----------
        kwargs : dict
            channel (int), sweep (bool)

        Returns
        -------
        skrf.Network
        """
        channel = self.active_channel
        sweep = kwargs.get("sweep", False)
        if sweep:
            self.sweep(channel=channel)

        ntwk = skrf.Network()
        sdata = self.scpi.query_data(channel)
        ntwk.s = sdata[::2] + 1j * sdata[1::2]
        ntwk.frequency = self.get_frequency(channel=channel)
        return ntwk
