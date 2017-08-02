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
        raise Exception("binary not working correctly")
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
        original_timeout = self.resource.timeout
        self.resource.timeout = kwargs.get("timeout", 15000)  # allow at least 10 seconds for every averaging sweep

        was_continuous = self.scpi.query_continuous_sweep()
        self.scpi.set_continuous_sweep(False)
        self.scpi.set_clear_averaging()

        try:
            for i in range(self.scpi.query_averaging_count()):
                self.scpi.query(":INIT;*OPC?")
        finally:
            self.resource.timeout = original_timeout
            self.scpi.set_continuous_sweep(was_continuous)

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
        f_start = self.scpi.query_f_start()
        f_stop = self.scpi.query_f_stop()
        f_npoints = self.scpi.query_sweep_n_points()
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
            self.initiate_twoport()

        port1, port2 = ports

        ntwk = skrf.Network()
        ntwk.frequency = self.get_frequency(f_unit="GHz")
        npoints = len(ntwk.f)
        ntwk.s = np.empty(shape=(npoints, 2, 2), dtype=complex)

        if kwargs.get("sweep", False) is True:
            self.sweep()

        sdata = []
        for i in range(1,5):
            self.scpi.set_current_trace(i)
            sdata.append(self.scpi.query_current_trace_data())

        ntwk.s[:, 0, 0] = sdata[0][::2] + 1j * sdata[0][1::2]
        ntwk.s[:, 1, 0] = sdata[1][::2] + 1j * sdata[1][1::2]
        ntwk.s[:, 0, 1] = sdata[2][::2] + 1j * sdata[2][1::2]
        ntwk.s[:, 1, 1] = sdata[3][::2] + 1j * sdata[3][1::2]

        ntwk.z0 = kwargs.get("z0", 50.)

        if port1 == 1 and port2 == 2:
            return ntwk
        elif port1 == 2 and port2 ==1:
            ntwk.flip()
            return ntwk
        else:
            raise Exception("ports must be a 2-length tuple with either 1, 2 or 2, 1")

    def get_forward_switch_terms(self, **kwargs):
        print("make sure source port is set to port2 manually")

        self.resource.clear()
        trace_config = self.scpi.query_trace_display_config()
        trace_measurements = []
        for i in range(self.scpi.query_trace_count()):
            trace_measurements.append(self.scpi.query_trace_measurement(i + 1))
        self.scpi.set_trace_count(2)
        self.scpi.set_trace_measurement(1, "R1")
        self.scpi.set_trace_measurement(2, "A")

        self.sweep(**kwargs)

        self.scpi.set_current_trace(1)
        R1 = self.get_active_trace_as_network()
        self.scpi.set_current_trace(2)
        A = self.get_active_trace_as_network()

        self.scpi.set_trace_display_config(trace_config)
        for i in range(len(trace_measurements)):
            self.scpi.set_trace_measurement(i + 1, trace_measurements[i])

        return R1 / A

    def get_reverse_switch_terms(self, **kwargs):
        print("make sure source port is set to port1 manually")

        self.resource.clear()
        trace_config = self.scpi.query_trace_display_config()
        trace_measurements = []
        for i in range(self.scpi.query_trace_count()):
            trace_measurements.append(self.scpi.query_trace_measurement(i + 1))
        self.scpi.set_trace_count(2)
        self.scpi.set_trace_measurement(1, "R2")
        self.scpi.set_trace_measurement(2, "B")

        self.sweep(**kwargs)

        self.scpi.set_current_trace(1)
        R2 = self.get_active_trace_as_network()
        self.scpi.set_current_trace(2)
        B = self.get_active_trace_as_network()

        self.scpi.set_trace_display_config(trace_config)
        for i in range(len(trace_measurements)):
            self.scpi.set_trace_measurement(i + 1, trace_measurements[i])

        return R2 / B

    def display_switch_terms(self):
        self.scpi.set_trace_count(4)
        self.scpi.set_trace_measurement(1, "R1")
        self.scpi.set_trace_measurement(2, "R2")
        self.scpi.set_trace_measurement(3, "A")
        self.scpi.set_trace_measurement(4, "B")

    def set_scale_all(self, bottom, top):
        for i in range(1, self.scpi.query_trace_count() + 1):
            self.write("DISP:WIND:TRAC{:d}:Y:TOP {:}".format(i, top))
            self.write("DISP:WIND:TRAC{:d}:Y:BOTT {:}".format(i, bottom))

    def get_switch_terms(self, **kwargs):
        msg = """the Field Fox VNA does not allow the automatic setting of source ports and so one cannot automatically
        measure the switch terms.  Instead, select Measure->More->Advanced to set the Source Port to port2, then use
        FieldFox.get_forward_switch_terms method to collect the forward switch terms.  Then change the source port to
        to port1, then use FieldFox.get_reverse_switch_terms to collect the reverse switch terms"""
        raise Exception(msg)

    def get_trace(self, tnum):
        self.scpi.set_current_trace(tnum)
        return self.get_active_trace_as_network()

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
        sweep = kwargs.get("sweep", False)
        if sweep:
            self.sweep()

        ntwk = skrf.Network()
        sdata = self.scpi.query_current_trace_data()
        ntwk.s = sdata[::2] + 1j * sdata[1::2]
        ntwk.frequency = self.get_frequency()
        return ntwk
