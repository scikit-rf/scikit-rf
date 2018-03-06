import warnings
from collections import OrderedDict, Iterable

import numpy as np
import skrf
import pyvisa

from . import abcvna
from . import keysight_pna_scpi


class PNA(abcvna.VNA):
    """
    Class for modern Keysight/Agilent Peformance Network Analyzers
    """

    DEFAULT_VISA_ADDRESS = "GPIB::16::INSTR"
    NAME = "Keysight PNA"
    NPORTS = 2
    NCHANNELS = 32
    SCPI_VERSION_TESTED = 'A.09.20.08'

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        """
        initialization of PNA Class

        Parameters
        ----------
        address : str
            visa resource string (full string or ip address)
        kwargs : dict
            interface (str), port (int), timeout (int),
        :param address:
        :param kwargs:
        """
        super(PNA, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi = keysight_pna_scpi.SCPI(self.resource)
        # self.use_binary()
        self.use_ascii()

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

    def get_snp_network(self, ports, **kwargs):
        """
        return n-port network as an Network object

        Parameters
        ----------
        ports : Iterable
            a iterable of integers designating the ports to query
        kwargs : dict
            channel(int), sweep(bool), name(str), f_unit(str), corrected(bool)

        Returns
        -------
        Network

        general function to take in a list of ports and return the full snp network as a Network object
        """
        self.resource.clear()

        # force activate channel to avoid possible errors:
        self.active_channel = channel = kwargs.get("channel", self.active_channel)

        sweep = kwargs.get("sweep", False)
        name = kwargs.get("name", "")
        f_unit = kwargs.get("f_unit", "GHz")
        raw_data = kwargs.get("raw_data", False)

        ports = [int(port) for port in ports] if type(ports) in (list, tuple) else [int(ports)]
        if not name:
            name = "{:}Port Network".format(len(ports))
        if sweep:
            self.sweep(channel=channel)

        npoints = self.scpi.query_sweep_n_points(channel)

        snp_fmt = self.scpi.query_snp_format()
        self.scpi.set_snp_format("RI")
        if raw_data is True:
            if self.scpi.query_channel_correction_state(channel):
                self.scpi.set_channel_correction_state(channel, False)
                data = self.scpi.query_snp_data(channel, ports)
                self.scpi.set_channel_correction_state(channel, True)
            else:
                data = self.scpi.query_snp_data(channel, ports)
        else:
            data = self.scpi.query_snp_data(channel, ports)
        self.scpi.set_snp_format(snp_fmt)  # restore the value before we got the RI data

        nrows = int(len(data) / npoints)
        nports = int(np.sqrt(nrows - 1))
        data = data.reshape([nrows, -1])

        fdata = data[0]
        sdata = data[1:]
        ntwk = skrf.Network()
        ntwk.frequency = skrf.Frequency.from_f(fdata, unit="Hz")
        ntwk.s = np.empty(shape=(sdata.shape[1], nports, nports), dtype=complex)
        for n in range(nports):
            for m in range(nports):
                i = n * nports + m
                ntwk.s[:, m, n] = sdata[i * 2] + 1j * sdata[i * 2 + 1]
        ntwk.frequency.unit = f_unit
        ntwk.name = name
        return ntwk

    def get_list_of_traces(self, **kwargs):
        self.resource.clear()
        traces = []
        channels = self.scpi.query_available_channels()
        for channel in channels:
            meas_list = self.scpi.query_meas_name_list(channel)
            if len(meas_list) == 1:
                continue  # if there isnt a single comma, then there aren't any measurments
            parameters = dict([(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)])

            meas_numbers = self.scpi.query_meas_number_list()
            for mnum in meas_numbers:
                name = self.scpi.query_meas_name_from_number(mnum)
                item = {"name": name, "channel": channel, "measurement number": mnum,
                        "parameter": parameters.get(name, name)}
                item["label"] = "{:s} - Chan{:},Meas{:}".format(
                    item["parameter"], item["channel"], item["measurement number"])
                traces.append(item)
        return traces

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
        sweep_type = self.scpi.query_sweep_type(channel)
        if sweep_type in ["LIN", "LOG", "SEGM"]:
            freqs = self.scpi.query_sweep_data(channel)
        else:
            freqs = np.array([self.scpi.query_f_start(channel)])

        frequency = skrf.Frequency.from_f(freqs, unit="Hz")
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

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        self.resource.clear()
        p1, p2 = ports

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

    def create_meas(self, mname, param, **kwargs):
        """
        Create a new measurement trace on the analyzer

        Parameters
        ----------
        mname: str
            name of the measurement  **WARNING**, not all names behave well
        param: str
            analyzer parameter, e.g.: S11 ; a1/b1,1 ; A/R1,1
        kwargs : dict
            channel(int)
        """
        channel = kwargs.get("channel", self.active_channel)
        self.scpi.set_create_meas(channel, mname, param)
        self.display_trace(mname)

    def display_trace(self, mname, **kwargs):
        """
        Display measurement name on the analyzer display window

        Parameters:
        mname : str
            the name of the measurement, e.g. 'CH1_S11_1'
        kwargs : dict
            channel(int), window_n(int), trace_n(int), display_format(str)

        Keyword Arguments
        ----------------
        display_format : str
            must be one of: MLINear, MLOGarithmic, PHASe, UPHase, IMAGinary, REAL, POLar, SMITh,
                            SADMittance, SWR, GDELay, KELVin, FAHRenheit, CELSius
        """
        channel = kwargs.get('channel', self.active_channel)
        window_n = kwargs.get("window_n", '')
        trace_n = kwargs.get("trace_n",
                             max(self.scpi.query_window_trace_numbers(window_n)) + 1)
        display_format = kwargs.get('display_format', 'MLOG')

        self.scpi.set_display_trace(window_n, trace_n, mname)

        self.scpi.set_selected_meas(channel, mname)
        self.scpi.set_display_format(channel, display_format)

    def get_meas_list(self, **kwargs):
        """
        Convenience function to return a nicely arranged list of the measurement, parameter catalogue

        Parameters
        ----------
        kwargs : dict
            channel : int

        Returns
        -------
        list
            list of tuples of the form: (name, measurement)

        Return a list of measurement names on all channels.
        If channel is provided to kwargs, then only measurements for that channel are queried
        """
        channel = kwargs.get("channel", self.active_channel)
        meas_list = self.scpi.query_meas_name_list(channel)
        if len(meas_list) == 1:
            return None  # if there isnt a single comma, then there arent any measurments
        return [(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)]

    @property
    def ntraces(self):
        """
        Get the number of traces on the active channel

        Returns
        -------
        int
            The number of measurement traces that exist on the current channel

        Notes
        -----
        Note that this may not be the same as the number of traces displayed because a measurement may exist,
        but not be associated with a trace.
        """
        meas_list = self.scpi.query_meas_number_list(self.active_channel)
        return 0 if meas_list is None else len(meas_list)


class PNAX(PNA):
    NAME = "Keysight PNA-X"
    NPORTS = 4
    NCHANNELS = 32
