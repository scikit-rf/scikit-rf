from collections import OrderedDict

import numpy as np
import skrf
from . import vna
from . import keysight_pna_scpi


class PNA(vna.VNA):
    DEFAULT_VISA_ADDRESS = "GPIB::16::INSTR"
    NAME = "Keysight PNA"
    NPORTS = 2
    NCHANNELS = 32
    SCPI_VERSION_TESTED = 'A.09.20.08'

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        super(PNA, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi = keysight_pna_scpi.SCPI(self.resource)
        self.use_binary()

        # convenience functions
        self.write = self.resource.write
        self.read = self.resource.read
        self.query = self.resource.query
        self.query_values = self.resource.query_values

    def use_binary(self):
        """setup the analyzer to transfer in binary which is faster, especially for large datasets"""
        self.resource.write(':FORM:BORD SWAP')
        self.resource.write(':FORM:DATA REAL,64')
        self.resource.values_format.use_binary(datatype='d', is_big_endian=False, container=np.array)

    def use_ascii(self):
        self.resource.write(':FORM:DATA ASCII')
        self.resource.values_format.use_ascii(converter='f', separator=',', container=np.array)

    @property
    def idn(self):
        return self.resource.query("*IDN?")

    def wait_until_finished(self):
        self.resource.query("*OPC?")

    @property
    def active_channel(self):
        return self.scpi.query_active_channel()

    @active_channel.setter
    def active_channel(self, channel):
        """
        :param channel: int
        :return:

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
        :param kwargs: channel("all", int or list of channels), timeout(milliseconds)
        :return:

        trigger a fresh sweep on the specified channels ("all" if no channel specified)
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
                number_of_sweeps *= self.nports  # TODO: this is right for 2-port, confirm also for 4-port
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
                self.write("sweep_mode", cnum=channel["cnum"], char=channel["sweep_mode"])
                self.wait_until_finished()
        finally:
            self.resource.clear()
            for channel in channels:
                if channel["was_continuous"]:
                    self.scpi.set_sweep_mode(channel["cnum"], "CONT")
            self.resource.timeout = original_timeout
        return

    def get_snp_network(self, ports, **kwargs):
        """
        :param ports: list of ports
        :param kwargs: channel(int), sweep(bool), name(str), f_unit(str)
        :return: an n-port skrf.Network object

        general function to take in a list of ports and return the full snp network as an skrf.Network for those ports
        """
        self.resource.clear()

        # force activate channel to avoid possible errors:
        self.active_channel = channel = kwargs.get("channel", self.active_channel)

        sweep = kwargs.get("sweep", False)
        name = kwargs.get("name", "")
        f_unit = kwargs.get("f_unit", "GHz")

        ports = [int(port) for port in ports] if type(ports) in (list, tuple) else [int(ports)]
        if not name:
            name = "{:}Port Network".format(len(ports))
        if sweep:
            self.sweep(channel=channel)

        npoints = self.scpi.query_sweep_n_points(channel)
        data = self.scpi.query_snp_data(channel, ports)
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
        """
        :return: list
         the purpose of this function is to query the analyzer for all available traces on all channels
         it then returns a list where each list-item represents one available trace.

         Each list item is a python dict with all necessary information to retrieve the trace as an skrf.Network object:
         list-item keys are:
         * "name": the name of the measurement e.g. "CH1_S11_1"
         * "channel": the channel number the measurement is on
         * "measurement number": the measurement number (MNUM in SCPI)
         * "parameter": the parameter of the measurement, e.g. "S11", "S22" or "a1b1,2"
         * "label": the text the item will use to identify itself to the user e.g "Measurement 1 on Channel 1"
        """
        self.resource.clear()
        traces = []
        channels = self.query("available_channels").split(",")
        for channel in channels:
            meas_list = self.scpi.query_meas_name_list(channel)
            if len(meas_list) == 1:
                continue  # if there isnt a single comma, then there aren't any measurments
            parameters = dict([(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)])

            meas_numbers = self.query("meas_number_list").split(",")
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
        :param traces: list of type that is exported by self.get_list_of_traces
        :param kwargs: sweep(bool), name_prefix(str)
        :return:
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
        """
        :param f_start: start frequency in units = f_unit
        :param f_stop: start frequency in units = f_unit
        :param f_npoints: number of points in the frequency sweep
        :return:
        """
        f_unit = kwargs.get("f_unit", "hz").lower()
        if f_unit != "hz":
            f_start = self.to_hz(f_start, f_unit)
            f_stop = self.to_hz(f_stop, f_unit)
        channel = kwargs.get("channel", self.active_channel)
        self.scpi.set_f_start(channel, f_start)
        self.scpi.set_f_stop(channel, f_stop)
        self.scpi.set_sweep_n_points(f_npoints)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        """
        Get switch terms and return them as a tuple of Network objects.

        Returns
        --------
        forward, reverse : oneport switch term Networks
        """

        self.resource.clear()
        p1, p2 = ports

        self.active_channel = channel = kwargs.get("channel", self.active_channel)

        # TODO: this is not a reliable way to determine the maximum trace number
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

        self.write("delete_meas", cnum=channel, mname=forward_name)
        self.write("delete_meas", cnum=channel, mname=reverse_name)
        self.scpi.set_delete_meas(channel, forward_name)
        self.scpi.set_delete_meas(channel, reverse_name)
        return forward, reverse

    def get_measurement(self, mname=None, mnum=None, **kwargs):
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
        :param kwargs:
        :return:
        """
        channel = self.active_channel
        sweep = kwargs.get("sweep", False)
        if sweep:
            self.sweep(channel=channel)

        ntwk = skrf.Network()
        sdata = self.scpi.query_data(channel, "SDATA")
        ntwk.s = sdata[::2] + 1j * sdata[1::2]
        ntwk.frequency = self.get_frequency(cnum=channel)
        return ntwk

    def create_meas(self, mname, param, **kwargs):
        """
        :param mname: str, name of the measurement  **WARNING**, not all names behave well
        :param param: str, analyzer parameter, e.g.: S11 ; a1/b1,1 ; A/R1,1
        :param kwargs: channel
        :return:
        """
        channel = kwargs.get("channel", self.active_channel)
        self.scpi.set_create_meas(channel, mname, param)
        self.display_trace(mname)

    def display_trace(self, mname, **kwargs):
        """
        :param mname: str
        :param kwargs: channel(int), window_n(int), trace_n(int), display_format(str)
        :return:

        Display a given measurement on specified trace number.

        display_format must be one of: MLINear, MLOGarithmic, PHASe, UPHase, IMAGinary, REAL, POLar, SMITh,
            SADMittance, SWR, GDELay, KELVin, FAHRenheit, CELSius
        """
        channel = kwargs.get('channel', self.active_channel)
        window_n = kwargs.get("window_n", '')
        trace_n = kwargs.get("trace_n", self.ntraces + 1)
        display_format = kwargs.get('display_format', 'MLOG')

        self.scpi.set_display_trace(window_n, trace_n, mname)
        self.scpi.set_selected_meas(channel, mname)
        self.scpi.set_display_format(channel, display_format)

    def get_meas_list(self, **kwargs):
        """
        :param kwargs: channel
        :type kwargs: dict
        :return: list of tuples of the form, (name, measurement)
        :rtype: list

        Return a list of measurement names on all channels.  If channel is provided to kwargs, then only measurements
        for that channel are queried
        """
        channel = kwargs.get("channel", self.active_channel)
        meas_list = self.scpi.query_meas_name_list(channel)
        if len(meas_list) == 1:
            return None  # if there isnt a single comma, then there arent any measurments
        return [(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)]

    @property
    def ntraces(self, **kwargs):
        """
        :param kwargs: channel
        :return: The number of measurement traces that exist on the current channel

        Note that this may not be the same as the number of traces displayed because a measurement may exist,
        but not be associated with a trace.
        """
        channel = kwargs.get("channel", self.active_channel)
        meas_list = self.scpi.query_meas_number_list(channel)
        return 0 if meas_list is None else len(meas_list)


class PNAX(PNA):
    NAME = "Keysight PNA-X"
    NPORTS = 4
    NCHANNELS = 32
