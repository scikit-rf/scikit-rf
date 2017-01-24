from collections import OrderedDict
from copy import deepcopy

import numpy as np
import skrf
from .. import base_analyzer

from .import scpi_preprocessor

scpi_commands = {
    # "key": "SCPI command template"
    "data":                 ":CALC<cnum>:DATA <char>,<data>",
    "create_meas":          ":CALC<cnum>:PAR:DEF:EXT <'mname'>,<'param'>",
    "delete_meas":          ":CALC<cnum>:PAR:DEL <'mname'>",
    "select_meas_by_name":  ":CALC<cnum>:PAR:SEL <'mname'>",
    "select_meas_by_number": ":CALC<cnum>:PAR:MNUM:SEL <mnum>",
    "meas_format":          ":CALC<cnum>:FORM <char>",
    "meas_name_list":       ":CALC<cnum>:PAR:CAT:EXT",
    "snp_data":             ":CALC<cnum>:DATA:SNP:PORT <'ports'>",
    "display_trace":        ":DISP:WIND<wnum>:TRAC<tnum>:FEED <'mname'>",
    "averaging_count":      ":SENS<cnum>:AVER:COUN <num>",
    "averaging_state":      ":SENS<cnum>:AVER:STAT <onoff>",
    "averaging_mode":       ":SENS<cnum>:AVER:MODE <char>",
    "start_frequency":      ":SENS<cnum>:FREQ:STAR <num>",
    "stop_frequency":       ":SENS<cnum>:FREQ:STOP <num>",
    "groups_count":         ":SENS<cnum>:SWE:GRO:COUNt <num>",
    "sweep_mode":           ":SENS<cnum>:SWE:MODE <char>",
    "sweep_time":           ":SENS<cnum>:SWE:TIME <num>",
    "sweep_type":           ":SENS<cnum>:SWE:TYPE <char>",
    "sweep_points":         ":SENSE<cnum>:SWE:POIN <num>",
    "available_channels":   ":SYST:CHAN:CAT",
    "active_channel":       ":SYST:ACT:CHAN",
    "meas_name_from_number": ":SYST:MEAS<mnum>:NAME",
    "meas_number_list":     ":SYST:MEAS:CAT <cnum>",
    "trigger_source":       ":TRIG:SOUR <char>",
}


class Analyzer(base_analyzer.Analyzer):
    DEFAULT_VISA_ADDRESS = "GPIB0::16::INSTR"
    NAME = "Keysight PNA"
    NPORTS = 2
    NCHANNELS = 32
    SCPI_VERSION_TESTED = 'A.09.20.08'

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        super(Analyzer, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi_commands = deepcopy(scpi_commands)  # child classes can overwrite or add scpi commands as needed
        self.use_binary()

    def write(self, command, **kwargs):
        scpi_str = " ".join(scpi_preprocessor.preprocess(self.scpi_commands, command, **kwargs)).strip()
        self.resource.write(scpi_str)

    def read(self):
        return self.resource.read().replace('"', '')  # string returns often have extraneous double quotes

    def query(self, command, **kwargs):
        scpi_str = "? ".join(scpi_preprocessor.preprocess(self.scpi_commands, command, **kwargs)).strip()
        return self.resource.query(scpi_str).replace('"', '')

    def query_values(self, command, **kwargs):
        scpi_str = "? ".join(scpi_preprocessor.preprocess(self.scpi_commands, command, **kwargs)).strip()
        return self.resource.query_values(scpi_str)

    def scpi_string(self, command, query=True, **kwargs):
        """for debugging purposes, to see what SCPI string we are actually making"""
        cmd, arg = scpi_preprocessor.preprocess(self.scpi_commands, command, **kwargs)
        joiner = "? " if query is True else " "
        return joiner.join((cmd, arg)).strip()

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
        return int(self.query("active_channel"))

    @active_channel.setter
    def active_channel(self, channel):
        """
        :param channel: int
        :return:

        There is no specific command to activate a channel, so we ask which channel we want and then activate the first
        trace on that channel.  We do this because if the analyzer gets into a state where it doesn't recognize
        any activated measurements, the get_snp_network method will fail, and possibly others as well.  That is why in
        some methods you will see the fillowing line:
        >>> self.active_channel = channel = kwargs.get("channel", self.active_channel)
        this way we force this property to be set, even if it just resets itself to the same value, but then a trace
        will become active and our get_snp_network method will succeed.
        """
        # TODO: Good chance this will fail if no measurement is on the set channel, need to think about that...

        mnum = self.query("meas_number_list", cnum=channel).split(",")[0]
        self.write("select_meas_by_number", cnum=channel, mnum=mnum)
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
        self.write("trigger_source", char="IMMediate")
        original_timeout = self.resource.timeout

        channels_to_sweep = kwargs.get("channels", None)
        if not channels_to_sweep:
            channels_to_sweep = kwargs.get("channel", "all")
        if not type(channels_to_sweep) in (list, tuple):
            channels_to_sweep = [channels_to_sweep]
        channels_to_sweep = list(map(str, channels_to_sweep))

        channels = self.query("available_channels").split(",")
        for i, channel in enumerate(channels):
            sweep_mode = self.query("sweep_mode", cnum=channel)
            was_continuous = "CONT" in sweep_mode.upper()
            sweep_time = float(self.query("sweep_time", cnum=channel))
            averaging_on = int(self.query("averaging_state", cnum=channel))
            averaging_type = self.query("averaging_mode", cnum=channel)

            if averaging_on and "SWE" in averaging_type.upper():
                sweep_mode = "GROUPS"
                sweeps = int(self.query("averaging_count"))
                self.write("groups_count", cnum=channel, num=sweeps)
                sweeps *= self.nports  # TODO: this is right for 2-port, confirm also for 4-port
            else:
                sweep_mode = "SINGLE"
                sweeps = self.nports
            channels[i] = {
                "cnum": channel,
                "sweep_time": sweep_time,
                "sweeps": sweeps,
                "sweep_mode": sweep_mode,
                "was_continuous": was_continuous
            }
            self.write("sweep_mode", cnum=channel, char="HOLD")

        timeout = kwargs.get("timeout", None)  # recommend not setting this variable, as autosetting is preferred

        try:
            for channel in channels:
                import time
                if "all" not in channels_to_sweep and channel["cnum"] not in channels_to_sweep:
                    continue  # default for sweep is all, else if we specify, then sweep
                if not timeout:  # autoset timeout based on sweep time
                    sweep_time = channel["sweep_time"] * channel[
                        "sweeps"] * 1000  # convert to milliseconds, and double for buffer
                    self.resource.timeout = max(sweep_time * 2, 5000)  # give ourselves a minimum 5 seconds for a sweep
                else:
                    self.resource.timeout = timeout
                self.write("sweep_mode", cnum=channel["cnum"], char=channel["sweep_mode"])
                self.wait_until_finished()
        finally:
            self.resource.clear()
            for channel in channels:
                if channel["was_continuous"]:
                    self.write("sweep_mode", cnum=channel["cnum"], char="continuous")
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

        npoints = int(self.query("sweep_points", cnum=channel))
        data = self.query_values("snp_data", cnum=channel, ports=ports)
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
            meas_list = self.query("meas_name_list", cnum=channel).split(",")
            if len(meas_list) == 1:
                continue  # if there isnt a single comma, then there aren't any measurments
            parameters = dict([(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)])

            meas_numbers = self.query("meas_number_list").split(",")
            for mnum in meas_numbers:
                name = self.query("meas_name_from_number", mnum=mnum)
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
                self.write("select_meas_by_number", cnum=trace["channel"], mnum=trace["measurement number"])
                sdata = self.query_values("data", cnum=trace["channel"], char="SDATA")
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
        use_log = "LOG" in self.query("sweep_type", cnum=channel).upper()
        f_start = float(self.query("start_frequency", cnum=channel))
        f_stop = float(self.query("stop_frequency", cnum=channel))
        f_npoints = int(self.query("sweep_points"))
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
        :param f_unit: the frequnecy unit of the provided f_start and f_stop, default is Hz
        :param channel: channel of the analyzer
        :return:
        """
        f_unit = kwargs.get("f_unit", "hz").lower()
        if f_unit != "hz":
            f_start = self.to_hz(f_start, f_unit)
            f_stop = self.to_hz(f_stop, f_unit)
        channel = kwargs.get("channel", self.active_channel)
        self.write('start_frequency', cnum=channel, num=f_start)
        self.write('stop_frequency', cnum=channel, num=f_stop)
        self.write('sweep_points', cnum=channel, num=f_npoints)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        '''
        Get switch terms and return them as a tuple of Network objects.

        Returns
        --------
        forward, reverse : oneport switch term Networks
        '''

        # TODO: This tends to error if there are multiple channels operating, figure out how to fix
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

        self.write("delete_meas", cnum=channel, mname=forward_name)
        self.write("delete_meas", cnum=channel, mname=reverse_name)
        return forward, reverse

    def get_measurement(self, mname=None, number=None, **kwargs):
        if mname is None and number is None:
            raise ValueError("must provide either a measurement mname or a number")

        channel = kwargs.get("channel", self.active_channel)

        if type(mname) is str:
            self.write("select_meas_by_name", cnum=channel, mname=mname)
        else:
            self.write("select_meas_by_number", cnum=channel, mnum=number)
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
        sdata = self.query_values("data", cnum=channel, char="SDATA")
        ntwk.s = sdata[::2] + 1j * sdata[1::2]
        ntwk.frequency = self.get_frequency(cnum=channel)
        return ntwk

    def create_meas(self, name, param, **kwargs):
        '''
        :param name: str, name of the measurement  **WARNING**, not all names behave well
        :param param: str, analyzer parameter, e.g.: S11 ; a1/b1,1 ; A/R1,1
        :param kwargs: channel
        :return:
        '''
        channel = kwargs.get("channel", self.active_channel)
        self.write("create_meas", cnum=channel, mname=name, param=param)
        self.display_trace(name)

    def display_trace(self, name, **kwargs):
        '''
        :param name: str
        :param kwargs: channel(int), window_n(int), trace_n(int), display_format(str)
        :return:

        Display a given measurement on specified trace number.

        display_format must be one of: MLINear, MLOGarithmic, PHASe, UPHase, IMAGinary, REAL, POLar, SMITh,
            SADMittance, SWR, GDELay, KELVin, FAHRenheit, CELSius
        '''
        channel = kwargs.get('channel', self.active_channel)
        window_n = kwargs.get("window_n", '')
        trace_n = kwargs.get("trace_n", self.ntraces + 1)
        display_format = kwargs.get('display_format', 'MLOG')

        self.write('display_trace', wnum=window_n, tnum=trace_n, mname=name)
        self.write("select_meas_by_name", cnum=channel, mname=name)
        self.write("meas_format", cnum=channel, char=display_format)

    def get_meas_list(self, **kwargs):
        '''
        :param kwargs: channel
        :return: list of tuples of the form, (name, measurement)

        Return a list of measurement names on all channels.  If channel is provided to kwargs, then only measurements
        for that channel are queried
        '''
        channel = kwargs.get("channel", self.active_channel)
        meas_list = self.query("meas_name_list", cnum=channel).split(",")
        if len(meas_list) == 1:
            return None  # if there isnt a single comma, then there arent any measurments
        return [(meas_list[k], meas_list[k + 1]) for k in range(0, len(meas_list) - 1, 2)]

    @property
    def ntraces(self, **kwargs):
        '''
        :param kwargs: channel
        :return: The number of measurement traces that exist on the current channel

        Note that this may not be the same as the number of traces displayed because a measurement may exist,
        but not be associated with a trace.
        '''
        channel = kwargs.get("channel", self.active_channel)
        meas = self.query("meas_number_list", cnum=channel).split(",")
        return 0 if meas is None else len(meas)
