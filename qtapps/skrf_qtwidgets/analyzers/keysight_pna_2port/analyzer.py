from collections import OrderedDict

import numpy as np
import skrf
from skrf_qtwidgets.analyzers import base_analyzer

SCPI_COMMANDS = {
    # "key": ["command", ("default_key": default_value) ...]
    # defaults convention:
    # if value is read/write, default scpi parameters will be null strings
    "active_channel":       ["SYSTem:ACTive:CHANnel?", ()],
    "measurement_list":     ["SYSTem:MEASurement:CATalog? {:}", ("channel", "")],
    "select_meas_by_num":   [":CALC{:}:PAR:MNUM:SEL {:}", ("channel", ""), ("mnum", "")],
    "trigger_source":       ["TRIGger:SEQuence:SOURce? {:}", ("source", "")],
    "sweep_mode":           ["SENSe{:}:SWEep:MODE? {:}", ("channel", ""), ("mode", "")],
    "averaging_state":      ["SENSe{:}:AVERage:STATe? {:}", ("channel", ""), ("onoff", "")],
    "averaging_count":      ["SENSe{:}:AVERage:COUNt? {:}", ("channel", ""), ("count", "")],
    "groups_count":         ["SENSe{:}:SWEep:GROups:COUNt? {:}", ("channel", ""), ("count", "")],
    "sweep_time":           ["SENSe{:}:SWEep:TIME? {:}", ("channel", ""), ("seconds", "")],
}


class Analyzer(base_analyzer.Analyzer):
    DEFAULT_VISA_ADDRESS = "GPIB0::16::INSTR"
    NAME = "Keysight PNA"
    NPORTS = 2
    NCHANNELS = 32
    SCPI_VERSION_TESTED = 'A.09.20.08'

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        super(Analyzer, self).__init__(address)
        self.scpi = SCPI_COMMANDS
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.use_binary()

    def process_scpi(self, command, **kwargs):
        scpi_str = self.scpi[command][0]
        defaults = self.scpi[command][1:]
        args = []
        for key, value in defaults:
            args.append(kwargs.get(key, value))
        return scpi_str.format(args)

    def write(self, command, **kwargs):
        scpi_str = self.process_scpi(command, **kwargs).replace('?', '').strip()
        self.resource.write(scpi_str)

    def read(self):
        return self.resource.read().replace('"', '')

    def query(self, command, **kwargs):
        scpi_str = self.process_scpi(command, **kwargs).strip()
        return self.resource.query(scpi_str).replace('"', '')

    def query_values(self, command, **kwargs):
        scpi_str = self.process_scpi(command, **kwargs)
        return self.resource.query_values(scpi_str)

    def use_binary(self):
        """setup the analyzer to transfer in binary which is faster, especially for large datasets"""
        self.resource.write(':FORM:BORD SWAP')
        self.resource.write(':FORM:DATA REAL,64')
        self.resource.values_format.use_binary(datatype='d', is_big_endian=False, container=np.array)

    def use_ascii(self):
        self.resource.write(':FORM:DATA ASCII')
        self.resource.values_format.use_ascii(converter='f', separator=',', container=np.array)

    @property
    def channel(self):
        return int(self.query("active_channel"))

    @channel.setter
    def channel(self, channel):
        """
        :param channel: int
        :return:

        There is no specific command to activate a channel, so we ask which channel we want and then activate the first
        trace on that channel.  We do this because if the analyzer gets into a state where it doesn't recognize
        any activated measurements, the get_snp_network method will fail, and possibly others as well.  That is why in
        some methods you will see the fillowing line:
        >>> self.channel = channel = kwargs.get("channel", self.channel)
        this way we force this property to be set, even if it just resets itself to the same value, but then a trace
        will become active and our get_snp_network method will succeed.
        """
        mnum = self.query("measurement_list", channel=channel).split(",")[0]
        self.write("select_meas_by_num", channel=channel, mnum=mnum)
        return

    def sweep(self, **kwargs):
        """
        :param kwargs: channel(int), timeout(milliseconds)
        :return:

        trigger a fresh sweep, based upon the analyzers current averaging setting, and then return to the prior state
        of continuous trigger or hold.
        """
        self.resource.clear()
        channel = kwargs.get("channel", self.channel)

        self.write("trigger_source", source="IMMediate")
        sweep_mode = self.query("sweep_mode", channel=channel)

        was_continuous = "CONT" in sweep_mode.upper()

        if int(self.query("averaging_state", channel=channel)):
            sweep_mode = "GROUPS"
            sweeps = int(self.query("averaging_count"))
            self.write("groups_count", channel=channel, count=sweeps)
            sweeps *= 2
        else:
            sweep_mode = "SINGLE"
            sweeps = 2  # 2 sweeps for each port

        original_timeout = self.resource.timeout
        timeout = kwargs.get("timeout", None)
        # if no timeout is specified attempt to autoset the sweep time to avoid unintentional timeouts

        # TODO: This works for 1 channel, but fails to account for the sweep time of many channels
        if not timeout:
            sweep_time = float(self.query("sweep_time"))
            total_time = sweep_time * sweeps
            self.resource.timeout = total_time * 2000  # convert to milliseconds, and double for buffer
        else:
            self.resource.timeout = timeout

        try:
            self.write("sweep_mode", channel=channel, mode=sweep_mode)
            self.query("*OPC?")  # TODO: universal way to query if the instrument is ready, a method perhaps?
        finally:
            self.resource.clear()
            if was_continuous:
                self.write("sweep_mode", mode="continuous")
            self.resource.timeout = original_timeout
        return

    def sweep_all_channels(self, **kwargs):
        pass

    def get_snp_network(self, ports, **kwargs):
        """
        :param ports: list of ports
        :param kwargs: channel(int), sweep(bool), name(str), f_unit(str)
        :return: an n-port skrf.Network object

        general function to take in a list of ports and return the full snp network as an skrf.Network for those ports
        """
        # assigning this way, activate the channel, see channel.setter for more information about weirdness
        # that can happen if we don't assign a measurement to be active
        self.resource.clear()
        self.channel = channel = kwargs.get("channel", self.channel)
        sweep = kwargs.get("sweep", False)
        name = kwargs.get("name", "")
        f_unit = kwargs.get("f_unit", "GHz")

        ports = [int(port) for port in ports] if type(ports) in (list, tuple) else [int(ports)]
        if not name:
            name = "{:d}Port Network".format(len(ports))

        if sweep:
            self.sweep()

        npoints = int(self.resource.query(":SENSE{:}:SWEEP:POINTS?".format(channel)))
        data = self.resource.query_values(
            ':CALCulate{:}:DATA:SNP:PORTs? "{:s}"'.format(channel, ",".join(map(str, ports))))
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
         it then returns a list where each list-item represents one available trace through.

         Each list item is a python dict with all necessary information to retrieve the trace as an skrf.Network object:
         list-item keys are:
         * "name": the name of the measurement e.g. "CH1_S11_1"
         * "channel": the channel number the measurement is on
         * "measurement": the measurement number (MNUM in SCPI)
         * "parameter": the parameter of the measurement, e.g. "S11", "S22" or "a1b1,2"
         * "label": the text the item will use to identify itself to the user e.g "Measurement 1 on Channel 1"
        """
        self.resource.clear()
        traces = []
        channels = self.resource.query("SYSTem:CHANnels:CATalog?")[1:-1].split(",")
        for channel in channels:
            meas_list = self.resource.query("CALC{:}:PAR:CAT:EXT?".format(channel))
            meas = meas_list[1:-1].split(',')
            if len(meas) == 1:
                continue  # if there isnt a single comma, then there arent any measurments
            parameters = dict([(meas[k], meas[k + 1]) for k in range(0, len(meas) - 1, 2)])

            measurements = self.resource.query("SYSTem:MEASurement:CATalog? " + channel)[1:-1].split(",")
            for measurement in measurements:
                name = self.resource.query("SYST:MEAS{:s}:NAME?".format(measurement))[1:-1]
                item = {"name": name, "channel": channel, "measurement": measurement,
                        "parameter": parameters.get(name, name)}
                item["label"] = "{:s} - Chan{:},Meas{:}".format(item["parameter"], item["channel"], item["measurement"])
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
        if sweep is True:
            self.sweep()

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
        traces = []
        for ch, ch_data in channels.items():
            frequency = ch_data["frequency"] = self.get_frequency()
            for trace in ch_data["traces"]:
                self.resource.write("CALC{:s}:PAR:MNUM:SEL {:s}".format(trace["channel"], trace["measurement"]))
                sdata = self.resource.query_values(":CALC{:s}:DATA? SDATA".format(trace["channel"]))
                s = sdata[::2] + 1j * sdata[1::2]
                ntwk = skrf.Network()
                ntwk.s = s
                ntwk.frequency = frequency
                ntwk.name = name_prefix + trace.get("parameter", "trace")
                traces.append(ntwk)
        return traces

    def get_frequency(self, **kwargs):
        self.resource.clear()
        channel = kwargs.get("channel", self.channel)
        use_log = "LOG" in self.resource.query("SENSe{:}:SWEep:TYPE?".format(channel)).upper()
        f_start = float(self.resource.query("SENSe:FREQuency:STARt?"))
        f_stop = float(self.resource.query("SENSe:FREQuency:STOP?"))
        f_npoints = int(self.resource.query("SENSe:SWEep:POINts?"))
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
        :param f_unit: the frequnecy unit of the provided f_start and f_stop
        :param channel: channel of the analyzer
        :return:
        """
        f_unit = kwargs.get("f_unit", "hz")
        channel = kwargs.get("channel", 1)
        self.resource.write('SENS{:}:FREQ:STAR {:f}'.format(channel, self.to_hz(f_start, f_unit)))
        self.resource.write('SENS{:}:FREQ:STOP {:f}'.format(channel, self.to_hz(f_stop, f_unit)))
        self.resource.write('SENS{:}:SWE:POIN {:d}'.format(channel, f_npoints))

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        '''
        Get switch terms and return them as a tuple of Network objects.

        Returns
        --------
        forward, reverse : oneport switch term Networks
        '''

        # TODO: This tends to error if there are multiple channels operating

        self.resource.clear()
        p1, p2 = ports

        channel = kwargs.get("channel", self.channel)

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

        self.sweep(**kwargs)

        forward = self.get_measurement(name=forward_name, sweep=False)  # type: skrf.Network
        forward.name = "forward switch terms"
        reverse = self.get_measurement(name=reverse_name, sweep=False)  # type: skrf.Network
        reverse.name = "reverse switch terms"

        self.delete_measurement(name=forward_name)
        self.delete_measurement(name=reverse_name)
        return forward, reverse

    def get_measurement(self, name=None, number=None, **kwargs):
        if name is None and number is None:
            raise ValueError("must provide either a measurement name or a number")

        if type(name) is str:
            self.select_meas_by_name(name, **kwargs)
        else:
            self.select_meas_by_number(number, **kwargs)
        return self.get_active_trace_as_network(**kwargs)

    def get_active_trace_as_network(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        sweep = kwargs.get("sweep", False)
        if sweep:
            self.sweep(**kwargs)

        channel = self.channel
        ntwk = skrf.Network()
        sdata = self.resource.query_values(":CALC{:}:DATA? SDATA".format(channel))
        ntwk.s = sdata[::2] + 1j * sdata[1::2]
        ntwk.frequency = self.get_frequency(channel=channel)
        return ntwk

    def create_meas(self, name, param, **kwargs):
        '''
        :param name: str, name of the measurement  **WARNING**, not all names behave well
        :param param: str, analyzer parameter, e.g.: S11 ; a1/b1,1 ; A/R1,1
        :param kwargs: channel
        :return:
        '''
        channel = kwargs.get("channel", self.channel)
        self.resource.write('CALCulate{:}:PARameter:DEFine:EXTended "{:}", "{:}"'.format(channel, name, param))
        self.display_trace(name)

    def delete_measurement(self, name, **kwargs):
        '''
        :param name: str, name of the measurement  **WARNING**, not all names behave well
        :param kwargs: channel
        :return:
        '''
        channel = kwargs.get("channel", self.channel)
        self.resource.write('CALCulate{:}:PARameter:DELete {:s}'.format(channel, name))

    def display_trace(self, name, **kwargs):
        '''
        :param name: str
        :param kwargs: channel(int), window_n(int), trace_n(int), display_format(str)
        :return:

        Display a given measurement on specified trace number.

        display_format must be one of: MLINear, MLOGarithmic, PHASe, UPHase, IMAGinary, REAL, POLar, SMITh,
            SADMittance, SWR, GDELay, KELVin, FAHRenheit, CELSius
        '''
        channel = kwargs.get('channel', self.channel)
        window_n = kwargs.get("window_n", '')
        trace_n = kwargs.get("trace_n", self.ntraces + 1)
        display_format = kwargs.get('display_format', 'MLOG')
        self.resource.write('DISPlay:WINDow{:}:TRACe{:}:FEED "{:s}"'.format(window_n, trace_n, name))
        self.resource.write('CALCulate{:}:PARameter:SELect "{:}"'.format(channel, name))
        self.resource.write('CALCulate{:}:FORMat {:}'.format(channel, display_format))

    def get_meas_list(self, **kwargs):
        '''
        :param kwargs: channel
        :return: list of tuples of the form, (name, measurement)

        Return a list of measurement names on all channels.  If channel is provided to kwargs, then only measurmeents
        for that channel are queried
        '''
        channel = kwargs.get("channel", "")
        meas_list = self.resource.query("CALC{:}:PAR:CAT:EXT?".format(channel))
        meas = meas_list[1:-1].split(',')
        if len(meas) == 1:
            return None  # if there isnt a single comma, then there arent any measurments
        return [(meas[k], meas[k + 1]) for k in range(0, len(meas) - 1, 2)]

    @property
    def ntraces(self, **kwargs):
        '''
        :param kwargs: channel
        :return: The number of measurement traces that exist on the current channel

        Note that this may not be the same as the number of traces displayed because a measurement may exist,
        but not be associated with a trace.
        '''
        channel = kwargs.get("channel", self.channel)
        # meas = self.get_meas_list()
        meas = self.resource.query("SYSTem:MEASurement:CATalog?")[1:-1].split(",")
        return 0 if meas is None else len(meas)

    def select_meas_by_number(self, meas_number, **kwargs):
        """
        :param meas_number: int
        :param kwargs: channel
        :return:
        """
        channel = kwargs.get("channel", self.channel)
        self.resource.write("CALC{:}:PAR:MNUM:SEL {:}".format(channel, meas_number))

    def select_meas_by_name(self, name, **kwargs):
        '''
        :param name: str
        :param kwargs: channel
        :return:
        '''
        channel = kwargs.get("channel", self.channel)
        self.resource.write('CALCulate{:}:PARameter:SELect "{:s}"'.format(channel, name))
