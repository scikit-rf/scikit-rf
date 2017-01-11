from collections import OrderedDict

import numpy as np
import skrf.vi.vna_pyvisa
from skrf_qtwidgets.analyzers import base_analyzer


class Analyzer(skrf.vi.vna_pyvisa.PNA, base_analyzer.Analyzer):
    DEFAULT_VISA_ADDRESS = "TCPIP0::192.168.1.50::5025::SOCKET"
    NAME = "Agilent E8363C"
    NPORTS = 2

    def __init__(self, address=None):
        if not address:
            address = self.DEFAULT_VISA_ADDRESS
        super(Analyzer, self).__init__(address)
        self.resource.timeout = 2000
        self.use_binary()

    def use_binary(self):
        """setup the analyzer to transfer in binary which is much faster for large datasets"""
        self.resource.write(':FORM:BORD SWAP')
        self.resource.write(':FORM:DATA REAL,64')
        self.resource.values_format.use_binary(datatype='d', is_big_endian=False, container=np.array)

    def use_ascii(self):
        self.resource.write(':FORM:DATA ASCII')
        self.resource.values_format.use_ascii(converter='f', separator=',', container=np.array)

    def measure_twoport_ntwk(self, ports=(1, 2), sweep=True):
        return self.measure_snp(ports, sweep=sweep)

    def measure_oneport_ntwk(self, port=1, sweep=True):
        return self.measure_snp(port, sweep=sweep)

    def measure_switch_terms(self, ports=(1, 2), sweep=True):
        return self.get_switch_terms(ports)

    def measure_snp(self, ports=(1, 2), channel=None, sweep=True, name="", funit="GHz"):
        if not name:
            name = "network"

        channel = str(channel) if channel else str(self.channel)
        self.activate_channel(channel)
        ports = [int(port) for port in ports] if type(ports) in (list, tuple) else int(ports)

        npoints = int(self.resource.query(":SENSE{:s}:SWEEP:POINTS?".format(channel)))
        data = self.resource.query_values(
            ':CALCulate{:s}:DATA:SNP:PORTs? "{:s}"'.format(channel, ",".join(map(str, ports))))
        nrows = int(len(data) / npoints)
        nports = int(np.sqrt(nrows - 1))
        data = data.reshape([nrows, -1])

        fdata = data[0]
        sdata = data[1:]
        ntwk = skrf.Network()
        ntwk.frequency = skrf.Frequency.from_f(fdata, unit="Hz")
        s = np.empty(shape=(sdata.shape[1], nports, nports), dtype=complex)
        for n in range(nports):
            for m in range(nports):
                i = n * nports + m
                s[:, m, n] = sdata[i * 2] + 1j * sdata[i * 2 + 1]
        ntwk.frequency.unit = funit
        ntwk.s = s
        ntwk.name = name
        return ntwk

    def get_channel_measurement_numbers(self, channel=1):
        channel = str(channel)
        meas_list = self.resource.query("CALC{:s}:PAR:CAT:EXT?".format(channel))
        meas = meas_list[1:-1].split(',')
        if len(meas) == 1:
            return None
        parameters = [(meas[k], meas[k + 1]) for k in range(0, len(meas) - 1, 2)]
        return parameters

    def activate_channel(self, channel):
        """weird errors sometimes happen, so make sure a measurement is selected on the desired channel"""
        channel = str(channel)
        measurements = self.resource.query("SYSTem:MEASurement:CATalog? " + channel)[1:-1].split(",")
        self.resource.write(":CALC{:s}:PAR:SEL {:s}".foramt(measurements[0]))
        return

    def measure_traces(self, traces, name_prefix=""):
        if name_prefix:
            name_prefix += " - "

        channels = OrderedDict()
        for trace in traces:
            ch = trace["channel"]
            if ch not in channels.keys():
                channels[ch] = {
                    "frequency": None,
                    "traces": []}
            channels[ch]["traces"].append(trace)
        traces = []
        for ch, ch_data in channels.items():
            self.channel = int(ch)
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

    def get_list_of_traces(self):
        traces = []
        channels = self.resource.query("SYSTem:CHANnels:CATalog?")[1:-1].split(",")
        for channel in channels:
            meas_list = self.resource.query("CALC{:s}:PAR:CAT:EXT?".format(channel))
            meas = meas_list[1:-1].split(',')
            if len(meas) == 1:
                continue  # if there isnt a single comma, then there arent any measurments
            parameters = dict([(meas[k], meas[k + 1]) for k in range(0, len(meas) - 1, 2)])

            measurements = self.resource.query("SYSTem:MEASurement:CATalog? " + channel)[1:-1].split(",")
            for measurement in measurements:
                name = self.resource.query("SYST:MEAS{:s}:NAME?".format(measurement))[1:-1]
                item = {"name": name, "channel": channel, "measurement": measurement,
                        "parameter": parameters.get(name, name)}
                item["label"] = "{:s} - Chan{:s},Meas{:s}".format(item["parameter"], item["channel"], item["measurement"])
                traces.append(item)
        return traces

    def get_frequency_sweep(self, channel=None):
        pass

    def set_frequency_sweep(self, start_freq, stop_freq, num_points, channel=None):
        pass
