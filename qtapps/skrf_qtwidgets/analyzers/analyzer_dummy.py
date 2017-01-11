from skrf_qtwidgets.analyzers import base_analyzer

import os
import skrf
from skrf_qtwidgets.cfg import example_data_dir


class DummyResource(object):
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Analyzer(base_analyzer.Analyzer):
    DEFAULT_VISA_ADDRESS = "GPIB0::16::INSTR"
    NAME = "Dummy Analyzer"
    NPORTS = 2

    def __init__(self, resource):
        self.resource = DummyResource()

    def measure_snp(self, ports):
        if len(ports) == 1:
            return self.measure_oneport_ntwk(ports)
        elif len(ports) == 2:
            return self.measure_twoport_ntwk(ports)

    def measure_twoport_ntwk(self, ports=(1, 2), sweep=True):
        # return self.get_twoport(ports, sweep=sweep)
        return skrf.Network(os.path.join(example_data_dir, 'ring slot array simulation.s2p'))

    def measure_oneport_ntwk(self, port=1, sweep=True):
        # return self.get_oneport(port, sweep=sweep)
        return skrf.Network(os.path.join(example_data_dir, 'ring slot array measured.s1p'))

    def measure_switch_terms(self, ports=(1, 2), sweep=True):
        return self.get_switch_terms(ports)

    def get_list_of_traces(self, channel=None):
        traces = [
            {"text": "S11"},
            {"text": "S21"},
            {"text": "S12"},
            {"text": "S22"},
        ]
        return traces

    def measure_traces(self, traces=[], channel=None, sweep=False):
        ntwk = skrf.Network(os.path.join(example_data_dir, 'ring slot array simulation.s2p'))
        if not traces:
            return

        result = []
        for trace in traces:
            if trace in ["S11", "S21", "S12", "S22"]:
                result.append(getattr(ntwk, trace.lower()))

        return result
