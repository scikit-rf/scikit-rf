import numpy as np
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
    NAME = "Analyzer"
    NPORTS = 2

    def __init__(self, resource):
        self.resource = DummyResource()

    def measure_twoport_ntwk(self, ports=(1, 2), sweep=True, **kwargs):
        # return self.get_twoport(ports, sweep=sweep)
        return self.get_snp_network(ports)

    def measure_oneport_ntwk(self, port=1, sweep=True, **kwargs):
        # return self.get_oneport(port, sweep=sweep)
        return self.get_snp_network(port)

    def get_snp_network(self, ports, **kwargs):
        ntwk = skrf.Network(os.path.join(example_data_dir, 'ring slot array simulation.s2p'))
        if type(ports) == int:
            ports = [ports]

        if len(ports) == 1:
            port = ports[0]
            if port == 1:
                return ntwk.s11
            elif port == 2:
                return ntwk.s22
            else:
                return None
        elif len(ports) == 2:
            port1, port2 = ports
            if port1 == 1 and port2 == 2:
                return ntwk
            elif port2 == 2 and port1 == 1:
                s = np.zeros_like(ntwk.s)
                s[:, 0, 0] = ntwk.s[:, 1, 1]
                s[:, 1, 0] = ntwk.s[:, 1, 0]
                s[:, 0, 1] = ntwk.s[:, 0, 1]
                s[:, 1, 1] = ntwk.s[:, 0, 0]
            else:
                raise Exception("ports must be 1 or 2 or 1,2 or 2,1")
        else:
            raise Exception("ports must be 1 or 2 or 1,2 or 2,1")

    def get_switch_terms(self, ports=(1, 2), sweep=True):
        return self.get_switch_terms(ports)

    def get_list_of_traces(self, channel=None):
        traces = [
            {"label": "S11"},
            {"label": "S21"},
            {"label": "S12"},
            {"label": "S22"},
        ]
        return traces

    def get_traces(self, traces=[], channel=None, sweep=False, **kwargs):
        name_prefix = kwargs.get('name_prefix', "")
        if name_prefix:
            name_prefix += " - "
        ntwk = skrf.Network(os.path.join(example_data_dir, 'ring slot array simulation.s2p'))
        if not traces:
            return

        result = []
        for trace in traces:
            if trace["label"] in ["S11", "S21", "S12", "S22"]:
                t = getattr(ntwk, trace["label"].lower())
                t.name = name_prefix + trace["label"]
                result.append(t)

        return result
