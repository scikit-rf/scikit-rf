from skrf_qtwidgets.analyzers import base_analyzer


class Analyzer(base_analyzer.Analyzer):
    DEFAULT_VISA_ADDRESS = "GPIB0::16::INSTR"
    NAME = "Agilent E8364C"
    NPORTS = 2

    def measure_twoport_ntwk(self, ports=(1, 2), sweep=True):
        return self.get_twoport(ports, sweep=sweep)

    def measure_oneport_ntwk(self, port=1, sweep=True):
        return self.get_oneport(port, sweep=sweep)

    def measure_switch_terms(self, ports=(1, 2), sweep=True):
        return self.get_switch_terms(ports)
