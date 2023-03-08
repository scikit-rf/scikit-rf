import itertools
from enum import Enum

import numpy as np

import skrf
from skrf.vi import vna
from skrf.vi.validators import (BooleanValidator, EnumValidator,
                                FloatValidator, FreqValidator, IntValidator,
                                SetValidator)


class WindowFormat(Enum):
    ONE_TRACE = "D1"
    TWO_TRACES = "D2"
    THREE_TRACES = "D3"
    TWO_VERTICAL = "D12H"
    ONE_FIRST_ROW_TWO_SECOND_ROW = "D11_23"
    TWO_BY_TWO = "D12_34"


class FieldFox(vna.VNA):

    freq_start = vna.VNA.command(
        get_cmd="SENS:FREQ:STAR?",
        set_cmd="SENS:FREQ:STAR <arg>",
        doc="""The start frequency [Hz]""",
        validator=FreqValidator(),
    )

    freq_stop = vna.VNA.command(
        get_cmd="SENS:FREQ:STOP?",
        set_cmd="SENS:FREQ:STOP <arg>",
        doc="""The start frequency [Hz]""",
        validator=FreqValidator(),
    )

    freq_center = vna.VNA.command(
        get_cmd="SENS:FREQ:CENT?",
        set_cmd="SENS:FREQ:CENT <arg>",
        doc="""The center frequency [Hz]""",
        validator=FreqValidator(),
    )

    freq_span = vna.VNA.command(
        get_cmd="SENS:FREQ:SPAN?",
        set_cmd="SENS:FREQ:SPAN <arg>",
        doc="""The frequency span [Hz]""",
        validator=FreqValidator(),
    )

    npoints = vna.VNA.command(
        get_cmd="SENS:SWE:POIN?",
        set_cmd="SENS:SWE:POIN <arg>",
        doc="""The number of frequency points""",
        validator=IntValidator(),
    )

    sweep_time = vna.VNA.command(
        get_cmd="SENS:SWE:TIME?",
        set_cmd="SENS:SWE:TIME <arg>",
        doc="""The sweep time [s]""",
        validator=FloatValidator(),
    )

    if_bandwidth = vna.VNA.command(
        get_cmd="SENS:BWID?",
        set_cmd="SENS:BWID <arg>",
        doc="""The center frequency [Hz]""",
        validator=SetValidator([10, 30, 100, 300, 1000, 10_000, 30_000, 100_000]),
    )

    window_configuration = vna.VNA.command(
        get_cmd="DISP:WIND:SPL?",
        set_cmd="DISP:WIND:SPL <arg>",
        doc="""How multiple trace windows appear on screen""",
        validator=EnumValidator(WindowFormat),
    )

    n_traces = vna.VNA.command(
        get_cmd="CALC:PAR:COUN?",
        set_cmd="CALC:PAR:COUN <arg>",
        doc="""The number of active traces.""",
        validator=IntValidator(min=1, max=4),
    )

    active_trace = vna.VNA.command(
        set_cmd="CALC:PAR<arg>:SEL",
        doc="""Set the active trace. There is no command to read the active
            trace.""",
        validator=IntValidator(min=1, max=4),
    )

    active_trace_sdata = vna.VNA.command(
        get_cmd="CALC:DATA:SDATA?",
        doc="""Get the current trace data as a network""",
        values=True,
    )

    is_continuous = vna.VNA.command(
        get_cmd="INIT:CONT?",
        set_cmd="INIT:CONT <arg>",
        doc="""Get the current trace data as a network""",
        validator=BooleanValidator()
    )

    _cal_term_map = {
        "forward directivity": "ed,1,1",
        "reverse directivity": "ed,2,2",
        "forward source match": "es,1,1",
        "reverse source match": "es,2,2",
        "forward reflection tracking": "er,1,1",
        "reverse reflection tracking": "er,2,2",
        "forward transmission tracking": "et,2,1",
        "reverse transmission tracking": "et,1,2",
        "forward load match": "el,2,1",
        "reverse load match": "el,1,2",
        "forward isolation": "ex,2,1",
        "reverse isolation": "ex,1,2"
    }

    def __init__(self, address: str, backend: str = "@py") -> None:
        super().__init__(address, backend)

        self._resource.read_termination = "\n"
        self._resource.write_termination = "\n"

        _ = self.query_format # calling the getter sets _values_format to make sure we're in sync with the instrument

    @property
    def freq_step(self) -> int:
        f = self.frequency
        return int(f.step)

    @freq_step.setter
    def freq_step(self, f: int) -> None:
        freq = self.frequency
        self.npoints = len(range(int(freq.start), int(freq.stop) + f, f))


    @property
    def frequency(self) -> skrf.Frequency:
        return skrf.Frequency(
            start=self.freq_start, stop=self.freq_stop, npoints=self.npoints, unit='Hz'
        )

    @frequency.setter
    def frequency(self, f: skrf.Frequency):
        self.freq_start = f.start
        self.freq_stop = f.stop
        self.npoints = f.npoints

    @property
    def calibration(self) -> skrf.Calibration:
        cal_dict = {}
        for cal_key, term in self._cal_term_map.items():
            raw = self.query_values(f"SENS:CORR:COEF? {term}", container=np.array)
            cal_dict[cal_key] = raw[::2] + 1j * raw[1::2]

        return skrf.Calibration.from_coefs(self.frequency, cal_dict)

    @calibration.setter
    def calibration(self, cal: skrf.Calibration) -> None:
        cal_dict = cal.coefs_12term
        for cal_key, term in self._cal_term_map.items():
            vals = np.array([(x.real, x.imag) for x in cal_dict[cal_key]]).flatten()
            self.write_values(f"SENS:CORR:COEF {term},", vals)

    @property
    def query_format(self) -> vna.ValuesFormat:
        fmt = self.query("FORM?")
        if fmt == "ASC,0":
            self._values_fmt = vna.ValuesFormat.ASCII
        elif fmt == "REAL,32":
            self._values_fmt = vna.ValuesFormat.BINARY_32
        elif fmt == "REAL,64":
            self._values_fmt = vna.ValuesFormat.BINARY_64
        return self._values_fmt

    @query_format.setter
    def query_format(self, fmt: vna.ValuesFormat) -> None:
        if fmt == vna.ValuesFormat.ASCII:
            self._values_fmt = vna.ValuesFormat.ASCII
            self.write("FORM ASC,0")
        elif fmt == vna.ValuesFormat.BINARY_32:
            self._values_fmt = vna.ValuesFormat.BINARY_32
            self.write("FORM REAL,32")
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write("FORM REAL,64")

    def get_measurement_parameter(self, trace: int) -> str:
        if trace not in range(1, 5):
            raise ValueError("Trace must be between 1 and 4")

        return self.query(f"CALC:PAR{trace}:DEF?")

    def define_measurement(self, trace: int, parameter: str) -> None:
        if trace not in range(1, self.n_traces + 1):
            self.n_traces = trace

        self.write(f"CALC:PAR{trace}:DEF {parameter}")

    def sweep(self) -> None:
        self._resource.clear()
        was_continuous = self.is_continuous
        self.is_continuous = False
        self.write("INIT")
        self.is_continuous = was_continuous

    def get_snp_network(self, ports=[1, 2], restore_settings: bool = True) -> skrf.Network:
        msmnts = list(itertools.product(ports, repeat=2))
        msmnt_params = [f"S{a}{b}" for a, b in msmnts]

        if restore_settings:
            original_config = {
                "n_traces": self.n_traces,
                "window_configuration": self.window_configuration,
                "trace_params": [self.get_measurement_parameter(i+1) for i in range(self.n_traces)]
            }

        self.n_traces = len(msmnts)
        for i, param in enumerate(msmnt_params):
            self.define_measurement(i+1, param)

        ntwk = skrf.Network()
        ntwk.frequency = self.frequency
        ntwk.s = np.empty(
            shape=(ntwk.frequency.npoints, len(ports), len(ports)), dtype=complex
        )

        self.sweep()
        for tr, ((i, j), param) in enumerate(zip(msmnts, msmnt_params)):
            self.active_trace = tr + 1

            raw = self.active_trace_sdata
            if len(msmnts) == 1:
                ntwk.s[:, 0, 0] = raw[::2] + 1j * raw[1::2]
            else:
                ntwk.s[:, i - 1, j - 1] = raw[::2] + 1j * raw[1::2]

        if restore_settings:
            for i, param in enumerate(original_config.pop('trace_params')):
                self.define_measurement(i+1, param)
            for key, val in original_config.items():
                setattr(self, key, val)

        return ntwk
