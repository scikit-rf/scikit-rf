import itertools
from enum import Enum

import numpy as np

import skrf
from skrf.vi import vna
from skrf.vi.validators import (EnumValidator, FloatValidator, FreqValidator,
                                IntValidator, SetValidator)


class WindowFormat(Enum):
    ONE_TRACE = "D1"
    TWO_TRACES = "D2"
    THREE_TRACES = "D3"
    TWO_VERTICAL = "D12H"
    ONE_FIRST_ROW_TWO_SECOND_ROW = "D11_23"
    TWO_BY_TWO = "D12_34"


class MeasurementParameter(Enum):
    S11 = "S11"
    S12 = "S12"
    S21 = "S21"
    S22 = "S22"
    A = "A"
    B = "B"
    R1 = "R1"
    R2 = "R2"


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

    @property
    def freq_step(self) -> int:
        f = self.frequency
        return f.step

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
            cal_dict[cal_key] = raw[::2] + 1j * raw

        return skrf.Calibration.from_coefs(self.frequency, cal_dict)

    @calibration.setter
    def calibration(self, cal: skrf.Calibration) -> None:
        for cal_key, term in self._cal_term_map.items():
            vals = np.array([(x.real, x.imag) for x in cal[cal_key]]).flatten()
            self.write_values(f"SENS:CORR:COEF {term},", vals)

    @property
    def query_format(self) -> vna.ValuesFormat:
        fmt = self.query("FORM?")
        if fmt == "ASC,0":
            self._query_values_fmt = vna.ValuesFormat.ASCII
        elif fmt == "REAL,32":
            self._query_values_fmt = vna.ValuesFormat.BINARY
        return self._query_values_fmt

    @query_format.setter
    def query_format(self, fmt: vna.ValuesFormat) -> None:
        if fmt == vna.ValuesFormat.ASCII:
            self._query_values_fmt = vna.ValuesFormat.ASCII
            self.write("FORM ASC,0")
        elif fmt == vna.ValuesFormat.BINARY:
            self._query_values_fmt = vna.ValuesFormat.BINARY
            self.write("FORM REAL,32")

    def define_measurement(self, trace: int, parameter: MeasurementParameter) -> None:
        if trace not in range(1, self.n_traces + 1):
            self.n_traces = trace

        self.write(f"CALC:PAR{trace}:DEF {parameter.value}")

    def get_snp_network(self, ports=[1, 2]) -> skrf.Network:
        msmnts = list(itertools.product(ports, repeat=2))
        msmnt_params = [f"S{a}{b}" for a, b in msmnts]

        original_config = {
            "n_traces": self.n_traces,
            "window_configuration": self.window_configuration,
        }

        self.n_traces = len(msmnts)
        ntwk = skrf.Network()
        ntwk.frequency = self.frequency
        ntwk.s = np.empty(
            shape=(ntwk.frequency.npoints, len(msmnts), len(msmnts)), dtype=complex
        )

        for tr, ((i, j), msmnt) in enumerate(zip(msmnts, msmnt_params)):
            self.active_trace = tr + 1

            raw = self.active_trace_sdata
            ntwk.s[:, i - 1, j - 1] = raw[::2] + 1j * raw[1::2]

        for key, val in original_config.items():
            setattr(self, key, val)

        return ntwk
