from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import itertools
import sys
from enum import Enum

import numpy as np

import skrf
from skrf.vi import vna
from skrf.vi.validators import (
    BooleanValidator,
    DelimitedStrValidator,
    EnumValidator,
    FloatValidator,
    FreqValidator,
    IntValidator,
)
from skrf.vi.vna import VNA, ValuesFormat


class SweepType(Enum):
    LINEAR = "LIN"
    LOG = "LOG"
    POWER = "POW"
    CW = "CW"
    SEGMENT = "SEGM"
    PHASE = "PHAS"


class SweepMode(Enum):
    HOLD = "HOLD"
    CONTINUOUS = "CONT"
    GROUPS = "GRO"
    SINGLE = "SING"


class TriggerSource(Enum):
    EXTERNAL = "EXT"
    IMMEDIATE = "IMM"
    MANUAL = "MAN"


class AveragingMode(Enum):
    POINT = "POIN"
    SWEEP = "SWE"


class PNA(VNA):
    '''
    Keysight PNA, PNA-L, PNA-X.

    PNA Models
    ==========
    E8361A, E8362B, E8363B, E8364B, N5221A, N5222A, N5224A, N5225A, N5227A
    N5224B, N5222B, N5227B, N5225B, N5221B, E8356A, E8357A, E8358A, E8361A
    E8361C, E8362A, E8362C, E8363A, E8363B, E8363C, E8364A, E8364B, E8364C
    E8801A, E8802A, E8803A, N3381A, N3382A, N3383A, N5250C

    PNA-L Models
    ============
    N5230A, N5230C, N5231A, N5232A, N5234A, N5235A, N5239A, N5234B, N5235B
    N5231B, N5232B, N5239B

    PNA-X Models
    ============
    N5241A, N5242A, N5244A, N5245A, N5247A, N5249A, N5247B, N5245B, N5244B
    N5242B, N5241B, N5249B, N5264A, N5264B
    '''
    # TODO:
    # - active_calset_name: SENS<self:cnum>:CORR:CSET:ACT? NAME
    # - create_calset: SENS<self:cnum>:CORR:CSET:CRE <name>
    # - calset_data: SENS<self:cnum>:CORR:CSET:DATA?  <eterm>,<port a>,<port b> '<receiver>'


    _models = {
        "default": {"nports": 2, "unsupported": []},
        "E8362C": {"nports": 2, "unsupported": ["nports", "freq_step", "fast_sweep"]},
        "N5227B": {"nports": 4, "unsupported": []},
    }

    class Channel(vna.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)

            if cnum != 1:
                default_msmnt = f"CH{self.cnum}_S11_1"
                self.create_measurement(default_msmnt, "S11")

        def _on_delete(self):
            self.write(f"SYST:CHAN:DEL {self.cnum}")

        freq_start = VNA.command(
            get_cmd="SENS<self:cnum>:FREQ:STAR?",
            set_cmd="SENS<self:cnum>:FREQ:STAR <arg>",
            doc="""The start frequency [Hz]""",
            validator=FreqValidator(),
        )

        freq_stop = VNA.command(
            get_cmd="SENS<self:cnum>:FREQ:STOP?",
            set_cmd="SENS<self:cnum>:FREQ:STOP <arg>",
            doc="""The stop frequency [Hz]""",
            validator=FreqValidator(),
        )

        freq_span = VNA.command(
            get_cmd="SENS<self:cnum>:FREQ:SPAN?",
            set_cmd="SENS<self:cnum>:FREQ:SPAN <arg>",
            doc="""The frequency span [Hz].""",
            validator=FreqValidator(),
        )

        freq_center = VNA.command(
            get_cmd="SENS<self:cnum>:FREQ:CENT?",
            set_cmd="SENS<self:cnum>:FREQ:CENT <arg>",
            doc="""The frequency span [Hz].""",
            validator=FreqValidator(),
        )

        npoints = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:POIN?",
            set_cmd="SENS<self:cnum>:SWE:POIN <arg>",
            doc="""The number of frequency points. Sets the frequency step as a
                side effect
            """,
            validator=IntValidator(),
        )

        if_bandwidth = VNA.command(
            get_cmd="SENS<self:cnum>:BWID?",
            set_cmd="SENS<self:cnum>:BWID <arg>",
            doc="""The IF bandwidth [Hz]""",
            validator=FreqValidator(),
        )

        sweep_time = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:TIME?",
            set_cmd="SENS<self:cnum>:SWE:TIME <arg>",
            doc="""The time in seconds for a single sweep [s]""",
            validator=FloatValidator(decimal_places=2),
        )

        sweep_type = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:TYPE?",
            set_cmd="SENS<self:cnum>:SWE:TYPE <arg>",
            doc="""The type of sweep (linear, log, etc)""",
            validator=EnumValidator(SweepType),
        )

        sweep_mode = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:MODE?",
            set_cmd="SENS<self:cnum>:SWE:MODE <arg>",
            doc="""This channel's trigger mode""",
            validator=EnumValidator(SweepMode),
        )

        measurement_numbers = VNA.command(
            get_cmd="SYST:MEAS:CAT? <self:cnum>",
            set_cmd=None,
            doc="""The list of measurement numbers on this channel""",
            validator=DelimitedStrValidator(int),
        )

        averaging_on = VNA.command(
            get_cmd="SENS<self:cnum>:AVER:STATE?",
            set_cmd="SENS<self:cnum>:AVER:STATE <arg>",
            doc="""Whether averaging is on or off""",
            validator=BooleanValidator(),
        )

        averaging_count = VNA.command(
            get_cmd="SENS<self:cnum>:AVER:COUN?",
            set_cmd="SENS<self:cnum>:AVER:COUN <arg>",
            doc="""The number of measurements combined for an average""",
            validator=IntValidator(1, 65536),
        )

        averaging_mode = VNA.command(
            get_cmd="SENS<self:cnum>:AVER:MODE?",
            set_cmd="SENS<self:cnum>:AVER:MODE <arg>",
            doc="""How measurements are averaged together""",
            validator=EnumValidator(AveragingMode),
        )

        n_sweep_groups = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:GRO:COUN?",
            set_cmd="SENS<self:cnum>:SWE:GRO:COUN <arg>",
            doc="""The number of triggers sent for one trigger command""",
            validator=IntValidator(1, int(2e6)),
        )

        @property
        def freq_step(self) -> int:
            # Not all instruments support SENS:FREQ:STEP
            f = self.frequency
            return int(f.step)

        @freq_step.setter
        def freq_step(self, f: int | float | str) -> None:
            validator = FreqValidator()
            f = validator.validate_input(f)
            freq = self.frequency
            self.npoints = len(range(int(freq.start), int(freq.stop) + f, f))

        @property
        def frequency(self) -> skrf.Frequency:
            f = skrf.Frequency(
                start=self.freq_start,
                stop=self.freq_stop,
                npoints=self.npoints,
                unit="hz",
            )
            return f

        @frequency.setter
        def frequency(self, f: skrf.Frequency) -> None:
            self.freq_start = f.start
            self.freq_stop = f.stop
            self.npoints = f.npoints

        @property
        def measurements(self) -> list[tuple[str, str]]:
            msmnts = self.query(f"CALC{self.cnum}:PAR:CAT:EXT?").replace('"', "")
            msmnts = msmnts.split(",")
            return list(zip(msmnts[::2], msmnts[1::2]))

        @property
        def measurement_names(self) -> list[str]:
            return [msmnt[0] for msmnt in self.measurements]

        @property
        def calibration(self) -> skrf.Calibration:
            raise NotImplementedError()
            # orig_query_fmt = self.parent.query_format
            # self.parent.query_format = ValuesFormat.BINARY_64

            # eterm_re = re.compile(r"(\w+)\((\d),(\d)\)")
            # cset_terms = self.query(f"SENS{self.cnum}:CORR:CSET:ETER:CAT?")
            # terms = re.findall(eterm_re, cset_terms)

            # if len(terms) == 3:
            #     cal = skrf.OnePort
            # elif len(terms) == 12:
            #     # cal = skrf.TwoPort
            #     raise NotImplementedError()
            # else:
            #     # cal = skrf.MultiPort
            #     raise NotImplementedError()

            # coeffs = {}
            # for term in terms:
            #     pna_name = term[0]
            #     skrf_name = re.sub(
            #         r"([A-Z])", lambda pat: f" {pat.group(1).lower()}", pna_name
            #     ).strip()

            #     coeffs[skrf_name] = self.query_values(
            #         f"SENS{self.cnum}:CORR:CSET:ETERM? '{pna_name}({a},{b})'",
            #         complex_values=True,
            #         container=np.array,
            #     )

            # self.parent.query_format = orig_query_fmt

            # freq = self.frequency
            # return cal.from_coefs(freq, coeffs)

        @calibration.setter
        def calibration(self, cal: skrf.Calibration) -> None:
            raise NotImplementedError()

        @property
        def active_trace_sdata(self) -> np.ndarray:
            active_measurement = (
                self.query(f"CALC{self.cnum}:PAR:SEL?").replace('"', "").split(",")[0]
            )
            if active_measurement == "":
                raise RuntimeError("No trace is active. Must select measurement first.")
            return self.query_values(
                f"CALC{self.cnum}:DATA? SDATA", complex_values=True
            )

        def clear_averaging(self) -> None:
            self.write(f"SENS{self.cnum}:AVER:CLE")

        def create_measurement(self, name: str, parameter: str) -> None:
            self.write(f"CALC{self.cnum}:PAR:EXT '{name}',{parameter}")
            # Not all instruments support DISP:WIND:TRAC:NEXT
            traces = self.query("DISP:WIND:CAT?").replace('"', "")
            traces = [int(tr) for tr in traces.split(",")] if traces != "EMPTY" else [0]
            next_tr = traces[-1] + 1
            self.write(f"DISP:WIND:TRAC{next_tr}:FEED '{name}'")

        def delete_measurement(self, name: str) -> None:
            self.write(f"CALC{self.cnum}:PAR:DEL '{name}'")

        def get_measurement(self, name: str) -> skrf.Network:
            if name not in self.measurement_names:
                raise KeyError(f"{name} does not exist")

            self.parent.active_measurement = name
            ntwk = self.get_active_trace()
            ntwk.name = name
            return ntwk

        def get_active_trace(self) -> skrf.Network:
            self.sweep()
            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64

            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = self.active_trace_sdata

            self.parent.query_format = orig_query_fmt

            return ntwk

        def get_sdata(self, a: int | str, b: int | str) -> skrf.Network:
            self.sweep()
            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            param = f"S{a}{b}"
            self.create_measurement("SKRF_TMP", param)
            self.parent.active_measurement = "SKRF_TMP"

            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = self.active_trace_sdata

            self.delete_measurement("SKRF_TMP")
            self.parent.query_format = orig_query_fmt

            return ntwk

        def get_snp_network(
            self,
            ports: Sequence | None = None,
        ) -> skrf.Network:
            if ports is None:
                ports = list(range(1, self.parent.nports + 1))

            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            self.parent.active_channel = self
            orig_snp_fmt = self.query("MMEM:STOR:TRAC:FORM:SNP?")
            self.write("MMEM:STOR:TRACE:FORM:SNP RI") # Expect Real/Imaginary data

            msmnt_params = [f"S{a}{b}" for a, b in itertools.product(ports, repeat=2)]

            names = []
            # Make sure the ports specified are driven
            for param in msmnt_params:
                # Not all models support CALC:PAR:TAG:NEXT
                name = f"CH{self.cnum}_SKRF_{param}"
                names.append(name)
                self.create_measurement(name, param)

            self.sweep()
            port_str = ",".join(str(port) for port in ports)
            raw = self.query_values(
                f"CALC{self.cnum}:DATA:SNP:PORTS? '{port_str}'", container=np.array
            )
            self.parent.wait_for_complete()

            for name in names:
                self.delete_measurement(name)

            # The data is sent back as:
            # [
            #   [frequency points],
            #   [s11.real],
            #   [s11.imag],
            #   [s12.real],
            #   [s12.imag],
            # ...
            # ]
            # but flattened. So we recreate the above shape from the flattened data
            npoints = self.npoints
            nrows = len(raw) // npoints
            nports = len(ports)
            data = raw.reshape((nrows, -1))[1:]

            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = np.empty(
                shape=(len(ntwk.frequency), nports, nports), dtype=complex
            )
            real_rows = data[::2]
            imag_rows = data[1::2]
            for n in range(nports):
                for m in range(nports):
                    i = n * nports + m
                    ntwk.s[:, n, m] = real_rows[i] + 1j * imag_rows[i]

            self.parent.query_format = orig_query_fmt
            self.write(f"MMEM:STOR:TRACE:FORM:SNP {orig_snp_fmt}")

            return ntwk

        def sweep(self) -> None:
            self.parent.trigger_source = TriggerSource.IMMEDIATE
            self.parent._resource.clear()

            sweep_mode = self.sweep_mode
            sweep_time = self.sweep_time
            avg_on = self.averaging_on
            avg_mode = self.averaging_mode

            original_config = {
                "sweep_mode": sweep_mode,
                "sweep_time": sweep_time,
                "averaging_on": avg_on,
                "averaging_mode": avg_mode,
                "timeout": self.parent._resource.timeout,
            }

            if avg_on and avg_mode == AveragingMode.SWEEP:
                self.sweep_mode = SweepMode.GROUPS
                n_sweeps = self.averaging_count
                self.n_sweep_groups = n_sweeps
                n_sweeps *= self.parent.nports
            else:
                self.sweep_mode = SweepMode.SINGLE
                n_sweeps = self.parent.nports

            try:
                sweep_time *= n_sweeps * 1_000  # 1s per port
                self.parent._resource.timeout = max(sweep_time, 5_000)  # minimum of 5s
                self.parent.wait_for_complete()
            finally:
                self.parent._resource.clear()
                self.parent._resource.timeout = original_config.pop("timeout")
                for k, v in original_config.items():
                    setattr(self, k, v)

    def __init__(self, address: str, backend: str = "@py") -> None:
        super().__init__(address, backend)

        self._resource.read_termination = "\n"
        self._resource.write_termination = "\n"

        self.create_channel(1, "Channel 1")
        self.active_channel = self.ch1

        self.model = self.id.split(",")[1]
        if self.model not in self._models:
            print(
                f"WARNING: This model ({self.model}) has not been tested with "
                "scikit-rf. By default, all features are turned on but older "
                "instruments might be missing SCPI support for some commands "
                "which will cause errors. Consider submitting an issue on GitHub to "
                "help testing and adding support.",
                file=sys.stderr,
            )

    def _supports(self, feature: str) -> bool:
        model_config = self._models.get(self.model, self._models["default"])
        return feature not in model_config["unsupported"]

    def _model_param(self, param: str):
        model_config = self._models.get(self.model, self._models["default"])
        return model_config[param]

    trigger_source = VNA.command(
        get_cmd="TRIG:SOUR?",
        set_cmd="TRIG:SOUR <arg>",
        doc="""The source of the sweep trigger signal""",
        validator=EnumValidator(TriggerSource),
    )

    nerrors = VNA.command(
        get_cmd="SYST:ERR:COUN?",
        set_cmd=None,
        doc="""The number of errors since last cleared (see
            :func:`PNA.clear_errors`)""",
        validator=IntValidator(),
    )

    channel_numbers = VNA.command(
        get_cmd="SYST:CHAN:CAT?",
        set_cmd=None,
        doc="""The channel numbers currently in use""",
        validator=DelimitedStrValidator(int),
    )

    @property
    def nports(self) -> int:
        if self._supports("nports"):
            return int(self.query("SYST:CAP:HARD:PORT:COUN?"))
        else:
            return self._model_param("nports")

    @property
    def active_channel(self) -> Channel | None:
        num = int(self.query("SYST:ACT:CHAN?"))
        return getattr(self, f"ch{num}", None)

    @active_channel.setter
    def active_channel(self, ch: Channel) -> None:
        if self.active_channel.cnum == ch.cnum:
            return

        msmnt = ch.measurement_numbers[0]
        self.write(f"CALC{ch.cnum}:PAR:MNUM {msmnt}")

    @property
    def query_format(self) -> vna.ValuesFormat:
        fmt = self.query("FORM?").replace("+", "")
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
            self.write("FORM:BORD SWAP")
            self.write("FORM REAL,32")
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write("FORM:BORD SWAP")
            self.write("FORM REAL,64")

    @property
    def active_measurement(self) -> str:
        return self.query("SYST:ACT:MEAS?").replace('"', "")

    @active_measurement.setter
    def active_measurement(self, name: str) -> None:
        measurements = {
            name: channel
            for channel in self.channels
            for name in channel.measurement_names
        }

        if name not in measurements:
            raise KeyError(f"{name} does not exist")

        if self._supports("fast_sweep"):
            self.write(f"CALC{measurements[name].cnum}:PAR:SEL '{name}',fast")
        else:
            self.write(f"CALC{measurements[name].cnum}:PAR:SEL '{name}'")
