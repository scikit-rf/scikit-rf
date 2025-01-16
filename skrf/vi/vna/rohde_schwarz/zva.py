from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import sys
from enum import Enum, auto

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
    Linear = "LIN"
    Log = "LOG"
    Segment = "SEGM"
    Power = "POW"
    Cw = "CW"
    Point = "POIN"
    Pulse = "PULS"
    AmplitudeImbalance = "IAMP"
    PhaseImbalance = "IPH"


class SweepMode(Enum):
    Single = auto()
    Continuous = auto()


class ZVA(VNA):
    """
    Rohde & Schwarz ZVA.

    ZVA Models
    ==========
    ZVA40, ..., others

    """

    _models = {
        "default": {"nports": 2, "unsupported": []},
    }

    class Channel(vna.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)

            if cnum != 1:
                default_msmnt = f"CH{self.cnum}_S11_1"
                self.create_measurement(default_msmnt, "S11")

        def _on_delete(self):
            self.write(f"CONF:CHAN{self.cnum} OFF")

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
            doc="""The frequency center [Hz].""",
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

        sweep_step = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:STEP?",
            set_cmd="SENS<self:cnum>:SWE:STEP <arg>",
            doc="""The frequency step size [Hz]""",
            validator=FreqValidator(),
        )

        sweep_time = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:TIME?",
            set_cmd="SENS<self:cnum>:SWE:TIME <arg>",
            doc="""The time in seconds for a single sweep [s]""",
            validator=FloatValidator(),
        )

        sweep_type = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:TYPE?",
            set_cmd="SENS<self:cnum>:SWE:TYPE <arg>",
            doc="""The type of sweep (linear, log, etc)""",
            validator=EnumValidator(SweepType),
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
            msmnts = self.query(f"CALC{self.cnum}:PAR:CAT?").split(",")
            return list(zip(msmnts[::2], msmnts[1::2]))

        @property
        def measurement_names(self) -> list[str]:
            return [msmnt[0] for msmnt in self.measurements]

        @property
        def calibration(self) -> skrf.Calibration:
            raise NotImplementedError()

        @calibration.setter
        def calibration(self, cal: skrf.Calibration) -> None:
            raise NotImplementedError()

        @property
        def sweep_mode(self) -> SweepMode:
            resp = self.query(f"INIT{self.cnum}:CONT?").lower()
            return SweepMode.Continuous if resp == "on" else SweepMode.Single

        @sweep_mode.setter
        def sweep_mode(self, mode: SweepMode) -> None:
            if mode == SweepMode.Continuous:
                state = "ON"
            elif mode == SweepMode.Single:
                state = "OFF"
            self.write(f"INIT{self.cnum}:CONT {state}")

        @property
        def active_trace_sdata(self) -> np.ndarray:
            active_measurement = self.query(f"CALC{self.cnum}:PAR:SEL?").replace("'", "").split(",")[0]
            if active_measurement == "":
                raise RuntimeError("No trace is active. Must select measurement first.")
            return self.query_values(f"CALC{self.cnum}:DATA? SDATA", complex_values=True)

        def clear_averaging(self) -> None:
            self.write(f"SENS{self.cnum}:AVER:CLE")

        def create_measurement(self, name: str, parameter: str) -> None:
            self.write(f"CALC{self.cnum}:PAR:SDEF '{name}',{parameter}")
            traces = self.query("DISP:WIND:TRAC:CAT?")
            traces = [int(tr) for tr in traces.split(",")] if traces != "" else [0]
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

        def create_sparam_group(self, ports: Sequence[int]) -> None:
            self.write(f"CALC{self.cnum}:PAR:DEF:SGR {','.join(ports)}")

        def get_snp_network(
            self,
            ports: Sequence | None = None,
        ) -> skrf.Network:
            if ports is None:
                ports = list(range(1, self.parent.nports + 1))

            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            self.parent.active_channel = self

            self.create_sparam_group(ports)

            self.sweep()

            raw = self.query_values(f"CALC{self.cnum}:DATA:SGR SDAT", container=np.array, complex_values=True)
            self.parent.wait_for_complete()

            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = raw.reshape((-1, len(ports), len(ports)))

            self.parent.query_format = orig_query_fmt

            return ntwk

        def sweep(self) -> None:
            orig_sweep_mode = self.sweep_mode
            self.sweep_mode = SweepMode.IMMEDIATE
            self.parent._resource.clear()

            self.write(f"INIT{self.cnum}:IMM")
            self.parent.wait_for_complete()
            self.sweep_mode = orig_sweep_mode

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

    channels = VNA.command(
        get_cmd="CONF:CHAN<self:cnum>:CAT?",
        set_cmd=None,
        doc="""The channel numbers and names currently in use""",
        validator=DelimitedStrValidator(),
    )

    @property
    def nports(self) -> int:
        if self._supports("nports"):
            return int(self.query("INST:PORT:COUN?"))
        else:
            return self._model_param("nports")

    @property
    def active_channel(self) -> Channel | None:
        active_ch = int(self.query("INST:NSEL?"))
        return next(ch for ch in self.channels if ch.cnum == active_ch)

    @active_channel.setter
    def active_channel(self, ch: Channel) -> None:
        if self.active_channel.cnum == ch.cnum:
            return

        self.write(f"INST:NSEL {ch.cnum}")

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
            self.write("FORM:BORD SWAP")
            self.write("FORM REAL,32")
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write("FORM:BORD SWAP")
            self.write("FORM REAL,64")

    @property
    def active_measurement(self) -> str:
        return self.query(f"CALC{self.cnum}:PAR:SEL?")

    @active_measurement.setter
    def active_measurement(self, name: str) -> None:
        measurements = {name: channel for channel in self.channels for name in channel.measurement_names}

        if name not in measurements:
            raise KeyError(f"{name} does not exist")

        self.write(f"CALC{measurements[name].cnum}:PAR:SEL '{name}'")
