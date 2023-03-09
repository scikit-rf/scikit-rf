from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Sequence

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
    SWEEP = "SWEEP"

class PNA(VNA):
    class Channel(vna.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)

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

        freq_step = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:STEP?",
            set_cmd="SENS<self:cnum>:SWE:STEP <arg>",
            doc="""The frequency step [Hz]. Sets the number of points as a side
                effect
            """,
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
            validator=FloatValidator(),
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
            get_cmd="SENS<self:cnum>:AVER:MODE?",
            set_cmd="SENS<self:cnum>:AVER:MODE <arg>",
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
            get_cmd="SENS<self:cnum>:AVER:COUN?",
            set_cmd="SENS<self:cnum>:AVER:COUN <arg>",
            doc="""The number of measurements combined for an average""", 
            validator=EnumValidator(AveragingMode),
        )

        n_sweep_groups = VNA.command(
            get_cmd="SENS<self:cnum>:SWE:GRO:COUN?",
            set_cmd="SENS<self:cnum>:SWE:GRO:COUN <arg>",
            doc="""The number of triggers sent for one trigger command""", 
            validator=IntValidator(1, int(2E6)),
        )

        active_trace_sdata = vna.VNA.command(
            get_cmd="CALC<self:cnum>:DATA:SDATA?",
            doc="""Get the current trace data as a network""",
            values=True,
        )

        @property
        def frequency(self) -> skrf.Frequency:
            f = skrf.Frequency(
                start=self.freq_start, 
                stop=self.freq_stop,
                npoints=self.npoints,
                unit='hz'
            )
            return f

        @frequency.setter
        def frequency(self, f: skrf.Frequency) -> None:
            self.freq_start = f.start
            self.freq_stop = f.stop
            self.npoints = f.npoints

        @property
        def measurements(self) -> list[tuple[str, str]]:
            msmnts = self.query(f"CALC{self.cnum}:PAR:CAT:EXT?")
            msmnts = msmnts.split(',')
            return list(zip(msmnts[::2], msmnts[1::2]))

        @property
        def measurement_names(self) -> list[str]:
            return [msmnt[0] for msmnt in self.measurements]

        def clear_averaging(self) -> None:
            self.write(f"SENS{self.cnum}:AVER:CLE")

        def create_measurement(self, name: str, parameter: str) -> None:
            self.write(f"CALC{self.cnum}:PAR:EXT '{name}',{parameter}")
            next_tr = int(self.query("DISP:WIND:TRAC:NEXT?"))
            self.write(f"DISP:WIND:TRAC:FEED{next_tr} '{name}'")

        def delete_measurement(self, name: str) -> None:
            self.write(f"CALC{self.cnum}:PAR:DEL '{name}'")
        
        def get_measurement(self, name: str) -> skrf.Network:
            if name not in self.measurement_names:
                raise KeyError(f"{name} does not exist")
                return

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
            ntwk.s = np.empty(
                shape=(ntwk.frequency.npoints, 1, 1), dtype=complex
            )
            raw = self.active_trace_sdata
            ntwk.s[:, 0, 0] = raw[::2] + 1j * raw[1::2]

            self.parent.query_format = orig_query_fmt

            return ntwk

        def get_snp_network(
            self, 
            ports: Optional[Sequence]=None, 
        ) -> skrf.Network:
            if not ports:
                ports = list(range(1, self.parent.n_ports+1))
            
            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            self.parent.active_channel = self

            msmnt_params = [f"S{p}{p}" for p in ports]

            names = []
            # Make sure the ports specified are driven
            for param in msmnt_params:
                name = self.query(f"CALC{self.cnum}:PAR:TAG:NEXT?").strip()
                names.append(name)
                self.create_measurement(name, param)

            self.sweep()
            port_str = ','.join(str(port) for port in ports)
            raw = self.query_values(f"CALC{self.cnum}:DATA:SNP:PORTS? '{port_str}'", container=np.array)
            self.parent.wait_for_complete()

            for name in names:
                self.delete_measurement(name)

            npoints = self.npoints
            nrows = len(raw) // npoints
            nports = int(((nrows - 1) / 2) ** (1 / 2))
            data = np.array(raw)
            data = data.reshape((nrows, -1))[1:]

            ntwk = skrf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = np.empty(
                shape=(len(ntwk.frequency), nports, nports), dtype=complex
            )
            for n in range(nports):
                for m in range(nports):
                    i = n * nports + m
                    ntwk.s[:, n, m] = data[i*2] + 1j * data[i*2+1]

            self.parent.query_format = orig_query_fmt

            return ntwk

        def sweep(self) -> None:
            self.parent.trigger_source = TriggerSource.IMMEDIATE
            self.parent._resource.clear()

            sweep_mode = self.sweep_mode
            sweep_time = self.sweep_time,
            avg_on = self.averaging_on,
            avg_mode = self.averaging_mode

            original_config = {
                'sweep_mode': sweep_mode,
                'sweep_time': sweep_time,
                'averaging_on': avg_on,
                'averaging_mode': avg_mode,
                'timeout': self.parent._resource.timeout
            }

            if avg_on and avg_mode == AveragingMode.SWEEP:
                self.sweep_mode = SweepMode.GROUPS
                n_sweeps = self.averaging_count
                self.n_sweep_groups = n_sweeps
                n_sweeps *= self.parent.n_ports
            else:
                self.sweep_mode = SweepMode.SINGLE
                n_sweeps = self.parent.n_ports

            try:
                sweep_time *= (n_sweeps * 1_000) # 1s per port
                self.parent._resource.timeout = max(sweep_time, 5_000) # minimum of 5s
                self.parent.wait_for_complete()
            finally:
                self.parent._resource.clear()
                self.parent._resource.timeout = original_config.pop('timeout')
                for k,v in original_config.items():
                    setattr(self, k, v)


    def __init__(self, address: str, backend: str = '@py') -> None:
        super().__init__(address, backend)
        self.create_channel(1, "Channel 1")

    trigger_source = VNA.command(
        get_cmd='TRIG:SOUR?',
        set_cmd='TRIG:SOUR <arg>',
        doc="""The source of the sweep trigger signal""",
        validator=EnumValidator(TriggerSource)
    )

    n_ports = VNA.command(
        get_cmd='SYST:CAP:HARD:PORT:COUN?',
        set_cmd=None,
        doc="""Number of ports this instrument has""",
        validator=IntValidator()
    )

    @property
    def active_channel(self) -> Optional[Channel]:
        num = self.query('SYST:ACT:CHAN?')
        return getattr(self, f"ch{num}", None)

    @active_channel.setter
    def active_channel(self, ch: Channel) -> None:
        if self.active_channel.cnum == ch.cnum:
            return

        msmnt = ch.measurement_numbers[0]
        self.write(f'CALC{ch.cnum}:PAR:MNUM {msmnt},fast')

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
            self.write("FORM:BORD SWAP;FORM REAL,32")
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write("FORM:BORD SWAP;FORM REAL,64")

    @property
    def active_measurement(self) -> str:
        return self.query("SYST:ACT:MEAS?")

    @active_measurement.setter
    def active_measurement(self, name: str) -> None:
        measurements = {
            name: channel
            for channel in self.channels 
            for name in channel.measurement_names
        }

        if name not in measurements:
            raise KeyError(f"{name} does not exist")

        self.write(f"CALC{measurements[name].cnum}:PAR:SEL '{name}',fast")
