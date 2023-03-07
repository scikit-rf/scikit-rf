from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

from enum import Enum

import skrf
from skrf.vi import vna
from skrf.vi.validators import (
    DelimitedStrValidator,
    EnumValidator,
    FloatValidator,
    FreqValidator,
    IntValidator,
)
from skrf.vi.vna import VNA


class SweepType(Enum):
    LINEAR = "LIN"
    LOG = "LOG"
    POWER = "POW"
    CW = "CW"
    SEGMENT = "SEGM"
    PHASE = "PHAS"


class PNA(VNA):
    class Channel(vna.Channel):
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
            doc="""The time in seconds for a single sweep [s]""",
            validator=EnumValidator(SweepType),
        )

        measurements = VNA.command(
            get_cmd="SYST:MEAS:CAT? <self:cnum>",
            set_cmd=None,
            doc="""The time in seconds for a single sweep [s]""",
            validator=DelimitedStrValidator(int),
        )

        @property
        def frequency(self) -> skrf.Frequency:
            f = skrf.Frequency(
                start=self.freq_start, stop=self.freq_stop, npoints=self.npoints
            )
            return f

        @frequency.setter
        def frequency(self, f: skrf.Frequency) -> None:
            self.freq_start = f.start
            self.freq_stop = f.stop
            self.npoints = f.npoints

    def __init__(self, address: str, backend: str) -> None:
        self.create_channel(1, "Channel 1")

    @property
    def active_channel(self) -> Optional[Channel]:
        num = self.query('SYST:ACT:CHAN?')
        return getattr(self, f"ch{num}", None)

    @active_channel.setter
    def active_channel(self, ch: Channel) -> None:
        if self.active_channel.cnum == ch.cnum:
            return

        msmnt = ch.measurements[0]
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
            self.write("FORM REAL,32")
        elif fmt == vna.ValuesFormat.BINARY_64:
            self._values_fmt = vna.ValuesFormat.BINARY_64
            self.write("FORM REAL,64")