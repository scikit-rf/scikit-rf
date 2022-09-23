from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Union

import itertools
import pprint

import numpy as np
from skrf import Calibration, Frequency, Network
from skrf.network import four_oneports_2_twoport

from ..vna import VNA


class FieldFox(VNA):
    """
    Keysight's FieldFox analyzer.

    This class assumes the FieldFox has been put into NA mode before initializing
    """

    @property
    def start_freq(self) -> float:
        return float(self.query("sense:frequency:start?"))

    @start_freq.setter
    def start_freq(self, freq: float) -> None:
        self.write(f"sense:frequency:start {freq}")

    @property
    def stop_freq(self) -> float:
        return float(self.query("sense:frequency:stop?"))

    @stop_freq.setter
    def stop_freq(self, freq: float) -> None:
        self.write(f"sense:frequency:stop {freq}")

    @property
    def npoints(self) -> int:
        return int(self.query("sense:sweep:points?"))

    @npoints.setter
    def npoints(self, n: int) -> None:
        self.write(f"sense:sweep:points {n}")

    @property
    def freq_step(self) -> float:
        return float(self.query("sense:sweep:step?"))

    @freq_step.setter
    def freq_step(self, f: float) -> None:
        if "lin" not in self.sweep_type.lower():
            raise ValueError("Can only set frequency step in linear sweep mode")

        self.write(f"sense:sweep:step {f}")

    @property
    def frequency(self) -> Frequency:
        return Frequency(
            start=self.start_freq, stop=self.stop_freq, npoints=self.npoints, unit="Hz"
        )

    @frequency.setter
    def frequency(self, frequency: Frequency) -> None:
        start = frequency.start
        stop = frequency.stop
        npoints = frequency.npoints

        self.write(
            f"sense:frequency:start {start};"
            f"frequency:stop {stop};"
            f"sweep:points {npoints}"
        )

    @property
    def sweep_time(self) -> float:
        return float(self.query("sense:sweep:time?"))

    @sweep_time.setter
    def sweep_time(self, time: float) -> None:
        self.write(f"sense:sweep:time {time}")

    @property
    def if_bandwidth(self) -> float:
        return float(self.query("sense:bwidth?"))

    @if_bandwidth.setter
    def if_bandwidth(self, bw: float) -> None:
        self.write(f"sense:bwidth {bw}")

    @property
    def averaging_on(self) -> bool:
        return self.average_count != 1

    @averaging_on.setter
    def averaging_on(self, state: bool) -> None:
        if state is False:
            self.average_count = 1
        else:
            raise NotImplementedError(
                "No command to turn averaging on directly. Set average_count instead"
            )

    @property
    def average_count(self) -> int:
        return int(self.query("sense:average:count?"))

    @average_count.setter
    def average_count(self, count: int) -> None:
        self.write(f"sense:average:count {count}")

    @property
    def average_mode(self) -> str:
        return self.query("sense:average:mode?")

    @average_mode.setter
    def average_mode(self, mode: str) -> None:
        if mode.lower() not in ["swe", "sweep", "point"]:
            raise ValueError(f"Unrecognized averaging mode: {mode}")
        self.write(f"sense:average:mode {mode}")

    def clear_averaging(self) -> None:
        self.write("sense:average:clear")

    def measurement_numbers(self) -> List[int]:
        resp = int(self.query("calculate:parameter:count?"))
        return [i + 1 for i in range(resp)]

    @property
    def active_measurement(self) -> Optional[str]:
        return self.query("calculate:parameter:define?")

    @active_measurement.setter
    def active_measurement(self, id_: Union[int, str]) -> None:
        assert isinstance(id_, int), "FieldFox only supports measurement numbers"
        self.write(f"calculate:parameter{id_}:select")

    def create_measurement(self, id_: Union[str, int], param: str) -> None:
        assert isinstance(id_, int), "FieldFox only supports measurement numbers"
        if (id_ < 1) or (id_ > 4):
            raise ValueError("Trace must be between 1 and 4")
        if param.lower() not in ["s11", "s12", "s21", "s22", "a", "b", "r1", "r2"]:
            raise ValueError(f"Unrecognized measurement type: {param}")

        self.write(f"calculate:parameter{id_}:define {param}")

    def delete_measurement(self, id_: Union[str, int]) -> None:
        # TODO: Check this is correct in testing
        assert isinstance(id_, int), "FieldFox only supports measurement numbers"
        self.write(f"display:window:trace{id_}:trace:state 0")

    def get_measurement(self, trace: int) -> Network:
        self.active_measurement = trace
        raw = np.array(self.query_ascii("calculate:data:sdata?"), dtype=np.complex64)
        self.query("*OPC?")
        ntwk = Network()
        ntwk.frequency = self.frequency
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    def get_active_trace(self) -> Network:
        raw = np.array(self.query_ascii("calculate:data:sdata?"), dtype=np.complex64)
        self.query("*OPC?")
        ntwk = Network()
        ntwk.frequency = self.frequency
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    @property
    def ports(self) -> List[str]:
        return ["1", "2"]

    @property
    def sysinfo(self) -> str:
        info = {"ID": self.id_string, "SCPI Version": self.query("system:version?")}

        return pprint.pformat(info)

    # TODO: Implement for multiple channels
    def sweep(self) -> None:
        self.resource.clear()
        self.write("initiate:immediate")
        self.write("*OPC?")

    def get_snp_network(self, ports: Optional[Sequence]) -> Network:
        self.resource.clear()

        if not ports:
            ports = self.ports

        _ports = set(ports)
        msmts = itertools.product(_ports, repeat=2)
        params = [f"S{a}{b}" for a, b in msmts]
        # Create all measurements
        self.write("display:window:split D12_34")
        for i, param in enumerate(params):
            self.create_measurement(i, param)

        self.sweep()

        data = dict()
        for i, (n, m) in enumerate(msmts):
            data[(n, m)] = self.get_measurement(i)

        ntwk = four_oneports_2_twoport(
            s11=data[(1, 1)], s12=data[(1, 2)], s21=data[(2, 1)], s22=data[(2, 2)]
        )

        return ntwk

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        raise NotImplementedError
