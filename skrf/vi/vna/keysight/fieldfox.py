from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Union

import pprint

import numpy as np
from skrf.calibration import Calibration
from skrf.network import Network
from skrf.vi.vna import VNA, Measurement


class FieldFox(VNA):
    def __init__(self, address: str, timeout: int = 2000, backend: str = "@py"):
        super().__init__(address, timeout, backend)

    def start_freq(self) -> float:
        return float(self.query("sense:frequency:start?"))

    def set_start_freq(self, f: float) -> None:
        self.write(f"sense:frequency:start {f}")

    def stop_freq(self) -> float:
        return float(self.query("sense:frequency:stop?"))

    def set_stop_freq(self, f: float) -> None:
        self.write(f"sense:frequency:stop {f}")

    def npoints(self) -> int:
        return int(self.query("sense:sweep:points?"))

    def set_npoints(self, n: int) -> None:
        self.write(f"sense:sweep:points {n}")

    def freq_step(self) -> float:
        return float(self.query("sense:sweep:step?"))

    def set_freq_step(self, f: float) -> None:
        if "LIN" not in self.sweep_type().upper():
            raise ValueError("Can only set frequency step in linear sweep mode")
        self.write(f"sense:sweep:step {f}")

    def sweep_mode(self) -> str:
        resp = self.query("initate:continuous?")
        return "Continuous" if resp else "Single"

    def set_sweep_mode(self, mode: str) -> None:
        if mode.lower() not in ["single", "continuous"]:
            raise ValueError(f"Unrecognized sweep mode: {mode}")
        self.write(f"initiate:continuous {0 if mode.lower() == 'single' else 1}")

    def sweep_type(self) -> str:
        raise NotImplementedError

    def set_sweep_type(self, _type: str) -> None:
        raise NotImplementedError

    def sweep_time(self) -> float:
        return float(self.query("sense:sweep:time?"))

    def set_sweep_time(self, time: float) -> None:
        self.write(f"sense:sweep:time {time}")

    def if_bandwidth(self) -> float:
        return float(self.query("sense:bwidth?"))

    def set_if_bandwidth(self, bw: float) -> None:
        self.write(f"sense:bwidth {bw}")

    def averaging_on(self) -> bool:
        return int(self.query("sense:average:count?")) != 1

    def set_averaging_on(self, onoff: bool) -> None:
        if onoff is False:
            self.write("sense:average:count 1")
        else:
            raise NotImplementedError(
                "No command to turn averaging on and off. Set average count instead"
            )

    def average_count(self) -> int:
        return int(self.query("sense:average:count?"))

    def set_average_count(self, n: int) -> None:
        self.write(f"sense:average:count {n}")

    def average_mode(self) -> str:
        return self.query("sense:average:mode?")

    def set_average_mode(self, mode: str) -> None:
        if mode.lower() not in ["sweep", "point"]:
            raise ValueError(f"Unrecognized averaging mode: {mode}")
        self.write(f"sense:average:mode {mode.lower()}")

    def clear_averaging(self) -> None:
        self.write("sense:average:clear")

    def num_sweep_groups(self) -> int:
        raise NotImplementedError

    def set_num_sweep_groups(self, n: int) -> None:
        raise NotImplementedError

    @property
    def channels_in_use(self) -> List[int]:
        raise NotImplementedError("FieldFox doesn't have channels")

    @property
    def active_channel(self) -> Optional[int]:
        raise NotImplementedError("FieldFox doesn't have channels")

    @active_channel.setter
    def active_channel(self, channel: int) -> None:
        raise NotImplementedError("FieldFox doesn't have channels")

    def measurements_on_channel(self, channel: int = 1) -> List[Measurement]:
        raise NotImplementedError("FieldFox doesn't have channels")

    def measurement_numbers(self) -> List[int]:
        resp = int(self.query("calculate:parameter:count?"))
        return [i + 1 for i in range(resp)]

    @property
    def active_measurement(self) -> Optional[str]:
        raise NotImplementedError

    def set_active_measurement(
        self, meas: Union[int, str], channel: int = 1, fast: bool = False
    ) -> None:
        raise NotImplementedError

    def create_measurement(self, name: str, param: str, trace: int = 1) -> None:
        if (trace < 1) or (trace > 4):
            raise ValueError("Trace must be between 1 and 4")
        if param.lower() not in ["s11", "s12", "s21", "s22", "a", "b", "r1", "r2"]:
            raise ValueError(f"Unrecognized measurement type: {param}")

        self.write(f"calculate:parameter{trace}:define {param}")

    def delete_measurement(self, name: str, channel: int = 1) -> None:
        raise NotImplementedError

    def get_measurement(self, trace: int) -> Network:
        self.write(f"calculate:parameter{trace}:select")
        raw = np.array(self.query_ascii("calculate:data:sdata?"), dtype=np.complex64)
        self.query("*OPC?")
        ntwk = Network()
        ntwk.frequency = self.frequency()
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    def get_active_trace(self) -> Network:
        raw = np.array(self.query_ascii("calculate:data:sdata?"), dtype=np.complex64)
        self.query("*OPC?")
        ntwk = Network()
        ntwk.frequency = self.frequency()
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    @property
    def snp_format(self) -> str:
        raise NotImplementedError

    @snp_format.setter
    def snp_format(self, format: str) -> None:
        raise NotImplementedError

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

    def get_snp_network(self) -> Network:
        # TODO: Seems like the best way to do this would be to create 4
        # measurements capture all of their data, and turn into a 2 port network
        raise NotImplementedError

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        raise NotImplementedError
