import pprint
from typing import List, Optional, Sequence

import numpy as np

from ...calibration import Calibration
from ...frequency import Frequency
from ...network import Network
from .vna import VNA


class PNA(VNA):
    def __init__(self, address: str, timeout: int = 2000, backend: str='@py'):
        super().__init__(address, timeout, backend)

    @property
    def start_freq(self) -> float:
        return float(self.query("sense:frequency:start?"))

    @start_freq.setter
    def start_freq(self, f: float) -> None:
        self.write(f"sense:frequency:start {f}")

    @property
    def stop_freq(self) -> float:
        return float(self.query("sense:frequency:stop?"))

    @stop_freq.setter
    def stop_freq(self, f: float) -> None:
        self.write(f"sense:frequency:stop {f}")

    @property
    def npoints(self) -> int:
        return int(self.query("sense:sweep:points?"))

    @npoints.setter
    def npoints(self, n: int) -> None:
        self.write(f"sense:sweep:points {n}")


    @property
    def sweep_type(self) -> str:
        return self.query("sense:sweep:type?")

    @sweep_type.setter
    def sweep_type(self, _type: str) -> None:
        self.write(f"sense:sweep:type {_type}")

    @property
    def sweep_time(self) -> float:
        return float(self.query("sense:sweep:time?"))

    @sweep_time.setter
    def sweep_time(self, time: float) -> None:
        self.write(f"sense:sweep:time {time}")

    @property
    def if_bandwidth(self) -> float:
        return float(self.query("sense:bandwidth?"))

    @if_bandwidth.setter
    def if_bandwidth(self, bw: float) -> None:
        self.write(f"sense:bandwidth {bw}")

    @property  # ???
    def averaging(self) -> bool:
        return bool(self.query("sense:average?"))

    @averaging.setter
    def averaging(self, onoff: bool) -> None:
        self.write(f"sense:average {int(onoff)}")

    @property
    def average_count(self) -> int:
        return int(self.query("sense:average:count?"))

    @average_count.setter
    def average_count(self, n: int) -> None:
        self.write(f"sense:average:count {n}")

    def clear_averaging(self) -> None:
        self.write("sense:average:clear")

    @property
    def num_sweep_groups(self) -> int:
        return int(self.query("sense:sweep:groups:count?"))

    @num_sweep_groups.setter
    def num_sweep_groups(self, n: int) -> None:
        self.write(f"sense:sweep:groups:count {n}")

    @property
    def channels_in_use(self) -> str:
        return self.query("system:channels:catalog?")

    @property
    def active_channel(self) -> Optional[int]:
        active = self.query("system:active:channel?")
        if active == "0":
            return None
        else:
            return int(active)

    @property
    def active_measurement(self) -> Optional[str]:
        active = self.query("system:active:measurement?")
        return None if active == "" else active

    @property
    def measurement_numbers(self) -> str:
        return self.query("system:measurement:catalog?")

    @property
    def snp_format(self) -> str:
        return self.query("mmemory:store:trace:format:snp?")

    @snp_format.setter
    def snp_format(self, format: str) -> None:
        if format.lower() not in ["ma", "db", "ri", "auto"]:
            raise ValueError(f"Invalid SNP format: {format.upper()}")
        self.write(f"mmemory:store:trace:format:snp {format}")

    @property
    def ports(self) -> List:
        query = "system:capability:hardware:ports:internal:catalog?"
        return self.query(query).split(",")

    @property
    def sysinfo(self) -> str:
        def capability(setting: str) -> str:
            return self.query(f"system:capability:{setting}?")

        info = {
            "ID": self.id_string,
            "Ports": capability("hardware:ports:internal:catalog").split(","),
            "Frequency": {
                "Minimum": capability("frequency:minimum"),
                "Maximum": capability("frequency:maximum"),
            },
            "IF Bandwidth": capability("ifbw:catalog").split(","),
            "Power": {
                "Minimum": capability("alc:power:minimum"),
                "Maximum": capability("alc:power:maximum"),
            },
        }

        return pprint.pformat(info)

    def get_snp_network(self, ports: Sequence = (1, 2)) -> Network:
        self.resource.clear()

        old_snp_format = self.snp_format
        port_str = ",".join(map(str, ports))
        self.snp_format = "RI"
        raw_data = self.query_ascii(
            f"calculate:data:snp:ports? {port_str}", container=np.array
        )
        self.query("*OPC?")
        self.snp_format = old_snp_format

        npoints = self.npoints
        nrows = len(raw_data) // npoints
        nports = int(((nrows - 1) / 2) ** (1 / 2))
        data = np.array(raw_data)
        data = data.reshape((nrows, -1))

        freq_data = data[0]
        s_data = data[1:]
        ntwk = Network()
        ntwk.frequency = Frequency.from_f(freq_data, unit="hz")
        ntwk.s = np.empty(shape=(s_data.shape[1], nports, nports), dtype=np.complex64)
        for n in range(nports):
            for m in range(nports):
                i = n * nports + m
                ntwk.s[:, m, n] = s_data[i * 2] + 1j * s_data[i * 2 + 1]

        return ntwk

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        pass

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        pass
