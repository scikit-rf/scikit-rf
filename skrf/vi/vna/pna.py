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
    def channels_in_use(self) -> List[int]:
        response = self.query("system:channels:catalog?")
        response = response.strip().replace('"', '')
        return [int(c) for c in response]

    def measurements_on_channel(self, channel: int) -> List[int]:
        channels = self.channels_in_use
        if channel not in channels:
            return list()
        else:
            response = self.query(f"system:measurement:catalog? {channel}")
            response = response.strip().replace('"', '').split(',')
            return [int(m) for m in response]

    @property
    def active_channel(self) -> Optional[int]:
        active = self.query("system:active:channel?")
        if active == "0":
            return None
        else:
            return int(active)

    @active_channel.setter
    def active_channel(self, channel: int) -> None:
        channels = self.channels_in_use
        if channel not in channels:
            raise IndexError(f"Channel {channel} doesn't exist")
        else:
            # No command to set active channel, so get first measurement on `channel` 
            # and set the selected measurement to that to hack set the active channel
            mnum = self.measurements_on_channel(channel)[0]
            self.active_measurement = mnum

    @property
    def active_measurement(self) -> Optional[str]:
        active = self.query("system:active:measurement?")
        return None if active == "" else active

    @active_measurement.setter
    def active_measurement(self, mnum: int) -> None:
        self.write(f"calculate:parameter:mnumber {mnum}")

    def measurement_numbers(self, chan: int) -> str:
        return self.query(f"system:measurement:catalog? {chan}")

    @property
    def snp_format(self) -> str:
        return self.query("mmemory:store:trace:format:snp?").strip()

    @snp_format.setter
    def snp_format(self, format: str) -> None:
        if format.lower() not in ["ma", "db", "ri", "auto"]:
            raise ValueError(f"Invalid SNP format: {format.upper()}")
        self.write(f"mmemory:store:trace:format:snp {format}")

    @property
    def ports(self) -> List:
        query = "system:capability:hardware:ports:internal:catalog?"
        ports = self.query(query).split(',')
        return [p.lstrip('"').rstrip('"\n')[-1] for p in ports]

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

    def create_measurement(self, name: str, measurement: str) -> None:
        self.write(f"calculate:parameter:extended '{name}', '{measurement}'")

    # OPTIMIZE: create and getting measurement data might be faster than 
    # e.g. getting a whole four port network
    def get_snp_network(self, ports: Sequence = (1, 2)) -> Network:
        self.resource.clear()

        old_snp_format = self.snp_format
        port_str = ",".join(map(str, ports))
        self.snp_format = "RI"
        self.active_channel = 1
        raw_data = self.query_ascii(
            f"calculate:data:snp:ports? '{port_str}'", container=np.array
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
        # Can we speed this up?
        for n in range(nports):
            for m in range(nports):
                i = n * nports + m
                ntwk.s[:, n, m] = s_data[i * 2] + 1j * s_data[i * 2 + 1]

        return ntwk


    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        pass

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        pass
