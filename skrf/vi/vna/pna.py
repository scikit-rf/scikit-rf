import pprint
from multiprocessing.sharedctypes import Value
from typing import List, Optional, Sequence, Union

import numpy as np

from ...calibration import Calibration
from ...frequency import Frequency
from ...network import Network
from .vna import VNA, Measurement


class PNA(VNA):
    def __init__(self, address: str, timeout: int = 2000, backend: str = "@py"):
        super().__init__(address, timeout, backend)

    def start_freq(self, channel: int = 1) -> float:
        return float(self.query("sense{channel}:frequency:start?"))

    def set_start_freq(self, f: float, channel: int = 1) -> None:
        self.write(f"sense{channel}:frequency:start {f}")

    def stop_freq(self, channel: int = 1) -> float:
        return float(self.query("sense{channel}:frequency:stop?"))

    def set_stop_freq(self, f: float, channel: int = 1) -> None:
        self.write(f"sense{channel}:frequency:stop {f}")

    def npoints(self, channel: int = 1) -> int:
        return int(self.query("sense{channel}:sweep:points?"))

    def set_npoints(self, n: int, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:points {n}")

    def freq_step(self, channel: int = 1) -> float:
        return float(self.query(f"sense{channel}:sweep:step?"))

    def set_freq_step(self, f: float, channel: int = 1) -> None:
        if "LIN" not in self.sweep_type(channel).upper():
            raise ValueError("Can only set frequency step in linear sweep mode")
        self.write(f"sense{channel}:sweep:step {f}")

    def sweep_mode(self, channel: int = 1) -> str:
        return self.query("sense{channel}:sweep:mode?")

    def set_sweep_mode(self, mode: str, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:mode {mode}")

    def sweep_type(self, channel: int = 1) -> str:
        return self.query("sense{channel}:sweep:type?")

    def set_sweep_type(self, _type: str, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:type {_type}")

    def sweep_time(self, channel: int = 1) -> float:
        return float(self.query("sense{channel}:sweep:time?"))

    def set_sweep_time(self, time: float, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:time {time}")

    def if_bandwidth(self, channel: int = 1) -> float:
        return float(self.query("sense{channel}:bandwidth?"))

    def set_if_bandwidth(self, bw: float, channel: int = 1) -> None:
        self.write(f"sense{channel}:bandwidth {bw}")

    def averaging_on(self, channel: int = 1) -> bool:
        avg = self.query("sense{channel}:average?").strip()
        return avg != "0"

    def set_averaging_on(self, onoff: bool, channel: int = 1) -> None:
        self.write(f"sense{channel}:average {int(onoff)}")

    def average_count(self, channel: int = 1) -> int:
        return int(self.query("sense{channel}:average:count?"))

    def set_average_count(self, n: int, channel: int = 1) -> None:
        self.write(f"sense{channel}:average:count {n}")

    def average_mode(self, channel: int = 1) -> str:
        return self.query("sense{channel}:average:mode?")

    def set_average_mode(self, mode: str, channel: int = 1) -> None:
        self.write("sense{channel}:average:mode {mode}")

    def clear_averaging(self, channel: int = 1) -> None:
        self.write("sense{channel}:average:clear")

    def num_sweep_groups(self, channel: int = 1) -> int:
        return int(self.query("sense{channel}:sweep:groups:count?"))

    def set_num_sweep_groups(self, n: int, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:groups:count {n}")

    @property
    def channels_in_use(self) -> List[int]:
        response = self.query("system:channels:catalog?")
        response = response.strip().replace('"', "")
        return [int(c) for c in response]

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
            msmts = self.measurements_on_channel(channel)
            self.set_active_measurement(msmts[0].name, channel)

    def measurements_on_channel(self, channel: int = 1) -> List[Measurement]:
        channels = self.channels_in_use
        if channel not in channels:
            return []
        else:
            response = self.query(f"calculate{channel}:parameter:catalog:extended?")
            response = response.strip().replace('"', "").split(",")
            names = response[::2]
            params = response[1::2]
            return [
                Measurement(channel, name, param)
                for (name, param) in zip(names, params)
            ]

    def measurement_numbers(self, channel: int = 1) -> List[int]:
        ret = self.query(f"system:measurement:catalog? {channel}").strip()
        return list(map(int, ret.split(",")))

    @property
    def active_measurement(self) -> Optional[str]:
        active = self.query("system:active:measurement?")
        return None if active == "" else active

    def set_active_measurement(
        self, meas: Union[int, str], channel: int = 1, fast: bool = False
    ) -> None:
        fast_str = "fast" if fast else ""
        if isinstance(meas, int):
            if meas not in self.measurement_numbers(channel):
                raise KeyError(f"Measurement doesn't exist on channel {channel}")
            self.write(f"calculate{channel}:parameter:mnumber {meas},{fast_str}")
        elif isinstance(meas, str):
            if meas not in self.measurement_numbers(channel):
                raise KeyError(f"Measurement doesn't exist on channel {channel}")
            self.write(f"calculate{channel}:parameter:select '{meas}',{fast_str}")

    def create_measurement(self, name: str, param: str, channel: int = 1) -> None:
        self.write(f"calculate{channel}:parameter:extended '{name}', '{param}'")

    def delete_measurement(self, name: str, channel: int = 1) -> None:
        self.write(f"calculate{channel}:parameter:delete {name}")

    def get_measurement(self, meas: Union[int, str], channel: int = 1) -> Network:
        self.set_active_measurement(meas, channel, True)
        raw = np.array(
            self.query_ascii("calculate{channel}:data? sdata"), dtype=np.complex64
        )
        ntwk = Network()
        ntwk.frequency = self.frequency(channel)
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    def get_active_trace(self) -> Network:
        assert self.active_channel is not None, "No channel is active"
        raw = np.array(self.query_ascii("calculate{self.active_channel}:data? sdata"))
        ntwk = Network()
        ntwk.frequency = self.frequency(self.active_channel)
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    @property
    def snp_format(self) -> str:
        return self.query("mmemory:store:trace:format:snp?").strip()

    @snp_format.setter
    def snp_format(self, format: str) -> None:
        if format.lower() not in ["ma", "db", "ri", "auto"]:
            raise ValueError(f"Invalid SNP format: {format.upper()}")
        self.write(f"mmemory:store:trace:format:snp {format}")

    @property
    def ports(self) -> List[str]:
        query = "system:capability:hardware:ports:internal:catalog?"
        ports = self.query(query).split(",")
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

    # TODO: Implement for multiple channels
    def sweep(self, channel: int = 1) -> None:
        self.resource.clear()
        orig_timeout = self.resource.timeout
        self.write("trigger:source immediate")
        sweep_mode = self.sweep_mode()
        continuous = "CONT" in sweep_mode.upper()
        sweep_time = self.sweep_time()
        avg_on = self.averaging_on()
        avg_mode = self.average_mode()

        if avg_on and "SWE" in avg_mode.upper():
            sweep_mode = "groups"
            n_sweeps = self.average_count(channel)
            self.set_num_sweep_groups(n_sweeps, channel)
            n_sweeps *= 4
        else:
            sweep_mode = "single"
            n_sweeps = 4

        try:
            sweep_time = sweep_time * n_sweeps * 1000
            self.resource.timeout = max(sweep_time * 2, 5000)
            self.set_sweep_mode(sweep_mode, channel)
            self.query("*OPC?")
        finally:
            self.resource.clear()
            if continuous:
                self.set_sweep_mode("continuous", channel)
            self.resource.timeout = orig_timeout

    def get_snp_network(
        self, ports: Optional[Sequence] = None, channel: int = 1
    ) -> Network:
        self.resource.clear()

        if not ports:
            ports = self.ports

        self.active_channel = channel
        msmts = self.measurements_on_channel(channel)
        params = [m.param for m in msmts]
        if not all([f"S{p}{p}" in params for p in ports]):
            raise KeyError(
                "Missing measurement. Must create S_ii for all i's in ports."
            )
        self.sweep(channel=channel)

        old_snp_format = self.snp_format
        port_str = ",".join(map(str, ports))
        self.snp_format = "RI"
        raw_data = self.query_ascii(
            f"calculate{channel}:data:snp:ports? '{port_str}'", container=np.array
        )
        self.query("*OPC?")
        self.snp_format = old_snp_format

        npoints = self.npoints()
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
