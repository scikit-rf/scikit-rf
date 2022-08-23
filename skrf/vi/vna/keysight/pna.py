from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

import pprint

import numpy as np
from skrf.calibration import Calibration
from skrf.frequency import Frequency
from skrf.network import Network

from ..vna import VNA


class PNA(VNA):
    def __init__(self, address: str, timeout: int = 2000, backend: str = "@py"):
        super().__init__(address, timeout, backend)

    def set_freq_step(self, f: float, channel: int = 1) -> None:
        if "lin" not in self.sweep_type(channel).lower():
            raise ValueError("Can only set frequency step in linear sweep mode")

        super().set_freq_step(f, channel)

    def average_mode(self, channel: int = 1) -> str:
        return self.query(f"sense{channel}:sweep:average:mode?")

    def set_average_mode(self, mode: str, channel: int = 1) -> None:
        if mode.lower() not in ["point, sweep"]:
            raise ValueError(f"Unrecognized averaging mode: {mode}")
        self.write(f"sense{channel}:sweep:average:mode {mode}")

    def num_sweep_groups(self, channel: int = 1) -> int:
        return int(self.query(f"sense{channel}:sweep:groups:count?"))

    def set_num_sweep_groups(self, n: int, channel: int = 1) -> None:
        self.write(f"sense{channel}:sweep:groups:count {n}")

    @property
    def channels_in_use(self) -> List[int]:
        response = self.query("system:channels:catalog?").strip().replace('"', "")
        return [int(c) for c in response.split(",")]

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
            self.set_active_measurement(msmts[0][0], channel)

    def measurements_on_channel(self, channel: int = 1) -> List[Tuple]:
        channels = self.channels_in_use
        if channel not in channels:
            return []
        else:
            response = self.query(f"calculate{channel}:parameter:catalog:extended?")
            response = response.strip().replace('"', "").split(",")
            names = response[::2]
            params = response[1::2]
            return [(name, param, channel) for (name, param) in zip(names, params)]

    def measurement_numbers(self, channel: int = 1) -> List[int]:
        ret = self.query(f"system:measurement:catalog? {channel}").strip().strip('"')
        return list(map(int, ret.split(",")))

    def measurements(self) -> List[Tuple]:
        msmnts = []
        chans = self.channels_in_use
        for chan in chans:
            msmnts.extend(self.measurements_on_channel(chan))

        return msmnts

    def active_measurement(self) -> Optional[Union[str, int]]:
        active = self.query("system:active:measurement?")
        return None if active == "" else active

    def set_active_measurement(
        self, id_: Union[int, str], channel: int = 1, fast: bool = False
    ) -> None:
        fast_str = "fast" if fast else ""
        if isinstance(id_, int):
            if id_ not in self.measurement_numbers(channel):
                raise KeyError(f"Measurement doesn't exist on channel {channel}")
            self.write(f"calculate{channel}:parameter:mnumber {id_},{fast_str}")
        elif isinstance(id_, str):
            meas_names = [m[0] for m in self.measurements_on_channel(channel)]
            if id_ not in meas_names:
                raise KeyError(f"Measurement doesn't exist on channel {channel}")
            self.write(f"calculate{channel}:parameter:select '{id_}',{fast_str}")

    def create_measurement(
        self, id_: Union[str, int], param: str, channel: int = 1
    ) -> None:
        self.write(f"calculate{channel}:parameter:extended '{id_}', '{param}'")
        next_tr = int(self.query("display:window:trace:next?").strip())
        self.write(f"display:window:trace{next_tr}:feed {id_}")
        self.set_active_measurement(id_, channel=channel)

    def delete_measurement(self, id_: Union[str, int], channel: int = 1) -> None:
        assert isinstance(id_, str), "Can only delete measurement by name"
        self.write(f"calculate{channel}:parameter:delete {id_}")

    def get_measurement(self, id_: Union[str, int], channel: int = 1) -> Network:
        self.set_active_measurement(id_, channel, True)
        self.query("*OPC?")
        raw = np.array(
            self.query_ascii(f"calculate{channel}:data? sdata"), dtype=np.complex64
        )
        ntwk = Network()
        ntwk.frequency = self.frequency(channel)
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    def get_active_trace(self) -> Network:
        assert self.active_channel is not None, "No channel is active"
        raw = np.array(self.query_ascii(f"calculate{self.active_channel}:data? sdata"))
        self.query("*OPC?")
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
        continuous = "cont" in sweep_mode.lower()
        sweep_time = self.sweep_time()
        avg_on = self.averaging_on()
        avg_mode = self.average_mode()

        if avg_on and "swe" in avg_mode.lower():
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

        _ports = set(ports)
        self.active_channel = channel
        params = [f"S{p}{p}" for p in _ports]
        names = []
        # Make sure the ports specified are driven
        for param in params:
            name = self.query(f"calculate{channel}:parameter:tag:next?").strip()
            names.append(name)
            self.create_measurement(name, param, channel)

        self.sweep(channel=channel)

        old_snp_format = self.snp_format
        port_str = ",".join(map(str, ports))
        self.snp_format = "RI"
        raw_data = self.query_ascii(
            f"calculate{channel}:data:snp:ports? '{port_str}'", container=np.array
        )
        self.query("*OPC?")
        self.snp_format = old_snp_format

        for name in names:
            self.delete_measurement(name, channel)

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
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        raise NotImplementedError
