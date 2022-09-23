from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

import pprint

import numpy as np
from skrf import Calibration, Frequency, Network

from ..vna import VNA


class ZVA(VNA):
    """
    Rhode and Schwarz's ZVA analyzer
    """

    @property
    def start_freq(self) -> float:
        ch = self.active_channel
        return float(self.query(f"sense{ch}:frequency:start?"))

    @start_freq.setter
    def start_freq(self, f: float) -> None:
        ch = self.active_channel
        self.query(f"sense{ch}:frequency:start {f}")

    @property
    def stop_freq(self) -> float:
        ch = self.active_channel
        return float(self.query(f"sense{ch}:frequency:stop?"))

    @stop_freq.setter
    def stop_freq(self, f: float) -> None:
        ch = self.active_channel
        self.query(f"sense{ch}:frequency:stop {f}")

    @property
    def npoints(self) -> int:
        ch = self.active_channel
        return int(self.query(f"sense{ch}:sweep:points?"))

    @npoints.setter
    def npoints(self, n: int) -> None:
        ch = self.active_channel
        self.query(f"sense{ch}:sweep:points {n}")

    @property
    def freq_step(self) -> float:
        ch = self.active_channel
        return float(self.query(f"sense{ch}:sweep:step?"))

    @freq_step.setter
    def freq_step(self, f: float) -> None:
        ch = self.active_channel
        self.query(f"sense{ch}:sweep:step {f}")

    @property
    def frequency(self) -> Frequency:
        return Frequency(
            start=self.start_freq, stop=self.stop_freq, npoints=self.npoints, unit="Hz"
        )

    @frequency.setter
    def frequency(self, frequency: Frequency) -> None:
        ch = self.active_channel

        start = frequency.start
        stop = frequency.stop
        npoints = frequency.npoints

        self.write(
            f"sense{ch}:frequency:start {start};"
            f"frequency:stop {stop};"
            f"sweep:points {npoints}"
        )

    @property
    def sweep_mode(self) -> str:
        raise NotImplementedError

    @sweep_mode.setter
    def sweep_mode(self) -> str:
        raise NotImplementedError

    @property
    def sweep_type(self) -> str:
        ch = self.active_channel
        return self.query(f"sense{ch}:sweep:type?")

    @sweep_type.setter
    def sweep_type(self, type_: str) -> None:
        allowed_sweep_types = [
            "lin",
            "linear",
            "log",
            "logarithmic",
            "segm",
            "segment",
            "pow",
            "power",
            "cw",
            "poin",
            "point",
            "puls",
            "pulse",
            "iamp",
            "iamplitude",
            "iph",
            "iphase",
        ]
        if type_.lower() not in allowed_sweep_types:
            raise ValueError(f"Unrecognized sweep type: {type_}")
        pass

    @property
    def sweep_time(self) -> float:
        ch = self.active_channel
        return float(self.query(f"sense{ch}:sweep:time?"))

    @sweep_time.setter
    def sweep_time(self, time: float) -> None:
        ch = self.active_channel
        self.write(f"sense{ch}:sweep:time {time}")

    @property
    def if_bandwidth(self) -> float:
        ch = self.active_channel
        return float(self.query(f"sense{ch}:bwidth?"))

    @if_bandwidth.setter
    def if_bandwidth(self, bw: float) -> None:
        ch = self.active_channel
        self.write(f"sense{ch}:bwidth {bw}")

    @property
    def averaging_on(self) -> bool:
        ch = self.active_channel
        query = self.query(f"sense{ch}:state?").lower()
        return query == "1" or query == "on"

    @averaging_on.setter
    def averaging_on(self, state: bool) -> None:
        ch = self.active_channel
        onoff = "ON" if state else "OFF"
        self.write(f"sense{ch}:state {onoff}")

    @property
    def average_count(self) -> int:
        ch = self.active_channel
        return int(self.query(f"sense{ch}:average:count?"))

    @average_count.setter
    def average_count(self, n: int) -> None:
        ch = self.active_channel
        self.query(f"sense{ch}:average:count {n}")

    @property
    def average_mode(self) -> str:
        ch = self.active_channel
        return self.query(f"sense{ch}:average:mode?")

    @average_mode.setter
    def average_mode(self, mode: str) -> None:
        if mode.lower() not in [
            "auto",
            "flat",
            "flatten",
            "red",
            "reduce",
        ]:
            raise ValueError("Unrecognized averaging mode: {mode}")

        ch = self.active_channel
        self.write(f"sense{ch}:average:mode {mode}")

        ch = self.active_channel
        self.write(f"sense{ch}:average:mode {mode}")

    def clear_averaging(self) -> None:
        ch = self.active_channel
        self.write(f"sense{ch}:average:clear")

    @property
    def channels_in_use(self) -> List[int]:
        resp = self.query("configure:channel:catalog?").strip().replace('"', "")
        return [int(c) for c in resp.split(",")]

    @property
    def active_channel(self) -> Optional[int]:
        active = self.query("instrument:nselect?").strip()
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
            self.write(f"instrument:nselect {channel}")

    def measurements_on_channel(self, channel: int = 1) -> List[Tuple]:
        channels = self.channels_in_use
        if channel not in channels:
            return []
        else:
            resp = self.query(f"calculate{channel}:parameter:catalog?")
            resp = resp.strip().replace('"', "").split("'")
            names = resp[::2]
            params = resp[1::2]
            return [(name, param, channel) for (name, param) in zip(names, params)]

    @property
    def measurements(self) -> List[Tuple]:
        msmnts = []
        chans = self.channels_in_use
        for chan in chans:
            msmnts.extend(self.measurements_on_channel(chan))

        return msmnts

    @property
    def active_measurement(self) -> Optional[Union[str, int]]:
        active = self.query("calculate{channel}:parameter:select?").strip()
        return None if active == "" else active

    @active_measurement.setter
    def active_measurement(self, msmnt: Tuple[int, Union[int, str]]) -> None:
        ch = msmnt[0]
        id_ = msmnt[1]
        if isinstance(id_, int):
            raise NotImplementedError
        elif isinstance(id_, str):
            meas_names = [m[0] for m in self.measurements_on_channel(ch)]
            if id_ not in meas_names:
                raise KeyError(f"Measurement {id_} doesn't exist on channel {ch}")
            self.write(f"calculate{ch}:parameter:select '{id_}'")

    def create_measurement(
        self, id_: Union[str, int], param: str, channel: int = 1
    ) -> None:
        # TODO: ...there's a lot of allowed parameters...how / should we verify?
        # TODO: This selects but does not display the new measurement. Should it?
        self.write(f"calculate{channel}:parameter:sdefine '{id_}', '{param}'")

    def delete_measurement(self, id_: Union[str, int], channel: int = 1) -> None:
        assert isinstance(id_, str), "Can only delete measurement by name"
        self.write(f"calculate{channel}:parameter:delete '{id_}'")

    def get_measurement(self, id_: Union[int, str], channel: int = 1) -> Network:
        self.active_measurement = (channel, id_)
        self.query("*OPC?*")
        raw = np.array(
            self.query_ascii(f"calculate{channel}:data? sdata"), dtype=np.complex64
        )
        ntwk = Network()
        ntwk.frequency = self.frequency
        ntwk.s = raw[::2] + 1j * raw[1::2]
        return ntwk

    def get_active_trace(self) -> Network:
        assert self.active_channel is not None, "No channel is active"
        raw = np.array(
            self.query_ascii(f"calculate{self.active_channel}:data? sdata"),
            dtype=np.complex64,
        )
        self.query("*OPC?")
        ntwk = Network()
        ntwk.frequency = self.frequency
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
        resp = int(self.query("instrument:port:count?").strip())
        return [str(p + 1) for p in range(resp)]

    @property
    def sysinfo(self) -> str:
        info = {
            "ID": self.id_string,
            "OS Version": self.query("system:osversion?"),
            "SCPI Version": self.query("system:version?"),
            "Ports": self.ports,
            "Frequency": {
                "Minimum": self.query("system:frequency? minimum"),
                "Maximum": self.query("system:frequency? maximum"),
            },
        }

        return pprint.pformat(info)

    def sweep(self) -> None:
        raise NotImplementedError

    def get_snp_network(
        self, ports: Optional[Sequence] = None, channel: int = 1
    ) -> Network:
        # TODO: This needs testing!
        # TODO: Need to trigger freesh sweep
        self.resource.clear()

        if not ports:
            ports = self.ports

        _ports = set(ports)
        port_str = ",".join(_ports)
        _ports = list(map(int, _ports))
        self.active_channel = channel
        # Create traces
        self.write(f"calculate{channel}:parameter:define:sgroup {port_str}")
        # Ensure we're transferring data in ascii
        self.write("format:ascii")
        # Sweep
        self.write("initiate:continuous off")
        self.write("initiate:immediate *OPC")
        # Query data
        raw = cast(
            np.ndarray,
            self.query_ascii(
                f"calculate{channel}:data:sgroup? sdata", container=np.ndarray
            ),
        )
        # Delete traces
        self.write(f"calculate{channel}:parameter:delete:sgroup")

        sdata = raw.reshape((len(raw) // self.npoints, -1))

        ntwk = Network()
        ntwk.frequency = self.frequency
        for n in _ports:
            for m in _ports:
                i = n * len(_ports) + m
                ntwk.s[:, n, m] = sdata[i * 2] + 1j * sdata[i * 2 + 1]

        return ntwk

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: int, cal: Calibration) -> None:
        raise NotImplementedError
