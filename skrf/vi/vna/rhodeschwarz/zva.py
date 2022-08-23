from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

import itertools
import pprint

import numpy as np
from skrf.calibration import Calibration
from skrf.network import Network, four_oneports_2_twoport

from ..vna import VNA


class ZVA(VNA):
    """
    Rhode and Schwarz's ZVA analyzer
    """

    def sweep_mode(self, channel: int = 1) -> str:
        raise NotImplementedError

    def set_sweep_mode(self, channel: int = 1) -> str:
        raise NotImplementedError

    def set_sweep_type(self, type_: str, channel: int = 1) -> None:
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
        super().set_sweep_type(type_, channel)

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

    def measurements(self) -> List[Tuple]:
        msmnts = []
        chans = self.channels_in_use
        for chan in chans:
            msmnts.extend(self.measurements_on_channel(chan))

        return msmnts

    def active_measurement(self, channel: int = 1) -> Optional[Union[str, int]]:
        active = self.query("calculate{channel}:parameter:select?").strip()
        return None if active == "" else active

    def set_active_measurement(self, id_: Union[int, str], channel: int = 1) -> None:
        if isinstance(id_, int):
            pass
        elif isinstance(id_, str):
            meas_names = [m[0] for m in self.measurements_on_channel(channel)]
            if id_ not in meas_names:
                raise KeyError(f"Measurement {id_} doesn't exist on channel {channel}")
            self.write(f"calculate{channel}:parameter:select '{id_}'")

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
        self.set_active_measurement(id_, channel)
        self.query("*OPC?*")
        raw = np.array(
            self.query_ascii(f"calculate{channel}:data? sdata"), dtype=np.complex64
        )
        ntwk = Network()
        ntwk.frequency = self.frequency(channel)
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
        ntwk.frequency = self.frequency(self.active_channel)
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

    def sweep(self, channel: int = 1) -> None:
        raise NotImplementedError

    def get_snp_network(
        self, ports: Optional[Sequence] = None, channel: int = 1
    ) -> Network:
        # TODO: This needs testing!
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
        self.write(f"format:ascii")
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

        sdata = raw.reshape((len(raw) // self.npoints(), -1))

        ntwk = Network()
        ntwk.frequency = self.frequency(channel)
        for n in _ports:
            for m in _ports:
                i = n * len(_ports) + m
                ntwk.s[:, n, m] = sdata[i * 2] + 1j * sdata[i * 2 + 1]

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: int, cal: Calibration) -> None:
        raise NotImplementedError
