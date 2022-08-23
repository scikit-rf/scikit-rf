from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

import itertools
import pprint

import numpy as np
from skrf.calibration import Calibration
from skrf.network import Network, four_oneports_2_twoport

from ..vna import VNA


class FieldFox(VNA):
    """
    Keysight's FieldFox analyzer.

    This class assumes the FieldFox has been put into NA mode before initializing
    """

    def __init__(self, address: str, timeout: int = 2000, backend: str = "@py"):
        super().__init__(address, timeout, backend)

    def sweep_mode(self) -> str:
        raise NotImplementedError

    def set_sweep_mode(self, mode: str) -> None:
        raise NotImplementedError

    def sweep_type(self) -> str:
        raise NotImplementedError

    def set_sweep_type(self, _type: str) -> None:
        raise NotImplementedError

    def averaging_on(self) -> bool:
        return self.average_count() != 1

    def set_averaging_on(self, state: bool) -> None:
        if state is False:
            self.set_average_count(1)
        else:
            raise NotImplementedError(
                "No command to turn averaging on and off. Set average count instead"
            )

    def average_mode(self) -> str:
        return self.query("sense:average:mode?")

    def set_average_mode(self, mode: str) -> None:
        if mode.lower() not in ["sweep", "point"]:
            raise ValueError(f"Unrecognized averaging mode: {mode}")
        self.write(f"sense:average:mode {mode.lower()}")

    def measurement_numbers(self) -> List[int]:
        resp = int(self.query("calculate:parameter:count?"))
        return [i + 1 for i in range(resp)]

    def measurements(self) -> List[Tuple]:
        raise NotImplementedError

    def active_measurement(self) -> Optional[str]:
        raise NotImplementedError

    def set_active_measurement(
        self, id_: Union[int, str], channel: int = 1, fast: bool = False
    ) -> None:
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
        self.set_active_measurement(trace)
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
