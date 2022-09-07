from __future__ import annotations

from typing import TYPE_CHECKING

from skrf.frequency import Frequency

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union
    from pyvisa.resources import SerialInstrument

import numpy as np
from skrf.calibration import Calibration
from skrf.network import Network

from ..vna import VNA


class _Cmd:
    # This is just a class of constants to make the rest of the code more clear
    #
    # Commands are byte strings of length 1-5
    # Users should not have to interact with these commands at all. These are an
    # implementation detail.
    # Commands are of the form
    # --- B0 --- B1 --- B2 --- B3 --- B4 --- B5 ---
    # | OpCode| Arguments ------------------------|
    # For example, READFIFO with NN bytes from address AA
    # b"\x18\xAA\xNN

    # Opcodes
    NOP = b"\x00"
    INDICATE = b"\x0d"
    READ = b"\x10"  # B1 = address
    READ2 = b"\x11"  # B1 = address
    READ4 = b"\x12"  # B1 = address
    READFIFO = b"\x18"  # B1 = address B2 = N bytes
    WRITE = b"\x20"  # B1 = address B2 = value
    WRITE2 = b"\x21"  # B1 = address B2->B3 = value
    WRITE4 = b"\x22"  # B1 = address B2 -> B5 = value
    WRITE8 = b"\x22"  # B1 = address B2->B9 = value
    WRITEFIFO = b"\x28"  # B1 = address B2 = n bytes B3->... = bytes

    # Frequency addresses
    ADDR_FSTART = b"\x00"
    ADDR_FSTEP = b"\x10"
    ADDR_FPOINTS = b"\x20"
    ADDR_FIFO = b"\x30"


class NanoVNAV2(VNA):
    """NanoRFE's NanoVNA v2."""

    resource: SerialInstrument  # The nanovna can only communicate over serial. Making that explicit helps the linter

    def __init__(self, address: str, timeout: int = 2000, backend: str = "@py") -> None:
        super().__init__(address, timeout, backend)
        self.reset()

        self.read = self.resource.read_bytes
        self.write = self.resource.write_raw

        self._start_freq: float = 1e6
        self._stop_freq: float = 1e6
        self._npoints: int = 201

        self._frequency = Frequency(
            start=self._start_freq,
            stop=self._stop_freq,
            npoints=self._npoints,
            unit="Hz",
        )

        self._freq_step = self._frequency.step

    def reset(self) -> None:
        self.write(b"".join([_Cmd.NOP] * 8))

    def id_string(self) -> str:
        self.write(b"".join([_Cmd.INDICATE, b"\xf0"]))
        resp_bytes = self.read(1)
        resp = int.from_bytes(resp_bytes, byteorder="little")

        if resp == 2:
            return "NanoVNA v2"
        else:
            return f"Unknown device variant: {resp}"

    def start_freq(self, channel: int = 1) -> float:
        return float(self._start_freq)

    def set_start_freq(self, f: float, channel: int = 1) -> None:
        self.write(
            _Cmd.WRITE8
            + _Cmd.ADDR_FSTART
            + int.to_bytes(int(f), 8, byteorder="little", signed=False)
        )
        self._start_freq = f

        self._frequency = Frequency(
            start=self._start_freq, stop=self._stop_freq, npoints=self._npoints
        )

    def stop_freq(self, channel: int = 1) -> float:
        return float(self._stop_freq)

    def set_stop_freq(self, f: float, channel: int = 1) -> None:
        # There is no command to set the stop frequency directly, so instead we
        # just update the internal frequency object, then update the frequency
        # step to conform to the updated range
        self._stop_freq = f

        self._frequency = Frequency(
            start=self._start_freq, stop=self._stop_freq, npoints=self._npoints
        )

        self.set_freq_step(self._frequency.step)

    def npoints(self, channel: int = 1) -> int:
        return self._npoints

    def set_npoints(self, n: int, channel: int = 1) -> None:
        self.write(
            _Cmd.WRITE2
            + _Cmd.ADDR_FPOINTS
            + int.to_bytes(n, 2, byteorder="little", signed=False)
        )
        self._npoints = n

    def freq_step(self, channel: int = 1) -> float:
        return self._freq_step

    def set_freq_step(self, f: float, channel: int = 1) -> None:
        self.write(
            _Cmd.WRITE8
            + _Cmd.ADDR_FSTEP
            + int.to_bytes(int(f), 8, byteorder="little", signed=False)
        )
        self._freq_step = f

    def frequency(self, channel: int = 1) -> Frequency:
        return self._frequency

    def set_frequency(self, frequency: Frequency, channel: int = 1) -> None:
        self._frequency = frequency

        start = self._frequency.start
        step = self._frequency.step
        npoints = self._frequency.npoints

        cmd = (
            _Cmd.WRITE8
            + _Cmd.ADDR_FSTART
            + int.to_bytes(int(start), 8, byteorder="little", signed=False)
            + _Cmd.WRITE8
            + _Cmd.ADDR_FSTEP
            + int.to_bytes(int(step), 8, byteorder="little", signed=False)
            + _Cmd.WRITE2
            + _Cmd.ADDR_FPOINTS
            + int.to_bytes(npoints, 2, byteorder="little", signed=False)
        )
        self.write(cmd)

    def sweep_mode(self, channel: int = 1) -> str:
        raise NotImplementedError

    def set_sweep_mode(self, mode: str, channel: int = 1) -> None:
        raise NotImplementedError

    def sweep_type(self, channel: int = 1) -> str:
        raise NotImplementedError

    def set_sweep_type(self, type_: str, channel: int = 1) -> None:
        raise NotImplementedError

    def sweep_time(self, channel: int = 1) -> float:
        raise NotImplementedError

    def set_sweep_time(self, time: Union[float, str], channel: int = 1) -> None:
        raise NotImplementedError

    def if_bandwidth(self, channel: int = 1) -> float:
        raise NotImplementedError

    def set_if_bandwidth(self, bw: float, channel: int = 1) -> None:
        raise NotImplementedError

    def averaging_on(self, channel: int = 1) -> bool:
        raise NotImplementedError

    def set_averaging_on(self, state: bool, channel: int = 1) -> None:
        raise NotImplementedError

    def average_count(self, channel: int = 1) -> int:
        raise NotImplementedError

    def set_average_count(self, n: int, channel: int = 1) -> None:
        raise NotImplementedError

    def clear_averaging(self, channel: int = 1) -> None:
        raise NotImplementedError

    def measurements(self) -> List[Tuple]:
        # The Nona VNA only supports these two measurements
        return [(1, "S11"), (2, "S21")]

    def active_measurement(self, channel: int = 1) -> Optional[Union[str, int]]:
        raise NotImplementedError

    def set_active_measurement(self, id_: Union[int, str], channel: int = 1) -> None:
        raise NotImplementedError

    def create_measurement(
        self, id_: Union[str, int], param: str, channel: int = 1
    ) -> None:
        raise NotImplementedError

    def delete_measurement(self, id_: Union[str, int], channel: int = 1) -> None:
        raise NotImplementedError

    def get_s11_s21(self) -> Tuple[np.ndarray, np.ndarray]:
        self.write(_Cmd.WRITE + _Cmd.ADDR_FIFO + b"\x00")

        data_raw = []
        npoints = self._frequency.npoints
        f_remaining = npoints

        while f_remaining > 0:
            len_segment = f_remaining if f_remaining <= 255 else 255
            f_remaining = f_remaining - len_segment

            self.write(
                _Cmd.READFIFO
                + b"\x30"
                + int.to_bytes(len_segment, 1, byteorder="little", signed=False)
            )
            data_raw.extend(self.read(32 * len_segment))

        data_s11 = np.zeros(self._frequency.npoints, dtype=complex)
        data_s21 = np.zeros_like(data_s11)

        for i in range(npoints):
            i_start = i * 32
            i_stop = (i + 1) * 32
            data_chunk = data_raw[i_start:i_stop]
            fwd0re = int.from_bytes(data_chunk[0:4], "little", signed=True)
            fwd0im = int.from_bytes(data_chunk[4:8], "little", signed=True)
            rev0re = int.from_bytes(data_chunk[8:12], "little", signed=True)
            rev0im = int.from_bytes(data_chunk[12:16], "little", signed=True)
            rev1re = int.from_bytes(data_chunk[16:20], "little", signed=True)
            rev1im = int.from_bytes(data_chunk[20:24], "little", signed=True)
            freqIndex = int.from_bytes(data_chunk[24:26], "little", signed=False)

            a1 = complex(fwd0re, fwd0im)
            b1 = complex(rev0re, rev0im)
            b2 = complex(rev1re, rev1im)

            data_s11[freqIndex] = b1 / a1
            data_s21[freqIndex] = b2 / a1

        return data_s11, data_s21

    def get_measurement(self, id_: Union[int, str], channel: int = 1) -> Network:
        if id_ not in [1, 2, "s11", "S11", "s21", "S21"]:
            raise ValueError(
                "Can only measure S11 or S21."
                "Request by passing 1, 2, 's11', or 's21'"
            )

        s11, s21 = self.get_s11_s21()
        freq = self._frequency.copy()
        ntwk = Network(frequency=freq)

        if id_ in [1, "s11", "S11"]:
            ntwk.s = s11
        elif id_ in [2, "s21", "S21"]:
            ntwk.s = s21

        return ntwk

    def get_active_trace(self) -> Network:
        raise NotImplementedError

    @property
    def snp_format(self) -> str:
        raise NotImplementedError

    @snp_format.setter
    def snp_format(self, format: str) -> str:
        raise NotImplementedError

    @property
    def ports(self) -> List[str]:
        return ["1", "2"]

    @property
    def sysinfo(self) -> str:
        self.write(b"".join([_Cmd.INDICATE, b"\xf2"]))
        resp_bytes = self.read(1)
        hw_rev = int.from_bytes(resp_bytes, byteorder="little")

        self.write(b"".join([_Cmd.INDICATE, b"\xf3"]))
        resp_bytes = self.read(1)
        firmware_major = int.from_bytes(resp_bytes, byteorder="little")

        self.write(b"".join([_Cmd.INDICATE, b"\xf4"]))
        resp_bytes = self.read(1)
        firmware_minor = int.from_bytes(resp_bytes, byteorder="little")

        info_str = (
            "Variant: NanoVNA V2\n"
            "Protocol Version: 1\n"
            f"Hardware Revision: {hw_rev}\n"
            f"Firmware Revision: {firmware_major}.{firmware_minor}"
        )
        return info_str

    def sweep(self) -> None:
        raise NotImplementedError

    def get_snp_network(
        self, ports: Optional[Sequence] = None, channel: int = 1
    ) -> Network:
        raise NotImplementedError(
            "You can't get a full .s2p network with the NanoVNA."
            "Instead, capture the individual 1-port measurements"
            "and combine them with Network.n_oneports_2_nport"
        )

    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        raise NotImplementedError

    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        raise NotImplementedError
