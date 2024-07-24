from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import functools
from enum import Enum

import numpy as np
import pyvisa

import skrf
from skrf.vi import vna


class OP(bytes, Enum):
    NOP = b"\x00"
    INDICATE = b"\x0d"
    READ = b"\x10"
    READ2 = b"\x11"
    READ4 = b"\x12"
    READFIFO = b"\x18"
    WRITE = b"\x20"
    WRITE2 = b"\x21"
    WRITE4 = b"\x22"
    WRITE8 = b"\x23"
    WRITEFIFO = b"\x28"


class REG_ADDR(bytes, Enum):
    SWEEP_START = b"\x00"
    SWEEP_STEP = b"\x10"
    SWEEP_POINTS = b"\x20"
    VALS_PER_FREQ = b"\x22"
    RAW_SAMPLES_MODE = b"\x26"
    VALS_FIFO = b"\x30"
    DEVICE_VARIANT = b"\xf0"
    PROTOCOL_VERSION = b"\xf1"
    HARDWARE_REV = b"\xf2"
    FIRMWARE_MAJOR = b"\xf3"
    FIRMWARE_MINOR = b"\xf4"


class NanoVNAv2(vna.VNA):
    """NanoVNAv2.

    This class connects to the NanoVNA V2 using a binary protocol over USB.
    It should also be compatible with other devices, if they use the same `protocol <https://nanorfe.com/nanovna-v2-user-manual.html#__RefHeading___Toc2537_2953165397>`_.
    Some variants of the NanoVNA use a text protocol and are not supported.

    .. warning::

        The NanoVNA V2 only returns uncalibrated networks, regardless of the calibration on the device itself.
        See section `Examples`_.

    Notes
    -----

    Tested devices:

    * `NanoVNA V2 <https://nanorfe.com/de/nanovna-v2.html>`_
    * `LiteVNA <https://www.zeenko.tech/litevna>`_

    .. _Examples:
    Examples
    --------

    Basic S11 and S21 measurement:

    .. code-block:: python

        import skrf
        from skrf.vi.vna.nanovna import NanoVNAv2

        vna = NanoVNAv2("ASRL/dev/ttyACM0::INSTR")  # Linux
        # vna = NanoVNAv2("ASRL1::INSTR")  # Windows
        freq = skrf.Frequency(start=1, stop=2, unit='GHz', npoints=101)
        s11, s21 = vna.get_s11_s21()

    1-port calibration assuming ideal reference standards:

    .. code-block:: python

        import skrf

        # load uncalibrated measurement
        raw_s1p = skrf.Network("measurement.s1p")

        # load s1p networks of measured standards
        cal_measured = [skrf.Network("short.s1p"), skrf.Network("open.s1p"), skrf.Network("load.s1p"]

        # get ideal standards
        line = skrf.DefinedGammaZ0(frequency=cal_measured[0].frequency, z0=50)
        cal_ideals = [line.short(nports=1), line.open(nports=1), line.match(nports=1)]

        # run calibration
        cal = skrf.OnePort(ideals=cal_ideals, measured=cal_measured)
        cal.run()

        # apply calibration
        calibrated_s1p = cal.apply_cal(raw_s1p)
    """

    _scpi = False

    def __init__(self, address, backend: str = "@py"):
        super().__init__(address, backend)
        if not isinstance(self._resource, pyvisa.resources.SerialInstrument):
            raise RuntimeError(
                "NanoVNAv2 can only be a serial instrument. "
                f"{address} yields a {self._resource.__class__.__name__}"
            )

        self.read_bytes = self._resource.read_bytes
        self.write_raw = self._resource.write_raw

        self._reset_protocol()

        self.frequency = skrf.Frequency(start=1e6, stop=10e6, npoints=201)

    def _reset_protocol(self):
        self.write_raw(b"\x00\x00\x00\x00\x00\x00\x00\x00")

    def query(self, op: OP, addr: REG_ADDR | bytes, nbytes: int) -> bytes:
        cmd = op + addr
        self.write_raw(cmd)
        return self.read_bytes(nbytes)

    def write(self, op: OP, addr: REG_ADDR | bytes, nbytes: int, arg: int) -> None:
        arg = int(arg).to_bytes(nbytes, byteorder="little", signed=False)

        if op == OP.WRITEFIFO:
            cmd = op + addr + nbytes.to_bytes(1) + arg
        else:
            cmd = op + addr + arg

        self.write_raw(cmd)

    @property
    def id(self) -> str:
        var = self.query(OP.READ, REG_ADDR.DEVICE_VARIANT, 1)
        var = int.from_bytes(var, 'little')
        return str(var)

    @property
    def device_info(self) -> str:
        variant = self.id
        protocol = self.query(OP.READ, REG_ADDR.PROTOCOL_VERSION, 1)
        protocol = int.from_bytes(protocol, 'little')
        hardware = self.query(OP.READ, REG_ADDR.HARDWARE_REV, 1)
        hardware = int.from_bytes(hardware, 'little')
        fw_major = self.query(OP.READ, REG_ADDR.FIRMWARE_MAJOR, 1)
        fw_major = int.from_bytes(fw_major, 'little')
        fw_minor = self.query(OP.READ, REG_ADDR.FIRMWARE_MAJOR, 1)
        fw_minor = int.from_bytes(fw_minor, 'little')

        return (
            f"NanoVNAv2\n"
            f"\tVariant:{variant}\n"
            f"\tProtocol Version:{protocol}\n"
            f"\tHardware Version: {hardware}\n"
            f"\tFirmware Version: {fw_major}.{fw_minor}"
        )


    @property
    def freq_start(self) -> float:
        return self._freq.start

    @freq_start.setter
    def freq_start(self, f: int) -> None:
        self.write(OP.WRITE8, REG_ADDR.SWEEP_START, 8, f)
        self._freq = skrf.Frequency(start=f, stop=self._freq.stop, npoints=self._freq.npoints)

    @property
    def freq_stop(self) -> float:
        return self._freq.stop

    @freq_stop.setter
    def freq_stop(self, f: int) -> None:
        self._freq = skrf.Frequency(start=self._freq.start, stop=f, npoints=self._freq.npoints)
        self.write(OP.WRITE8, REG_ADDR.SWEEP_STEP, 8, self._freq.step)

    @property
    def freq_step(self) -> float:
        return self._freq.step

    @freq_step.setter
    def freq_step(self, f: int) -> None:
        npoints = (self._freq.stop - self._freq.start + f) / f
        npoints = int(npoints.round())
        self._freq = skrf.Frequency(start=self._freq.start, stop=self._freq.stop, npoints=npoints)
        self.write(OP.WRITE2, REG_ADDR.SWEEP_POINTS, 2, npoints)

    @property
    def npoints(self) -> int:
        return self._freq.npoints

    @npoints.setter
    def npoints(self, n: int) -> None:
        self.write(OP.WRITE2, REG_ADDR.SWEEP_POINTS, 2, n)
        self._freq = skrf.Frequency(start=self._freq.start, stop=self._freq.stop, npoints=n)

    @property
    def frequency(self) -> skrf.Frequency:
        return self._freq

    @frequency.setter
    def frequency(self, f: skrf.Frequency):
        self.write(OP.WRITE8, REG_ADDR.SWEEP_START, 8, f.start)
        self.write(OP.WRITE8, REG_ADDR.SWEEP_STEP, 8, f.step)
        self.write(OP.WRITE2, REG_ADDR.SWEEP_POINTS, 2, f.npoints)
        self._freq = f

    def clear_fifo(self) -> None:
        self.write(OP.WRITE, REG_ADDR.VALS_FIFO, 1, 0)

    def _convert_bytes_to_sparams(n: int, raw: bytearray) -> tuple[np.ndarray, np.ndarray]:
        s11 = np.zeros(n, dtype=complex)
        s21 = np.zeros_like(s11)

        from_bytes = functools.partial(int.from_bytes, byteorder='little', signed=True)

        for i in range(n):
            start = i * 32
            stop = (i+1) * 32
            chunk = raw[start:stop]

            fwd0re = from_bytes(chunk[0:4])
            fwd0im = from_bytes(chunk[4:8])
            rev0re = from_bytes(chunk[8:12])
            rev0im = from_bytes(chunk[12:16])
            rev1re = from_bytes(chunk[16:20])
            rev1im = from_bytes(chunk[20:24])
            freqIndex = from_bytes(chunk[24:26])

            a1 = complex(fwd0re, fwd0im)
            b1 = complex(rev0re, rev0im)
            b2 = complex(rev1re, rev1im)

            s11[freqIndex] = b1 / a1
            s21[freqIndex] = b2 / a1

        return s11, s21

    def get_s11_s21(self) -> tuple[skrf.Network, skrf.Network]:
        n = self._freq.npoints
        self.clear_fifo()

        raw = bytearray()
        n_remaining = n

        while n_remaining > 0:
            len_segment = 255 if n_remaining > 255 else n_remaining
            n_remaining = n_remaining - len_segment
            self.write(OP.READFIFO, REG_ADDR.VALS_FIFO, 1, len_segment)
            raw.extend(self.read_bytes(32 * len_segment))

        s11, s21 = skrf.Network(), skrf.Network()
        s11.frequency = self._freq.copy()
        s21.frequency = self._freq.copy()

        s11.s, s21.s = NanoVNAv2._convert_bytes_to_sparams(n, raw)

        return s11, s21

    def get_sdata(self, a: int = 1, b: int = 1) -> skrf.Network:
        """
        Get S-parameter as 1-port :class:`skrf.Network`.

        Parameters
        ----------
        a:
            Input port of VNA. Can be 1 or 2, default is 1.
        b:
            Output port of VNA. Can only be 1, default is 1.

        Returns
        -------
        :class:`skrf.Network`
            Measured S-parameter
        """
        if a not in (1, 2) or b != 1:
            raise RuntimeError(
                "NanoVNA V2 can only measure S11 and S21"
            )

        return self.get_s11_s21()[a-1]

    def get_snp_network(self, ports: Sequence[int | None]) -> skrf.Network:
        """
        Get custom n-port :class:`skrf.Network`.

        Parameters
        ----------
        ports:
            Port mapping. Can be any sequence of 1, 2 and None

        Returns
        -------
        :class:`skrf.Network`
            Measured S-parameters in custom network
        """
        nwk_s11, nwk_s21 = self.get_s11_s21()
        frequency = self._freq.copy()

        s_parameters = np.zeros((len(frequency), len(ports), len(ports)), dtype=complex)
        snp_network = skrf.Network(frequency=frequency, s=s_parameters)

        for i, i_port in enumerate(ports):
            if i_port is not None:
                for j, j_port in enumerate(ports):
                    if i_port == 1 and j_port == 1:
                        snp_network.s[:, i, j] = nwk_s11.s[:, 0, 0]
                    elif i_port == 2 and j_port == 1:
                        snp_network.s[:, i, j] = nwk_s21.s[:, 0, 0]

        return snp_network
