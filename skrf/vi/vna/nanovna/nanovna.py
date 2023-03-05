from enum import Enum

import pyvisa

import skrf
from skrf.vi import vna


class _OP(bytes, Enum):
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


class _REG_ADDR(bytes, Enum):
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


class NanoVNA(vna.VNA):
    _scpi = False

    id = vna.VNA.command(
        get_cmd="READ;DEVICE_VARIANT;1",
        set_cmd=None,
        doc="""Device identification string. i.e. the device model""",
    )

    npoints = vna.VNA.command(
        get_cmd="READ2;SWEEP_POINTS;2",
        set_cmd="WRITE2;SWEEP_POINTS;2;<arg>",
        doc="""""",
    )

    def __init__(self, address, backend):
        super().__init__(address, backend)
        if not isinstance(self._resource, pyvisa.resources.SerialInstrument):
            raise RuntimeError(
                f"NanoVNA can only be a serial instrument. {address} yields a {self._resource.__class__.__name__}"
            )

        self.read_bytes = self._resource.read_bytes
        self.write_raw = self._resource.write_raw

        self._freq = skrf.Frequency(start=1e6, stop=10e6, npoints=201)
        self._protocol_reset()

    def _reset_protocol(self):
        self.write_raw(b"\x00\x00\x00\x00\x00\x00\x00\x00")

    def query(self, cmd: str) -> bytes:
        op, addr, nbytes = cmd.split(";")
        op_code = _OP[op]
        try:
            addr = _REG_ADDR[addr]
        except KeyError:
            addr = bytes.fromhex(addr)

        nbytes = int(nbytes)

        cmd = op_code + addr
        self.write_raw(cmd)
        return self.read_bytes(nbytes)

    def write(self, cmd: str) -> None:
        op, addr, nbytes, arg = cmd.split(";")
        op_code = _OP[op]
        try:
            addr = _REG_ADDR[addr]
        except KeyError:
            addr = bytes.fromhex(addr)
        nbytes = int(nbytes)
        arg = int(arg).to_bytes(nbytes, byteorder="little", signed=False)

        if op_code == _OP.WRITEFIFO:
            cmd = op_code + addr + nbytes.to_bytes(1) + arg
        else:
            cmd = op_code + addr + arg

        self.write_raw(cmd)

    @property
    def freq_start(self) -> int:
        pass

    @freq_start.setter
    def freq_start(self, f: int) -> None:
        pass

    @property
    def freq_stop(self) -> int:
        pass

    @freq_stop.setter
    def freq_stop(self, f: int) -> None:
        pass

    @property
    def freq_step(self) -> int:
        pass

    @freq_step.setter
    def freq_step(self, f: int) -> None:
        pass

    @property
    def frequency(self) -> skrf.Frequency:
        return self._freq

    @frequency.setter
    def frequency(self, f: skrf.Frequency):
        pass

    def get_s11_s12(self) -> tuple[skrf.Network, skrf.Network]:
        pass
