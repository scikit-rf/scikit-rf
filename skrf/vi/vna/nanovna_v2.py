from . import abcvna
import numpy as np
import skrf
from time import sleep


# Communication commands and register addresses are listed in the user manual at
# https://nanorfe.com/nanovna-v2-user-manual.html
# See also `python/nanovna.py` in the NanoVNA project repository at https://github.com/nanovna-v2/NanoVNA2-firmware
#
#
# COMMAND SUMMARY:
#
# No operation:
# cmd = [0x00]
#
# Indicate; 1 byte reply (always 0x32)
# cmd = [0x0d]
#
# Read 1 byte from address 0xAA; 1 byte reply
# cmd = [0x10, 0xAA]
#
# Read 2 bytes from address 0xAA; 2 byte reply
# cmd = [0x11, 0xAA]
#
# Read 4 bytes from address 0xAA; 4 byte reply
# cmd = [0x12, 0xAA]
#
# Read 0xNN values from FIFO at address 0xAA; 0xNN byte reply
# cmd = [0x18, 0xAA, 0xNN]
#
# Write 1 byte (0xBB) to address 0xAA; no reply
# cmd = [0x20, 0xAA, 0xBB]
#
# Write 2 bytes (0xB0 0xB1) to addresses 0xAA and following; no reply
# 0xB0 will be written to 0xAA; 0xB1 will be written to 0xAB
# cmd = [0x21, 0xAA, 0xB0, 0xB1]
#
# Write 4 bytes (0xB0 to 0xB3) to addresses 0xAA and following; no reply
# 0xB0 will be written to 0xAA; 0xB1 will be written to 0xAB; ...
# cmd = [0x22, 0xAA, 0xB0, 0xB1, 0xB2, 0xB3]
#
# Write 8 bytes (0xB0 to 0xB7) to addresses 0xAA and following; no reply
# 0xB0 will be written to 0xAA; 0xB1 will be written to 0xAB; ...
# cmd = [0x23, 0xAA, 0xB0, 0xB1, ..., 0xB7]
#
# Write 0xNN bytes to FIFO at address 0xAA and following; no reply
# cmd = [0x28, 0xAA, 0xNN, 0xB0, 0xB1, ..., 0xBNN]
#
#
# FIFO DATA FORMAT (encoding: little endian):
# 0x03 to 0x00: real part of channel 0 outgoing wave; fwd0re (4 bytes; int32)
# 0x07 to 0x04: imaginary part of channel 0 outgoing wave; fwd0im (4 bytes; int32)
# 0x0b to 0x08: real part of channel 0 incoming wave; rev0re (4 bytes; int32)
# 0x0f to 0x0c: imaginary part of channel 0 incoming wave; rev0im (4 bytes; int32)
# 0x13 to 0x10: real part of channel 1 incoming wave; rev1re (4 bytes; int32)
# 0x17 to 0x14: imaginary part of channel 1 incoming wave; rev1im (4 bytes; int32)
# 0x19 0x18: frequency index of the sample (0 to sweep_points - 1); 2 bytes; uint16
#
#
# REGISTER ADDRESSES (encoding: little endian):
# 0x07 to 0x00: sweep start frequency in Hz (8 bytes; uint64)
# 0x17 to 0x10: sweep step in Hz (8 bytes; uint64)
# 0x21 0x20: number of sweep frequency points (2 bytes; uint16)
# 0x23 0x22: number of data points to output for each frequency (2 bytes; uint16)

class NanoVNAv2(abcvna.VNA):
    """
    Class for NanoVNA_v2 network analyzers.
    """

    def __init__(self, serial_port: str = '/dev/ttyACM0'):
        super().__init__(address='ASRL{}::INSTR'.format(serial_port), kwargs={'visa_library': 'py'})
        self._frequency = np.linspace(1e6, 10e6, 101)
        self.set_frequency_sweep(1e6, 10e6, 101)

    def idn(self) -> str:
        """
        Returns the identification string of the device.

        Returns
        -------
        str
        """

        # send 1-byte READ (0x10) of address 0xf0 to retrieve device variant code
        self.resource.write_raw([0x10, 0xf0])
        v_byte = self.resource.read_bytes(1)
        v = int.from_bytes(v_byte, byteorder='little')
        if v == 2:
            return 'NanoVNA_v2'
        else:
            return 'Unknown device, got deviceVariant={}'.format(v)

    def _protocol_reset(self):
        # send 8x NOP (0x00) to reset the communication protocol
        self.resource.write_raw([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    def get_traces(self, traces: list = None, **kwargs) -> list:
        """
        Returns the measured data of the sweep.
        :param traces:
        :param kwargs:
        :return:
        """

        # data is continuously being sampled and stored in the FIFO (address 0x30)
        # writing any value to 0x30 clears the FIFO, which enables on-demand readings

        f_points = len(self._frequency)

        self._protocol_reset()

        # write any byte to register 0x30 to clear FIFO
        self.resource.write_raw([0x20, 0x30, 0x00])

        # read 'f_point' values from FIFO (32 * f_points bytes)
        # can only read 256 values (f_points <= 256)
        self.resource.write_raw([0x18, 0x30, f_points])
        data_raw = self.resource.read_bytes(32 * f_points)

        # parse FIFO data
        data_s11 = np.zeros(f_points, dtype=complex)
        data_s21 = np.zeros_like(data_s11)

        for i in range(f_points):
            i_start = i * 32
            i_stop = (i + 1) * 32
            data_chunk = data_raw[i_start:i_stop]
            fwd0re = int.from_bytes(data_chunk[0:4], 'little', signed=True)
            fwd0im = int.from_bytes(data_chunk[4:8], 'little', signed=True)
            rev0re = int.from_bytes(data_chunk[8:12], 'little', signed=True)
            rev0im = int.from_bytes(data_chunk[12:16], 'little', signed=True)
            rev1re = int.from_bytes(data_chunk[16:20], 'little', signed=True)
            rev1im = int.from_bytes(data_chunk[20:24], 'little', signed=True)
            freqIndex = int.from_bytes(data_chunk[24:26], 'little', signed=False)

            a1 = complex(fwd0re, fwd0im)
            b1 = complex(rev0re, rev0im)
            b2 = complex(rev1re, rev1im)

            data_s11[freqIndex] = b1 / a1
            data_s21[freqIndex] = b2 / a1

        frequency = skrf.Frequency.from_f(self._frequency, unit='hz')
        nw_s11 = skrf.Network(frequency=frequency, s=data_s11, name='Trace0_S11')
        nw_s21 = skrf.Network(frequency=frequency, s=data_s21, name='Trace1_S21')
        return [nw_s11, nw_s21]

    def set_frequency_sweep(self, start_freq: float, stop_freq: float, num_points: int = 201, **kwargs) -> None:
        """
        Configures the frequency sweep.

        Parameters
        ----------
        start_freq : float
            Start frequency in Hertz
        stop_freq : float
            Stop frequency in Hertz
        num_points : int, optional
            Number of frequencies in the sweep.
        kwargs : dict, optional

        Returns
        -------
        None
        """

        if num_points > 255:
            raise ValueError('Number of frequency points cannot be greater than 255.')

        f_step = 0.0
        if num_points > 1:
            f_step = (stop_freq - start_freq) / (num_points - 1)

        self._protocol_reset()

        self._frequency = np.linspace(start_freq, stop_freq, num_points)

        # set f_start by writing 8 bytes (cmd=0x23) to (0x00...0x07)
        cmd = b'\x23\x00' + int.to_bytes(int(start_freq), 8, byteorder='little', signed=False)
        self.resource.write_raw(cmd)

        # set f_step by writing 8 bytes (cmd=0x23) to (0x10...0x17)
        cmd = b'\x23\x10' + int.to_bytes(int(f_step), 8, byteorder='little', signed=False)
        self.resource.write_raw(cmd)

        # set f_points by writing 2 bytes (cmd=0x21) to (0x20 0x21)
        cmd = b'\x21\x20' + int.to_bytes(int(num_points), 2, byteorder='little', signed=False)
        self.resource.write_raw(cmd)

        # wait 1s for changes to be effective
        sleep(1)

    def get_list_of_traces(self, **kwargs) -> list:
        """
        Returns a list of dictionaries describing all available measurement traces. In case of the NanoVNA_v2, this is
        just a static list of the two measurement channels `[{'channel': 0, 'parameter': 'S11'},
        {'channel': 1, 'parameter': 'S21'}]`.

        Parameters
        ----------
        kwargs : dict, optional

        Returns
        -------
        list
        """

        return [{'channel': 0, 'parameter': 'S11'},
                {'channel': 1, 'parameter': 'S21'}]

    def get_snp_network(self, ports, **kwargs):
        pass

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        pass
