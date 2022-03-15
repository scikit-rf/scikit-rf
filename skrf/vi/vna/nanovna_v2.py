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
    Python class for NanoVNA_v2 network analyzers [#website]_.

    Examples
    --------
    Load and initialize NanoVNA on `COM1` (Windows OS, see Device Manager):

    >>> from skrf.vi import vna
    >>> nanovna = vna.NanoVNAv2('ASRL1::INSTR')

    Load and initialize NanoVNA on `/dev/ttyACM0` (Linux OS, see dmesg):

    >>> from skrf.vi import vna
    >>> nanovna = vna.NanoVNAv2('ASRL/dev/ttyACM0::INSTR')

    Configure frequency sweep (from 20 MHz to 4 GHz with 200 points, i.e. 20 MHz step):

    >>> nanovna.set_frequency_sweep(20e6, 4e9, 200)

    Get S11 and S21 as NumPy arrays:

    >>> s11, s21 = nanovna.get_s11_s21()

    Get list of available traces (will always return both channels, regardless of trace configuration):

    >>> traces_avail = nanovna.get_list_of_traces()

    Get 1-port networks of one or both of the traces listed in `get_list_of_traces()`:

    >>> nws_all = nanovna.get_traces(traces_avail)
    >>> nw_s11 = nws_all[0]
    >>> nw_s21 = nws_all[1]

    Get S11 as a 1-port skrf.Network:

    >>> nw_1 = nanovna.get_snp_network(ports=(0,))

    Get S11 and S12 as s 2-port skrf.Network (incomplete with S21=S22=0):

    >>> nw_2 = nanovna.get_snp_network(ports=(0, 1))

    Get S21 and S22 in a 2-port skrf.Network (incomplete with S11=S12=0):

    >>> nw_3 = nanovna.get_snp_network(ports=(1, 0))

    References
    ----------
    .. [website] Website of NanoVNA V2: https://nanorfe.com/nanovna-v2.html
    """

    def __init__(self, serial_port: str = '/dev/ttyACM0'):
        super().__init__(address='ASRL/dev/ttyACM0::INSTR', kwargs={'visa_library': 'py'})
        self._protocol_reset()
        self._frequency = np.linspace(1e6, 10e6, 101)
        self.set_frequency_sweep(1e6, 10e6, 101)

    def idn(self) -> str:
        """
        Returns the identification string of the device.

        Returns
        -------
        str
            Identification string, e.g. `NanoVNA_v2`.
        """

        # send 1-byte READ (0x10) of address 0xf0 to retrieve device variant code
        self.resource.write_raw([0x10, 0xf0])
        v_byte = self.resource.read_bytes(1)
        v = int.from_bytes(v_byte, byteorder='little')

        if v == 2:
            return 'NanoVNA_v2'
        else:
            return 'Unknown device, got deviceVariant={}'.format(v)

    def reset(self):
        raise NotImplementedError

    def wait_until_finished(self):
        raise NotImplementedError

    def _protocol_reset(self):
        # send 8x NOP (0x00) to reset the communication protocol
        self.resource.write_raw([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    def get_s11_s21(self) -> (np.ndarray, np.ndarray):
        """
        Returns individual NumPy arrays of the measured data of the sweep. Being a 1.5-port analyzer, the results only
        include :math:`S_{1,1}` and :math:`S_{2,1}`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            List of NumPy arrays with :math:`S_{1,1}` and :math:`S_{2,1}`.

        Notes
        -----
        Regardless of the calibration state of the NanoVNA, the results returned by this method are always raw, i.e.
        uncalibrated. The user needs to apply a manual calibration in postprocessing, if required.

        See Also
        --------
        set_frequency_sweep
        get_traces
        :mod:`Calibration`
        """

        # data is continuously being sampled and stored in the FIFO (address 0x30)
        # writing any value to 0x30 clears the FIFO, which enables on-demand readings

        f_points = len(self._frequency)

        # write any byte to register 0x30 to clear FIFO
        self.resource.write_raw([0x20, 0x30, 0x00])

        data_raw = []
        f_remaining = f_points

        while f_remaining > 0:

            # can only read 255 values in one take
            if f_remaining > 255:
                len_segment = 255
            else:
                len_segment = f_remaining
            f_remaining = f_remaining - len_segment

            # read 'len_segment' values from FIFO (32 * len_segment bytes)
            self.resource.write_raw([0x18, 0x30, len_segment])
            data_raw.extend(self.resource.read_bytes(32 * len_segment))

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

        return data_s11, data_s21

    def set_frequency_sweep(self, start_freq: float, stop_freq: float, num_points: int = 201, **kwargs) -> None:
        """
        Configures the frequency sweep. Only linear spacing is supported.

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

        f_step = 0.0
        if num_points > 1:
            f_step = (stop_freq - start_freq) / (num_points - 1)

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

    def get_traces(self, traces: list = None, **kwargs) -> list:
        """
        Returns the data of the traces listed in `traces` as 1-port networks.

        Parameters
        ----------
        traces: list of dict, optional
            Traces selected from :func:`get_list_of_traces`.

        kwargs: list

        Returns
        -------
        list of skrf.Network
            List with trace data as individual 1-port networks.
        """

        data_s11, data_s21 = self.get_s11_s21()
        frequency = skrf.Frequency.from_f(self._frequency, unit='hz')
        nw_s11 = skrf.Network(frequency=frequency, s=data_s11, name='Trace0')
        nw_s21 = skrf.Network(frequency=frequency, s=data_s21, name='Trace1')

        traces_valid = self.get_list_of_traces()
        networks = []
        for trace in traces:
            if trace in traces_valid:
                if trace['channel'] == 0:
                    networks.append(nw_s11)
                elif trace['channel'] == 1:
                    networks.append(nw_s21)

        return networks

    def get_snp_network(self, ports: tuple = (1, 2), **kwargs) -> skrf.Network:
        """
        Returns a :math:`N`-port network with the measured responses saved at the positions specified in `ports`.
        The remaining responses will be 0. This can be useful for sliced measurements of larger networks with an
        analyzer that does not have enough ports.
        For example when measuring a 3-port (e.g a balun) with the 1.5-port NanoVNA, the required six individual
        measurements with the slices (s11, s21), (_, s31), (s22, s12), (_, s32), (s33, s13), and (_, s23) can be
        obtained directly as (incomplete) 3-port networks, which can later be combined very easily by adding them
        together: nw_balun = nw_s1 + nw_s2 + ... + nw_s6.

        Parameters
        ----------
        ports: tuple of int or None, optional
            Specifies the position, order, and type of the measured responses in the returned N-port network. The
            length of the tuple defines the size N of the network, the entries define the type and position. Valid
            entries are `0`, `1`, or `None`. Number `0` refers to `s11` from the measurement, `1` refers to `s21`,
            and `None` skips this position (required to increase `N`). See examples below.

        kwargs: list

        Returns
        -------
        skrf.Network

        Examples
        --------
        >>> from skrf.vi import vna
        >>> nanovna = vna.NanoVNAv2()

        Get a 1-port network with `s11`, i.e. [s11_meas]:

        >>> nw = nanovna.get_snp_network(ports=(0, ))

        Get a 1-port network with `s21`, i.e. [s21_meas]:

        >>> nw = nanovna.get_snp_network(ports=(1, ))

        Get a 2-port network (incomplete) with `(s11, s12) = measurement, (s21, S22) = 0`,
        i.e. [[s11_meas, s21_meas], [0, 0]]:

        >>> nw = nanovna.get_snp_network(ports=(0, 1))

        Get a 2-port network (incomplete) with `(s21, s22) = measurement, (s11, S12) = 0`,
        i.e. [[0, 0], [s21_meas, s11_meas]]:

        >>> nw = nanovna.get_snp_network(ports=(1, 0))

        Get a 3-port network (incomplete) with `(s31, s33) = measurement, rest = 0`,
        i.e. [[0, 0, 0], [0, 0, 0], [s21_meas, 0, s11_meas]]:

        >>> nw = nanovna.get_snp_network(ports=(1, None, 0))

        Get a 3-port network (incomplete) with `(s32, s33) = measurement, rest = 0`,
        i.e. [[0, 0, 0], [0, 0, 0], [0, s21_meas, s11_meas]]:

        >>> nw = nanovna.get_snp_network(ports=(None, 1, 0))

        Get a 3-port network (incomplete) with `(s22, s23) = measurement, rest = 0`,
        i.e. [[0, 0, 0], [0, s11_meas, s21_meas], [0, 0, 0]]:

        >>> nw = nanovna.get_snp_network(ports=(None, 0, 1))

        Get a 3-port network (incomplete) with `(s11, s12) = measurement, rest = 0`,
        i.e. [[s11_meas, s21_meas, 0], [0, 0, 0], [0, 0, 0]]:

        >>> nw = nanovna.get_snp_network(ports=(0, 1, None))

        Get a 3-port network (incomplete) with only `(s13) = measurement, rest = 0`,
        i.e. [[0, 0, s21_meas], [0, 0, 0], [0, 0, 0]]:

        >>> nw = nanovna.get_snp_network(ports=(None, None, 1))
        """

        data_s11, data_s21 = self.get_s11_s21()
        frequency = skrf.Frequency.from_f(self._frequency, unit='hz')
        s = np.zeros((len(frequency), len(ports), len(ports)), dtype=complex)

        # get trace indices from 'ports' (without None)
        j = []
        i = 0
        for i_port, port in enumerate(ports):
            if port is not None:
                j.append(i_port)
                if port == 0:
                    i = i_port
        if max(j) > 2:
            raise ValueError('Port index p>2 is not possible with a 1.5 port analyzer.')

        # populate N-port network with s11 and s21
        k = 0
        for _, port in enumerate(ports):
            if port is not None:
                if port == 0:
                    s[:, i, j[k]] = data_s11
                elif port == 1:
                    s[:, i, j[k]] = data_s21
                else:
                    raise ValueError('Invalid port index `{}` in ports'.format(port))
                k += 1

        return skrf.Network(frequency=frequency, s=s)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        raise NotImplementedError
