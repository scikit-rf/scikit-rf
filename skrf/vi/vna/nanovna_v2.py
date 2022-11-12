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
    Python class for NanoVNA V2 network analyzers [#website]_.

    Parameters
    ----------
    address : str
            SCPI identifier of the serial port for the NanoVNA. For example `'ASRL1::INSTR'` for `COM1` on Windows, or
            `'ASRL/dev/ttyACM0::INSTR'` for `/dev/ttyACM0` on Linux.

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
    .. [#website] Website of NanoVNA V2: https://nanorfe.com/nanovna-v2.html
    """

    def __init__(self, address: str = 'ASRL/dev/ttyACM0::INSTR'):
        super().__init__(address=address, visa_library='@py')
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
            return f'Unknown device, got deviceVariant={v}'

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

    def get_snp_network(self, ports: tuple = (0, 1), **kwargs) -> skrf.Network:
        """
        Returns a :math:`N`-port network containing the measured parameters at the positions specified in `ports`.
        The remaining responses will be 0. The rows and the column to be populated in the network are selected
        implicitly based on the position and the order of the entries in `ports`. See the parameter desciption for
        details.
        This function can be useful for sliced measurements of larger networks with an analyzer that does not have
        enough ports, for example when measuring a 3-port (e.g a balun) with the 1.5-port NanoVNA (example below).

        Parameters
        ----------
        ports: tuple of int or None, optional
            Specifies the position and order of the measured responses in the returned `N`-port network. Valid entries
            are `0`, `1`, or `None`. The length of the tuple defines the size `N` of the network, the entries
            define the type (forward/reverse) and position (indices of the rows and the column to be populated).
            Number `0` refers to the source port (`s11` from the NanoVNA), `1` refers to the receiver port (`s21` from
            the NanoVNA), and `None` skips this position (required to increase `N`). For `N>1`, the colum index is
            determined by the position of the source port `0` in `ports`. See examples below.

        kwargs: list
            Additional parameters will be ignored.

        Returns
        -------
        skrf.Network

        Examples
        --------
        To get the measured S-matrix of a 3-port from six individual measurements, the slices (s11, s21), (s11, s31),
        (s12, s22), (s22, s32), (s13, s33), and (s23, s33) can be obtained directly as (incomplete) 3-port networks
        with the results stored at the correct positions, which helps combining them afterwards.

        >>> from skrf.vi import vna
        >>> nanovna = vna.NanoVNAv2()

        1st slice: connect VNA_P1=P1 and VNA_P2=P2 to measure s11 and s21:

        >>> nw_s1 = nanovna.get_snp_network(ports=(0, 1, None))

        This will return a 3-port network with [[s11_vna, 0, 0], [s21_vnas, 0, 0], [0, 0, 0]].

        2nd slice: connect VNA_P1=P1 and VNA_P2=P3 to measure s11 and s31:

        >>> nw_s2 = nanovna.get_snp_network(ports=(0, None, 1))

        This will return a 3-port network with [[s11_vna, 0, 0], [0, 0, 0], [s21_vna, 0, 0]].

        3rd slice: connect VNA_P1=P2 and VNA_P2=P1 to measure s22 and s12:

        >>> nw_s3 = nanovna.get_snp_network(ports=(1, 0, None))

        This will return a 3-port network with [[0, s21_vna, 0], [0, s11_vna, 0], [0, 0, 0]].

        4th slice: connect VNA_P1=P2 and VNA_P2=P3 to measure s22 and s32:

        >>> nw_s4 = nanovna.get_snp_network(ports=(None, 0, 1))

        This will return a 3-port network with [[0, 0, 0], [0, s11_vna, 0], [0, s21_vna, 0]].

        5th slice: connect VNA_P1=P3 and VNA_P2=P1 to measure s13 and s33:

        >>> nw_s5 = nanovna.get_snp_network(ports=(1, None, 0))

        This will return a 3-port network with [[0, 0, s21_vna], [0, 0, 0], [0, 0, s11_vna]].

        6th slice: connect VNA_P1=P3 and VNA_P2=P2 to measure s23 and s33:

        >>> nw_s6 = nanovna.get_snp_network(ports=(None, 1, 0))

        This will return a 3-port network with [[0, 0, 0], [0, 0, s21_vna], [0, 0, s11_vna]].

        Now, the six incomplete networks can simply be added to get to complete network of the 3-port:

        >>> nw = nw_s1 + nw_s2 + nw_s3 + nw_s4 + nw_s5 + nw_s6

        The reflection coefficients s11, s22, s33 have been measured twice, so the sum still needs to be divided by 2
        to get the correct result:

        >>> nw.s[:, 0, 0] = 0.5 * nw.s[:, 0, 0]
        >>> nw.s[:, 1, 1] = 0.5 * nw.s[:, 1, 1]
        >>> nw.s[:, 2, 2] = 0.5 * nw.s[:, 2, 2]

        This gives the average, but you could also replace it with just one of the measurements.

        This function can also be used for smaller networks:

        Get a 1-port network with `s11`, i.e. [s11_meas]:

        >>> nw = nanovna.get_snp_network(ports=(0, ))

        Get a 1-port network with `s21`, i.e. [s21_meas]:

        >>> nw = nanovna.get_snp_network(ports=(1, ))

        Get a 2-port network (incomplete) with `(s11, s21) = measurement, (s12, S22) = 0`,
        i.e. [[s11_meas, 0], [s21_meas, 0]]:

        >>> nw = nanovna.get_snp_network(ports=(0, 1))

        Get a 2-port network (incomplete) with `(s12, s22) = measurement, (s11, S21) = 0`,
        i.e. [[0, s21_meas], [0, s11_meas]]:

        >>> nw = nanovna.get_snp_network(ports=(1, 0))
        """

        # load s11, s21 from NanoVNA
        data_s11, data_s21 = self.get_s11_s21()
        frequency = skrf.Frequency.from_f(self._frequency, unit='hz')

        # prepare empty S matrix to be populated
        s = np.zeros((len(frequency), len(ports), len(ports)), dtype=complex)

        # get trace indices from 'ports' (without None)
        rows = []
        col = -1
        for i_port, port in enumerate(ports):
            if port is not None:
                # get row indices directly from entries in `ports`
                rows.append(i_port)

                # try to get column index from from position of `0` entry (if present)
                if port == 0:
                    col = i_port

        if col == -1:
            # `0` entry was not present to specify the column index
            if len(ports) == 1:
                # not a problem; it's a 1-port
                col = 0
            else:
                # problem: column index is ambiguous
                raise ValueError('Source port index `0` is missing in `ports` with length > 1. Column index is ambiguous.')

        # populate N-port network with s11 and s21
        k = 0
        for _, port in enumerate(ports):
            if port is not None:
                if port == 0:
                    s[:, rows[k], col] = data_s11
                elif port == 1:
                    s[:, rows[k], col] = data_s21
                else:
                    raise ValueError(f'Invalid port index `{port}` in `ports`')
                k += 1

        return skrf.Network(frequency=frequency, s=s)

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        raise NotImplementedError

    @property
    def s11(self) -> skrf.Network:
        """
        Measures :math:`S_{1,1}` and returns it as a 1-port Network.

        Returns
        -------
        skrf.Network
        """
        traces = self.get_list_of_traces()
        ntwk = self.get_traces([traces[0]])[0]
        ntwk.name = 'NanoVNA_S11'
        return ntwk

    @property
    def s21(self) -> skrf.Network:
        """
        Measures :math:`S_{2,1}` and returns it as a 1-port Network.

        Returns
        -------
        skrf.Network
        """
        traces = self.get_list_of_traces()
        ntwk = self.get_traces([traces[1]])[0]
        ntwk.name = 'NanoVNA_S21'
        return ntwk
