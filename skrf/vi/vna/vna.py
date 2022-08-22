"""
.. module:: skrf.vi.vna

Abstract base class for VNAs. Do not use directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import pyvisa
from pyvisa.constants import RENLineOperation
from skrf.calibration import Calibration
from skrf.frequency import Frequency
from skrf.network import Network


@dataclass
class Measurement:
    name: str
    param: str
    channel: Optional[int] = None


class VNA(ABC):
    """
    Abstract base class for VNAs

    This class defines the interface to be provided by all network analyzer
    implementations.
    """

    def __init__(self, address: str, timeout: int = 3000, backend: str = "@py") -> None:
        """
        Initialize a network analyzer object

        Parameters
        ----------
        address : str
            a visa resource string. See `NI's documentation on syntax`_
        timeout : int
            communication timeout in milliseconds
        backend: str
            path to visa implentation to be used as the backend. "@py" means use
            the pyvisa Python implementation (the default).

        Notes
        -----

        .. _NI VISA Address syntax:
            https://www.ni.com/docs/en-US/bundle/ni-visa/page/ni-visa/visaresourcesyntaxandexamples.html
        """
        rm = pyvisa.ResourceManager(backend)
        self.resource = rm.open_resource(address)

        if isinstance(self.resource, pyvisa.resources.MessageBasedResource):
            self.read_termination = self.resource.read_termination
            self.write_termination = self.resource.write_termination

            self.read_termination = "\n"
            self.write_termination = "\n"

            self.write = self.resource.write
            self.write_ascii = self.resource.write_ascii_values
            self.write_binary = self.resource.write_binary_values

            self.query = self.resource.query
            self.query_ascii = self.resource.query_ascii_values
            self.query_binary = self.resource.query_binary_values

        else:
            raise NotImplementedError(
                "Only message based resources are implemented at this time."
            )

        if isinstance(self.resource, pyvisa.resources.GPIBInstrument):
            self.resource.control_ren(RENLineOperation.deassert_gtl)

    def __enter__(self) -> VNA:
        """
        context manager entry point

        Returns
        -------
        VNA
            the Analyzer Driver Object
        """
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        context manager exit point

        Parameters
        ----------
        *args
        **kwargs
        """
        self.resource.close()

    def close(self) -> None:
        """
        Close the device connection
        """
        self.__exit__(None, None, None)

    def reset(self) -> None:
        """
        Reset the device to default
        """
        self.write("*RST")

    def id_string(self) -> str:
        """
        Query the device identifier string

        Returns
        -------
        str
            device identifier string
        """
        return self.query("*IDN?")

    @abstractmethod
    def start_freq(self) -> float:
        """
        Get start frequency

        Returns
        -------
        float
            start frequency [Hz]
        """
        pass

    @abstractmethod
    def set_start_freq(self, f: float) -> None:
        """
        Set start frequency

        Parameters
        ----------
        f : float
            start frequency [Hz]
        """
        pass

    @abstractmethod
    def stop_freq(self) -> float:
        """
        Get stop frequency

        Returns
        -------
        float
            stop frequency [Hz]
        """
        pass

    @abstractmethod
    def set_stop_freq(self, f: float) -> None:
        """
        Set stop frequency

        Parameters
        ----------
        f : float
            stop frequency [Hz]
        """
        pass

    @abstractmethod
    def npoints(self) -> int:
        """
        Get number of frequency points

        Returns
        -------
        int
            number of frequency points
        """
        pass

    @abstractmethod
    def set_npoints(self, n: int) -> None:
        """
        Set number of frequency points

        Parameters
        ----------
        n : int
            number of frequency points
        """
        pass

    @abstractmethod
    def freq_step(self) -> float:
        """
        Get frequency step

        Returns
        -------
        float
            frequency step [Hz]
        """
        pass

    @abstractmethod
    def set_freq_step(self, f: float) -> None:
        """
        Set frequency step

        Parameters
        ----------
        f : float
            frequency step [Hz]
        """
        pass

    def frequency(self) -> Frequency:
        """
        Get current frequency as :class:`Frequency` object

        Returns
        -------
        Frequency
            current frequency settings as frequency object
        """
        start = self.start_freq()
        stop = self.stop_freq()
        npoints = self.npoints()
        return Frequency(start, stop, npoints, unit="hz")

    def set_frequency(
        self,
        freq: Optional[Frequency] = None,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        npoints: Optional[int] = None,
    ) -> None:
        """
        Set current frequency

        The user can provide **either** a frequency object **or** start, stop
        **and** npoints

        Parameters
        ----------
        freq :
            Frequency object
        start :
            start frequency [Hz]
        stop :
            stop frequency [Hz]
        npoints :
            number of frequency points

        Raises
        ------
        ValueError
            If a frequency object is passed with any other parameters or if not
            all of start, stop, and npoints are provided
        """
        if freq and any((start, stop, npoints)):
            raise ValueError(
                "Got too many arguments. Pass either Frequency object or start, stop, and step."
            )
        if not freq and not all((start, stop, npoints)):
            raise ValueError(
                "Got too few arguments. Pass either Frequency object or start, stop, and step."
            )

        if freq:
            self.set_start_freq(freq.start)
            self.set_stop_freq(freq.stop)
            self.set_npoints(freq.npoints)
        else:
            self.set_start_freq(start)  # type: ignore
            self.set_stop_freq(stop)  # type: ignore
            self.set_npoints(npoints)  # type: ignore

    @abstractmethod
    def sweep_mode(self) -> str:
        """
        Get the current sweep mode

        This is typically to hold, continuous, etc.

        Returns
        -------
        str
            current sweep mode
        """
        pass

    @abstractmethod
    def set_sweep_mode(self, mode: str) -> None:
        """
        Set the sweep mode

        This is typically to set the sweep to hold, continuous, etc.

        Parameters
        ----------
        mode : str
            sweep mode
        """
        pass

    @abstractmethod
    def sweep_type(self) -> str:
        """
        Get the current frequency sweep type

        This is typically linear, logarithmic, etc.

        Returns
        -------
        str
            Current sweep type
        """
        pass

    @abstractmethod
    def set_sweep_type(self, _type: str) -> None:
        """
        Set the type of frequency sweep

        This is typically to set to linear, logarithmic, etc.

        Parameters
        ----------
        _type : str
            type of frequency sweep
        """
        pass

    @abstractmethod
    def sweep_time(self) -> float:
        """
        Get the current sweep time

        Returns
        -------
        float
            duration of a single sweep [s]
        """
        pass

    @abstractmethod
    def set_sweep_time(self, time: Union[float, str]) -> None:
        """
        Set the duration of a single sweep

        Parameters
        ----------
        time : Union[float, str]
            length of time to set a single sweep [s]
        """
        pass

    @abstractmethod
    def if_bandwidth(self) -> float:
        """
        Get the current IF bandwidth

        Returns
        -------
        float
            current IF bandwidth [Hz]
        """
        pass

    @abstractmethod
    def set_if_bandwidth(self, bw: float) -> None:
        """
        Set the IF bandwidth

        Parameters
        ----------
        bw : float
            desired IF bandwidth [Hz]
        """
        pass

    @abstractmethod
    def averaging_on(self) -> bool:
        """
        Checks if averaging is on or off

        Returns
        -------
        bool
            True if averaging is on, False otherwise
        """
        pass

    @abstractmethod
    def set_averaging_on(self, onoff: bool) -> None:
        """
        Sets averaging on or off

        Parameters
        ----------
        onoff : bool
            True to turn on averaging, False to turn it off
        """
        pass

    @abstractmethod
    def average_count(self) -> int:
        """
        Get the current averaging count

        Returns
        -------
        int
            The current averaging count
        """
        pass

    @abstractmethod
    def set_average_count(self, n: int) -> None:
        """
        Sets the averaging count

        Parameters
        ----------
        n : int
            desired averaging count
        """
        pass

    @abstractmethod
    def average_mode(self) -> str:
        """
        Get the current averaging mode

        This is typically point, sweep, etc.

        Returns
        -------
        str
            current averaging mode
        """
        pass

    @abstractmethod
    def set_average_mode(self, mode: str) -> None:
        """
        Set the averaging mode

        Parameters
        ----------
        mode : str
            desired averaging mode
        """
        pass

    @abstractmethod
    def clear_averaging(self) -> None:
        """
        Clear the averaging values
        """
        pass

    @abstractmethod
    def num_sweep_groups(self) -> int:
        """
        Get the current number of sweep groups

        Returns
        -------
        int
            current number of sweep groups
        """
        pass

    @abstractmethod
    def set_num_sweep_groups(self, n: int) -> None:
        """
        Set the number of sweep groups

        Parameters
        ----------
        n : int
            desired number of sweep groups
        """
        pass

    @property
    @abstractmethod
    def channels_in_use(self) -> List[int]:
        """
        Get the channels currently in use

        Returns
        -------
        List[int]
            list of the channels currently in use
        """
        pass

    @property
    @abstractmethod
    def active_channel(self) -> Optional[int]:
        """
        Get the active channel

        Returns
        -------
        Optional[int]
            the number of the active channel. `None` if no channels are active
        """
        pass

    @active_channel.setter
    @abstractmethod
    def active_channel(self, channel: int) -> None:
        """
        Set the active channel

        Parameters
        ----------
        channel : int
            the desired channel
        """
        pass

    @abstractmethod
    def measurements_on_channel(self, channel: int) -> List[Measurement]:
        """
        Get the list of measurements active on the specified channel

        Parameters
        ----------
        channel : int
            the channel in question

        Returns
        -------
        List[Measurement]
            List of measurement objects with measurement name, parameter, and channel
        """
        pass

    @property
    def measurements(self) -> List[Measurement]:
        """
        Get a list of all current measurements

        Returns
        -------
        List[Measurement]
            List of all measurements (name, parameter, channel) currently
            defined on all channels
        """
        channels = self.channels_in_use
        msmts = []
        for chan in channels:
            msmts += self.measurements_on_channel(chan)

        return msmts

    def measurement_names(self, channel: int = 1) -> List[str]:
        """
        Get the names of all measurements on a channel

        Parameters
        ----------
        channel : int
            the channel in question

        Returns
        -------
        List[str]
            list of measurement names
        """
        return [m.name for m in self.measurements if m.channel == channel]

    @property
    @abstractmethod
    def active_measurement(self) -> Optional[str]:
        """
        Get the active measurement

        Returns
        -------
        Optional[str]
            the active measurement name. None if no measurement is active
        """
        pass

    @abstractmethod
    def set_active_measurement(self, meas: Union[int, str]) -> None:
        """
        Set the active measurement

        Parameters
        ----------
        meas : Union[int, str]
            the name or number of the desired measurement
        """
        pass

    @abstractmethod
    def create_measurement(self, name: str, param: str) -> None:
        """
        Create a measurement

        Parameters
        ----------
        name : str
            what to name the measurement
        param : str
            the measurement parameter (S11, S21, A, etc.)
        """
        pass

    @abstractmethod
    def delete_measurement(self, name: str) -> None:
        """
        Delete a measurement

        Parameters
        ----------
        name : str
            name of the measurement to be deleted
        """
        pass

    @abstractmethod
    def get_measurement(self, meas: Union[int, str]) -> Network:
        """
        Get measurement data as a `Network`

        Parameters
        ----------
        meas : Union[int, str]
            name or number of the measurement to get

        Returns
        -------
        Network
            one-port network representing measurement data
        """
        pass

    @abstractmethod
    def get_active_trace(self) -> Network:
        """
        Get the active trace data as a `Network`

        Returns
        -------
        Network
            one-port network representing measurement data
        """
        pass

    @property
    @abstractmethod
    def snp_format(self) -> str:
        """
        Get the current SNP format

        This is one of Real/Imaginary, Magnitude/Angle, etc.

        Returns
        -------
        str:
            current SNP format
        """
        pass

    @snp_format.setter
    @abstractmethod
    def snp_format(self, format: str) -> None:
        """
        Set the SNP format

        Parameters
        ----------
        format : str
            desired SNP format
        """
        pass

    @property
    @abstractmethod
    def ports(self) -> List[str]:
        """
        Get the list of available ports

        Returns
        -------
        List[str]
            available ports
        """
        pass

    @property
    @abstractmethod
    def sysinfo(self) -> str:
        """
        Get interesting / important information about the instrument

        This can be information like the instrument name, SCPI version,
        available ports, frequency limitations, power limitations, etc.

        Returns
        -------
        str
            instrument information
        """
        pass

    @abstractmethod
    def sweep(self) -> None:
        """
        Trigger a fresh sweep
        """
        pass

    @abstractmethod
    def get_snp_network(self, ports: Optional[Sequence] = None) -> Network:
        """
        Get the full SNP network of the specified ports

        For example, if [1, 2] is passed, the Network would be the full 2-port
        network (i.e. S11, S12, S21, S22)

        Parameters
        ----------
        ports : Optional[Sequence]
            list (or set or tuple...) of ports to get S-network of. `None` gets
            all ports

        Returns
        -------
        Network
            network object containing full S network data
        """
        pass

    @abstractmethod
    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        """
        Upload one-port calibration to instrument and apply to a port

        Parameters
        ----------
        port : int
            the port to apply the calibration to
        cal : Calibration
            the calibration to apply
        """
        pass

    @abstractmethod
    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        """
        Upload a twoport calibration to the instrument and apply to the ports
        specified

        Parameters
        ----------
        port : Sequence
            the ports to apply the calibration to
        cal : Calibration
            the calibration to apply
        """
        pass
