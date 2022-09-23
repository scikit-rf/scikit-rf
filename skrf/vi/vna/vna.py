"""
Abstract base class for VNAs. Do not use directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

from abc import ABC, abstractmethod

import pyvisa
from pyvisa.constants import RENLineOperation
from skrf.calibration import Calibration
from skrf.frequency import Frequency
from skrf.network import Network


class UnsupportedError(RuntimeError):
    """Error raised when an instrument doesn't support something"""

    pass


class VNA(ABC):
    """
    Abstract base class for VNAs

    This class defines the interface to be provided by all network analyzer
    implementations.

    The instrument's manual should be consulted when subclassing. This base
    class only provides the expected interface any subclass should implement. If
    a device does not support a function, the default is to raise `UnsupportedError`.

    Finally, subclasses should check that arguments are sensible to the
    instrument they represent. For example, if an instrument only has two ports,
    the user should only be able to create measurements with S11, S12, S21, S22
    as parameters (as well as other supported parameters like A, B, etc.)
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

        .. _visa-address-syntax: https://www.ni.com/docs/en-US/bundle/ni-visa/page/ni-visa/visaresourcesyntaxandexamples.html
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

    @property
    def start_freq(self) -> float:
        """
        Get start frequency

        Returns
        -------
        float
            start frequency [Hz]
        """
        raise UnsupportedError

    @start_freq.setter
    def start_freq(self, f: float) -> None:
        """
        Set start frequency

        Parameters
        ----------
        f : float
            start frequency [Hz]
        """
        raise UnsupportedError

    @property
    def stop_freq(self) -> float:
        """
        Get stop frequency

        Returns
        -------
        float
            stop frequency [Hz]
        """
        raise UnsupportedError

    @stop_freq.setter
    def stop_freq(self, f: float, channel: int = 1) -> None:
        """
        Set stop frequency

        Parameters
        ----------
        f : float
            stop frequency [Hz]
        """
        raise UnsupportedError

    @property
    def npoints(self) -> int:
        """
        Get number of frequency points

        Returns
        -------
        int
            number of frequency points
        """
        raise UnsupportedError

    @npoints.setter
    def npoints(self, n: int) -> None:
        """
        Set number of frequency points

        Parameters
        ----------
        n : int
            number of frequency points
        """
        raise UnsupportedError

    @property
    def freq_step(self) -> float:
        """
        Get frequency step

        Returns
        -------
        float
            frequency step [Hz]
        """
        raise UnsupportedError

    @freq_step.setter
    def freq_step(self, f: float) -> None:
        """
        Set frequency step

        Parameters
        ----------
        f : float
            frequency step [Hz]
        """
        raise UnsupportedError

    @property
    def frequency(self) -> Frequency:
        """
        Get current frequency as :class:`Frequency` object

        Returns
        -------
        Frequency
            current frequency settings as frequency object
        """
        raise UnsupportedError

    @frequency.setter
    def frequency(
        self,
        frequency: Frequency,
    ) -> None:
        """
        Set current frequency from a skrf Frequency object

        Parameters
        ----------
        frequency :
            Frequency object
        """
        raise UnsupportedError

    @property
    def sweep_mode(self) -> str:
        """
        Get the current sweep mode

        This is typically hold, continuous, etc.

        Returns
        -------
        str
            current sweep mode
        """
        raise UnsupportedError

    @sweep_mode.setter
    def sweep_mode(self, mode: str) -> None:
        """
        Set the sweep mode

        This is typically to set the sweep to hold, continuous, etc.

        Parameters
        ----------
        mode : str
            sweep mode
        """
        raise UnsupportedError

    @property
    def sweep_type(self) -> str:
        """
        Get the current frequency sweep type

        This is typically linear, logarithmic, etc.

        Returns
        -------
        str
            Current sweep type
        """
        raise UnsupportedError

    @sweep_type.setter
    def set_sweep_type(self, type_: str) -> None:
        """
        Set the type of frequency sweep

        This is typically to set to linear, logarithmic, etc.

        Parameters
        ----------
        type_ : str
            type of frequency sweep
        """
        raise UnsupportedError

    @property
    def sweep_time(self) -> float:
        """
        Get the current sweep time

        Returns
        -------
        float
            duration of a single sweep [s]
        """
        raise UnsupportedError

    @sweep_time.setter
    def sweep_time(self, time: float) -> None:
        """
        Set the duration of a single sweep

        Parameters
        ----------
        time : Union[float, str]
            length of time to set a single sweep [s]
        """
        raise UnsupportedError

    @property
    def if_bandwidth(self) -> float:
        """
        Get the current IF bandwidth

        Returns
        -------
        float
            current IF bandwidth [Hz]
        """
        raise UnsupportedError

    @if_bandwidth.setter
    def if_bandwidth(self, bw: float) -> None:
        """
        Set the IF bandwidth

        Parameters
        ----------
        bw : float
            desired IF bandwidth [Hz]
        """
        raise UnsupportedError

    @property
    def averaging_on(self) -> bool:
        """
        Checks if averaging is on or off

        Returns
        -------
        bool
            True if averaging is on, False otherwise
        """
        raise UnsupportedError

    @averaging_on.setter
    def averaging_on(self, state: bool) -> None:
        """
        Sets averaging on or off

        Parameters
        ----------
        state : bool
            True to turn on averaging, False to turn it off
        """
        raise UnsupportedError

    @property
    def average_count(self) -> int:
        """
        Get the current averaging count

        Returns
        -------
        int
            The current averaging count
        """
        raise UnsupportedError

    @average_count.setter
    def average_count(self, n: int) -> None:
        """
        Sets the averaging count

        Parameters
        ----------
        n : int
            desired averaging count
        """
        raise UnsupportedError

    @property
    def average_mode(self) -> str:
        """
        Get the current averaging mode

        Returns
        -------
        str
            The current averaging mode
        """
        raise UnsupportedError

    @average_mode.setter
    def average_mode(self, mode: str) -> None:
        """
        Sets the current averaging mode

        Parameters
        -------
        mode : str
            The desired averaging mode
        """

    def clear_averaging(self) -> None:
        """
        Clear the averaging values
        """
        raise UnsupportedError

    @property
    def active_channel(self) -> int:
        """
        Get the currently active channel

        Returns
        -------
        int
            Currently active channel
        """
        raise UnsupportedError

    @active_channel.setter
    def active_channel(self, channel: int) -> None:
        """
        Set the currently active channel

        Parameters
        ----------
        channel : int
            Desired channel
        """
        raise UnsupportedError

    def measurements_on_channel(self, channel: int) -> List[Tuple]:
        """
        Get a list of measurements on the specified channel

        Parameters
        ----------
        channel : int
            The channel in question

        Returns
        -------
        List[Tuple]
            List of measurements currently defined on the specified channel.
            Each element of the list is a Tuple containing (in order), the
            measurement name or number, measurement parameter
        """
        raise UnsupportedError

    @property
    def measurements(self) -> List[Tuple]:
        """
        Get a list of all current measurements

        Returns
        -------
        List[Tuple]
            List of all measurements currently defined. Each element of the list
            is a Tuple containing (in order), the measurement name or number,
            measurement parameter, and channel if available.
        """
        raise UnsupportedError

    @property
    def active_measurement(self) -> Optional[Union[str, int]]:
        """
        Get the active measurement

        Returns
        -------
        Optional[str, int]
            the active measurement or trace number. None if no measurement is
            active
        """
        raise UnsupportedError

    @active_measurement.setter
    def active_measurement(self, id_: Union[int, str]) -> None:
        """
        Set the active measurement

        Parameters
        ----------
        id_ : Union[int, str]
            the name or number of the desired measurement
        """
        raise UnsupportedError

    def create_measurement(self, id_: Union[str, int], param: str) -> None:
        """
        Create a measurement

        Parameters
        ----------
        id_ : Union[str, int]
            the name or id of the new measurement
        param : str
            the measurement parameter (S11, S21, A, etc.)
        """
        raise UnsupportedError

    def delete_measurement(self, id_: Union[str, int]) -> None:
        """
        Delete a measurement

        Parameters
        ----------
        id_ : Union[str, int]
            name or number of the measurement to be deleted
        """
        raise UnsupportedError

    def get_measurement(self, id_: Union[int, str]) -> Network:
        """
        Get measurement data as a `Network`

        Parameters
        ----------
        id_ : Union[int, str]
            name or number of the measurement to get

        Returns
        -------
        Network
            one-port network representing measurement data
        """
        raise UnsupportedError

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
    def snp_format(self) -> str:
        """
        Get the current SNP format

        This is one of Real/Imaginary, Magnitude/Angle, etc.

        Returns
        -------
        str:
            current SNP format
        """
        raise UnsupportedError

    @snp_format.setter
    def snp_format(self, format: str) -> None:
        """
        Set the SNP format

        Parameters
        ----------
        format : str
            desired SNP format
        """
        raise UnsupportedError

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
        raise UnsupportedError

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
        raise UnsupportedError

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
        raise UnsupportedError
