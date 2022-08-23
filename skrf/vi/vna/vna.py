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


class VNA(ABC):
    """
    Abstract base class for VNAs

    This class defines the interface to be provided by all network analyzer
    implementations.

    The instrument's manual and the SCPI manual should both be consulted when
    subclassing. This base class only provides implementations of official SCPI
    commands and the virutal instrument interface. Simple SCPI commands that are
    specific to the instrument or manufacturer should only be defined in those
    classes.

    If an instrument does support a SCPI command that is defined in this class,
    it should raise a `NotImplmentedError`.

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

    def start_freq(self, channel: int = 1) -> float:
        """
        Get start frequency

        Parameters
        ---------
        channel : int
            channel to get frequency (if supported)

        Returns
        -------
        float
            start frequency [Hz]
        """
        return float(self.query(f"sense{channel}:frequency:start?"))

    def set_start_freq(self, f: float, channel: int = 1) -> None:
        """
        Set start frequency

        Parameters
        ----------
        f : float
            start frequency [Hz]
        channel : int
            channel to set frequency (if supported)
        """
        self.write(f"sense{channel}:frequency:start {f}")

    def stop_freq(self, channel: int = 1) -> float:
        """
        Get stop frequency

        Parameters
        ----------
        channel : int
            channel to get frequency (if supported)

        Returns
        -------
        float
            stop frequency [Hz]
        """
        return float(self.query(f"sense{channel}:frequency:stop?"))

    def set_stop_freq(self, f: float, channel: int = 1) -> None:
        """
        Set stop frequency

        Parameters
        ----------
        f : float
            stop frequency [Hz]
        channel : int
            channel to set frequency (if supported)
        """
        self.write(f"sense{channel}:frequency:stop {f}")

    def npoints(self, channel: int = 1) -> int:
        """
        Get number of frequency points

        Parameters
        ----------
        channel : int
            channel to get npoints (if supported)

        Returns
        -------
        int
            number of frequency points
        """
        return int(self.query(f"sense{channel}:sweep:points?"))

    def set_npoints(self, n: int, channel: int = 1) -> None:
        """
        Set number of frequency points

        Parameters
        ----------
        n : int
            number of frequency points
        channel : int
            channel to set npoints (if supported)
        """
        self.write(f"sense{channel}:sweep:points {n}")

    def freq_step(self, channel: int = 1) -> float:
        """
        Get frequency step

        Parameters
        ----------
        channel : int
            channel to get frequency step (if supported)

        Returns
        -------
        float
            frequency step [Hz]
        """
        return float(self.query(f"sense{channel}:sweep:step?"))

    def set_freq_step(self, f: float, channel: int = 1) -> None:
        """
        Set frequency step

        Parameters
        ----------
        f : float
            frequency step [Hz]
        channel : int
            channel to set frequency step (if supported)
        """
        self.write(f"sense{channel}:sweep:step {f}")

    def frequency(self, channel: int = 1) -> Frequency:
        """
        Get current frequency as :class:`Frequency` object

        Parameters
        ----------
        channel : int
            channel to get frequency (if supported)

        Returns
        -------
        Frequency
            current frequency settings as frequency object
        """
        start = self.start_freq(channel)
        stop = self.stop_freq(channel)
        npoints = self.npoints(channel)
        return Frequency(start, stop, npoints, unit="hz")

    def set_frequency(
        self,
        frequency: Optional[Frequency] = None,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        npoints: Optional[int] = None,
        channel: int = 1,
    ) -> None:
        """
        Set current frequency

        The user can provide **either** a frequency object **or** start, stop
        **and** npoints

        Parameters
        ----------
        frequency :
            Frequency object
        start :
            start frequency [Hz]
        stop :
            stop frequency [Hz]
        npoints :
            number of frequency points
        channel : int
            channel to set frequency (if supported)

        Raises
        ------
        ValueError
            If a frequency object is passed with any other parameters or if not
            all of start, stop, and npoints are provided
        """
        if frequency and any((start, stop, npoints)):
            raise ValueError(
                "Got too many arguments. Pass either Frequency object or start, stop, and step."
            )
        if not frequency and not all((start, stop, npoints)):
            raise ValueError(
                "Got too few arguments. Pass either Frequency object or start, stop, and step."
            )

        if frequency:
            self.set_start_freq(frequency.start, channel)
            self.set_stop_freq(frequency.stop, channel)
            self.set_npoints(frequency.npoints, channel)
        else:
            # We've already checked our arguments at this point,
            # so we can safely ignore the possibility of None
            self.set_start_freq(start, channel)  # type: ignore
            self.set_stop_freq(stop, channel)  # type: ignore
            self.set_npoints(npoints, channel)  # type: ignore

    def sweep_mode(self, channel: int = 1) -> str:
        """
        Get the current sweep mode

        This is typically to hold, continuous, etc.

        Parameters
        ----------
        channel : int
            channel to get sweep mode (if supported)

        Returns
        -------
        str
            current sweep mode
        """
        return self.query(f"sense{channel}:sweep:mode?")

    def set_sweep_mode(self, mode: str, channel: int = 1) -> None:
        """
        Set the sweep mode

        This is typically to set the sweep to hold, continuous, etc.

        Parameters
        ----------
        mode : str
            sweep mode
        channel : int
            channel to set sweep mode (if supported)
        """
        self.write(f"sense{channel}:sweep:mode {mode}")

    def sweep_type(self, channel: int = 1) -> str:
        """
        Get the current frequency sweep type

        This is typically linear, logarithmic, etc.

        Parameters
        ----------
        channel : int
            channel to get sweep type (if supported)

        Returns
        -------
        str
            Current sweep type
        """
        return self.query(f"sense{channel}:sweep:type?")

    def set_sweep_type(self, type_: str, channel: int = 1) -> None:
        """
        Set the type of frequency sweep

        This is typically to set to linear, logarithmic, etc.

        Parameters
        ----------
        type_ : str
            type of frequency sweep
        channel : int
            channel to set sweep type (if supported)
        """
        self.write(f"sense{channel}:sweep:type {type_}")

    def sweep_time(self, channel: int = 1) -> float:
        """
        Get the current sweep time

        Parameters
        ----------
        channel : int
            channel to get sweep time (if supported)

        Returns
        -------
        float
            duration of a single sweep [s]
        """
        return float(self.query(f"sense{channel}:sweep:time?"))

    def set_sweep_time(self, time: Union[float, str], channel: int = 1) -> None:
        """
        Set the duration of a single sweep

        Parameters
        ----------
        time : Union[float, str]
            length of time to set a single sweep [s]
        channel : int
            channel to set sweep time (if supported)
        """
        self.write(f"sense{channel}:sweep:time {time}")

    def if_bandwidth(self, channel: int = 1) -> float:
        """
        Get the current IF bandwidth

        Parameters
        ----------
        channel : int
            channel to get IF bandwidth (if supported)

        Returns
        -------
        float
            current IF bandwidth [Hz]
        """
        return float(self.query(f"sense{channel}:sweep:bwidth?"))

    def set_if_bandwidth(self, bw: float, channel: int = 1) -> None:
        """
        Set the IF bandwidth

        Parameters
        ----------
        bw : float
            desired IF bandwidth [Hz]
        channel : int
            channel to set IF bandwidth (if supported)
        """
        self.write(f"sense{channel}:bwidth {bw}")

    def averaging_on(self, channel: int = 1) -> bool:
        """
        Checks if averaging is on or off

        Parameters
        ----------
        channel : int
            channel to get averaging state (if supported)

        Returns
        -------
        bool
            True if averaging is on, False otherwise
        """
        return self.query(f"sense{channel}:average?").strip() != "0"

    def set_averaging_on(self, state: bool, channel: int = 1) -> None:
        """
        Sets averaging on or off

        Parameters
        ----------
        state : bool
            True to turn on averaging, False to turn it off
        channel : int
            channel to set averaging state (if supported)
        """
        self.write(f"sense{channel}:average {'on' if state else 'off'}")

    def average_count(self, channel: int = 1) -> int:
        """
        Get the current averaging count

        Parameters
        ----------
        channel : int
            channel to get average count (if supported)

        Returns
        -------
        int
            The current averaging count
        """
        return int(self.query(f"sense{channel}:average:count?"))

    def set_average_count(self, n: int, channel: int = 1) -> None:
        """
        Sets the averaging count

        Parameters
        ----------
        n : int
            desired averaging count
        channel : int
            channel to set average count (if supported)
        """
        self.write(f"sense{channel}:average:count {n}")

    def clear_averaging(self, channel: int = 1) -> None:
        """
        Clear the averaging values

        Parameters
        ----------
        channel : int
            channel to clear averaging (if supported)
        """
        self.write(f"sense{channel}:average:clear")

    @abstractmethod
    def measurements(self) -> List[Tuple]:
        """
        Get a list of all current measurements

        Returns
        -------
        List[Tuple]
            List of all measurements currently defined on all channels. Each
            element of the list is a Tuple containing (in order), the
            measurement name or number, measurement parameter, and
            channel if available.
        """
        pass

    @abstractmethod
    def active_measurement(self, channel: int = 1) -> Optional[Union[str, int]]:
        """
        Get the active measurement

        Parameters
        ----------
        channel : int
            channel to get active measurement from (if supported)

        Returns
        -------
        Optional[str, int]
            the active measurement or trace number. None if no measurement is
            active
        """
        pass

    @abstractmethod
    def set_active_measurement(self, id_: Union[int, str], channel: int = 1) -> None:
        """
        Set the active measurement

        Parameters
        ----------
        id_ : Union[int, str]
            the name or number of the desired measurement
        channel : int
            channel to set active measurement on (if supported)
        """
        pass

    @abstractmethod
    def create_measurement(
        self, id_: Union[str, int], param: str, channel: int = 1
    ) -> None:
        """
        Create a measurement

        Parameters
        ----------
        id_ : Union[str, int]
            the name or id of the new measurement
        param : str
            the measurement parameter (S11, S21, A, etc.)
        channel : int
            channel to create measurement on (if supported)
        """
        pass

    @abstractmethod
    def delete_measurement(self, id_: Union[str, int], channel: int = 1) -> None:
        """
        Delete a measurement

        Parameters
        ----------
        id_ : Union[str, int]
            name or number of the measurement to be deleted
        channel : int
            channel to delete measurement on (if supported)
        """
        pass

    @abstractmethod
    def get_measurement(self, id_: Union[int, str], channel: int = 1) -> Network:
        """
        Get measurement data as a `Network`

        Parameters
        ----------
        id_ : Union[int, str]
            name or number of the measurement to get
        channel : int
            channel to get measurement from (if supported)

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

    def sweep(self) -> None:
        """
        Trigger a fresh sweep
        """
        self.resource.clear()
        self.write("initiate:immediate")
        self.query("*OPC?")

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
