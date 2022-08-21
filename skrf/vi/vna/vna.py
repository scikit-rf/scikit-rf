"""
Abstract base class for VNAs. Do not use directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import pyvisa
from pyvisa.constants import RENLineOperation

from ...calibration import Calibration
from ...frequency import Frequency
from ...network import Network


@dataclass
class Measurement:
    name: str
    param: str
    channel: Optional[int] = None


class VNA(ABC):
    """
    class defining a base analyzer for using with scikit-rf

    This class defines the base functionality expected for all Network
    Analyzers. To keep this manageable, it defines only the most commonly used
    applications, which currently are:
    * grabbing data (SNP networks, and traces)
    * grabbing a catalogue of available traces
    * instantiating and grabbing switch terms for calibration
    * setting the frequency parameters (start, stop, number of points, type)

    Attributes
    ----------
    idn

    Methods
    -------
    get_oneport(ports=(1, 2), **kwargs)
        get an return a 1-port Network object
    get_twoport(ports=1, **kwargs)
        get an return a 2-port Network object
    enter/exit - for using python's with statement
    >>> with Analyzer("GPIB::16::ISNTR") as nwa:
    >>>     ntwk = nwa.measure_twoport_ntwk()

    Methods that must be subclassed
    -------------------------------
    get_traces
    get_list_of_traces
    get_snp_network
    get_switch_terms
    set_frequency_sweep
    """

    def __init__(self, address: str, timeout: int = 3000, backend: str = "@py") -> None:
        """
        Initialize a network analyzer object

        Parameters
        ----------
        address : str
            a visa resource string, or an ip address
        timeout : int
            communication timeout in milliseconds

        Notes
        -----
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
        self.__exit__(None, None, None)

    def reset(self) -> None:
        self.write("*RST")

    def id_string(self) -> str:
        return self.query("*IDN?")

    @abstractmethod
    def start_freq(self) -> float:
        pass

    @abstractmethod
    def set_start_freq(self, f: float) -> None:
        pass

    @abstractmethod
    def stop_freq(self) -> float:
        pass

    @abstractmethod
    def set_stop_freq(self, f: float) -> None:
        pass

    @abstractmethod
    def npoints(self) -> int:
        pass

    @abstractmethod
    def set_npoints(self, n: int) -> None:
        pass

    @abstractmethod
    def freq_step(self) -> float:
        pass

    @abstractmethod
    def set_freq_step(self, f: float) -> None:
        pass

    def frequency(self) -> Frequency:
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
        pass

    @abstractmethod
    def set_sweep_mode(self, mode: str) -> None:
        pass

    @abstractmethod
    def sweep_type(self) -> str:
        pass

    @abstractmethod
    def set_sweep_type(self, _type: str) -> None:
        pass

    @abstractmethod
    def sweep_time(self) -> float:
        pass

    @abstractmethod
    def set_sweep_time(self, time: Union[float, str]) -> None:
        pass

    @abstractmethod
    def if_bandwidth(self) -> float:
        pass

    @abstractmethod
    def set_if_bandwidth(self, bw: float) -> None:
        pass

    @abstractmethod
    def averaging_on(self) -> bool:
        pass

    @abstractmethod
    def set_averaging_on(self, onoff: bool) -> None:
        pass

    @abstractmethod
    def average_count(self) -> int:
        pass

    @abstractmethod
    def set_average_count(self, n: int) -> None:
        pass

    @abstractmethod
    def average_mode(self) -> str:
        pass

    @abstractmethod
    def set_average_mode(self, mode: str) -> None:
        pass

    @abstractmethod
    def clear_averaging(self) -> None:
        pass

    @abstractmethod
    def num_sweep_groups(self) -> int:
        pass

    @abstractmethod
    def set_num_sweep_groups(self, n: int) -> None:
        pass

    @property
    @abstractmethod
    def channels_in_use(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def active_channel(self) -> Optional[int]:
        pass

    @active_channel.setter
    @abstractmethod
    def active_channel(self, channel: int) -> None:
        pass

    @abstractmethod
    def measurements_on_channel(self, channel: int) -> List[Measurement]:
        pass

    @property
    def measurements(self) -> List[Measurement]:
        channels = self.channels_in_use
        msmts = []
        for chan in channels:
            msmts += self.measurements_on_channel(chan)

        return msmts

    def measurement_names(self, channel: int = 1) -> List[str]:
        return [m.name for m in self.measurements if m.channel == channel]

    @property
    @abstractmethod
    def active_measurement(self) -> Optional[str]:
        pass

    @abstractmethod
    def set_active_measurement(self, meas: Union[int, str]) -> None:
        pass

    @abstractmethod
    def create_measurement(self, name: str, param: str) -> None:
        pass

    @abstractmethod
    def delete_measurement(self, name: str) -> None:
        pass

    @abstractmethod
    def get_measurement(self, meas: Union[int, str]) -> Network:
        pass

    @abstractmethod
    def get_active_trace(self) -> Network:
        pass

    @property
    @abstractmethod
    def snp_format(self) -> str:
        pass

    @snp_format.setter
    @abstractmethod
    def snp_format(self, format: str) -> None:
        pass

    @property
    @abstractmethod
    def ports(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def sysinfo(self) -> str:
        pass

    @abstractmethod
    def sweep(self) -> None:
        pass

    @abstractmethod
    def get_snp_network(self, ports: Optional[Sequence] = None) -> Network:
        pass

    @abstractmethod
    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        pass

    @abstractmethod
    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        pass
