"""
Abstract base class for VNAs. Do not use directly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import pyvisa
from pyvisa.constants import RENLineOperation

from ...calibration import Calibration
from ...frequency import Frequency
from ...network import Network


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

    def __init__(self, address: str, timeout: int = 3000, backend: str='@py') -> None:
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

    @classmethod
    def available(cls, backend: str='@py') -> List[str]:
        rm = pyvisa.ResourceManager(backend)
        avail = rm.list_resources()
        rm.close()
        return list(avail)

    @property
    @abstractmethod
    def start_freq(self) -> float:
        pass

    @start_freq.setter
    @abstractmethod
    def start_freq(self, f: float) -> None:
        pass

    @property
    @abstractmethod
    def stop_freq(self) -> float:
        pass

    @stop_freq.setter
    @abstractmethod
    def stop_freq(self, f: float) -> None:
        pass

    @property
    @abstractmethod
    def npoints(self) -> int:
        pass

    @npoints.setter
    @abstractmethod
    def npoints(self, n: int) -> None:
        pass

    @property
    @abstractmethod
    def sweep_type(self) -> str:
        pass

    @sweep_type.setter
    @abstractmethod
    def sweep_type(self, _type: str) -> None:
        pass

    @property
    @abstractmethod
    def sweep_time(self) -> float:
        pass

    @sweep_time.setter
    @abstractmethod
    def sweep_time(self, time: Union[float, str]) -> None:
        pass

    @property
    @abstractmethod
    def if_bandwidth(self) -> float:
        pass

    @if_bandwidth.setter
    @abstractmethod
    def if_bandwidth(self, bw: float) -> None:
        pass

    @property
    @abstractmethod
    def averaging(self) -> bool:
        pass

    @averaging.setter
    @abstractmethod
    def averaging(self, onoff: bool) -> None:
        pass

    @property
    @abstractmethod
    def average_count(self) -> int:
        pass

    @average_count.setter
    @abstractmethod
    def average_count(self, n: int) -> None:
        pass

    @abstractmethod
    def clear_averaging(self) -> None:
        pass

    @property
    @abstractmethod
    def num_sweep_groups(self) -> int:
        pass

    @num_sweep_groups.setter
    @abstractmethod
    def num_sweep_groups(self, n: int) -> None:
        pass

    @property
    @abstractmethod
    def channels_in_use(self) -> str:
        pass

    @property
    @abstractmethod
    def active_channel(self) -> Optional[int]:
        pass

    @property
    @abstractmethod
    def active_measurement(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def measurement_numbers(self) -> str:
        pass

    @property
    @abstractmethod
    def snp_format(self) -> str:
        pass

    @snp_format.setter
    @abstractmethod
    def snp_format(self) -> None:
        pass

    @property
    @abstractmethod
    def sysinfo(self) -> str:
        pass

    @property
    def frequency(self) -> Frequency:
        start = self.start_freq
        stop = self.stop_freq
        npoints = self.npoints
        return Frequency(start, stop, npoints, unit="hz")

    @frequency.setter
    def frequency(
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
            self.start_freq = freq.start
            self.stop_freq = freq.stop
            self.npoints = freq.npoints
        else:
            self.start_freq = start
            self.stop_freq = stop
            self.npoints = npoints

    @abstractmethod
    def get_snp_network(self, ports: Sequence) -> Network:
        pass

    @abstractmethod
    def upload_oneport_calibration(self, port: int, cal: Calibration) -> None:
        pass

    @abstractmethod
    def upload_twoport_calibration(self, ports: Sequence, cal: Calibration) -> None:
        pass
