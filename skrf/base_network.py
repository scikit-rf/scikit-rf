from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import numpy as npy

from .frequency import Frequency

class BaseNetwork(ABC):
    PRIMARY_PROPERTIES: List[str]

    frequency: Frequency
    passivity: npy.ndarray
    name: str
    nports: int
    number_of_ports: int
    reciprocity: npy.ndarray
    reciprocity2: npy.ndarray
    s: npy.ndarray
    z0: npy.ndarray

    @abstractmethod
    def impulse_response(self, *args, **kwargs):
        pass

    @abstractmethod
    def step_response(self, *args, **kwargs):
        pass

    @abstractmethod
    def attribute(self, *args, **kwargs):
        pass

    @abstractmethod
    def windowed(self, *args, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def _generated_functions(cls) -> Dict[str, Tuple[Callable, str, str]]:
        pass