from dataclasses import dataclass
import numpy as np
from typing import List,Union

class SweepSection:
    def get_hz(self):
        raise NotImplementedError()
    def apply_8510(self, hp8510c):
        raise NotImplementedError()
    def mask_8510(self, network):
        return network

@dataclass
class LinearBuiltinSweepSection(SweepSection):
    hz_min : float
    hz_max : float
    n_points : int
    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)
    def apply_8510(self, hp8510c):
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)

@dataclass
class LinearMaskedSweepSection(SweepSection):
    hz_min : float
    hz_max : float
    n_points : int
    mask : object
    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)
    def apply_8510(self, hp8510c):
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)
    def mask_8510(self, network):
        return network[self.mask]

@dataclass
class LinearCustomSweepSection(SweepSection):
    hz_min : float
    hz_max : float
    n_points : int
    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)
    def apply_8510(self, hp8510c):
        assert self.n_points <= 792
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)

@dataclass
class RandomSweepSection(SweepSection):
    hz_list : List[float]
    def get_hz(self):
        return self.hz_list

@dataclass
class SweepPlan:
    """
    The user requests a big sweep with different spacings in different frequency blocks,
    but the instrument only supports smaller sweeps with fixed spacing. What to do?
    Break up the big sweep, naturally.
    Each SweepSection represents a sweep the instrument can actually perform.
    Together, the SweepSections in a SweepPlan satisfy the user's request.
    """
    sections : List[SweepSection]
    def get_hz(self):
        """Get a list of all frequency points in the entire sweep plan."""
        ret = []
        for s in self.sections:
            ret.extend(s.get_hz())
        return ret

    def matches_f_list(self, hz):
        """
        True if the frequencies sweepned by self equal those in the list hz.
        """
        plan_hz = np.array(sorted(self.get_hz()))
        return good
