import dataclasses
import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod


class SweepSection(ABC):
    @abstractmethod
    def get_hz(self):
        pass

    def apply_8510(self, hp8510c):
        pass

    def mask_8510(self, network):
        return network


@dataclasses.dataclass
class LinearBuiltinSweepSection(SweepSection):
    hz_min: float
    hz_max: float
    n_points: int

    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)

    def apply_8510(self, hp8510c):
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)


@dataclasses.dataclass
class LinearMaskedSweepSection(SweepSection):
    hz_min: float
    hz_max: float
    n_points: int
    mask: object

    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)

    def apply_8510(self, hp8510c):
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)

    def mask_8510(self, network):
        return network[self.mask]


@dataclasses.dataclass
class LinearCustomSweepSection(SweepSection):
    hz_min: float
    hz_max: float
    n_points: int

    def get_hz(self):
        return np.linspace(self.hz_min, self.hz_max, self.n_points)

    def apply_8510(self, hp8510c):
        assert self.n_points <= 792
        hp8510c._set_instrument_step_state(self.hz_min, self.hz_max, self.n_points)


@dataclasses.dataclass
class RandomSweepSection(SweepSection):
    hz_list: List[float]

    def get_hz(self):
        return self.hz_list


@dataclasses.dataclass
class SweepPlan:
    """
    The user requests a big sweep with different spacings in different frequency
    blocks, but the HP8510 instrument only supports smaller sweeps with fixed
    spacing. What to do? Break up the big sweep, naturally.

    SweepPlan.from_hz(array_of_hz)
      ->
        SweepPlan
            SweepSection
            SweepSection
            SweepSection

    Each SweepSection represents a sweep the instrument can actually perform.
    Together, the SweepSections in a SweepPlan satisfy the user's request.

    There is room for cleverness: conducting an 800 point sweep by running an
    801 point sweep and discarding a point saved 15 minutes in a recent script.
    """

    sections: List[SweepSection]

    @classmethod
    def from_hz(cls, hz):
        """
        Take a list of hz, return a SweepPlan, a list of 8510C compatible
        SweepSections that can be executed to cover the original list of frequencies.
        Limitations: No overlapping linear sweeps. No log sweeps.
        """
        hz = np.array(sorted(hz))
        n = len(hz)
        misfits = []  # hz that don't fit into a linear sweep
        growing_window = []
        growing_window_d = -1
        sweep_sections = []

        def finalize_window(growing_window, misfits, sweep_sections):
            """ When a growing_window has grown as far as it can, finalize_window is called to turn it into a sweep section + misfits"""
            # Runt windows aren't really linear sweeps -- we should just add their points to the misfit pile
            if len(growing_window) <= 2:
                misfits.extend(growing_window)
                growing_window.clear()
                return
            # Certain window lengths are preferred. Try to use these as much as possible
            def try_builtin_window_len(builtin_len):
                while len(growing_window) >= builtin_len:
                    chunk = growing_window[0:builtin_len]
                    del growing_window[0:builtin_len]
                    ns = LinearBuiltinSweepSection(
                        hz_min=chunk[0], hz_max=chunk[-1], n_points=len(chunk)
                    )
                    sweep_sections.append(ns)

            try_builtin_window_len(801)
            try_builtin_window_len(401)
            try_builtin_window_len(201)
            try_builtin_window_len(101)
            try_builtin_window_len(51)

            # Remainder get a custom linear window:
            while len(growing_window):
                chunk = growing_window[0:792]
                del growing_window[0:792]
                ns = LinearCustomSweepSection(
                    hz_min=chunk[0], hz_max=chunk[-1], n_points=len(chunk)
                )
                assert len(chunk) == len(ns.get_hz())
                assert np.allclose(chunk, ns.get_hz())
                sweep_sections.append(ns)

        # Start growing windows!
        for i in range(n):
            wl = len(growing_window)
            if wl == 0:
                growing_window.append(hz[i])
            if wl == 1:
                growing_window.append(hz[i])
                growing_window_d = hz[i] - hz[i - 1]
            if wl >= 2:
                i_fits_in_growing_window = (
                    abs((hz[i] - hz[i - 1]) - growing_window_d) < 0.5
                )  # Hz tolerance
                if i_fits_in_growing_window:
                    growing_window.append(hz[i])
                else:
                    finalize_window(growing_window, misfits, sweep_sections)
                    growing_window.append(hz[i])
        finalize_window(growing_window, misfits, sweep_sections)
        if len(misfits):
            sweep_sections.append(RandomSweepSection(hz_list=misfits))
        plan = SweepPlan(sections=sweep_sections)
        assert plan._matches_f_list(hz)
        return plan

    def get_hz(self):
        """Get a list of all frequency points in the entire sweep plan."""
        ret = []
        for s in self.sections:
            ret.extend(s.get_hz())
        return ret

    def _matches_f_list(self, golden_hz):
        """
        Returns True iff the frequencies this SweepPlan intends to sweep
        equal those in the list golden_hz.
        """
        plan_hz = np.array(sorted(self.get_hz()))
        if len(golden_hz) != len(plan_hz):
            print(f"Planner output has different length.")
            return False
        good = True
        for h in golden_hz:
            if not np.any(np.isclose(h, plan_hz)):
                print(f"In original list but not plan: {h}")
                good = False
        for ph in plan_hz:
            if not np.any(np.isclose(ph, golden_hz)):
                print(f"In plan but not in original list: {h}")
                good = False
        if not np.allclose(golden_hz, plan_hz):
            print("All were not close.")
            good = False
        return good

    @classmethod
    def from_ssn(cls, hz_start, hz_stop, n):
        return cls.from_hz(np.linspace(hz_start, hz_stop, n))
