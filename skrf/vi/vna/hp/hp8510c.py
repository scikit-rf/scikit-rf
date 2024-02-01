"""
.. module:: skrf.vi.vna.hp.8510c
=================================================
HP 8510C (:mod:`skrf.vi.vna.hp.8510c`)
=================================================

Instrument Class
================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    HP8510C
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import numpy as np

import skrf
from skrf.vi import vna
from skrf.vi.validators import BooleanValidator, FreqValidator, SetValidator
from skrf.vi.vna import VNA


class HP8510C(VNA):
    _scpi = False

    _supported_npoints = [51, 101, 201, 401, 801]

    id = VNA.command(
        get_cmd="OUTPIDEN;",
        set_cmd=None,
        doc="""Instrument ID string""",
    )

    last_error = VNA.command(
        get_cmd="OUTPERRO",
        set_cmd=None,
        doc="""Last error message""",
    )

    freq_start = VNA.command(
        get_cmd='STAR;OUTPACTI;',
        set_cmd="STEP; STAR <arg>;",
        doc="""Start frequency [hz]""",
        validator=FreqValidator()
    )

    freq_stop = VNA.command(
        get_cmd='STOP;OUTPACTI;',
        set_cmd="STEP; STOP <arg>;",
        doc="""Stop frequency [hz]""",
        validator=FreqValidator()
    )

    npoints = VNA.command(
        get_cmd='POIN;OUTPACTI;',
        set_cmd="STEP; POIN <arg>;",
        doc="""Number of frequency points""",
        validator=SetValidator(_supported_npoints)
    )

    is_continuous = VNA.command(
        get_cmd="GROU?",
        set_cmd="<arg>", # This is blank on purpose. The command sent is in the BooleanValidator constructor
        doc="""The trigger mode of the instrument""",
        validator=BooleanValidator(
            true_response='"CONTINUAL"',
            false_response='"HOLD"',
            true_setting='CONT;',
            false_setting='SING;'
        )
    )

    def __init__(self, address: str, backend='@py') -> None:
        super().__init__(address, backend)

        self._resource.timeout = 2_000
        assert 'HP8510' in self.id

        self._resource.timeout = 60_000
        self.query_delay = 10.

        self.read_raw = self._resource.read_raw

        self.reset()

    @property
    def frequency(self) -> skrf.Frequency:
        return skrf.Frequency(
            start=self.freq_start,
            stop=self.freq_stop,
            npoints=self.npoints,
            unit='hz'
        )

    @frequency.setter
    def frequency(self, f: skrf.Frequency) -> None:
        if f.npoints not in self._supported_npoints:
            raise ValueError("The HP8510C only supports one of {self._supported_npoints}.")

        self._resource.clear()
        self.write(f'STEP; STAR {int(f.start)}; STOP {int(f.stop)}; POIN{f.npoints};')

    @property
    def query_format(self) -> vna.ValuesFormat:
        return self._values_fmt

    @query_format.setter
    def query_format(self, arg: vna.ValuesFormat) -> None:
        fmt = {
            vna.ValuesFormat.ASCII: "FORM4;",
            vna.ValuesFormat.BINARY_32: "FORM2;",
            vna.ValuesFormat.BINARY_64: "FORM3;"
        }[arg]

        self._values_fmt = arg
        self.write(fmt)

    def reset(self) -> None:
        self.write('FACTPRES;')
        self.wait_until_finished()
        self.query_format = vna.ValuesFormat.ASCII

    def wait_until_finished(self) -> None:
        _ = self.id

    def get_complex_data(self, cmd: str) -> np.ndarray:
        self.query_format = vna.ValuesFormat.BINARY_32
        # Query values will interpret the response as floats, but the first 4
        # bytes are a header. Since it gets cast to a 4-byte float, we can just
        # ignore the first "value"
        # TODO: Is this correct? or is the header already handled in query_binary_values?
        raw = self.query_values(cmd, container=np.array, delay=self.query_delay)[1:]
        vals = raw.reshape((-1, 2))
        vals_complex = (vals[:,0] + 1j * vals[:,1]).flatten()
        return vals_complex

    def _get_oneport(self, parameter: tuple[int, int], sweep: bool=True):
        if any(p not in {1,2} for p in parameter):
            raise ValueError("The elements of parameter must be 1, or 2.")

        self.write(f"s{parameter[0]}{parameter[1]}")

        if sweep:
            self.sweep()

        ntwk = skrf.Network(name=f"S{parameter[0]}{parameter[1]}")
        ntwk.s = self.get_complex_data("OUTPDATA")
        ntwk.frequency = self.frequency

        return ntwk

    def _get_twoport(self, sweep: bool=True) -> skrf.Network:
        s11 = self._get_oneport((1, 1))
        freq = s11.frequency
        s11 = s11.s[:, 0, 0]
        s12 = self._get_oneport((1, 2)).s[:, 0, 0]
        s21 = self._get_oneport((2, 1)).s[:, 0, 0]
        s22 = self._get_oneport((2, 2)).s[:, 0, 0]

        ntwk = skrf.Network()
        ntwk.s = np.array([
            [s11, s21],
            [s12, s22]
        ]).transpose().reshape((-1, 2, 2))
        ntwk.frequency = freq

        return ntwk

    def get_snp_network(self, ports: Sequence = {1,2}) -> skrf.Network:
        ports = set(ports)
        if not ports.issubset({1,2}):
            raise ValueError("This instrument only has two ports. Must pass 1, 2, or (1,2)")

        if len(ports) == 1:
            p = ports[0]
            return self._get_oneport((p, p))
        else:
            return self._get_twoport()
            return self._get_twoport()
