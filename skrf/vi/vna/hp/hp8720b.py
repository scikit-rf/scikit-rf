"""
<<<<<<< HEAD
.. module:: skrf.vi.vna.hp.hp8720b
=================================================
HP 8720B (:mod:`skrf.vi.vna.hp.hp8720b`)
=================================================

HP8720B Class
=======
.. module:: skrf.vi.vna.hp.8720b
=================================================
HP 8720B (:mod:`skrf.vi.vna.hp.8720b`)
=================================================

Instrument Class
>>>>>>> cf81f554787ae19e36dc7b99e44d5403bbb171ab
================

.. autosummary::
    :nosignatures:
    :toctree: generated/
<<<<<<< HEAD
"""

import time

import numpy as np
import pyvisa

import skrf
import skrf.network
=======

    HP8720B
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import numpy as np

import skrf
from skrf.vi import vna
from skrf.vi.validators import BooleanValidator, FreqValidator, SetValidator
>>>>>>> cf81f554787ae19e36dc7b99e44d5403bbb171ab
from skrf.vi.vna import VNA


class HP8720B(VNA):
<<<<<<< HEAD
    '''
    HP 8720B driver, created by modifiying the 8510C driver and commenting out 
    the additional sweep options. The instrument natively supports 
    (3/11/21/51/101/201/401/801/1601pts). 

    Segmented sweeps occur automatically when the user requests a short or
    irregularly spaced sweep (see "Advanced Example" below). The 8510 actually
    does support sweeps other than 51/101/201/401/801pts, but only in a separate
    mode (segmented sweep mode) and with significant restrictions. This driver
    knows how to handle segmented sweep mode to get what it wants.


    Examples
    ============

    Basic one-port:

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8720B(address='GPIB0::16::INSTR', backend='C:\\WINDOWS\\system32\\visa32.dll')
        vna.set_frequency_sweep(2e9,3e9,201)
        vna.get_snp_network(ports=(1,))



    Basic two-port:

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8720B(address='GPIB0::16::INSTR', backend='C:\\WINDOWS\\system32\\visa32.dll')
        vna.set_frequency_sweep(2e9,3e9,201)
        vna.get_snp_network(ports=(1,2))



    Intermediate example -- note that 1001 point sweeps are not natively supported by the instrument; this driver
    takes multiple sweeps and stitches the results together.

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8720B(address='GPIB0::16::INSTR', backend='C:\\WINDOWS\\system32\\visa32.dll')
        vna.set_frequency_sweep(2e9,3e9,1001)
        vna.get_snp_network(ports=(1,2))

    Advanced example. The driver is handed a bucket of frequencies containing
    two separate bands mashed together. Behind the scenes it will construct a
    sweep plan consisting of one native sweep and one segmented sweep, perform
    both, and stitch the results together -- but all the user need worry about
    is constructing the request and interpreting the results.

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8510C(address='TCPIP::ad007-right.lan::gpib0,16::INSTR', backend='@py')
        freq_block_1 = np.linspace(1e9,2e9,801)
        freq_block_2 = [10e9,11e9,12e9]
        freqs = np.concatenate((freq_block_1, freq_block_2))
        vna.frequency = skrf.Frequency.from_f(freqs)
        vna.get_snp_network(ports=(1,2))
    '''
    min_hz = None  #: Minimum frequency supported by instrument
    max_hz = None  #: Maximum frequency supported by instrument
    # compound_sweep_plan = None
    #: If None, get_snp_network()/one_port()/two_port() just ask the VNA for data.
    # If populated, those methods perform the multiple sweeps in the plan and stitch together the results.

    def __init__(self, address : str, backend : str = "@py", **kwargs):
        super().__init__(address, backend, **kwargs)
        # Tested 2024-05-29:
        #     HP8720B
        #     NI GPIB to USB
        #     address="gpib0,16::INSTR", backend='C:\\WINDOWS\\system32\\visa32.dll'

        # 8720s are slow. This check ensures we won't wait 60s for connection error.
        self._resource.timeout = 2_000 * (3_000 / self.if_bandwidth) # Speed dependant on IF bandwidth selected
        id_str = self.query('OUTPIDEN;')
        assert('8720' in id_str) # example: 'HP8720B,0,1.01'

        self._resource.read_termination = '\n'
        self.read_raw = self._resource.read_raw
        self.min_hz = self.freq_start
        self.max_hz = self.freq_stop
        self.write('CONT;')

        self.write('DEBUON;') # Debug HPIB mode ON to displace instrument commands on the instrument screen
        #  DEBUOFF to turn off (or just remove command)

        # If compound_sweep_plan is None, we rely on the internal state of the analyzer.
        # If compound_sweep_plan is a SweepPlan object, we must take multiple short sweeps
        # on the 8510 and stitch them into a single compound sweep ourselves.
        self.compound_sweep_plan = None

        # Compound sweeps can be added in the future using the subsweep menu commands

    @property
    def id(self):
        ''' Instrument ID string '''
        return self.query("OUTPIDEN;")

    def reset(self):
        ''' Preset instrument. '''
        self.write("PRES;")
        self.wait_until_finished()

    def clear(self):
        self._resource.clear()

    def wait_until_finished(self):
        self.query("OUTPIDEN;")

    def get_snp_network(self, ports, **kwargs):
        ''' MAIN METHOD for obtaining S parameters, like get_snp_network((1,)) or get_snp_network((1,2)). '''
        ports = tuple(ports)
        sweep = kwargs.get("sweep", True)
        # name = kwargs.get("name", "")
        # raw_data = kwargs.get("raw_data", True)
        if ports==(1,):
            self.write('S11;')
            return self.one_port(fresh_sweep=sweep)
        elif ports==(2,):
            self.write('S22;')
            return self.one_port(fresh_sweep=sweep)
        elif ports==(1,2) or ports==(2,1):
            return self.two_port(fresh_sweep=sweep)
        else:
            raise(ValueError("Invalid ports "+str(ports)+". Options: (1,) (2,) (1,2)."))

    def get_switch_terms(self, ports=(1, 2), **kwargs):  # NOT NEEDED?
        '''
        Returns (forward_one_port,reverse_one_port) switch terms.
        The ports short be connected with a half decent THRU before calling.
        These measure how much signal is reflected from the imperfect switched
        termination on the non-stimulated port.
        '''
        return self.switch_terms()

    @property
    def error(self):
        ''' Error from OUTPERRO '''
        return self.query('OUTPERRO')
    
    @property
    def if_bandwidth(self):
        ''' Current IF Bandwidth [Hz] '''
        return float(self.query('IFBW?'))
    
    @if_bandwidth.setter
    def if_bandwidth(self, if_bw):
        ''' Allowed values (in Hz): 3000, 1000, 300, 100, 30, and 10'''
        if if_bw in [3,10,30,100,300,1000,3000]:
            self.write(f'IFBW {if_bw}')
            self._resource.timeout = 2_000 * (3_000 / self.if_bandwidth) # Changing the timeout due to change IF BW
        else:
            raise(ValueError('Takes a value from [3,10,30,100,300,1000,3000]'))

    @property
    def is_continuous(self):
        ''' True if sweep mode is continuous. Can be set. '''
        answer_dict={'1':False,'0':True}
        return answer_dict[self.query('TRIG?')]

    @is_continuous.setter
    def is_continuous(self, choice):
        if choice:
            self.write('CONT;')
        elif not choice:
            self.write('SING;')
        else:
            raise(ValueError('takes a boolean'))

    @property
    def averaging(self):
        ''' Averaging factor for AVERON '''
        return int(float(self.query('AVERFACT?')))

    @averaging.setter
    def averaging(self, factor):
        if not factor:
            self.write('AVEROFF')
        elif isinstance(factor, int):
            self.write(f'AVERON; AVERFACT {factor};')
        else:
            raise(ValueError('takes an interger'))

    @property
    def _frequency(self):
        ''' Frequencies of non-compound sweep '''
        freq=skrf.Frequency( self.freq_start, self.freq_stop, self._npoints, unit='hz' )
        return freq

    @property
    def frequency(self):
        ''' Frequencies of compound sweep '''
        if self.compound_sweep_plan is None:
            return self._frequency
        # return skrf.Frequency.from_f(self.compound_sweep_plan.get_hz(), unit='hz')

    @frequency.setter
    def frequency(self, frequency_obj: skrf.Frequency):
        ''' List sweep using hz, an array of frequencies.
        If hz is too long, multiple sweeps will automatically be performed.'''
        hz = frequency_obj.f
        valid = (self.min_hz<=hz) & (hz<=self.max_hz)
        if not np.all(valid):
            print("set_frequency called with %i/%i points out of VNA frequency range. Dropping them."
                  %(np.sum(valid),len(valid)))
            hz = hz[valid]
        # self.compound_sweep_plan = SweepPlan.from_hz(hz)

    def set_frequency_sweep(self, f_start, f_stop, f_npoints, **kwargs):
        ''' Interprets units and calls set_frequency_step '''
        f_unit = kwargs.get("f_unit", "hz").lower()
        if f_unit != "hz":
            hz_start = self.to_hz(f_start, f_unit)
            hz_stop = self.to_hz(f_stop, f_unit)
        else:
            hz_start, hz_stop = f_start, f_stop
        self.set_frequency_step(hz_start, hz_stop, f_npoints)

    def set_frequency_step(self, hz_start, hz_stop, npoint=801):
        ''' Step (slow, synthesized) sweep + logic to handle lots of npoint. '''
        if (self._instrument_natively_supports_steps(npoint)):
            self.compound_sweep_plan = None
            self._set_instrument_step_state(hz_start, hz_stop, npoint)
        # else:
        #     self.compound_sweep_plan = SweepPlan.from_ssn(hz_start, hz_stop, npoint)

    @property
    def freq_start(self):
        ''' Start frequency [hz] '''
        return float(self.query('STAR;OUTPACTI;'))

    @freq_start.setter
    def freq_start(self, new_start_hz):
        self.write(f'STAR {new_start_hz};')

    @property
    def freq_stop(self):
        ''' Stop frequency [hz] '''
        return float(self.query('STOP;OUTPACTI;'))

    @freq_stop.setter
    def freq_stop(self, new_stop_hz):
        self.write(f'STOP {new_stop_hz};')

    @property
    def _npoints(self):
        ''' Number of points in non-compound sweep '''
        instrument_ret_val = self.query('POIN;OUTPACTI;')
        v1 = instrument_ret_val.strip()
        v2 = float(v1)
        v3 = int(v2)
        return v3

    @property
    def npoints(self):
        ''' Number of points in compound sweep (if programmed) or non-compound sweep '''
        if self.compound_sweep_plan is None:
            return self._npoints
        # return len(self.compound_sweep_plan.get_hz())

    @npoints.setter
    def npoints(self, npoint):
        ''' Set number of points in sweep'''
        hz_start, hz_stop = self.freq_start, self.freq_stop
        if (self._instrument_natively_supports_steps(npoint)):
            self.compound_sweep_plan = None
            self._set_instrument_step_state(hz_start, hz_stop, npoint)
        # else:
        #     self.compound_sweep_plan = SweepPlan.from_ssn(hz_start, hz_stop, npoint)

    def _instrument_natively_supports_steps(self, npoint):
        if npoint in [3,11,21,51,101,201,401,801,1601]:
            return True
        if npoint<=1601: # Supported with a single native list sweep commented as only work once then breaks connection
            return True
        return False

    def _set_instrument_step_state(self, hz_start, hz_stop, npoint=801):
        assert(self._instrument_natively_supports_steps(npoint))
        if self._resource is not None:
            self._resource.clear()
        if npoint in [3,11,21,51,101,201,401,801,1601]:
            # If it's a directly supported step sweep npoints, use regular sweep
            self.write(f'STAR {hz_start}; STOP {hz_stop}; POIN {npoint};')
        elif npoint<=1601:
            # List sweep lets us support, for example, 401<npoints<801 CURRENTLY BREAKS connection
            self.write('EDITLIST;')
            self.write('SDEL;')
            self.write('SADD;')
            self.write(f'STAR {hz_start}; STOP {hz_stop}; POIN {npoint};')
            self.write('SDON; EDITDONE; LISFREQ;')

    def _set_instrument_cwstep_state(self, hz_list):
        assert(len(hz_list)<=30) # 8720 only supports CW lists up to length 30
        self.write('EDITLIST;')
        self.write('CLEL;')
        for hz in hz_list:
            self.write('SADD;')
            self.write(f'CWFREQ {int(hz)};')
        self.write('SDON; EDITDONE; LISFREQ;')

    def ask_for_cmplx(self, outp_cmd, timeout_s=30):
        """Like ask_for_values, but use FORM2 binary transfer, much faster than ASCII for HP8720. 
        Also could not get FORM4 working for HP8720B"""

        self._resource.read_termination = False  # Binary mode doesn't work if we allow premature termination on \n

        # Benchmarks:
        #  %time i.write('FORM4; OUTPDATA'); i._read(); None      # 543ms
        #  %time i.write('FORM2; OUTPDATA'); i._read_raw(); None  #  97.5ms
        # self.wait_for_status(max_wait_seconds=timeout_s)
        self.write('FORM2;')
        self.write(outp_cmd)
        buf = self.read_raw()
        float_bin = buf[4:] # Skip 4 header bytes and trailing newline
        try:
            floats = np.frombuffer(float_bin, dtype='>f4').reshape((-1,2))
        except ValueError as e:
            print(buf)
            print("len(buf): %i"%(len(buf),))
            raise(e)
        cmplxs = (floats[:,0] + 1j*floats[:,1]).flatten()

        self._resource.read_termination = '\n'  # Switching back for other outputs
        
        return cmplxs

    def _one_port(self, expected_hz=None, fresh_sweep=True):
        ''' Perform a single sweep and return Network data. '''
        is_current_sweep_continous = self.is_continuous
        if fresh_sweep:
             self.write('SING;') # Poll for sweep status
        s =  self.ask_for_cmplx('OUTPDATA')
        ntwk = skrf.Network()
        ntwk.s = s
        hz = expected_hz if expected_hz is not None else self._frequency.f
        assert(len(s)==len(hz))
        ntwk.frequency = skrf.Frequency.from_f(hz,unit='hz')
        self.is_continuous = is_current_sweep_continous
        return ntwk

    def one_port(self, **kwargs):
        ''' Performs a single sweep OR COMPOUND SWEEP and returns Network data. '''
        if self.compound_sweep_plan is None:
            return self._one_port()
        # old_start_hz, old_stop_hz = self.freq_start, self.freq_stop
        # stitched_network = None
        # for sweep_section in self.compound_sweep_plan.get_sections():
        #     sweep_section.apply_8510(self)
        #     chunk_net_nomask = self._one_port(expected_hz=sweep_section.get_raw_hz())
        #     chunk_net = sweep_section.mask_8510(chunk_net_nomask)
        #     stitched_network = (chunk_net if stitched_network is None
        #                         else skrf.network.stitch(stitched_network, chunk_net))
        # self.freq_start = old_start_hz
        # self.freq_stop  = old_stop_hz
        # return stitched_network

    def _two_port(self, expected_hz=None, fresh_sweep=True):
        ''' Performs a single sweep and returns Network data. '''

        self.write('S11;')
        s11 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('S12;')
        s12 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('S22;')
        s22 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('S21;')
        s21 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]

        ntwk = skrf.Network()
        ntwk.s = np.array(\
                [[s11,s21],\
                [ s12, s22]]\
                ).transpose().reshape(-1,2,2)
        hz = expected_hz if expected_hz is not None else self._frequency.f
        assert(len(s11)==len(hz))
        ntwk.frequency= skrf.Frequency.from_f(hz,unit='hz')

        return ntwk

    def two_port(self, **kwargs):
        ''' Performs a single sweep OR COMPOUND SWEEP and returns Network data. '''
        if self.compound_sweep_plan is None:
            return self._two_port()
        # old_start_hz, old_stop_hz = self.freq_start, self.freq_stop
        # stitched_network = None
        # for sweep_chunk in self.compound_sweep_plan.get_sections():
        #     sweep_chunk.apply_8510(self)
        #     chunk_net_nomask = self._two_port(expected_hz=sweep_chunk.get_raw_hz())
        #     chunk_net = sweep_chunk.mask_8510(chunk_net_nomask)
        #     stitched_network = (chunk_net if stitched_network is None
        #                         else skrf.network.stitch(stitched_network,chunk_net))
        # self.freq_start = old_start_hz
        # self.freq_stop  = old_stop_hz
        # return stitched_network

    # def wait_for_status(self, max_wait_seconds=30):
    #     ''' Fetches status bytes respecting max_wait_seconds even if it must tank a few timeouts along the way. '''
    #     t0 = time.time()
    #     while True:
    #         try:
    #             status_str = self.query('OUTPSTAT')
    #             print(status_str)
    #             s0str, s1str = status_str.strip().split(',')
    #             s0 = int(s0str)
    #             s1 = int(s1str)
    #             break
    #         except pyvisa.errors.VisaIOError as e:
    #             waited = time.time()-t0
    #             if waited > max_wait_seconds:
    #                 raise e
    #     return s0,s1

    def switch_terms(self):
        '''
        Returns (forward_one_port,reverse_one_port) switch terms.
        The ports short be connected with a half decent THRU before calling.
        These measure how much signal is reflected from the imperfect switched
        termination on the non-stimulated port.
        '''
        print('forward')
        self.write('USER2;DRIVPORT1;LOCKA1;NUMEB2;DENOA2;CONV1S;')
        forward = self.one_port()
        forward.name = 'forward switch term'

        print ('reverse')
        self.write('USER1;DRIVPORT2;LOCKA2;NUMEB1;DENOA1;CONV1S;')
        reverse = self.one_port()
        reverse.name = 'reverse switch term'

        return (forward,reverse)
=======
    _scpi = False

    _supported_npoints = [3, 11, 21, 51, 101, 201, 401, 801, 1601,]

    id = VNA.command(
        get_cmd="OUTPIDEN;",
        set_cmd=None,
        doc="""Instrument ID string""",
    )

    last_error = VNA.command(
        get_cmd="OUTPERRO;",
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
        set_cmd="<arg>", # This is blank on purpose. The command sent to the instrument is in the BooleanValidator constructor
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
        #assert 'HP8720' in self.id

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
            raise ValueError("The HP8720B only supports one of {self._supported_npoints}.")

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

        #if sweep:
            #self.sweep()  # NEEDS DEFINING

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
>>>>>>> cf81f554787ae19e36dc7b99e44d5403bbb171ab
