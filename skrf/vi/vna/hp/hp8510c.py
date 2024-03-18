"""
.. module:: skrf.vi.vna.hp.hp8510c
=================================================
HP 8510C (:mod:`skrf.vi.vna.hp.hp8510c`)
=================================================

HP8510C Class
================

.. autosummary::
    :nosignatures:
    :toctree: generated/
"""

import time

import numpy as np
import pyvisa

import skrf
import skrf.network
from skrf.vi.vna import VNA

from .hp8510c_sweep_plan import SweepPlan


class HP8510C(VNA):
    '''
    HP 8510 driver that is capable of compound sweeps, segmented sweeps,
    and fast binary transfers. These features make this venerable old instrument
    much more pleasant to use in the 21st century. Likely works with "A" and
    "B" versions of the instrument as well.

    Compound sweeps occur automatically when the user requests a sweep larger
    than what the instrument natively supports (51/101/201/401/801pts). This
    driver takes multiple shorter sweeps and stitches them together.

    Segmented sweeps occur automatically when the user requests a short or
    irregularly spaced sweep (see "Advanced Example" below). The 8510 actually
    does support sweeps other than 51/101/201/401/801pts, but only in a separate
    mode (segmented sweep mode) and with significant restrictions. This driver
    knows how to handle segmented sweep mode to get what it wants.


    Examples
    ============

    Basic one-port:

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8510C(address='TCPIP::ad007-right.lan::gpib0,16::INSTR', backend='@py')
        vna.set_frequency_sweep(2e9,3e9,201)
        vna.get_snp_network(ports=(1,))



    Basic two-port:

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8510C(address='TCPIP::ad007-right.lan::gpib0,16::INSTR', backend='@py')
        vna.set_frequency_sweep(2e9,3e9,201)
        vna.get_snp_network(ports=(1,2))



    Intermediate example -- note that 1001 point sweeps are not natively supported by the instrument; this driver
    takes multiple sweeps and stitches the results together.

    .. code-block:: python

        vna = skrf.vi.vna.hp.HP8510C(address='TCPIP::ad007-right.lan::gpib0,16::INSTR', backend='@py')
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
    compound_sweep_plan = None
    #: If None, get_snp_network()/one_port()/two_port() just ask the VNA for data.
    # If populated, those methods perform the multiple sweeps in the plan and stitch together the results.

    def __init__(self, address : str, backend : str = "@py", **kwargs):
        super().__init__(address, backend, **kwargs)
        # Tested 2024-03-09:
        #     HP8510C.07.14
        #     AD007 (a VXI11-complaint GPIB-Ethernet adapter)
        #     address="TCPIP::ad007-right.lan::gpib0,16::INSTR", backend="@py"

        # 8510s are slow. This check ensures we won't wait 60s for connection error.
        self._resource.timeout = 2_000
        id_str = self.query('OUTPIDEN;')
        assert('HP8510' in id_str) # example: 'HP8510C.07.14: Aug 26  1998 '

        # 8510s are slow. Actual work might take acutal 60 seconds.
        self._resource.timeout = 60_000
        self._resource.read_termination = False # Binary mode doesn't work if we allow premature termination on \n
        self.read_raw = self._resource.read_raw
        self.reset()
        self.min_hz = self.freq_start
        self.max_hz = self.freq_stop
        self.write('STEP;')

        # If compound_sweep_plan is None, we rely on the internal state of the analyzer.
        # If compound_sweep_plan is a SweepPlan object, we must take multiple short sweeps
        # on the 8510 and stitch them into a single compound sweep ourselves.
        self.compound_sweep_plan = None

    @property
    def id(self):
        ''' Instrument ID string '''
        return self.query("OUTPIDEN;")

    def reset(self):
        ''' Preset instrument. '''
        self.write("FACTPRES;")
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
            self.write('s11;')
            return self.one_port(fresh_sweep=sweep)
        elif ports==(2,):
            self.write('s22;')
            return self.one_port(fresh_sweep=sweep)
        elif ports==(1,2) or ports==(2,1):
            return self.two_port(fresh_sweep=sweep)
        else:
            raise(ValueError("Invalid ports "+str(ports)+". Options: (1,) (2,) (1,2)."))

    def get_switch_terms(self, ports=(1, 2), **kwargs):
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
    def is_continuous(self):
        ''' True iff sweep mode is continuous. Can be set. '''
        answer_dict={'\"HOLD\"':False,'\"CONTINUAL\"':True}
        return answer_dict[self.query('GROU?')]

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
        raise NotImplementedError

    @averaging.setter
    def averaging(self, factor ):
        self.write('AVERON %i;'%factor )

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
        return skrf.Frequency.from_f(self.compound_sweep_plan.get_hz(), unit='hz')

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
        self.compound_sweep_plan = SweepPlan.from_hz(hz)

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
        else:
            self.compound_sweep_plan = SweepPlan.from_ssn(hz_start, hz_stop, npoint)

    def set_frequency_ramp(self, hz_start, hz_stop, npoint=801):
        ''' Ramp (fast, not synthesized) sweep. Must have standard npoint. '''
        if npoint not in [51,101,201,401,801]:
            print("Warning: 8510C only supports NPOINT in [51,101,201,401,801]")
        self.resource.clear()
        self.write('RAMP; STAR %f; STOP %f; POIN%i;'%(hz_start,hz_stop,npoint))

    @property
    def freq_start(self):
        ''' Start frequency [hz] '''
        return float(self.query('STAR;OUTPACTI;'))

    @freq_start.setter
    def freq_start(self, new_start_hz):
        self.write(f'STEP; STAR {new_start_hz};')

    @property
    def freq_stop(self):
        ''' Stop frequency [hz] '''
        return float(self.query('STOP;OUTPACTI;'))

    @freq_stop.setter
    def freq_stop(self, new_stop_hz):
        self.write(f'STEP; STOP {new_stop_hz};')

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
        return len(self.compound_sweep_plan.get_hz())

    @npoints.setter
    def npoints(self, npoint):
        ''' Set number of points in sweep'''
        hz_start, hz_stop = self.freq_start, self.freq_stop
        if (self._instrument_natively_supports_steps(npoint)):
            self.compound_sweep_plan = None
            self._set_instrument_step_state(hz_start, hz_stop, npoint)
        else:
            self.compound_sweep_plan = SweepPlan.from_ssn(hz_start, hz_stop, npoint)

    def _instrument_natively_supports_steps(self, npoint):
        if npoint in [51,101,201,401,801]:
            return True
        if npoint<=792: # Supported with a single native list sweep
            return True
        return False

    def _set_instrument_step_state(self, hz_start, hz_stop, npoint=801):
        assert(self._instrument_natively_supports_steps(npoint))
        if self._resource is not None:
            self._resource.clear()
        if npoint in [51,101,201,401,801]:
            # If it's a directly supported step sweep npoints, use regular sweep
            self.write('STEP; STAR %f; STOP %f; POIN%i;'%(hz_start,hz_stop,npoint))
        elif npoint<=792:
            # List sweep lets us support, for example, 401<npoints<801
            self.write('STEP;')
            self.write('EDITLIST;')
            self.write('CLEL;')
            self.write('SADD;')
            self.write('STAR %f; STOP %f; POIN %i;'%(hz_start,hz_stop,npoint))
            self.write('SDON; EDITDONE; LISFREQ;')

    def _set_instrument_cwstep_state(self, hz_list):
        assert(len(hz_list)<=30) # 8510 only supports CW lists up to length 30
        self.write('STEP;')
        self.write('EDITLIST;')
        self.write('CLEL;')
        for hz in hz_list:
            self.write('SADD;')
            self.write(f'CWFREQ {int(hz)};')
        self.write('SDON; EDITDONE; LISFREQ;')

    def ask_for_cmplx(self, outp_cmd, timeout_s=30):
        """Like ask_for_values, but use FORM2 binary transfer, much faster than ASCII."""
        # Benchmarks:
        #  %time i.write('FORM4; OUTPDATA'); i._read(); None      # 7960ms
        #  %time i.write('FORM2; OUTPDATA'); i._read_raw(); None  #  399ms
        self.wait_for_status(max_wait_seconds=timeout_s)
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
        return cmplxs

    def _one_port(self, expected_hz=None, fresh_sweep=True):
        ''' Perform a single sweep and return Network data. '''
        if fresh_sweep:
             self.write('SING;') # Poll for sweep status
        s =  self.ask_for_cmplx('OUTPDATA')
        ntwk = skrf.Network()
        ntwk.s = s
        hz = expected_hz if expected_hz is not None else self._frequency.f
        assert(len(s)==len(hz))
        ntwk.frequency = skrf.Frequency.from_f(hz,unit='hz')
        return ntwk

    def one_port(self, **kwargs):
        ''' Performs a single sweep OR COMPOUND SWEEP and returns Network data. '''
        if self.compound_sweep_plan is None:
            return self._one_port()
        old_start_hz, old_stop_hz = self.freq_start, self.freq_stop
        stitched_network = None
        for sweep_section in self.compound_sweep_plan.get_sections():
            sweep_section.apply_8510(self)
            chunk_net_nomask = self._one_port(expected_hz=sweep_section.get_raw_hz())
            chunk_net = sweep_section.mask_8510(chunk_net_nomask)
            stitched_network = (chunk_net if stitched_network is None
                                else skrf.network.stitch(stitched_network, chunk_net))
        self.freq_start = old_start_hz
        self.freq_stop  = old_stop_hz
        return stitched_network

    def _two_port(self, expected_hz=None, fresh_sweep=True):
        ''' Performs a single sweep and returns Network data. '''
        # ASCII vs Binary transfer performance:
        #  ascii_xfer separate_sweep: 32s
        #  bin_xfer   separate_sweep: 08s
        #  ------------------------------
        #  Decision: spend time to implement binary transfer. It matters.
        #
        # Sweep mode:
        #  source  mode   strategy  time
        #  83651   ramp   FOUPOVER  12.8s
        #  83651   ramp   4xSxx     11.6s  <-- in ramp mode + faststep, consecutive Sxx sweeps are fastest
        #  83651   ramp   FULL2PORT 14.5s
        #  83651   step   FOUPOVER  50.0s  <-- in step mode, FOUPOVER is a tad faster.
        #  83651   step   4xSxx     52.5s
        #  83651   step   FULL2PORT 52.3s
        #  --------------------------
        #  Decision: use consecutive Sxx sweeps everywhere for simplicity + max speed of fast sweeps
        self.write('s11;')
        s11 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('s12;')
        s12 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('s22;')
        s22 = self._one_port(expected_hz=expected_hz, fresh_sweep=fresh_sweep).s[:,0,0]
        self.write('s21;')
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
        old_start_hz, old_stop_hz = self.freq_start, self.freq_stop
        stitched_network = None
        for sweep_chunk in self.compound_sweep_plan.get_sections():
            sweep_chunk.apply_8510(self)
            chunk_net_nomask = self._two_port(expected_hz=sweep_chunk.get_raw_hz())
            chunk_net = sweep_chunk.mask_8510(chunk_net_nomask)
            stitched_network = (chunk_net if stitched_network is None
                                else skrf.network.stitch(stitched_network,chunk_net))
        self.freq_start = old_start_hz
        self.freq_stop  = old_stop_hz
        return stitched_network

    def wait_for_status(self, max_wait_seconds=30):
        ''' Fetches status bytes respecting max_wait_seconds even if it must tank a few timeouts along the way. '''
        t0 = time.time()
        while True:
            try:
                status_str = self.query('OUTPSTAT')
                s0str, s1str = status_str.strip().split(',')
                s0 = int(s0str)
                s1 = int(s1str)
                break
            except pyvisa.errors.VisaIOError as e:
                waited = time.time()-t0
                if waited > max_wait_seconds:
                    raise e
        return s0,s1

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
