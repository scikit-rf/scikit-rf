import numpy as npy
import pyvisa.constants
import time
from .abcvna import VNA
from ...frequency import *
from ...network import *

class HP8510C(VNA):
    '''
    Slightly more sophisticated 8510 driver than the one in vna_old.py, which is preserved for compatibility.

    Features:
    * Chunked Sweep -- sweep with any number of points, emulated by sweeping in chunks or list mode if necessary.
    * Tested with AD007 (a VXI11-compliant GPIB-Ethernet adapter) and pyvisa-py (not dependent on NI VISA or Keysight IO)
    '''

    # Some GPIB adapters have hardcoded timeouts below what we need.
    # This is a hack to wait a bit longer.
    delay_seconds_before_readback = 0.0

    def __init__(self, address="GPIB::16::INSTR", visa_library='@py', **kwargs):
        super(HP8510C, self).__init__(address, visa_library=visa_library, **kwargs)
        self.resource.set_visa_attribute(pyvisa.constants.ResourceAttribute.timeout_value, 2000)
        id_str = self.query('OUTPIDEN;')
        assert('HP8510' in id_str) # example: 'HP8510C.07.14: Aug 26  1998 '
        # 8510s are slow. The above check ensures we won't wait 60s for connection error.
        self.resource.set_visa_attribute(pyvisa.constants.ResourceAttribute.timeout_value, 60000)
        self.resource.read_termination = False # Binary mode doesn't work if we allow premature termination on \n
        self.chunked_sweep_params = None
        self.read_raw = self.resource.read_raw
        if kwargs.get("initial_reset",True):
            self.reset()

    @property
    def idn(self):
        return self.query("OUTPIDEN;")

    def reset(self):
        self.write("FACTPRES;")
        self.wait_until_finished()
    
    def clear(self):
        self.resource.clear()

    def wait_until_finished(self):
        self.query("OUTPIDEN;")
    
    def get_list_of_traces(self,**kwargs):
        ''' The 8510 doesn't really support multiple traces, so we just list S params. '''
        return ["S11", "S21", "S12", "S22"]

    def get_traces(self, traces, **kwargs):
        traces_out = []
        for trace in traces:
            if trace.upper()=='S11':
                trace_out = self.s11
            elif trace.upper()=='S22':
                trace_out = self.s22
            elif trace.upper()=='S12':
                trace_out = self.s12
            elif trace.upper()=='S21':
                trace_out = self.s21
            else:
                raise(ValueError(trace+" is not a valid trace. Options: "+' '.join(self.get_list_of_traces())))

    def get_snp_network(self, ports, **kwargs):
        sweep = kwargs.get("sweep", True)
        # name = kwargs.get("name", "")
        # raw_data = kwargs.get("raw_data", True)
        if ports==(1,):
            self.write('s11;')
            return self.one_port(fresh_sweep=sweep);
        elif ports==(2,):
            self.write('s22;')
            return self.one_port(fresh_sweep=sweep)
        elif ports==(1,2) or ports==(2,1):
            return self.two_port(fresh_sweep=sweep)
        else:
            raise(ValueError("Invalid ports "+str(ports)+". Options: (1,) (2,) (1,2)."))

    def get_switch_terms(self, ports=(1, 2), **kwargs):
        return self.switch_terms()

    def set_frequency_sweep(self, f_start, f_stop, f_npoints, **kwargs):
        f_unit = kwargs.get("f_unit", "hz").lower()
        if f_unit != "hz":
            hz_start = self.to_hz(f_start, f_unit)
            hz_stop = self.to_hz(f_stop, f_unit)
        else:
            hz_start, hz_stop = f_start, f_stop
        self.set_frequency_step(hz_start, hz_stop, f_npoints)
        
















    @property
    def error(self):
        return self.query('OUTPERRO')

    @property
    def continuous(self):
        answer_dict={'\"HOLD\"':False,'\"CONTINUAL\"':True}
        return answer_dict[self.query('GROU?')]

    @continuous.setter
    def continuous(self, choice):
        if choice:
            self.write('CONT;')
        elif not choice:
            self.write('SING;')
        else:
            raise(ValueError('takes a boolean'))
    @property
    def averaging(self):
        '''
        averaging factor
        '''
        raise NotImplementedError

    @averaging.setter
    def averaging(self, factor ):
        self.write('AVERON %i;'%factor )

    def get_frequency(self, unit='ghz'):
        freq=Frequency( float(self.query('star;outpacti;')),
                float(self.query('stop;outpacti;')),\
                int(float(self.query('poin;outpacti;'))),'hz')
        freq.unit = unit
        return freq
    
    def set_frequency_ramp(self, hz_start, hz_stop, npoint=801):
        ''' Ramp (fast, not synthesized) sweep. Must have standard npoint. '''
        self.delay_seconds_before_readback = 0  # These sweeps are fast and don't need extra wait beyond timeout
        if npoint not in [51,101,201,401,801]:
            print("Warning: 8510C only supports NPOINT in [51,101,201,401,801]")
        self.resource.clear()
        self.write('RAMP; STAR %f; STOP %f; POIN%i;'%(hz_start,hz_stop,npoint))
    
    def _instrument_natively_supports_steps(self, npoint):
        if npoint in [51,101,201,401,801]:
            return True
        if npoint<=792:
            return True
        return False
    
    def _set_instrument_step_state(self, hz_start, hz_stop, npoint=801):
        assert(self._instrument_natively_supports_steps(npoint))
        self.resource.clear()
        self.delay_seconds_before_readback = 0.0
        if npoint in [51,101,201,401,801]:
            if npoint>401:
                self.delay_seconds_before_readback = 10.0
            # If it's a directly supported step sweep npoints, use regular sweep
            self.write('STEP; STAR %f; STOP %f; POIN%i;'%(hz_start,hz_stop,npoint))
        elif npoint<=792:
            self.delay_seconds_before_readback = 10.0
            # List sweep lets us support, for example, 401<npoints<801
            self.write('RAMP;')
            self.write('EDITLIST;')
            self.write('CLEL;')
            self.write('SADD;')
            self.write('STAR %f; STOP %f; POIN %i;'%(hz_start,hz_stop,npoint))
            self.write('SDON; EDITDONE; LISFREQ;')

    def set_frequency_step(self, hz_start, hz_stop, npoint=801):
        ''' Step (slow, synthesized) sweep + logic to handle lots of npoint. '''
        if (self._instrument_natively_supports_steps(npoint)):
            self.chunked_sweep_params = None
            self._set_instrument_step_state(hz_start, hz_stop, npoint)
        else:
            self.chunked_sweep_params = (hz_start, hz_stop, npoint)

    def ask_for_cmplx(self, outp_cmd):
        """Like ask_for_values, but use FORM2 binary transfer, much faster than ASCII."""
        # Benchmarks:
        #  %time i.write('FORM4; OUTPDATA'); i._read(); None      # 7960ms
        #  %time i.write('FORM2; OUTPDATA'); i._read_raw(); None  #  399ms
        self.write('FORM2;')
        time.sleep(self.delay_seconds_before_readback)
        self.write(outp_cmd)
        buf = self.read_raw()
        float_bin = buf[4:] # Skip 4 header bytes and trailing newline
        try:
            floats = npy.frombuffer(float_bin, dtype='>f4').reshape((-1,2))
        except ValueError as e:
            print(buf)
            print("len(buf): %i"%(len(buf),))
            raise(e)
        cmplxs = (floats[:,0] + 1j*floats[:,1]).flatten()
        return cmplxs
    
    def _chunked_sweep_chunks(self):
        hz_start, hz_stop, npoints = self.chunked_sweep_params
        freqs = npy.linspace(hz_start, hz_stop, npoints)
        i = 0
        while i<npoints:
            pts_remaining = npoints - i
            if pts_remaining >= 801:
                chnk_size = 801
            elif pts_remaining <=792:
                chnk_size = pts_remaining
            else:
                chnk_size = 401
            freq_chnk = freqs[i:(i+chnk_size)]
            yield (freq_chnk[0],freq_chnk[-1],chnk_size)
            i += chnk_size

    def _one_port(self, fresh_sweep=True):
        ''' Perform a single sweep and return Network data. '''
        if fresh_sweep:
             self.write('SING;') # Poll for sweep status
        s =  self.ask_for_cmplx('OUTPDATA')
        ntwk = Network()
        ntwk.s = s
        ntwk.frequency= self.get_frequency()
        return ntwk

    def one_port(self, **kwargs):
        ''' Performs one or more sweeps as per set_frequency_*, returns network. '''
        if self.chunked_sweep_params is None:
            return self._one_port()
        stitched_network = None
        for (hz_start,hz_end,chnk_size) in self._chunked_sweep_chunks():
            self._set_instrument_step_state(hz_start,hz_end,chnk_size)
            chunk_net = self._one_port()
            stitched_network = chunk_net if stitched_network is None else stitch(stitched_network,chunk_net)
        return stitched_network

    def _two_port(self, fresh_sweep=True):
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
        s11 = self._one_port().s[:,0,0]
        self.write('s12;')
        s12 = self._one_port().s[:,0,0]
        self.write('s22;')
        s22 = self._one_port().s[:,0,0]
        self.write('s21;')
        s21 = self._one_port().s[:,0,0]

        ntwk = Network()
        ntwk.s = npy.array(\
                [[s11,s21],\
                [ s12, s22]]\
                ).transpose().reshape(-1,2,2)
        ntwk.frequency= self.get_frequency()

        return ntwk
    
    def two_port(self, **kwargs):
        ''' Performs one or more sweeps per set_frequency_*, returns network. '''
        if self.chunked_sweep_params is None:
            return self._two_port()
        stitched_network = None
        for (hz_start,hz_end,chnk_size) in self._chunked_sweep_chunks():
            self._set_instrument_step_state(hz_start,hz_end,chnk_size)
            chunk_net = self._two_port()
            stitched_network = chunk_net if stitched_network is None else stitch(stitched_network,chunk_net)
        return stitched_network


    ##properties for the super lazy
    @property
    def s11(self):
        self.write('s11;')
        ntwk =  self.one_port()
        ntwk.name = 'S11'
        return ntwk
    @property
    def s22(self):
        self.write('s22;')
        ntwk =  self.one_port()
        ntwk.name = 'S22'
        return ntwk
    @property
    def s12(self):
        self.write('s12;')
        ntwk =  self.one_port()
        ntwk.name = 'S12'
        return ntwk
    @property
    def s21(self):
        self.write('s21;')
        ntwk =  self.one_port()
        ntwk.name = 'S21'
        return ntwk

    def switch_terms(self):
        '''
        measures forward and reverse switch terms and returns them as a
        pair of one-port networks.

        returns:
                forward, reverse: a tuple of one ports holding forward and
                        reverse switch terms.

        see also:
                skrf.calibrationAlgorithms.unterminate_switch_terms

        notes:
                thanks to dylan williams for making me aware of this, and
                providing the gpib commands in his statistical help

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

        