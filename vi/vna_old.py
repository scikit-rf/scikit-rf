

'''
.. module:: skrf.vi.vna
++++++++++++++++++++++++++++++++++++++++++++++++++++
Vector Network Analyzers (:mod:`skrf.vi.vna`)
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. warning::

    As of 2017.02 a new architecture for vna drivers is being implemented.

New VNA drivers 
---------------

- VNA drivers will now have a common high level functionality across all vendors implemented in an ABCVNA class.
- Different vendor drivers will implement their own mid level functionality as needed to implement the ABC class
- The low level functions are all implemented as SCPI commands which have a new way of being generated and called

Legacy vna module
------------------------
The old vna.py module containing drivers for PNA, PNA-X, HP8510, etc. will be available as vna_old.py and can be used as
follows:

::

    from skrf.vi.vna_old import PNA

Available VNAs
------------------

.. autosummary::
    :toctree: generated/

    PNA
    ZVA40
    HP8510C
    HP8720
'''
import numpy as npy
#import visa
#from visa import Driver
from warnings import warn
from itertools import product
import re

from ..frequency import *
from ..network import *
from ..calibration.calibration import Calibration, SOLT, OnePort, \
                                      convert_pnacoefs_2_skrf,\
                                      convert_skrfcoefs_2_pna
from .. import mathFunctions as mf

from . ivihack import Driver

class PNA(Driver):
    '''
    Agilent PNA[X]


    Below are lists of some high-level commands sorted by functionality.


    Object IO

    .. hlist::
        :columns: 2

        * :func:`~PNA.get_oneport`
        * :func:`~PNA.get_twoport`
        * :func:`~PNA.get_frequency`
        * :func:`~PNA.get_network`
        * :func:`~PNA.get_network_all_meas`


    Simple IO

    .. hlist::
        :columns: 2

        * :func:`~PNA.get_data_snp`
        * :func:`~PNA.get_data`
        * :func:`~PNA.get_sdata`
        * :func:`~PNA.get_fdata`
        * :func:`~PNA.get_rdata`


    Calibration

    .. hlist::
        :columns: 2

        * :func:`~PNA.get_calibration`
        * :func:`~PNA.set_calibration`
        * :func:`~PNA.get_cal_coefs`
        * more below

    Examples
    -----------

    >>> from skrf.vi.vna import PNA
    >>> v = PNA()
    >>> n = v.get_oneport()
    >>> n = v.get_twoport()


    Notes
    --------
    This instrument references `measurements` and `traces`. Traces are
    displayed traces, while measurements are active measurements on the
    VNA which may or may not be displayed on screen.
    '''
    def __init__(self, address, channel=1,timeout = 3, echo = False,
        front_panel_lockout= False, **kwargs):
        '''
        Constructor

        Parameters
        -------------
        address : int or str
            GPIB address , or resource string
        channel : int
            set active channel. Most commands operate on the active channel
        timeout : number
            GPIB command timeout in seconds.
        echo : Boolean
            echo  all strings passed to the write command to stdout.
            useful for troubleshooting
        front_panel_lockout : Boolean
            lockout front panel during operation.
        \*\*kwargs :
            passed to  `visa.Driver.__init__`
        '''

        if isinstance(address,int):
            resource = 'GPIB::%i::INSTR'%address
        else:
            resource = address

        Driver.__init__(self,resource = resource, **kwargs)
        
        self.channel=channel
        self.port = 1
        self.echo = echo
        self.timeout=timeout
        if not front_panel_lockout:
            pass#self.gtl()


    def write(self, msg, *args, **kwargs):
        '''
        Write a msg to the instrument.
        '''
        if self.echo:
            print(msg)
        return Driver.write(self,msg, *args, **kwargs)

    write.__doc__ = Driver.write.__doc__
    
    @property
    def timeout(self):
        return self._interface.timeout/1000.
    @timeout.setter
    def timeout(self,val):
        self._interface.timeout=val*1000.
        
    ## BASIC GPIB
    @property
    def idn(self):
        '''
        Identifying string for the instrument
        '''
        return self.ask('*IDN?')

    def opc(self):
        '''
        Ask for indication that operations complete
        '''
        return self.ask('*OPC?')

    '''def gtl(self):
        ''''''
        Go to local.
        ''''''
        self._vpp43.gpib_control_ren(
            self.vi,
            self._vpp43.VI_GPIB_REN_DEASSERT_GTL,
            )'''

    def reset(self):
        '''
        reset
        '''
        self.write('*RST;')

    ## triggering
    @property
    def continuous(self):
        '''
        Set continuous sweeping ON/OFF
        '''
        return (self.ask('sense:sweep:mode?')=='CONT')

    @continuous.setter
    def continuous(self, val):
        '''
        '''
        if val:
            self.write('sense:sweep:mode cont')
        else:
            self.write('sense:sweep:mode hold')

    def sweep(self):
        '''
        Initiates a sweep and waits for it to complete before returning

        If vna is in continuous sweep mode then this puts it back
        '''
        was_cont = self.continuous
        out = bool(self.ask("SENS:SWE:MODE SINGle;*OPC?"))
        self.continuous = was_cont
        return out

    def get_sweep_type(self):
        '''
        Sets the type of analyzer sweep mode. First set sweep type, then set sweep
        parameters such as frequency or power settings.

        Parameters
        -------------
        val: str
            Choose from:
                LINear | LOGarithmic | POWer | CW | SEGMent | PHASe
                Note: SWEep TYPE cannot be set to SEGMent if there
                are no segments turned ON.
        '''
        return self.ask('sense%i:sweep:type?'%self.channel)

    def set_sweep_type(self,val):
        self.write('sense%i:sweep:type %s'%(self.channel,val))

    sweep_type = property(get_sweep_type, set_sweep_type)
    def get_sweep_mode(self):
        '''
        Sets the number of trigger signals the specified channel will ACCEPT.
        See Triggering the PNA Using SCPI.

        Parameters
        -------------
        val: str
            Trigger mode. Choose from:
            HOLD - channel will not trigger
            CONTinuous - channel triggers indefinitely
            GROups - channel accepts the number of triggers specified with the last
            SENS:SWE:GRO:COUN <num>. This is one of the PNA overlapped
            commands. Learn more.
            SINGle - channel accepts ONE trigger, then goes to HOLD.
        '''
        return self.ask('sense%i:sweep:mode?'%self.channel)

    def set_sweep_mode(self,val):
        self.write('sense%i:sweep:mode %s'%(self.channel,val))

    sweep_mode = property(get_sweep_mode, set_sweep_mode)


    def get_trigger_mode(self):
        '''
        Sets and reads the trigger mode for the specified channel.
        This determines what EACH signal will trigger.

        values
        ----------

        ['channel','sweep','point','trace']

        '''
        return self.ask('sense%i:sweep:trigger:mode?'%self.channel)

    def set_trigger_mode(self,val):
        if val.lower() not in ['channel','sweep','point','trace']:
            raise ValueError('value must be a boolean')

        self.write('sense%i:sweep:trigger:mode %s'%(self.channel, val))

    trigger_mode = property(get_trigger_mode, set_trigger_mode)

    def get_trigger_source(self):
        '''
        Sets the source of the sweep trigger signal. This command is
        a super-set of INITiate:CONTinuous which can NOT set the
        source to External.

        values
        --------

        EXTernal - external (rear panel) source.
        IMMediate - internal source sends continuous trigger signals
        MANual - sends one trigger signal when manually triggered from
            the front panel or INIT:IMM is sent.
        '''
        return self.ask('trigger:sequence:source?')

    def set_trigger_source(self,val):
        '''
        '''
        self.write('trigger:sequence:source %s'%val)

    trigger_source = property(get_trigger_source, set_trigger_source)

    def trigger(self):
        '''
        sent a manual trigger signal
        '''
        self.write('INIT:IMM')

    def trigger_and_wait_till_done(self):
        '''
        send a manual trigger signal, and don't return until operation
        is completed
        '''
        self.trigger()
        self.opc()

    ## power
    def get_power_level(self):
        '''
        Get the RF power level
        '''
        return float(self.ask('SOURce:POWer?'))

    def set_power_level(self, num, cnum=None, port=None):
        '''
        Set the RF power level

        Parameters
        -----------
        num : float
            Source power in dBm

        '''
        if cnum is None:
            cnum = self.channel

        if port is None:
            port = self.port

        self.write('SOURce%i:POWer%i %f'%(cnum, port, num))

    power_level = property(get_power_level, set_power_level)

    def toggle_port_power(self, mode = 'on', port =1):
        '''
        Turn a given port's power on or off or auto

        Parameters
        ----------
        mode  : str
            ['on','off', 'auto']
        port : int
            the port (duh)
        '''
        if mode is True:
            mode = 'on'
        elif mode is False:
            mode = 'off'

        self.write('source%i:power%i:mode %s'%(self.channel,port, mode))



    ## IO - Frequency related
    def get_f_start(self):
        '''
        Start frequency in Hz
        '''
        return float(self.ask('sens%i:FREQ:STAR?'%(self.channel)))

    def set_f_start(self,f):
        '''
        Start frequency in Hz
        '''
        self.write('sens%i:FREQ:STAR %f'%(self.channel,f))

    f_start = property(get_f_start, set_f_start)

    def get_f_stop(self):
        '''
        Stop frequency in Hz
        '''
        return float(self.ask('sens%i:FREQ:STOP?'%(self.channel)))

    def set_f_stop(self,f):
        '''
        Stop frequency in Hz
        '''
        self.write('sens%i:FREQ:STOP %f'%(self.channel,f))

    f_stop = property(get_f_stop, set_f_stop)

    def get_f_npoints(self):
        '''
        Number of points for the measurement
        '''
        return int(self.ask('sens%i:swe:poin?'%(self.channel)))

    def set_f_npoints(self, n):
        '''
        Number of points for the measurement
        '''
        self.write('sens%i:swe:poin %i'%(self.channel,n))

    f_npoints = property(get_f_npoints, set_f_npoints)
    npoints = f_npoints

    def get_frequency(self, unit='ghz'):
        '''
        Get frequency data for active meas.

        This Returns a :class:`~skrf.frequency.Frequency` object.

        Parameters
        -------------
        unit : ['khz','mhz','ghz','thz']
            the frequency unit of the Frequency object.

        See Also
        ---------
        select_meas
        get_meas_list
        '''
        freq = Frequency(self.f_start,
                         self.f_stop,
                         self.f_npoints,'hz')
        freq.unit = unit

        return freq

    def set_frequency(self, freq):
        self.f_start = freq.start
        self.f_stop = freq.stop
        self.f_npoints = freq.npoints

    frequency = property(get_frequency,set_frequency)

    def get_frequency_cw(self):
        '''
        Sets the Continuous Wave (or Fixed) frequency. Must also send
        SENS:SWEEP:TYPE CW to put the analyzer into CW sweep mode.

        Parameters
        --------------
        val : number
            CW frequency. Choose any number between the minimum and
            maximum frequency limits of the analyzer. Units are Hz.

        This command will accept MIN or MAX instead of a numeric
        parameter. See SCPI Syntax for more information
        '''
        return float(self.ask('sens%i:FREQ?'%(self.channel)))

    def set_frequency_cw(self, val):
        self.write('sens%i:FREQ %f'%(self.channel,val))

    frequency_cw = property(get_frequency_cw, set_frequency_cw)

    ##  IO - S-parameter and  Networks
    def get_snp_format(self):
        '''
        the output format for snp data.
        '''
        return self.ask('MMEM:STOR:TRAC:FORM:SNP?')

    def set_snp_format(self, val= 'ri'):
        '''
        the output format for snp data.
        '''
        if val.lower() not in ['ma','ri','auto','disp']:
            raise ValueError('bad value for `val`')

        self.write('MMEM:STOR:TRAC:FORM:SNP %s'%val)

    snp_format = property(get_snp_format, set_snp_format)

    def get_network(self, sweep=True, name = None):
        '''
        Returns a :class:`~skrf.network.Network` object representing the
        active trace.

        This can be used to get arbitrary traces, in the form of
        Network objects, so that they can be plotted/saved/etc.

        If you want to get s-parameter data, use :func:`get_twoport` or
        :func:`get_oneport`

        Parameters
        -----------
        sweep : Boolean
            trigger a sweep or not. see :func:`sweep`

        Examples
        ----------
        >>> from skrf.vi.vna import PNAX
        >>> v = PNAX()
        >>> dut = v.get_network()

        See Also
        ----------
        get_network_all_meas
        '''

        was_cont = self.continuous
        self.continuous   = False
        if sweep:
            self.sweep()

        ntwk = Network(
            s = self.get_sdata(),
            frequency = self.get_frequency(),
            )
        if name is None:
            name = self.get_active_meas()

        ntwk.name = name
        self.continuous = was_cont
        return ntwk

    def get_network_all_meas(self,sweep=True):
        '''
        Return list of Network Objects for all measurements.


        See Also
        -----------
        get_meas_list
        get_network
        '''

        out = []
        if sweep:
            self.sweep()
        for name,parm in self.get_meas_list():
            self.select_meas(name)
            out.append(self.get_network(sweep=False, name= name))

        return out

    def get_oneport(self, port=1, *args, **kwargs):
        '''
        Get a one-port Network object for given ports.

        This calls :func:`~PNA.get_data_snp` and :func:`~PNA.get_frequency`
        to retrieve data, and then creates and returns a
        :class:`~skrf.network.Network` object.

        Parameters
        ------------
        ports : list of ints
            list of port indecies to retrieve data from

        \*args,\*\*kwargs :
            passed to Network init

        See Also
        -----------
        get_twoport
        get_snp
        get_frequency
        '''
        was_cont = self.continuous
        self.continuous   = False
        self.sweep()
        ntwk = Network(
            s = self.get_data_snp(port),
            frequency = self.get_frequency(),
            *args, **kwargs
            )
        self.continuous = was_cont
        return ntwk

    def get_twoport(self, ports=[1,2], sweep=True, single=True, *args, **kwargs):
        '''
        Get a two-port Network object for given ports.

        This calls :func:`~PNA.get_data_snp` and :func:`~PNA.get_frequency`
        to retrieve data, and then creates and returns a
        :class:`~skrf.network.Network` object.

        Parameters
        ------------
        ports : list of ints
            list of port indecies to retrieve data from

        \*args,\*\*kwargs :
            passed to Network init

        '''
        if single:
            was_cont = self.continuous
            self.continuous= False
        if sweep:
            self.sweep()
        ntwk = Network(
            s = self.get_data_snp(ports),
            frequency = self.get_frequency(),
            *args, **kwargs
            )
        if single:
            self.continuous = was_cont
        return ntwk

    def get_data_snp(self,ports=[1,2]):
        '''
        Get n-port, s-parameter data.

        Returns s-parameter data of shape FXNXN where F is frequency
        length and N is number of ports. This does not do any timing
        see :func:`sweep` for that or use a higher level IO command,
        which are listed below in `see also`.

        Note, this uses the  `calc:data:snp:ports` command

        Parameters
        ------------
        ports : list of ints
            list of port indecies to retrieve data from

        See Also
        ----------
        get_oneport
        get_twoport
        get_frequency
        '''

        if type(ports) == int:
            ports = [ports]


        d = self.ask_for_values('calc%i:data:snp:ports? \"%s\"'\
            %(self.channel,str(ports)[1:-1]))


        npoints = len(self.get_frequency())
        nports = len(ports)


        ##TODO: this could be re-written in a general matrical way so
        # that testing for cases is not needed. i didnt have time.
        if nports==2:
            d = npy.array(d)
            d = d.reshape(9,-1).T
            s11 = d[:,1] +1j*d[:,2]
            s21 = d[:,3] +1j*d[:,4]
            s12 = d[:,5] +1j*d[:,6]
            s22 = d[:,7] +1j*d[:,8]
            s = npy.c_[s11,s12,s21,s22].reshape(-1,2,2)
        elif nports == 1:
            d = npy.array(d)
            d = d.reshape(3,-1).T
            s = (d[:,1] +1j*d[:,2]).reshape(-1,1,1)
        else:
            raise NotImplementedError()
        return s

    def get_data(self, char='SDATA', cnum = None):
        '''
        Get data for current active measuremnent

        Note that this doesnt do any sweep timing. It just gets whatever
        data is in the registers according to char.  If you want the
        data to be returned after a sweep has completed

        Parameters
        ------------
        char : [SDATA, FDATA, RDATA]
            type of data to return


        See Also
        ----------
        get_sdata
        get_fdata
        get_rdata
        get_snp_data

        '''
        if cnum is None:
            cnum = self.channel

        self.write('calc%i:par:sel \"%s\"'%(cnum, self.get_active_meas()))
        data = npy.array(self.ask_for_values('CALC%i:Data? %s'%(cnum, char)))

        if char.lower() == 'sdata':
            data = mf.scalar2Complex(data)

        return data

    def get_sdata(self, *args, **kwargs):
        '''
        Get complex data

        See Also
        ---------
        get_data

        '''
        out= self.get_data(char = 'SDATA', *args, **kwargs)

        return out

    def get_fdata(self, *args, **kwargs):
        '''
        Get formatted data

        See Also
        ----------
        get_data
        '''
        return self.get_data(char = 'fDATA', *args, **kwargs)

    def get_rdata(self, char='A', cnum = None):
        '''
        Get data directly from the recievers.

        Parameters
        -----------
        char : ['A', 'B', 'C', ... , 'REF']
            the reciever to measure, the 'REF' number  (like R1, R2)
            depends on the source port.
        cnum : int
            channel number

        '''
        if cnum is None:
            cnum = self.channel
        self.write('calc:par:sel %s'%(self.get_active_meas()))
        return npy.array(self.ask_for_values('CALC%i:RData? %s'%(cnum, char)))

    def get_switch_terms(self, ports = [1,2]):
        '''
        Get switch terms and return them as a tuple of Network objects.

        Returns
        --------
        forward, reverse : oneport switch term Networks
        '''

        p1,p2 = ports
        self.delete_all_meas()
        self.create_meas('forward switch term', 'a%i/b%i,%i'%(p2,p2,p1))
        forward = self.get_network()


        self.delete_all_meas()
        self.create_meas('reverse switch term', 'a%i/b%i,%i'%(p1,p1,p2))
        reverse = self.get_network()
        self.delete_all_meas()
        return forward, reverse

    ## MEASUREMENT/TRACES
    @property
    def ntraces(self):
        '''
        The number of measurement traces that exist on the current channel

        Note that this may not be the same as the number of traces
        displayed because a measurement may exist, but not be associated
        with a trace.

        '''
        n = self.get_meas_list()
        if n is None:
            return 0
        else:
            return len(n)

    @property
    def if_bw(self):
        '''
        IF bandwidth
        '''
        return float(self.ask('sens%i:band?'%self.channel))

    @if_bw.setter
    def if_bw(self,n):
        '''
        IF bandwidth
        '''
        self.write('sens%i:band %i'%(self.channel,n))


    def set_yscale_auto(self, window_n=None, trace_n=None):
        '''
        Display a given measurment on specified trace number.

        Parameters
        ------------

        window_n : int
            window number. If None, active window is used.
        trace_n : int
            trace number to display on. If None, a new trace is made.
        '''
        if window_n is None:
            window_n =''
        if trace_n is None:
            trace_n =self.ntraces+1
        self.write('disp:wind%s:trac%s:y:scale:auto'%(str(window_n), str(trace_n)))

    def get_win_trace(self, meas=''):
        '''
        Get window number and trace number either for the current measurement
        (if meas is empty) or for a specific measurement

        Returns
        ----------
        out :  tuple
            tuple of the form (window_n, trace_n)
        '''
        if meas:
            self.select_meas(meas)
        window_n = int(self.ask('CALC:PAR:WNUM?')[1:])
        trace_n = int(self.ask('CALC:PAR:TNUM?')[1:])
        return (window_n, trace_n)

    def get_meas_list(self):
        '''
        Get a list of existent measurements

        Returns
        ----------
        out :  list
            list of tuples of the form, (name, measurement)
        '''
        meas_list = self.ask("CALC%i:PAR:CAT:EXT?"%self.channel)

        meas = meas_list[1:-1].split(',')
        if len(meas)==1:
            # if there isnt a single comma, then there arent any measurments
            return None


        return [(meas[k],meas[k+1]) for k in range(0,len(meas)-1,2)]

    def get_window_list(self):
        '''
        Get list of existing window numbers

        Returns
        ----------
        out :  list
            list of window numbers
        '''
        window_list = self.ask("DISP:CAT?")

        windows = window_list[1:-1].split(',')
        if 'EMPTY' in window_list:
            # no windows defined
            return None

        return [int(k) for k in windows]

    def get_active_meas(self):
        '''
        Get the name of the active measurement
        '''
        out = self.ask("SYST:ACT:MEAS?")[1:-1]
        return out

    def delete_meas(self,name):
        '''
        Delete a measurement with name `name`


        '''
        self.write('calc%i:par:del %s'%(self.channel, name))

    def delete_all_meas(self):
        '''
        duh
        '''
        self.write('calc%i:par:del:all'%self.channel)

    def create_meas(self,name, meas):
        '''
        Create a new measurement.

        Parameters
        ------------
        name : str
            name given to measurment
        meas : str
            something like
            * S11
            * a1/b1,1
            * A/R1,1
            * ...

        Examples
        ----------
        >>> p = PNA()
        >>> p.create_meas('my_meas', 'A/R1,1')
        '''
        self.write('calc%i:par:def:ext \"%s\", \"%s\"'%(self.channel, name, meas))
        self.display_trace(name)

    def create_meas_hidden(self,name, meas):
        '''
        Create a new measurement but dont display it.

        Parameters
        ------------
        name : str
            name given to measurment
        meas : str
            something like
            * S11
            * a1/b1,1
            * A/R1,1
            * ...

        Examples
        ----------
        >>> p = PNA()
        >>> p.create_meas('my_meas', 'A/R1,1')
        '''
        self.write('calc%i:par:def:ext %s, %s'%(self.channel, name, meas))

    def select_meas(self,name):
        '''
        Make a specified measurement active

        Parameters
        ------------
        name : str
            name of measurement. See :func:`get_meas_list`


        '''
        self.write('calc%i:par:sel \"%s\"'%(self.channel, name))

    def set_window_arrangement(self, arr='overlay'):
        '''
        Set window arrangement for currently active measurements.

        Parameters
        ------------
        arr : str
            Window arrangement.

        Choose from:

        * TILE - tiles existing windows
        * CASCade - overlaps existing windows
        * OVERlay - all traces placed in 1 window
        * STACk - 2 windows
        * SPLit - 3 windows
        * QUAD - 4 windows
        '''
        self.write('disp:arr %s'%str(arr))

    def display_trace(self,  name = '',window_n = None, trace_n=None, form='MLOG'):
        '''
        Display a given measurement on specified trace number.

        Parameters
        ------------
        name : str
            name of measurement. See :func:`get_meas_list`
        window_n : int
            window number. If None, active window is used.
        trace_n : int
            trace number to display on. If None, a new trace is made.
        form : str
            display format (see set_display_format())
        '''
        if window_n is None:
            window_n =''
        if trace_n is None:
            trace_n =self.ntraces+1
        self.write('disp:wind%s:trac%s:feed \"%s\"'%(str(window_n), str(trace_n), name))
        self.select_meas(name)
        self.set_display_format(form)

    def set_display_format(self, form):
        '''
        Set the display format

        Choose from:

        * MLINear
        * MLOGarithmic
        * PHASe
        * UPHase 'Unwrapped phase
        * IMAGinary
        * REAL
        * POLar
        * SMITh
        * SADMittance 'Smith Admittance
        * SWR
        * GDELay 'Group Delay
        * KELVin
        * FAHRenheit
        * CELSius
        '''
        self.write('calc%i:form %s'%(self.channel,form))

    def set_display_format_all(self, form):
        '''
        Set the display format for all measurements

        Choose from:

        * MLINear
        * MLOGarithmic
        * PHASe
        * UPHase 'Unwrapped phase
        * IMAGinary
        * REAL
        * POLar
        * SMITh
        * SADMittance 'Smith Admittance
        * SWR
        * GDELay 'Group Delay
        * KELVin
        * FAHRenheit
        * CELSius

        '''
        self.func_on_all_traces(self.set_display_format, form)

    def func_on_all_traces(self,func, *args, **kwargs):
        '''
        Run a function on all traces are active

        Loop through all measurements, and making each active, then
        subsequently run a command.

        Parameters
        ------------
        func : func
            The function to run while each trace is active

        Examples
        ---------
        >>> p = PNA()
        >>> p.func_on_all_traces(p.set_display_format, 'smith')


        '''

        for name,parm in self.get_meas_list():
            self.select_meas(name)
            func(*args,**kwargs)

    def normalize(self, meas=[], op='DIV', write_memory=True):
        '''
        normalizes current measurement or specific measurements

        Parameters
        ------------
        meas : list of strings or single string
            measurements to normalize
        op : string
            math operation to be applied.
            Choose from:

            * normal (no normalization active)
            * add
            * subtract
            * multiply
            * DIVide
        write_memory: bool
            write current trace to memory (True) or just activate normalisation (False)
        '''
        if not meas:
            meas = self.get_active_meas()
        else: self.select_meas(meas)
        if write_memory:
            self.write('CALC:MATH:MEM')
        self.write('CALC:MATH:FUNC %s'%str(op))

    def enable_transform(self, meas=[], center=0, span=2e-9):
        '''
        Enables time domain transform for current trace

        Parameters
        ------------
        meas : list of strings or single string
            measurements to activate transform for
        center : float
            center time in [s]
        span : float
            span time in [s]
        '''
        if not meas:
            meas = self.get_active_meas()
        else: self.select_meas(meas)
        self.write('CALC:TRAN:TIME:CENT %e'%center)
        self.write('CALC:TRAN:TIME:SPAN %e'%span)
        self.write('CALC:TRAN:TIME:STAT ON')

    def enable_gating(self, meas=[], center=0, span=0.5e-9):
        '''
        Enables time domain gating for current measurement or specified measurements

        Parameters
        ------------
        meas : list of strings or single string
            measurements to enable gating for
        center : float
            center time in [s]
        span : float
            span time in [s]
        '''
        if not meas:
            meas = self.get_active_meas()
        else: self.select_meas(meas)
        self.write('CALC:FILT:TIME:CENT %e'%center)
        self.write('CALC:FILT:TIME:SPAN %e'%span)
        self.write('CALC:FILT:TIME:STAT ON')

    def disable_gating(self, meas=[]):
        '''
        Disables time domain gating for current measurement or specified measurements

        Parameters
        ------------
        meas : list of strings or single string
            measurements to disable gating for

        '''
        if not meas:
            meas = self.get_active_meas()
        else: self.select_meas(meas)
        self.write('CALC:FILT:TIME:STAT OFF')

    def set_yscale(self, meas='', pdiv=5, rlev=0, rpos=8):
        '''
        set y-scale 'per division' value for current measurement or the one specified if
        no measurement identifier is given.

        Parameters
        ------------
        pdiv : float
            per division
        rlev : float
            reference level
        rpos : int
            reference position on y-scale
        '''
        if not meas:
            meas = self.get_active_meas()
        (w, t) = self.get_win_trace(meas=meas)
        self.write('DISP:WIND%i:TRAC%i:Y:PDIV %f'%(w, t, pdiv))
        self.write('DISP:WIND%i:TRAC%i:Y:RLEV %f'%(w, t, rlev))
        self.write('DISP:WIND%i:TRAC%i:Y:RPOS %f'%(w, t, rpos))

    def set_yscale_couple(self, method= 'all' ,window_n = None, trace_n=None):
        '''
        set y-scale coupling

        Parameters
        ------------
        method : ['off','all','window']
            controls the coupling method
        '''
        if window_n is None:
            window_n =''
        if trace_n is None:
            trace_n =self.ntraces+1
        self.write('disp:wind%s:trac%s:y:coup:meth %s'%(str(window_n), str(trace_n), method))


    ## Correction related operations
    def get_corr_state_of_channel(self):
        '''
        correction status for give channel
        '''
        return bool(int(self.ask('sense%i:corr:state?'%self.channel)))

    def set_corr_state_of_channel(self, val):
        '''
        toggle correction for give channel
        '''
        val = 'on' if val else 'off'
        self.write('sense%i:corr:state %s'%(self.channel, val))

    corr_state_of_channel = property(get_corr_state_of_channel,
        set_corr_state_of_channel)

    def get_corr_state_of_meas(self):
        '''
        correction status for give channel
        '''
        return bool(int(self.ask('calc%i:corr:state?'%self.channel)))

    def set_corr_state_of_meas(self, val):
        '''
        toggle correction for give channel
        '''
        val = 'on' if val else 'off'
        self.write('calc%i:corr:state %s'%(self.channel, val))

    corr_state_of_meas = property(get_corr_state_of_meas,
        set_corr_state_of_meas)


    def get_cset_list(self, form = 'name'):
        '''
        Get list of calsets on current channel

        Parameters
        ------------
        form : 'name','guid'
            format of returned values

        Returns
        ---------
        cset : list
            list of csets by guid or name
        '''
        a =  self.ask(\
            'SENSE%i:CORRection:CSET:CATalog? %s'%(self.channel,form))
        return a[1:-1].split(',')

    def create_cset(self, name='skrfCal'):
        '''
        Creates an empty calset
        '''
        self.write('SENS%i:CORR:CSET:CRE \'%s\''%(self.channel,name))

    def delete_cset(self, name):
        '''
        Deletes a given calset
        '''
        self.write('SENS%i:CORR:CSET:DEL \'%s\''%(self.channel,name))

    def get_active_cset(self,channel =None, form = 'name'):
        '''
        Get the active calset name for a give channel.

        Parameters
        -------------
        chawornnel : int
            channel to apply cset to
        form : 'name' or 'guid'
            form of `cset` argument
        '''
        if channel is None:
            channel  = self.channel

        form = form.lower()
        if form not in ['name','guid']:
            raise ValueError ('bad value for `form`')

        out = self.ask('SENS%i:CORR:CSET:ACT? %s'\
                    %(channel,form) )[1:-1]

        if out =='No Cal Set selected':
            return None
        else:
            return out

    def set_active_cset(self, cset, channel=None, apply_stim_values=True ):
        '''
        Set the current  active calset.

        Parameters
        -------------
        cset: str
            name of calset
        channel : int
            channel to apply cset to
        apply_stim_values : bool
            should the cset stimulus values be applied to the channel.
        '''
        if channel is None:
            channel  = self.channel

        available_csets = self.get_cset_list()
        if cset not in available_csets:
            raise ValueError('%s not in list of available csets'%cset)

        self.write('SENS%i:CORR:CSET:ACT \'%s\',%i'\
                    %(channel,cset,int(apply_stim_values) ))
    def save_active_cset(self,channel=None):
        '''
        Save the activer calset
        '''
        if channel is None:
            channel  = self.channel
        self.write('sense%i:correction:cset:save'%channel)

    def get_cal_coefs(self):
        '''
        Get calibration coefficients for current calset


        Returns
        ----------
        coefs : dict
            keys are cal terms and values are complex numpy.arrays

        See Also
        -----------
        get_calibration
        get_cset
        create_cset
        '''
        coefs = {}
        coefs_list = self.cal_coefs_list
        if len(coefs_list) ==1:
            return None

        for k in coefs_list:
            s_ask = 'sense%i:corr:cset:eterm? \"%s\"'%(self.channel,k)
            data = self.ask_for_values(s_ask)
            coefs[k] = mf.scalar2Complex(data)

        return coefs


    def set_cal_coefs(self, coefs, channel=None):
        '''
        Set calibration coefficients for current calset

        See Also
        -----------
        get_cal_coefs
        '''
        if channel is None:
            channel  = self.channel

        for k in coefs:
            try:
                data = coefs[k].s
            except (AttributeError):
                data = coefs[k]
            data_flat = mf.complex2Scalar(data)
            data_str = ''.join([str(l)+',' for l in data_flat])[:-1]
            self.write('sense%i:corr:cset:eterm \"%s\",%s'\
                        %(channel,k, data_str))
            self.write('sense%i:corr:cset:save'%(channel))


    def get_cal_coefs_list(self):
        '''
        Get list of calibration coefficients for current calset

        '''
        out = self.ask('SENS%i:CORR:CSET:ETERM:cat?'%self.channel)[1:-1]
        # agilent mixes the delimiter with values! this requires that
        # we use regex to only split on comma's that follow parenthesis
        return re.split('(?<=\)),',out)

    cal_coefs_list = property(get_cal_coefs_list)

    def get_calibration(self,channel = None, **kwargs):
        '''
        Get :class:`~skrf.calibration.calibration.Calibration` object
        for the active cal set on a given channel.
        '''
        if channel is None:
            channel  = self.channel

        name = self.get_active_cset()
        if name is None:
            # there is no active calibration
            return None
        #if name not in **kwargs

        freq = self.frequency
        coefs = self.get_cal_coefs()

        skrf_coefs = convert_pnacoefs_2_skrf(coefs)

        if len(skrf_coefs) == 12:
            return SOLT.from_coefs(frequency = freq,
                               coefs = skrf_coefs,
                               name = name,
                               **kwargs)
        if len(skrf_coefs) == 3:
            return OnePort.from_coefs(frequency = freq,
                               coefs = skrf_coefs,
                               name = name,
                               **kwargs)
        else:
            raise NotImplementedError

    def set_calibration(self,cal, ports = (1,2), channel = None,
                        name =None, create_new=True):
        '''
        Upload class:`~skrf.calibration.calibration.Calibration` object
        to the active cal set on a given channel.

        Parameters
        -----------
        cal :  :class:`~skrf.calibration.calibration.Calibration` object
            the calibration to upload to the VNA
        ports : tuple
            ports which to apply calibration to. with respect to the
            skrf calibration the ports are in order (forward,reverse)
        channel : int
            channel on VNA to assign calibration
        name : str
            name of the cset on the VNA
        create_new : bool
            create a new cset called `name`, even if one exists.

        Examples
        ---------
        >>> p = PNA()
        >>> my_cal =  p.get_calbration()
        >>> p.set_calibration(my_cal, ports = (1,2))

        See Also
        --------
        get_calibration
        '''
        if channel is None:
            channel  = self.channel

        # figure out  a name for this calibration
        if name is None:
            if cal.name is None:
                name = 'skrfCal'
            else:
                name = cal.name

        if create_new:
            if name in self.get_cset_list():
                self.delete_cset(name)

            self.create_cset(name = name)

        if len(cal.coefs)==3:
            pna_coefs = convert_skrfcoefs_2_pna(cal.coefs_3term,
                                                ports = ports)
        else:
            pna_coefs = convert_skrfcoefs_2_pna(cal.coefs_12term,
                                                ports = ports)

        self.set_cal_coefs(pna_coefs, channel = channel)
        self.frequency = cal.frequency
        self.save_active_cset()

PNAX = PNA

class ZVA40(PNA):
    '''
    Rohde&Scharz ZVA40

    Examples
    -----------

    >>> from skrf.vi.vna import ZVA40
    >>> v = ZVA40()
    >>> dut = v.network
    '''
    def sweep(self):
        '''
        Initiates a sweep and waits for it to complete before returning

        If vna is in continuous sweep mode then this puts it back
        '''
        was_cont = self.continuous
        self.continuous = False
        self.write("INITiate%i:IMMediate;*WAI"%self.channel)
        
        self.continuous = was_cont

    def get_meas_list(self):
        '''
        Get a list of existent measurements

        Returns
        ----------
        out :  list
            list of tuples of the form, (name, measurement)
        '''
        meas_list = self.ask("CALC%i:PAR:CAT?"%self.channel)

        meas = meas_list[1:-1].split(',')
        if len(meas)==1:
            # if there isnt a single comma, then there arent any measurments
            return None


        return [(meas[k],meas[k+1]) for k in range(0,len(meas)-1,2)]

    def get_data(self, char='SDATA', cnum = None):
        '''
        Get data for current active measuremnent

        Note that this doesnt do any sweep timing. It just gets whatever
        data is in the registers according to char.  If you want the
        data to be returned after a sweep has completed

        Parameters
        ------------
        char : [SDATA, FDATA, RDATA]
            type of data to return

        '''
        if cnum is None:
            cnum = self.channel

        data = npy.array(self.ask_for_values('CALC%i:Data? %s'%(cnum, char)))

        if char.lower() == 'sdata':
            data = mf.scalar2Complex(data)

        return data

    def get_active_meas(self):
        '''
        Get the name of the active measurement
        '''
        warn('Retriving active trace is not functional. This is a stub.')

        return ''

    def create_meas(self,name, meas):
        '''
        Create a new measurement.

        Parameters
        ------------
        name : str
            name given to measurment
        meas : str
           measurement string

        '''
        self.write('calc%i:par:sdef \"%s\", \"%s\"'%(self.channel, name, meas))
        self.display_trace(name)

    def setup_twoport(self, ports=[1,2]):
        '''
        Sets up traces appropriate for 2-port s-parameter measurment

        Parameters
        -----------
        ports : tuple of ints
            the pair of ports on the VNA used in the measurement


        '''
        self.delete_all_meas()

        port_list = [(y,x) for x in ports for y in ports]
        #create traces
        for k in port_list:
            self.create_meas('s%i%i'%k,'s%i%i'%k)

    def get_twoport(self, *args, **kwargs):
        '''
        Retrieves a two-port  Network.

        This requires that all 4 s-parameters are already defined, and
        in correct order. see below for setup.

        See Also
        ----------
        setup_twoport
        '''
        n = self.get_network_all_meas()
        twoport = n_oneports_2_nport([n[0],n[2],n[1],n[3]], *args, **kwargs)
        return twoport

    def setup_oneport(self, port=1):
        '''
        sets up traces appropriate for 1-port s-parameter measurement

        Parameters
        -----------
        port : int
            the  port on the VNA used in the measurement

        '''
        self.setup_twoport(ports = [port])




    def set_source_freq_conversion(self, port, numer, denom, offset, mode='swe'):
        '''
        set source frequency for frequency converted measurements

        fs = `numer`/`denom`*fb. + `offset`

        Parameters
        --------------
        numer : int
            numerator
        denom : int
            denominator
        offset : float
            offset frequency in hz
        mode : ['swe','cw','fixed']
            sweep type
        '''
        self.write('SOUR%i:FREQ%i:CONV:ARB:IFR %i,%i,%f,%s'%\
            (self.channel, port, numer, denom, offset, mode))

    def set_source_power_permanent(self, port,val=True):
        '''
        set a given port to have its power permanently on

        the same as having the 'gen' box checked in the `port config`
        dialog

        Examples
        -----------
        >>>zva.set_source_power_permant(port =1, val= True)
        '''
        if val:
            val='on'
        else:
            val = 'off'
        self.write('source%i:power%i:perm %s'%(self.channel,port, val))

    def set_port_power_level(self, port, offset, only =True):
        '''
        Parameters
        -----------
        port : int
            port number
        offset : number
            power offset (dB)
        only : bool
            if true: only set port power. ignore channel power.
            if false: the port power is added to channel ower.
        '''

        if only:
            mode  = 'only'
        else:
            mode= 'cpad'
        self.write('source%i:power%i:offset %f, %s'\
            %(self.channel,port, offset, mode))



    def toggle_port_power(self,mode=True, port =1):
        '''
        Turn a given port's power on or off

        This  overides the PNA's implementation with because
        I dont think RS supports it. This uses the permanent power on
        option which is equivalent to clicking 'gen' in teh port config
        window

        Parameters
        ----------
        mode : bool
            turn power on or not
        port : int
            the port (duh)
        '''
        if mode == 'auto':
            warn('ZVA dont support \'auto\' power mode. using \'on\'')
            mode = True
        self.set_source_power_permanent(port=port,val=mode)

    def get_switch_terms(self, ports = [1,2]):
        '''
        Get switch terms and return them as a tuple of Network objects.

        Returns
        --------
        forward, reverse : oneport switch term Networks
        '''

        p1,p2 = ports
        self.delete_all_meas()
        self.create_meas('forward_switch_term', 'A%iD%i/B%iD%i'%(p2,p1,p2,p1))
        forward = self.get_network()
        forward.name = 'forward switch term'

        self.delete_all_meas()
        self.create_meas('reverse_switch_term', 'A%iD%i/B%iD%i'%(p1,p2,p1,p2))
        reverse = self.get_network()
        reverse.name = 'reverse switch term'
        self.delete_all_meas()
        return forward, reverse

    get_oneport = PNA.get_network

class VectorStar(PNA):
    '''
    '''


    def rtl(self):
        '''
        Return to local
        '''
        self.write('rtl')

    @property
    def continuous(self):
        out =  self.ask(':sense%i:hold:func?'%self.channel)
        if (out.lower() == 'hold' or out.lower() == 'sing'):
            return False
        else:
            return True


    @continuous.setter
    def continuous(self, mode):
        '''
        '''
        if mode:
            self.write(':sense%i:hold:func cont'%self.channel)
        else:
            self.write(':sense%i:hold:func hold'%self.channel)

    def sweep(self):
        '''
        Initiates a sweep and waits for it to complete before returning

        If vna is in continuous sweep mode then this puts it back
        '''
        was_cont = self.continuous
        self.continuous = False
        out = bool(self.ask("TRS;WFS;*IDN?"))
        self.continuous = was_cont
        return out

    def get_twoport(self, *args, **kwargs):
        '''
        Get a two-port Network using alternative command

        This method uses the `OS2P` command, which isn't documented,
        except for the examples, but its dang fast.



        '''
        self.write("LANG NATIVE")
        self.write(":FORM:SNP:FREQ HZ")
        self.write(":FORM:SNP:PAR REIM")
        d = self.ask_for_values("TRS;WFS;OS2P")[19:] # i dont know what the first 19 values are
        d = npy.array(d)
        d = d.reshape(-1,9)
        s11 = d[:,1] +1j*d[:,2]
        s21 = d[:,3] +1j*d[:,4]
        s12 = d[:,5] +1j*d[:,6]
        s22 = d[:,7] +1j*d[:,8]
        s = npy.c_[s11,s12,s21,s22].reshape(-1,2,2)
        freq = self.frequency
        return Network(s = s, frequency = freq,*args, **kwargs)

    def get_oneport(self, n=None, *args, **kwargs):
        was_cont = self.continuous
        self.continuous = False

        if n is not None:
            self.active_trace = n
        freq = self.frequency
        s = npy.array(self.get_sdata())
        s = mf.scalar2Complex(s)

        self.continuous = was_cont
        return Network(
            frequency = freq,
            s=s,
            *args,
            **kwargs)

    def get_twoport_slow(self, *args, **kwargs):
        was_cont = self.continuous
        self.continuous = False
        self.setup_s_parameters()
        self.active_trace = 1
        s11 = mf.scalar2Complex(self.get_sdata())
        self.active_trace = 3
        s21 = mf.scalar2Complex(self.get_sdata())
        self.active_trace = 2
        s12 = mf.scalar2Complex(self.get_sdata())
        self.active_trace = 4
        s22 = mf.scalar2Complex(self.get_sdata())

        s = npy.c_[s11,s12,s21,s22].reshape(-1,2,2)
        freq = self.frequency
        self.continuous=was_cont
        return Network(s = s, frequency = freq,*args, **kwargs)

    def get_network_all_meas(self):
        '''
        Return list of Network Objects for all measurements.


        See Also
        -----------
        get_meas_list
        get_network
        '''

        out = []
        self.sweep()
        for name,parm in self.get_meas_list():
            self.select_meas(name)
            out.append(self.get_network(sweep=False, name= parm))

        return out

    def get_ntraces(self):
        return int(self.ask(':calc%i:par:count?'%self.channel))

    def set_ntraces(self,val):
        self.write((':calc%i:par:count %i'%(self.channel,int(val))))

    ntraces = property(get_ntraces, set_ntraces)

    def get_active_meas(self):
        '''
        Get the name of the active measurement
        '''
        return self.ask(':calc%i:par%i:def?'%(self.channel, self.active_trace_num))

    def create_meas(self,name, meas):
        '''
        Create a new measurement.

        Parameters
        ------------
        name : str
            name given to measurement
        meas : str
            something like
            * S11
            * a1/b1,1
            * A/R1,1
            * ...

        Examples
        ----------
        >>> p = PNA()
        >>> p.create_meas('my_meas', 'A/R1,1')
        '''
        # translate agilent's semantics into Anritsu's
        # TODO: use regex to fully translate all combos
        translation_dict  = {
                             '/':',',
                             'R1':'A1',
                             'R2':'A2',
                             'R3':'A3',
                             'R4':'A4',
                             }

        if meas.lower().startswith('s'):
            # measuring an s-parameter, es simple.
            self.write('calc%i:par%i:def %s'%(self.channel,(self.ntraces+1), meas))
        else:
            # measuring something other than s-parameters, need to do
            # some translation .. .
            meas, port = meas.split(',')
            for k in translation_dict:
                meas = meas.replace(k,translation_dict[k])

            self.write('calc%i:par%i:def:usr %s,%s,port%s'%(self.channel,
                                                            self.ntraces+1,
                                                            meas, port))

    def get_active_trace_num(self):
        return int(self.ask(':calc%i:par:sel?'%self.channel))


    def set_active_trace_num(self,n):
        n = int(n)
        self.write(':calc%i:par%i:sel'%(self.channel, n))

    active_trace_num = property(get_active_trace_num,set_active_trace_num)

    select_meas = set_active_trace_num

    @property
    def trace_format(self):
        return self.ask('calc%i:form?'%self.channel)

    @trace_format.setter
    def trace_format(self, form):
        self.write(':calc%i:form %s'%(self.channel, form))



    def get_fdata(self):
        return npy.array(self.ask_for_values('trs;wfs;:calc%i:data:fdat?'%self.channel))[1:]

    def get_sdata(self):
        return npy.array(self.ask_for_values('trs;wfs;:calc%i:data:sdat?'%self.channel))[1:]

    def get_smem(self):
        return npy.array(self.ask_for_values('trs;wfs;:calc%i:data:smem?'%self.channel))[1:]

    def delete_all_meas(self):
        self.ntraces = 0



    def get_all_traces(self):
        ntwks = []
        for k in  range(1, self.ntraces+1):
            self.active_trace_num =k
            ntwks.append(self.get_oneport(name = self.get_active_meas()))

        return ntwks

    def get_meas_list(self):
        '''
        Get a list of existent measurements

        Returns
        ----------
        out :  list
            list of tuples of the form, (name, measurement)
        '''
        meas_list = []
        for k in  range(1, self.ntraces+1):
            self.active_trace_num =k
            meas_list.append(self.get_active_meas())



        return [(k+1,meas_list[k]) for k in range(self.ntraces)]

class HP8510C(Driver):
    '''
    good ole 8510
    '''
    def __init__(self, address=16,**kwargs):
        Driver.__init__(self,'GPIB::'+str(address),**kwargs)
        self.write('FORM4;')



    @property
    def error(self):
        return self.ask('OUTPERRO')
    @property
    def continuous(self):
        answer_dict={'\"HOLD\"':False,'\"CONTINUAL\"':True}
        return answer_dict[self.ask('GROU?')]

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

    @property
    def frequency(self, unit='ghz'):
        freq=Frequency( float(self.ask('star;outpacti;')),
                float(self.ask('stop;outpacti;')),\
                int(float(self.ask('poin;outpacti;'))),'hz')
        freq.unit = unit
        return freq


    @property
    def one_port(self):
        '''
        Initiates a sweep and returns a  Network type representing the
        data.

        If you are taking multiple sweeps, and want the sweep timing to
        work, put the turn continuous mode off. like pnax.continuous='off'
        '''
        #tmp_continuous = self.continuous
        #if self.continuous:
        #       tmp_continuous =True
        self.continuous = False
        s = npy.array(self.ask_for_values('OUTPDATA'))
        s.shape=(-1,2)
        s =  s[:,0]+1j*s[:,1]
        ntwk = Network()
        ntwk.s = s
        ntwk.frequency= self.frequency
        #self.continuous  = tmp_continuous
        return ntwk

    @property
    def two_port(self):
        '''
        Initiates a sweep and returns a  Network type representing the
        data.

        If you are taking multiple sweeps, and want the sweep timing to
        work, put the turn continuous mode off. like pnax.continuous='off'
        '''
        print ('s11')
        s11 = self.s11.s[:,0,0]
        print ('s12')
        s12 = self.s12.s[:,0,0]
        print ('s22')
        s22 = self.s22.s[:,0,0]
        print ('s21')
        s21 = self.s21.s[:,0,0]

        ntwk = Network()
        ntwk.s = npy.array(\
                [[s11,s21],\
                [ s12, s22]]\
                ).transpose().reshape(-1,2,2)
        ntwk.frequency= self.frequency

        return ntwk
    ##properties for the super lazy
    @property
    def s11(self):
        self.write('s11;')
        ntwk =  self.one_port
        ntwk.name = 'S11'
        return ntwk
    @property
    def s22(self):
        self.write('s22;')
        ntwk =  self.one_port
        ntwk.name = 'S22'
        return ntwk
    @property
    def s12(self):
        self.write('s12;')
        ntwk =  self.one_port
        ntwk.name = 'S12'
        return ntwk
    @property
    def s21(self):
        self.write('s21;')
        ntwk =  self.one_port
        ntwk.name = 'S21'
        return ntwk

    @property
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
        forward = self.one_port
        forward.name = 'forward switch term'

        print ('reverse')
        self.write('USER1;DRIVPORT2;LOCKA2;NUMEB1;DENOA1;CONV1S;')
        reverse = self.one_port
        reverse.name = 'reverse switch term'

        return (forward,reverse)

class HP8720(HP8510C):
    def __init__(self, address=16,**kwargs):
        HP8510C.__init__(self,address,**kwargs)
    @property
    def averaging(self):
        raise ( NotImplementedError)
    @averaging.setter
    def averaging(self,value):
        if value:
            self.write('AVEROON')
        else:
            self.write('AVEROFF')
    @property
    def ifbw(self):
        raise ( NotImplementedError)

    @ifbw.setter
    def ifbw(self,value):
        self.write('IFBW %i'%int(value))

    @property
    def frequency(self, unit='ghz'):
        f = npy.array(self.ask_for_values('OUTPLIML'))
        f.shape=(-1,4)
        freq=Frequency.from_f(f[:,0], unit='hz')
        freq.unit = unit
        return freq
