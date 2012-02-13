#          vna.py
#
#       This file holds all VNA models
#
#          Copyright 2010  lihan chen, alex arsenovic <arsenovic@virginia.edu>
#
#          This program is free software; you can redistribute it and/or modify
#          it under the terms of the GNU General Public License as published by
#          the Free Software Foundation; either version 2 of the License, or
#          (at your option) any later version.
#
#          This program is distributed in the hope that it will be useful,
#          but WITHOUT ANY WARRANTY; without even the implied warranty of
#          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#          GNU General Public License for more details.
#
#          You should have received a copy of the GNU General Public License
#          along with this program; if not, write to the Free Software
#          Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#          MA 02110-1301, USA.

'''
holds class's for VNA virtual instruments
'''
import numpy as npy
import visa
from visa import GpibInstrument

from ..frequency import *
from ..network import *

class PNAX(GpibInstrument):
    '''
    Agilent PNAX
    '''
    def __init__(self, address=16, channel=1,**kwargs):
        GpibInstrument.__init__(self,'GPIB::'+str(address),**kwargs)
        self.channel=channel
        self.write('calc:par:sel CH1_S11_1')

    @property
    def continuous(self):
        raise NotImplementedError

    @continuous.setter
    def continuous(self, mode):
        self.write('initiate:continuous '+ mode)

    @property
    def frequency(self, unit='ghz'):
        freq=Frequency( float(self.ask('sens:FREQ:STAR?')),
                float(self.ask('sens:FREQ:STOP?')),\
                int(self.ask('sens:sweep:POIN?')),'hz')
        freq.unit = unit
        return freq


    @property
    def network(self):
        '''
        Initiates a sweep and returns a  Network type represting the
        data.

        if you are taking multiple sweeps, and want the sweep timing to
        work, put the turn continuous mode off. like pnax.continuous='off'
        '''
        self.write('init:imm')
        self.write('*wai')
        s = npy.array(self.ask_for_values('CALCulate1:DATA? SDATa'))
        s.shape=(-1,2)
        s =  s[:,0]+1j*s[:,1]
        ntwk = Network()
        ntwk.s = s
        ntwk.frequency= self.frequency
        return ntwk


class ZVA40_lihan(object):
    '''
    Created on Aug 3, 2010
    @author: Lihan

    This class is create to remote control Rohde & Schwarz by using pyvisa.
    For detail about visa please refer to:
            http://pyvisa.sourceforge.net/

    After installing the pyvisa and necessary driver (GPIB to USB driver,
     for instance), please follow the pyvisa manual to set up the module

    This class only has several methods. You can add as many methods
    as possible by reading the Network Analyzer manual
    Here is an example

    In the manual,

            "CALC:DATA? FDAT"

    This is the SCPI command

            "Query the response values of the created trace. In the FDATa setting, N
            comma-separated ASCII values are returned."

    This descripes the function of the SCPI command above

    Since this command returns ASCII values, so we can use ask_for_values method in pyvisa

    temp=vna.ask_for_values('CALCulate1:DATA? SDATa')

    vna is a pyvisa.instrument instance


    '''
    def __init__(self,address=20, **kwargs):
        self.vna=visa.instrument('GPIB::'+str(address),**kwargs)
        self.spara=npy.array([],dtype=complex)

    def continuousOFF(self):
        self.vna.write('initiate:continuous OFF')


    def continuousON(self):
        self.vna.write('initiate:continuous ON')

    def displayON(self):
        self.vna.write('system:display:update ON')

    def setFreqBand(self,StartFreq,StopFreq):
        '''
        Set the frequency band in GHz
        setFreqBand(500,750)
        Start frequency 500GHz, Stop frequency 750GHz
        '''
        self.freqGHz=npy.linspace(StartFreq, StopFreq, 401)
        self.vna.write('FREQ:STAR '+'StartFreq'+'GHz')
        self.vna.write('FREQ:STOP '+'StopFreq'+'GHz')

    def sweep(self):
        '''
        Initiate a sweep under continuous OFF mode
        '''
        self.vna.write('initiate;*WAI')

    def getData(self):
        '''
        Get data from current trace
        '''
        temp=self.vna.ask_for_values('CALCulate1:DATA? SDATa')
        temp=npy.array(temp)
        temp.shape=(-1,2)
        self.spara=temp[:,0]+1j*temp[:,1]
        self.spara.shape=(-1,1,1)                       #this array shape is compatible to Network Class
        return self.spara

    def measure(self):
        '''
        Take one-port measurement
        1.turn continuous mode off
        2.initiate a single sweep
        3.get the measurement data
        4.turn continuous mode on
        '''
        self.continuousOFF()
        self.sweep()
        temp=self.getData()
        self.continuousON()
        return temp

    def saveSpara(self,fileName):
        '''
        Take one-port measurement and save the data as touchstone file, .s1p
        '''
        temp=self.spara
        formatedData=npy.array([self.freqGHz[:],temp[:,0,0].real,temp[:,0,0].imag],dtype=float)
        fid = open(fileName+'.s1p', 'w')
        fid.write("# GHz S RI R 50\n")
        npy.savetxt(fid,formatedData,fmt='%10.5f')
        fid.close()

class ZVA40(GpibInstrument):
    '''
    the rohde Swarz ZVA40
    '''
    def __init__(self, address=20, active_channel = 1, continuous=True,\
            **kwargs):
        GpibInstrument.__init__(self,address, **kwargs)
        self.active_channel = active_channel
        self.continuous = continuous
        self.traces = []
        #self.update_trace_list()

    @property
    def sdata(self):
        '''
        unformated s-parameter data [a numpy array]
        '''
        return npy.array(self.ask_for_values(\
                'CALCulate%i:DATA? SDATa'%(self.active_channel)))

    @property
    def fdata(self):
        '''
        formated s-parameter data [a numpy array]
        '''
        return npy.array(self.ask_for_values(\
                'CALCulate%i:DATA? FDATa'%(self.active_channel)))

    @property
    def continuous(self):
        '''
        set/get continuous sweep mode on/off [boolean]
        '''
        return self._continuous

    @continuous.setter
    def continuous(self, value):
        if value:
            self.write('INIT%i:CONT ON;'%(self.active_channel))
            self._continuous = True
        elif not value:
            self.write('INIT%i:CONT OFF;'%(self.active_channel))
            self._continuous = False
        else:
            raise ValueError('takes boolean')


    @property
    def frequency(self, unit='ghz'):
        '''
        a frequency object, representing the current frequency axis
        [skrf Frequency object]
        '''
        freq=Frequency(0,0,0)
        freq.f = self.ask_for_values(\
                'CALC%i:DATA:STIMulus?'%self.active_channel)
        freq.unit = unit
        return freq

    @property
    def one_port(self):
        '''
        a network representing the current active trace
        [skrf Network object]
        '''
        self.sweep()
        s = self.sdata
        s.shape=(-1,2)
        s =  s[:,0]+1j*s[:,1]
        ntwk = Network()
        ntwk.s = s
        ntwk.frequency= self.frequency
        return ntwk

    @property
    def s11(self):
        '''
        this is just for legacy support, there is no gurantee this
        will return s11. it just returns active trace
        '''
        return self.one_port


    @property
    def error(self):
        '''
        returns list errors stored on vna
        '''
        return self.ask('OUTPERROR?')

    def initiate(self):
        '''
        initiate a sweep on current channel (low level timing)
        '''
        self.write('INITiate%i'%self.active_channel)

    def sweep(self):
        '''
        initiate a sweep on current channel. if vna is in continous
        mode it will put in single sweep mode, then request a sweep,
        and then return sweep mode to continous.
        '''
        if self.continuous:
            self.continuous = False
            self.write(\
                    'INITiate%i:IMMediate;*WAI'%self.active_channel)
            self.continuous = True
        else:
            self.write(\
                    'INITiate%i:IMMediate;*WAI'%self.active_channel)

    def wait(self):
        '''
        wait for preceding command to finish before executing subsequent
        commands
        '''
        self.write('*WAIt')

    def add_trace(self, parameter, name):
        print ('CALC%i:PARA:SDEF \"%s\",\"%s\"'\
                %(self.active_channel, name, parameter))
        self.write('CALC%i:PARA:SDEF \"%s\",\"%s\"'\
                %(self.active_channel, name, parameter))
        self.traces[name] = parameter

    def set_active_trace(self, name):
        if name in self.traces:
            self.write('CALC%i:PAR:SEL %s'%(self.active_channel,name))
        else:
            raise ValueError('trace name does exist')
    def update_trace_list(self):
        raise(NotImplementedError)


class ZVA40_alex(GpibInstrument):
    '''
    the rohde Swarz zva40
    '''
    class Channel(object):
        def __init__(self, vna, channel_number):
            self.number = channel_number
            self.vna = vna
            self.traces = {}
            self.continuous = True

        @property
        def sdata(self):
            return npy.array(self.vna.ask_for_values('CALCulate%i:DATA? SDATa'%(self.number)))
        @property
        def fdata(self):
            return npy.array(self.vna.ask_for_values('CALCulate%i:DATA? FDATa'%(self.number)))
        @property
        def continuous(self):
            return self._continuous
        @continuous.setter
        def continuous(self, value):
            if value:
                self.vna.write('INIT%i:CONT ON;'%(self.number))
                self._continuous = True
            elif not value:
                self.vna.write('INIT%i:CONT OFF;'%(self.number))
                self._continuous = False
            else:
                raise ValueError('takes boolean')

        def initiate(self):
            self.vna.write('INITiate%i'%self.number)

        def sweep(self):
            if self.continuous:
                self.continuous = False
                self.vna.write('INITiate%i:IMMediate;*WAI'%self.number)
                self.continuous = True
            else:
                self.vna.write('INITiate%i:IMMediate;*WAI'%self.number)

        def add_trace(self, parameter, name):
            print ('CALC%i:PARA:SDEF \"%s\",\"%s\"'\
                    %(self.number, name, parameter))
            self.vna.write('CALC%i:PARA:SDEF \"%s\",\"%s\"'\
                    %(self.number, name, parameter))
            self.traces[name] = parameter

        def select_trace(self, name):
            if name in self.traces.keys():
                self.vna.write('CALC%i:PAR:SEL %s'%(self.number,name))
            else:
                raise ValueError('trace name does exist')

        @property
        def frequency(self, unit='ghz'):
            freq=Frequency(0,0,0)
            freq.f = self.vna.ask_for_values('CALC%i:DATA:STIMulus?'%self.number)
            freq.unit = unit
            return freq
        @property
        def one_port(self):
            self.sweep()
            s = self.sdata
            s.shape=(-1,2)
            s =  s[:,0]+1j*s[:,1]
            ntwk = Network()
            ntwk.s = s
            ntwk.frequency= self.frequency
            return ntwk

    def __init__(self, address=20,**kwargs):
        GpibInstrument.__init__(self,address, **kwargs)
        self.add_channel(1)

    def _set_property(self, name, value):
        setattr(self, '_' + name, value)
    def _get_property(self, name):
        return getattr(self, '_' + name)

    @property
    def error(self):
        return self.ask('OUTPERROR?')
    def add_channel(self,channel_number):
        channel = self.Channel(self, channel_number)
        fget = lambda self: self._get_property('ch'+str(channel_number))
        setattr(self.__class__,'ch'+str(channel_number), property(fget))
        setattr(self, '_'+'ch'+str(channel_number), channel)



    def wait(self):
        self.write('*WAIt')




class HP8510C(GpibInstrument):
    '''
    good ole 8510
    '''
    def __init__(self, address=16,**kwargs):
        GpibInstrument.__init__(self,'GPIB::'+str(address),**kwargs)
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
        Initiates a sweep and returns a  Network type represting the
        data.

        if you are taking multiple sweeps, and want the sweep timing to
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
        Initiates a sweep and returns a  Network type represting the
        data.

        if you are taking multiple sweeps, and want the sweep timing to
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
