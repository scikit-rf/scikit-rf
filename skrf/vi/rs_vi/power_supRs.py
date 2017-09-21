# -*- coding: utf-8 -*-
from viRs import viRs


class hmpx(viRs):
    '''
    Class derivate from viRs. Main propose is control Rohde&Schwarz / Hameg
    power supplyers.
    '''
    def set_iv_2channals_on(self, v1, i1, v2, i2):
        '''
        set_iv_2chenals(self, v1, i1, v2, i2)

        1. Sets voltage and current for channels 1 and 2,
        2. Select channals 1 and 2,
        3. Switch on channals 1 and  2.
        '''
        self.write('INST OUT1')
        self.write('VOLT %s' % v1)
        self.write('CURR %s' % i1)
        self.write('OUTP:SEL ON ')

        self.write('INST OUT2')
        self.write('VOLT %s' % v2)
        self.write('CURR %s' % i2)
        self.write('OUTP:SEL ON ')

        #self.write('SYSTem:BEEPe')
        self.write('OUTP:GEN ON')

    def set_iv_2channels(self, v1, i1, v2, i2):
        """
        set_iv_2chenals(self, v1, i1, v2, i2)

        1. Sets voltage and current for channels 1 and 2,
        2. Select channals 1 and 2,

        """
        self.write('INST OUT1')
        self.write('VOLT %s' % v1)
        self.write('CURR %s' % i1)
        self.write('OUTP:SEL ON ')

        self.write('INST OUT2')
        self.write('VOLT %s' % v2)
        self.write('CURR %s' % i2)
        self.write('OUTP:SEL ON ')



    def on_sel(self):
        """
        On selected channals.

        Example:
        xxx.on_sel()
        """
        if self.echo:
            print('OUTP:GEN ON')
        self.write('OUTP:GEN ON')


    def off_sel(self):
        """
        Off selected channals.

        Example:
        xxx.off_sel()
        """
        if self.echo:
            print('OUTP:GEN OFF')
        self.write('OUTP:GEN OFF')


    def set_volt(self, channelNun, volt):
        """"
        Set voltage to specified channel.

        xxx.set_volt(1, 0.1)
        """

        if self.echo:
            print('INST OUT%i' % channelNun)
            print('VOLT %s' % volt)

        self.write('INST OUT%i' % channelNun)
        self.write('VOLT %s' % volt)

    def set_curr(self, channelNun, curr):
        """"
        Set voltage to specified channel.

        xxx.set_volt(1, 0.1)
        """

        if self.echo:
            print('INST OUT%i' % channelNun)
            print('CURR %s' % curr)

        self.write('INST OUT%i' % channelNun)
        self.write('CURR %s' % curr)

    def meas_curr(self, channelNun):
        """
        Measurement current at channel

        i = xxx.meas_curr(1) #Measurement current on 1 channal.
        """
        if self.echo:
            print('INST OUT%i' % channelNun)
            print('MEAS:CURR?')

        self.write('INST OUT%i' % channelNun)
        return self.query('MEAS:CURR?')



class hmcx(hmpx):
    '''
    For R&S HMC series power supplyers.
    '''

    def set_iv_2channals_on(self, v1, i1, v2, i2):
        '''
        set_iv_2chenals(self, v1, i1, v2, i2)

        1. Sets voltage and current for channels 1 and 2,
        2. Select channals 1 and 2,
        3. Switch on channals 1 and  2.
        '''
        self.write('INST OUT1')
        self.write('VOLT %s' % v1)
        self.write('CURR %s' % i1)
        self.write('OUTP:CHAN ON ')

        self.write('INST OUT2')
        self.write('VOLT %s' % v2)
        self.write('CURR %s' % i2)
        self.write('OUTP:CHAN ON ')

        #self.write('SYSTem:BEEPe')
        self.write('OUTP:MAST ON')

    def set_iv_2channels(self, v1, i1, v2, i2):
        """
        set_iv_2chenals(self, v1, i1, v2, i2)

        1. Sets voltage and current for channels 1 and 2,
        2. Select channals 1 and 2,

        """
        self.write('INST OUT1')
        self.write('VOLT %s' % v1)
        self.write('CURR %s' % i1)
        self.write('OUTP:CHAN ON ')

        self.write('INST OUT2')
        self.write('VOLT %s' % v2)
        self.write('CURR %s' % i2)
        self.write('OUTP:CHAN ON ')


    def on_master(self):
        """
        On selected channals.

        Example:
        xxx.on_sel()
        """
        if self.echo:
            print('OUTP:GEN ON')
        self.write('OUTP:MAST ON')

    def off_master(self):
        """
        Off selected channals.

        Example:
        xxx.off_sel()
        """
        if self.echo:
            print('OUTP:CHAN OFF')
        self.write('OUTP:MAST OFF')
















