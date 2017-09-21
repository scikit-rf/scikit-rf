from viRs import viRs
import visa
import numpy as np
from time import sleep
import skrf as rf
from skrf.mathFunctions import scalar2Complex

class vnaRs(viRs):
    """
    This class is derevated from viRs to control Rohde&Schwarz VNA.

    ZVA
    """

    def __init__(self, address, channel=1, timeout=10, echo=False,
             front_panel_lockout=False, **kwargs):
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
            passed to  `visa.Driver.__init__`  - in work

        '''
    #        visa.__init__(self,resource = resource, **kwargs)

        #Initiate for GPIB device
        if isinstance(address,int):
            resource = 'GPIB::%i::INSTR' % address
        else:
            resource = address

        rm = visa.ResourceManager()
        self.resource = rm.open_resource(resource)  # type: pyvisa.resources.messagebased.MessageBasedResource


        if "socket" in resource.lower():
            #Didn't tested, just copied.
            self.resource.read_termination = "\n"
            self.resource.write_termination = "\n"

        if "gpib" in resource.lower():
            #Didn't tested, just copied.
            self.resource.control_ren(2)

        self.channel = channel
        self.port = 1
        self.echo = echo
        self.timeout = timeout
        if not front_panel_lockout:
            self.write("SYST:DISP:UPD ON")

        #Setting uotput data format.
        self.resource.write("format:data ascii,0")
#        self.set_snp_format("RI") maby not needed


    """Basic vnaRs setup."""

#    @property
#    def channel(self):
#        return self.channel
##    @channel.setter
##    def channel(self, val):
##        self.channel = val
#
#    @property
#    def echo(self):
#        return self.echo
#    @echo.setter
#    def echo(self, val):
#        self.echo = val



    """**********************************************************************
       **************************Frequency setup.****************************
       **********************************************************************
    """
    def get_f_start(self):
        """
        Get R&S VNA start frequency.
        """
        if self.echo:
            print(':SENS%i:FREQ%i:STAR?' % (self.channel, self.channel))
        return float(self.query(':SENS%i:FREQ%i:STAR?' %
                                (self.channel, self.channel)))


    def get_f_stop(self):
        """
        Get R&S VNA stop frequency.
        """
        if self.echo:
            print(':SENS%i:FREQ%i:STOP?' % (self.channel, self.channel))
        return float(self.query(':SENS%i:FREQ%i:STOP?' % (self.channel, self.channel)))

    def get_f_array(self):
        """Gets all frequency points in numpy array."""
        return np.linspace(self.get_f_start(), self.get_f_stop(), self.get_num_sweep_points())

    def set_f_start(self, startFrq):
        """
        Set R&S VNA start frequency.
        """
        if self.echo:
            print(':SENS%i:FREQ%i:STAR %s' % (self.channel,
                                              self.channel, startFrq))
        self.write(':SENS%i:FREQ%i:STAR %s' % (self.channel,
                                               self.channel, startFrq))


    def set_f_stop(self, stopFrq):
        """
        Set R&S VNA stop frequency.
        """
        if self.echo:
            print('FREQ%i:STOP %s' % (self.channel, stopFrq))
        self.write('FREQ%i:STOP %s' % (self.channel, stopFrq))

    frequncy = property(get_f_start, get_f_stop)
#    frequncy = property.setter(set_frequency_start, set_frequency_stop)


    """**********************************************************************
       **************************Sweep setup.********************************
       **********************************************************************
   """
    def get_num_sweep_points(self):
        """
        Get number of sweep points from R&S VNA.
        """
        if self.echo:
            print(":SENS%i:SWE:POIN?" % self.channel)
        return int(self.query(":SENS%i:SWE:POIN?" % self.channel))

    def set_num_sweep_points(self, nPoints):
        """
        Set number of sweep points from R&S VNA.
        """
        if self.echo:
            print(":SENS%i:SWE:POIN" % self.channel)

        self.write(":SENS%i:SWE:POIN %i" % (self.channel, int(nPoints)))

    def get_sweep_step(self, stepFrq):
        """
        Get frequncy step size from R&S VNA.
        """
        if self.echo:
            print(":SENS%i:SWE:STEP?" % self.channel)
        return float(self.query(":SENS%i:SWE:STEP?" % self.channel))

    def set_sweep_step(self, stepFrq):
        """
        Get frequncy step size from R&S VNA.
        """
        if self.echo:
            print(":SENS%i:SWE:STEP %s" % (self.channel, stepFrq))
        self.write(":SENS%i:SWE:STEP %s" % (self.channel, stepFrq))


    sweep = property(get_num_sweep_points)
#    sweep = property.setter(set_num_sweep_points)
    sweep = property(get_sweep_step)
#    sweep = property.setter(set_sweep_step)

    """Source power setup"""
    def power_off(self):
        self.write("OUTPut OFF")

    def power_on(self):
        self.write("OUTPut ON")

    def set_power(self, power, source = 1):
        """
        Set source power for R&S VNA.
        """
        if self.echo:
            print()
        self.write("SOURce%i:POWer%i %s" % (self.channel, source, power))

    def set_power_opc(self, power, source = 1):
        """
        Set source power for R&S VNA.
        """
        if self.echo:
            print()
        self.write("SOURce%i:POWer%i %s; *OPC?" % (self.channel, source, power))

    def get_power(self, source = 1):
        """
        Get source power for R&S VNA.
        """
        if self.echo:
            print("SOURce%i:POWer%i ?" % (self.channel, source))
        return float(self.query("SOURce%i:POWer%i ?" % (self.channel, source)))

    power = property(get_power)
#    powe = property.setter(set_source_power)

    def set_mes_bw(self, bw):

        """
        Function sets VNA measurement bandwidth.
        """
        if self.echo:
            print("SENSe%i:BANDwidth %f" % (self.channel, bw))
        self.write("SENSe%i:BANDwidth %f" % (self.channel, bw))

    def get_mes_bw(self):

        """
        Function sets VNA measurement bandwidth.
        """
        if self.echo:
            print("SENSe%i:BANDwidth %f" % self.channel)
        return float(self.query("SENSe%i:BANDwidth?" % self.channel))



    """**********************************************************************
       **************************ALC config.*********************************
       **********************************************************************
    """

    def alc_on(self, source = 1):
        """
        Activate ALC on selected channel.
        """
        if self.echo:
            print()
        self.write("SOURce%i:POWer:ALC:CSTAate ON" % (self.channel))
        self.write("SOURce%i:POWer%i:ALC ON" % (self.channel, source))

    def alc_off(self, source = 1):
        """
        Activate ALC on selected channel.
        """
        if self.echo:
            print("SOURce%i:POWer:ALC:CSTAate OFF" % (self.channel))
            print("SOURce%i:POWer%i:ALC OFF" % (self.channel, source))
        self.write("SOURce%i:POWer:ALC:CSTAate OFF" % (self.channel))
        self.write("SOURce%i:POWer%i:ALC OFF" % (self.channel, source))


    """Correction setup."""
    def load_correction(self, corrName):
        """
        Load correction from stored.
        """
        if self.echo:
            print("MMEMORY:LOAD:CORRection %i, '%s'" % (self.channel, corrName))
        self.write("MMEMORY:LOAD:CORRection %i, '%s'"% (self.channel, corrName))

    """Trigger mode setup."""

    def sweep_cont_off(self):
        """
        Switched continuous sweep off.
        """
        if self.echo:
            print("INITiate%i:CONTinuous OFF" % self.channel)
        self.write("INITiate%i:CONTinuous OFF" % self.channel)

    def swee_cont_on(self):
        """
        Switched continuous sweep on.
        """
        if self.echo:
            print("INITiate%i:CONTinuous ON" % self.channel)
        self.write("INITiate%i:CONTinuous ON" % self.channel)

    def sweep_singl_opc(self):
        """
        Starts single sweep.
        """
        if self.echo:
            print("INIT%i:IMM; *OPC?" % self.channel)
        self.query("INIT%i:IMM; *OPC?" % self.channel)


    """**********************************************************************
       **************************Output data settings************************
       **********************************************************************
    """

    def set_snp_format(self, format = "RI"):
        """
        Sets format of the output data.
        """
        pass



    """**********************************************************************
       **************************Display function****************************
       **********************************************************************
   """

    def disp_on(self):
        """
        Enable to update display
        """
        if self.echo:
            print("SYST:DISP:UPD ON")
        self.write("SYST:DISP:UPD ON")

    """**********************************************************************
       **********************Measurements setup******************************
       **********************************************************************
    """

    def get_1_port_ntwk(self, port = 1, name = 'mw'):
        '''
        This funnction setup one port meas.
        '''
        self.write('CALCulate%i:PARameter:DELete:ALL' % self.channel)
        self.write("CALCulate%i:PARameter:SDEFine 'Trc1', 'S%i%i'"
                   % (self.channel, port, port))
        self.write("DISPlay:WINDow1:TRACe1:FEED 'Trc1'")
        self.write(":DISPLAY:WINDOW1:STATE on")

        self.sweep_cont_off()
        self.sweep_singl_opc()
        mes = scalar2Complex(self.query_ascii_values(":CALCulate%i:DATA:ALL? SDATa"
        % self.channel))
        freq = self.get_f_array()
        self.swee_cont_on()

        ntwk = rf.Network(s= mes, z0=50, name = name)
        ntwk.frequency.unit = 'hz'
        ntwk.f = freq
        return ntwk

    def get_2_port_ntwk(self, port = [1,2], name = 'mw'):
        '''
        This function get data from two port network.
        '''
        self.write('CALCulate%i:PARameter:DELete:ALL' % self.channel)
        trcN = 0
        for i in port:
            for j in port:
                trcN+=1
                self.write("CALCulate%i:PARameter:SDEFine 'Trc%i', 'S%i%i'"
                           % (self.channel, trcN, i, j))
                self.write("DISPlay:WINDow1:TRACe%i:FEED 'Trc%i'" % (trcN, trcN))


        self.write(":DISPLAY:WINDOW1:STATE on")
        freq = self.get_f_array()
        self.sweep_cont_off()
        self.sweep_singl_opc()
        mes = scalar2Complex(self.query_ascii_values(":CALCulate%i:DATA:ALL? SDATa"
                                                     % self.channel))
        self.swee_cont_on()

        s11 = mes[:int(len(mes)/4)]
        s12 = mes[(int(len(mes)/4)):(int(len(mes)/2))]
        s21 = mes[(int(len(mes)/2)):(int(len(mes)*3/4))]
        s22 = mes[-int(len(mes)/4):]

        mes = np.column_stack((s11,s12,s21,s22)).reshape((int(len(mes)/4),2,2))
        ntwk = rf.Network(s= mes, z0=50, name = name)
        ntwk.frequency.unit = 'hz'
        ntwk.f = freq
        return ntwk















































