import re

null_parameter = re.compile(",{2,}")  # detect optional null parameter as two consecutive commas, and remove
converters = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda x: bool(int(x)),
}


def to_string(value):
    tval = type(value)
    if tval is str:
        return value
    elif tval is bool:
        return str(int(value))
    elif tval in (list, tuple):
        return ",".join(map(to_string, value))
    elif value is None:
        return ""
    else:
        return str(value)


def scpi_preprocess(command_string, *args):
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = to_string(arg)
    cmd = command_string.format(*args)
    return null_parameter.sub(",", cmd)


def process_query(query, csv=False, strip_outer_quotes=True, returns="str"):
    if strip_outer_quotes is True:
        if query[0] + query[-1] in ('""', "''"):
            query = query[1:-1]
    if csv is True:
        query = query.split(",")

    converter = None if returns == "str" else converters.get(returns, None)
    if converter:
        if csv is True:
            query = list(map(converter, query))
        else:
            query = converter(query)

    return query


class SCPI(object):
    def __init__(self, resource):
        self.resource = resource
        self.echo = False  # print scpi command string to scpi out

    def write(self, scpi, *args, **kwargs):
        if self.echo:
            print(scpi)
        self.resource.write(scpi, *args, **kwargs)

    def query(self, scpi, *args, **kwargs):
        if self.echo:
            print(scpi)
        return self.resource.query(scpi, *args, **kwargs)

    def write_values(self, scpi, *args, **kwargs):
        if self.echo:
            print(scpi)
        self.resource.write_values(scpi, *args, **kwargs)

    def query_values(self, scpi, *args, **kwargs):
        if self.echo:
            print(scpi)
        return self.resource.query_values(scpi, *args, **kwargs)

    def set_active_channel(self, channel=1):
        """Selects channel as the active channel.
    
         INSTrument:NSELect <Channel>
         """
        scpi_command = scpi_preprocess(":INST:NSEL {:}", channel)
        self.write(scpi_command)

    def set_averaging_clear(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:CLE", cnum)
        self.write(scpi_command)

    def set_averaging_count(self, cnum=1, FACTOR=10):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN {:}", cnum, FACTOR)
        self.write(scpi_command)

    def set_averaging_state(self, cnum=1, ONOFF="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT {:}", cnum, ONOFF)
        self.write(scpi_command)

    def set_channel_name(self, cnum=1, CNAME=""):
        """Assigns a name to channel number <Ch>.
    
         The channel must be created before (CONFigure:CHANnel<Ch>[:STATe] ON).
         Moreover it is not possible to assign the same name to two
         different channels. CONFigure:CHANnel<Ch>:CATalog? returns a list of
         all defined channels with their names.
    
         CONFigure:CHANnel<Ch>:NAME '<Ch_name>'
         """
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:NAME '{:}'", cnum, CNAME)
        self.write(scpi_command)

    def set_channel_state(self, cnum=1, STATE="ON"):
        """Creates or deletes channel no. <Ch> and selects it as the active channel.
    
         CONFigure:CHANnel<Ch>:NAME defines the channel name.
         """
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:STAT {:}", cnum, STATE)
        self.write(scpi_command)

    def set_channel_trace_name(self, cnum=1, TRNAME=""):
        """Assigns a (new) name to the active trace in channel <Ch>."""
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:TRAC:REN '{:}'", cnum, TRNAME)
        self.write(scpi_command)

    def set_converter_mode(self, cnum=1, MODE="RILI"):
        """Selects the test setup (internal or external sources)
    
         for the frequency converter measurement (with option ZVA-K8, Converter Control).
    
         [SENSe<Ch>:]FREQuency:CONVersion:DEVice:MODE RILI | RILE | RILI4 | RILI56
    
         The available test setups depend on the instrument type:
    
                             RILI    RILE    RILI4    RILI56
         ---------------------------------------------------
         R&S ZVA 24|40|50      x      x
         R&S ZVA 67                   x        x
         R&S ZVT 20                            x        x
    
         <Ch>    Channel number. This suffix is ignored, the command affects all channels.
         RILI    RF internal, LO internal (4-Port)
                 Ports 1 and 2: Converter RF Signal
                 Ports 3 and 4: Converter LO
         RILE    RF internal, LO external (4-Port)
                 Ports 1 to 4: Converter RF Signal
                 External Generator #1: Converter LO
         RILI4    RF internal, LO internal, Port4=LO (R&S ZVA 67 only)
                 Ports 1 and 2: Converter RF Signal
                 Port 4: Converter LO
         RILI56    RF internal, LO internal, 6-Port (R&S ZVT 20 only)
                 Ports 1 to 4: Converter RF Signal
                 Ports 5 and 6: Converter LO 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:CONV:DEV:MODE {:}", cnum, MODE)
        self.write(scpi_command)

    def set_converter_name(self, cnum=1, TYPE=""):
        """Selects the frequency converter type for enhanced frequency-converting measurements
    
         (with option ZVA-K8, Converter Control).
    
         The preset configuration will be reset to Instrument scope (SYSTem:PRESet:SCOPe ALL)
         and Factory Preset (SYSTem:PRESet:USER:STATe OFF).
    
         [SENSe<Ch>:]FREQuency:CONVersion:DEVice:NAME '<Converter Type>'
    
         | TYPE      | frequency range    |
         |-----------|--------------------|
         | R&S®ZC170 | 110 GHz to 170 GHz |
         | R&S®ZC220 | 140 GHz to 220 GHz |
         | R&S®ZC330 | 220 GHz to 330 GHz |
         | R&S®ZC500 | 330 GHz to 500 GHz |
         |-----------|--------------------|
     
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:CONV:DEV:NAME {:}", cnum, TYPE)
        self.write(scpi_command)

    def set_corr_connection(self, cnum=1, pnum=1, CONN=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:} {:}", cnum, pnum, CONN)
        self.write(scpi_command)

    def set_corr_connection_genders(self, cnum=1, pnum=1, GENDERS="ALL"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:}:GEND {:}", cnum, pnum, GENDERS)
        self.write(scpi_command)

    def set_corr_connection_ports(self, cnum=1, pnum=1, PORTS="ALL"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:}:PORT {:}", cnum, pnum, PORTS)
        self.write(scpi_command)

    def set_corr_state(self, cnum=1, STATE="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:STAT {:}", cnum, STATE)
        self.write(scpi_command)

    def set_data(self, cnum=1, SDATA="", data=None):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA {:},", cnum, SDATA)
        self.write_values(scpi_command, data)

    def set_disp_color(self, COLOR="DBAC"):
        """no help available"""
        scpi_command = scpi_preprocess(":SYST:DISP:COL {:}", COLOR)
        self.write(scpi_command)

    def set_disp_maximize(self, wnum=1, ONOFF="OFF"):
        """Maximizes all diagram areas in the active setup or restores the previous display configuration."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:MAX {:}", wnum, ONOFF)
        self.write(scpi_command)

    def set_disp_name(self, wnum=1, name=""):
        """Defines a name for diagram area <Wnd>.
    
         The name appears in the list of diagram areas, to be queried by DISPlay[:WINDow<Wnd>]:CAT?.
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:NAME {:}", wnum, name)
        self.write(scpi_command)

    def set_disp_state(self, wnum=1, ONOFF=""):
        """Creates or deletes a diagram area, identified by its area number <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:STATe <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:STAT {:}", wnum, ONOFF)
        self.write(scpi_command)

    def set_disp_title_data(self, wnum=1, TITLE=""):
        """Defines a title for diagram area <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:TITLe:DATA '<string>'
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TITL:DATA {:}", wnum, TITLE)
        self.write(scpi_command)

    def set_disp_title_state(self, wnum=1, ONOFF="ON"):
        """Displays or hides the title for area number <Wnd>, defined by means of DISPlay:WINDow<Wnd>:TITLe:DATA.
    
         DISPlay[:WINDow<Wnd>]:TITLe[:STATe] <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TITL:STAT {:}", wnum, ONOFF)
        self.write(scpi_command)

    def set_disp_trace_auto(self, wnum=1, tnum=1, TRACENAME="None"):
        """Displays the entire trace in the diagram area, leaving an appropriate display margin.
    
         The trace can be referenced either by its number <WndTr> or by its name <trace_name>.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y[:SCALe]:AUTO ONCE[, '<trace_name>']
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:AUTO ONCE,'{:}'", wnum, tnum, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_bottom(self, wnum=1, tnum=1, lower_value="", TRACENAME="None"):
        """Sets the lower (minimum) edge of the diagram area <Wnd>."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:BOTT {:},{:}", wnum, tnum, lower_value, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_delete(self, wnum=1, tnum=1):
        """Releases the assignment between a trace and a diagram area.
    
         As defined by means of DISPlay:WINDow<Wnd>:TRACe<WndTr>:FEED <Trace_Name>
         and expressed by the <WndTr> suffix. The trace itself is not deleted; this
         must be done via CALCulate<Ch>:PARameter:DELete <Trace_Name>.
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:DEL", wnum, tnum)
        self.write(scpi_command)

    def set_disp_trace_efeed(self, wnum=1, tnum=1, TRACENAME=""):
        """Assigns an existing trace to a diagram area.
    
         Assigns an existing trace (CALCulate<Ch>:PARameter:SDEFine <Trace_Name>)
         to a diagram area <Wnd>, and displays the trace. Use
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:FEED to assign the trace to a diagram area
         using a numeric suffix (e.g. in order to use the
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y:OFFSet command).
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:EFEed '<trace_name>'
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:EFE '{:}'", wnum, tnum, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_feed(self, wnum=1, tnum=1, TRACENAME=""):
        """Assigns an existing trace to a diagram area.
    
         Assigns an existing trace (CALCulate<Ch>:PARameter:SDEFine <Trace_Name>)
         to a diagram area, using the <WndTr> suffix, and displays the trace. Use
         DISPlay[:WINDow<Wnd>]:TRACe:EFEed to assign the trace to a diagram area
         without using a numeric suffix.
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:FEED '{:}'", wnum, tnum, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_pdiv(self, wnum=1, tnum=1, value="", TRACENAME="None"):
        """Sets the value between two grid lines (value “per division”) for the diagram area <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y[:SCALe]:PDIVision  <numeric_value>[, '<trace_name>']
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:PDIV {:},{:}", wnum, tnum, value, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_reflevel(self, wnum=1, tnum=1, value="", TRACENAME="None"):
        """Sets the reference level (or reference value) for a particular displayed trace.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y[:SCALe]:RLEVel  <numeric_value>[, '<trace_name>']
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:RLEV {:},{:}", wnum, tnum, value, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_refpos(self, wnum=1, tnum=1, value="", TRACENAME="None"):
        """Sets the point on the y-axis to be used as the reference position
    
         (as a percentage of the length of the y-axis).
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y[:SCALe]:RPOSition  <numeric_value>[, '<trace_name>']
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:RPOS {:},{:}", wnum, tnum, value, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_show(self, wnum=1, tnum=1, DMTRACES="", TRACENAME=""):
        """Displays or hides an existing trace, identified by its trace name, or a group of traces.
    
         Displays or hides an existing trace, identified by its trace name
         (CALCulate<Ch>:PARameter:SDEFine <Trace_Name>), or a group of traces.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:SHOW DALL | MALL | '<trace_name>', <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:SHOW {:} {:}", wnum, tnum, DMTRACES, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_top(self, wnum=1, tnum=1, upper_value="", TRACENAME="None"):
        """Sets the upper (maximum) edge of the diagram area <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:Y[:SCALe]:TOP  <upper_value>[, '<trace_name>']
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:SCAL:TOP {:},{:}", wnum, tnum, upper_value, TRACENAME)
        self.write(scpi_command)

    def set_disp_trace_x_offset(self, wnum=1, tnum=1, range=""):
        """Shifts the trace <WndTr> in horizontal direction, leaving the positions of all markers unchanged."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:X:OFFS {:}", wnum, tnum, range)
        self.write(scpi_command)

    def set_disp_trace_y_offset(self, wnum=1, tnum=1, MAG="", PHASE=0, REAL=0, IMAG=0):
        """Modifies all points of the trace <WndTr> by means of an added and/or a multiplied complex constant."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:OFFS {:}{:}{:}{:}", wnum, tnum, MAG, PHASE, REAL, IMAG)
        self.write(scpi_command)

    def set_disp_update(self, UPDATE="OFF"):
        """no help available"""
        scpi_command = scpi_preprocess(":SYST:DISP:UPD {:}", UPDATE)
        self.write(scpi_command)

    def set_f_start(self, cnum=1, freq=""):
        """Defines the start frequency for a frequency sweep which is equal to the left edge of a Cartesian diagram.
    
         [SENSe<Ch>:]FREQuency:STARt <start_frequency> 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STAR {:}", cnum, freq)
        self.write(scpi_command)

    def set_f_stop(self, cnum=1, freq=""):
        """Defines the stop frequency for a frequency sweep which is equal to the right edge of a Cartesian diagram.
    
         [SENSe<Ch>:]FREQuency:STOP <stop_frequency> 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP {:}", cnum, freq)
        self.write(scpi_command)

    def set_format_binary(self, ORDER="SWAP"):
        """Controls whether binary data is transferred in normal or swapped byte order.
    
         FORMat:BORDer NORMal | SWAPped
         """
        scpi_command = scpi_preprocess(":FORM:BORD {:}", ORDER)
        self.write(scpi_command)

    def set_format_data(self, DATA="ASCII"):
        """Selects the format for numeric data transferred to and from the analyzer.
    
         FORMat[:DATA] ASCii | REAL [,<length>]
    
         NOTE:
         The format setting is only valid for commands and queries whose
         description states that the response is formatted as described by
         FORMat[:DATA]. In particular, it affects trace data transferred by means
         of the commands in the TRACe:... system.
    
         The default length of REAL data is 32 bits (single precision).
         """
        scpi_command = scpi_preprocess(":FORM:DATA {:}", DATA)
        self.write(scpi_command)

    def set_init_continuous(self, cnum=1, STATE="ON"):
        """Qualifies whether the analyzer measures in single sweep or in continuous sweep mode.
    
         INITiate<Ch>:CONTinuous <Boolean>
         """
        scpi_command = scpi_preprocess(":INIT{:}:CONT {:}", cnum, STATE)
        self.write(scpi_command)

    def set_init_immediate(self, cnum=1):
        """Starts a new single sweep sequence.
    
         This command is available in single sweep mode only
         (INITiate<Ch>:CONTinuous OFF). The data of the last sweep (or previous
         sweeps, see Sweep History) can be read using
         CALCulate<Ch>:DATA:NSWeep:FIRSt? SDATa, <count>.
         """
        scpi_command = scpi_preprocess(":INIT{:}:IMM", cnum)
        self.write(scpi_command)

    def set_init_immediate_scope(self, cnum=1, SCOPE="ALL"):
        """Selects the scope of the single sweep sequence.
    
         The setting is applied in single sweep mode only (INITiate<Ch>:CONTinuous OFF).
    
         NITiate<Ch>[:IMMediate]:SCOPe ALL | SINGle
         """
        scpi_command = scpi_preprocess(":INIT{:}:IMM:SCOP {:}", cnum, SCOPE)
        self.write(scpi_command)

    def set_par_del(self, cnum=1, TRACE=""):
        """Deletes a trace with a specified trace name and channel.
    
         CALCulate<Ch>:PARameter:DELete '<trace>'
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL '{:}'", cnum, TRACE)
        self.write(scpi_command)

    def set_par_del_all(self, cnum=1):
        """Deletes all traces in all channels of the active setup,
    
         including the default trace Trc1 in channel no. 1. The manual
         control screen shows 'No Trace'
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL:ALL", cnum)
        self.write(scpi_command)

    def set_par_del_call(self, cnum=1):
        """Deletes all traces in channel no. <Ch>."""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL:CALL", cnum)
        self.write(scpi_command)

    def set_par_del_sgroup(self, cnum=1):
        """Deletes a group of logical ports (S-parameter group),
    
         previously defined via CALCulate<Ch>:PARameter:DEFine:SGRoup.
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL:SGR", cnum)
        self.write(scpi_command)

    def set_par_sdef(self, cnum=1, TRACE="", COEFF=""):
        """Creates a trace and assigns a channel number, a name and a measured quantity to it.
    
         The trace becomes the active trace in the channel but is not displayed.
    
         To select an existing trace as the active trace, use
         CALCulate:PARameter:SELect. You can open the trace manager
         (DISPlay:MENU:KEY:EXECute 'Trace Manager') to obtain an overview of
         all channels and traces, including the traces that are not displayed.
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SDEF '{:}', '{:}'", cnum, TRACE, COEFF)
        self.write(scpi_command)

    def set_par_select(self, cnum=1, TRACE=""):
        """Selects an existing trace as the active trace of the channel.
    
         All trace commands without explicit reference to the trace name act on
         the active trace (e.g. CALCulate<Chn>:FORMat).
         CALCulate<Ch>:PARameter:SELect is also necessary if the active trace of
         a channel has been deleted.
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL '{:}'", cnum, TRACE)
        self.write(scpi_command)

    def set_rbw(self, cnum=1, BW=""):
        """Defines the resolution bandwidth of the analyzer (Meas. Bandwidth).
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution] <bandwidth>
    
         <Ch>  Channel number. If unspecified the numeric suffix is set to 1.
         <bandwidth>
         Resolution bandwidth
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:RES {:}", cnum, BW)
        self.write(scpi_command)

    def set_rbw_reduction(self, cnum=1, ONOFF="OFF"):
        """Enables or disables dynamic bandwidth reduction at low frequencies.
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution]:DREDuction <Boolean>
    
         <Ch>        Channel number
         <Boolean>    ON | OFF – enable or disable dynamic bandwidth reduction.
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:DRED {:}", cnum, ONOFF)
        self.write(scpi_command)

    def set_rbw_select(self, cnum=1):
        """Defines the selectivity of the IF filter for an unsegmented sweep.
    
         The value is also used for all segments of a segmented sweep,
         provided that separate selectivity setting is disabled
         ([SENSe<Ch>:]SEGMent<Seg>:BWIDth[:RESolution]:SELect:CONTrol OFF).
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution]:SELect NORMal | HIGH
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:SEL", cnum)
        self.write(scpi_command)

    def set_sweep_count(self, cnum=1, NUMSWEEP=1):
        """Defines the number of sweeps to be measured in single sweep mode
    
         (INITiate<Ch>:CONTinuous OFF).
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:COUN {:}", cnum, NUMSWEEP)
        self.write(scpi_command)

    def set_sweep_n_points(self, cnum=1, NUMPOINTS=201):
        """Defines the total number of measurement points per sweep (Number of Points)."""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN {:}", cnum, NUMPOINTS)
        self.write(scpi_command)

    def set_sweep_spacing(self, cnum=1, SPACING="LIN"):
        """Defines the frequency vs. time characteristics of a frequency sweep
    
         (Lin. Frequency or Log Frequency). The command has no effect on segmented
         frequency, power or time sweeps.
    
         [SENSe<Ch>:]SWEep:SPACing  LINear | LOGarithmic
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:SPAC {:}", cnum, SPACING)
        self.write(scpi_command)

    def set_sweep_step(self, cnum=1, step_size=""):
        """Sets the distance between two consecutive sweep points.
    
         [SENSe<Ch>:]SWEep:STEP <step_size>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:STEP {:}", cnum, step_size)
        self.write(scpi_command)

    def set_sweep_time(self, cnum=1, duration=""):
        """Sets the duration of the sweep (Sweep Time).
    
         Setting a duration disables the automatic calculation of the (minimum) sweep time;
         see [SENSe<Ch>:]SWEep:TIME:AUTO.
    
         [SENSe<Ch>:]SWEep:TIME <duration>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME {:}", cnum, duration)
        self.write(scpi_command)

    def set_sweep_time_auto(self, cnum=1, ONOFF="ON"):
        """When enabled, the (minimum) sweep duration is calculated internally
    
         using the other channel settings and zero delay ([SENSe<Ch>:]SWEep:DWELl).
    
         [SENSe<Ch>:]SWEep:TIME:AUTO <Boolean>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME:AUTO {:}", cnum, ONOFF)
        self.write(scpi_command)

    def set_sweep_type(self, cnum=1, TYPE="LIN"):
        """Selects the sweep type,
    
         i.e. the sweep variable (frequency/power/time) and the position of the
         sweep points across the sweep range.
    
         [SENSe<Ch>:]SWEep:TYPE LINear | LOGarithmic | SEGMent | POWer | CW | POINt | PULSe | IAMPlitude | IPHase
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE {:}", cnum, TYPE)
        self.write(scpi_command)

    def query_active_channel(self):
        """Selects channel as the active channel.
    
         INSTrument:NSELect <Channel>
         """
        scpi_command = ":INST:NSEL?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_average_factor_current(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN:CURR?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_averaging_state(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_channel_catalog(self, cnum=1):
        """Returns the numbers and names of all channels in the current setup."""
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:CAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_channel_name(self, cnum=1):
        """Assigns a name to channel number <Ch>.
    
         The channel must be created before (CONFigure:CHANnel<Ch>[:STATe] ON).
         Moreover it is not possible to assign the same name to two
         different channels. CONFigure:CHANnel<Ch>:CATalog? returns a list of
         all defined channels with their names.
    
         CONFigure:CHANnel<Ch>:NAME '<Ch_name>'
         """
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:NAME?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_channel_name_id(self, cnum=1, CNAME="Ch1"):
        """Queries the channel number (numeric suffix) of a channel with known channel name.
    
         A channel name must be assigned before (CONFigure:CHANnel<Ch>NAME '<Ch_name>').
         CONFigure:CHANnel<Ch>:CATalog? returns a list of all defined
         channels with their names.
    
         CONFigure:CHANnel<Ch>:NAME:ID? '<Ch_name>'
         """
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:NAME:ID? '{:}'", cnum, CNAME)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_channel_state(self, cnum=1):
        """Creates or deletes channel no. <Ch> and selects it as the active channel.
    
         CONFigure:CHANnel<Ch>:NAME defines the channel name.
         """
        scpi_command = scpi_preprocess(":CONF:CHAN{:}:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_converter_mode(self, cnum=1):
        """Selects the test setup (internal or external sources)
    
         for the frequency converter measurement (with option ZVA-K8, Converter Control).
    
         [SENSe<Ch>:]FREQuency:CONVersion:DEVice:MODE RILI | RILE | RILI4 | RILI56
    
         The available test setups depend on the instrument type:
    
                             RILI    RILE    RILI4    RILI56
         ---------------------------------------------------
         R&S ZVA 24|40|50      x      x
         R&S ZVA 67                   x        x
         R&S ZVT 20                            x        x
    
         <Ch>    Channel number. This suffix is ignored, the command affects all channels.
         RILI    RF internal, LO internal (4-Port)
                 Ports 1 and 2: Converter RF Signal
                 Ports 3 and 4: Converter LO
         RILE    RF internal, LO external (4-Port)
                 Ports 1 to 4: Converter RF Signal
                 External Generator #1: Converter LO
         RILI4    RF internal, LO internal, Port4=LO (R&S ZVA 67 only)
                 Ports 1 and 2: Converter RF Signal
                 Port 4: Converter LO
         RILI56    RF internal, LO internal, 6-Port (R&S ZVT 20 only)
                 Ports 1 to 4: Converter RF Signal
                 Ports 5 and 6: Converter LO 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:CONV:DEV:MODE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_converter_name(self, cnum=1):
        """Selects the frequency converter type for enhanced frequency-converting measurements
    
         (with option ZVA-K8, Converter Control).
    
         The preset configuration will be reset to Instrument scope (SYSTem:PRESet:SCOPe ALL)
         and Factory Preset (SYSTem:PRESet:USER:STATe OFF).
    
         [SENSe<Ch>:]FREQuency:CONVersion:DEVice:NAME '<Converter Type>'
    
         | TYPE      | frequency range    |
         |-----------|--------------------|
         | R&S®ZC170 | 110 GHz to 170 GHz |
         | R&S®ZC220 | 140 GHz to 220 GHz |
         | R&S®ZC330 | 220 GHz to 330 GHz |
         | R&S®ZC500 | 330 GHz to 500 GHz |
         |-----------|--------------------|
     
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:CONV:DEV:NAME?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_corr_connection(self, cnum=1, pnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:}?", cnum, pnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_corr_connection_genders(self, cnum=1, pnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:}:GEND?", cnum, pnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_corr_connection_ports(self, cnum=1, pnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:COLL:CONN{:}:PORT?", cnum, pnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_corr_date(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:DATE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_corr_state(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_data(self, cnum=1, SDATA=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA? {:}", cnum, SDATA)
        return self.query_values(scpi_command)

    def query_data_all(self, cnum=1, SDATA=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:ALL? {:}", cnum, SDATA)
        return self.query_values(scpi_command)

    def query_data_call(self, cnum=1, SDATA=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:CALL? {:}", cnum, SDATA)
        return self.query_values(scpi_command)

    def query_data_call_catalog(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:CALL:CAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='str')
        return value

    def query_data_dall(self, cnum=1, SDATA=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:DALL? {:}", cnum, SDATA)
        return self.query_values(scpi_command)

    def query_data_sgroup(self, cnum=1, SDATA=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:SGR? {:}", cnum, SDATA)
        return self.query_values(scpi_command)

    def query_disp_catalog(self, wnum=1):
        """Returns the numbers and names of all diagram areas in the current setup."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:CAT?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_disp_name(self, wnum=1):
        """Defines a name for diagram area <Wnd>.
    
         The name appears in the list of diagram areas, to be queried by DISPlay[:WINDow<Wnd>]:CAT?.
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:NAME?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_disp_state(self, wnum=1):
        """Creates or deletes a diagram area, identified by its area number <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:STATe <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:STAT?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_disp_title_data(self, wnum=1):
        """Defines a title for diagram area <Wnd>.
    
         DISPlay[:WINDow<Wnd>]:TITLe:DATA '<string>'
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TITL:DATA?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_disp_title_state(self, wnum=1):
        """Displays or hides the title for area number <Wnd>, defined by means of DISPlay:WINDow<Wnd>:TITLe:DATA.
    
         DISPlay[:WINDow<Wnd>]:TITLe[:STATe] <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TITL:STAT?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_disp_trace_catalog(self, wnum=1, tnum=1):
        """Returns the numbers and names of all traces in diagram area no. <Wnd>."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:CAT?", wnum, tnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_disp_trace_show(self, wnum=1, tnum=1, DMTRACES=""):
        """Displays or hides an existing trace, identified by its trace name, or a group of traces.
    
         Displays or hides an existing trace, identified by its trace name
         (CALCulate<Ch>:PARameter:SDEFine <Trace_Name>), or a group of traces.
    
         DISPlay[:WINDow<Wnd>]:TRACe<WndTr>:SHOW DALL | MALL | '<trace_name>', <Boolean>
         """
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:SHOW? {:}", wnum, tnum, DMTRACES)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_disp_trace_x_offset(self, wnum=1, tnum=1):
        """Shifts the trace <WndTr> in horizontal direction, leaving the positions of all markers unchanged."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:X:OFFS?", wnum, tnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_disp_trace_y_offset(self, wnum=1, tnum=1):
        """Modifies all points of the trace <WndTr> by means of an added and/or a multiplied complex constant."""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:Y:OFFS?", wnum, tnum)
        return self.query_values(scpi_command)

    def query_f_start(self, cnum=1):
        """Defines the start frequency for a frequency sweep which is equal to the left edge of a Cartesian diagram.
    
         [SENSe<Ch>:]FREQuency:STARt <start_frequency> 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STAR?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_f_stop(self, cnum=1):
        """Defines the stop frequency for a frequency sweep which is equal to the right edge of a Cartesian diagram.
    
         [SENSe<Ch>:]FREQuency:STOP <stop_frequency> 
        """
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_format_binary(self):
        """Controls whether binary data is transferred in normal or swapped byte order.
    
         FORMat:BORDer NORMal | SWAPped
         """
        scpi_command = ":FORM:BORD?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_format_data(self):
        """Selects the format for numeric data transferred to and from the analyzer.
    
         FORMat[:DATA] ASCii | REAL [,<length>]
    
         NOTE:
         The format setting is only valid for commands and queries whose
         description states that the response is formatted as described by
         FORMat[:DATA]. In particular, it affects trace data transferred by means
         of the commands in the TRACe:... system.
    
         The default length of REAL data is 32 bits (single precision).
         """
        scpi_command = ":FORM:DATA?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_init_continuous(self, cnum=1):
        """Qualifies whether the analyzer measures in single sweep or in continuous sweep mode.
    
         INITiate<Ch>:CONTinuous <Boolean>
         """
        scpi_command = scpi_preprocess(":INIT{:}:CONT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_init_immediate_scope(self, cnum=1):
        """Selects the scope of the single sweep sequence.
    
         The setting is applied in single sweep mode only (INITiate<Ch>:CONTinuous OFF).
    
         NITiate<Ch>[:IMMediate]:SCOPe ALL | SINGle
         """
        scpi_command = scpi_preprocess(":INIT{:}:IMM:SCOP?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_par_catalog(self, cnum=1):
        """Returns the trace names and measured quantities of all traces assigned to a particular channel."""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:CAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='str')
        return value

    def query_par_select(self, cnum=1):
        """Selects an existing trace as the active trace of the channel.
    
         All trace commands without explicit reference to the trace name act on
         the active trace (e.g. CALCulate<Chn>:FORMat).
         CALCulate<Ch>:PARameter:SELect is also necessary if the active trace of
         a channel has been deleted.
         """
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_rbw(self, cnum=1):
        """Defines the resolution bandwidth of the analyzer (Meas. Bandwidth).
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution] <bandwidth>
    
         <Ch>  Channel number. If unspecified the numeric suffix is set to 1.
         <bandwidth>
         Resolution bandwidth
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:RES?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_rbw_reduction(self, cnum=1):
        """Enables or disables dynamic bandwidth reduction at low frequencies.
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution]:DREDuction <Boolean>
    
         <Ch>        Channel number
         <Boolean>    ON | OFF – enable or disable dynamic bandwidth reduction.
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:DRED?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_rbw_select(self, cnum=1):
        """Defines the selectivity of the IF filter for an unsegmented sweep.
    
         The value is also used for all segments of a segmented sweep,
         provided that separate selectivity setting is disabled
         ([SENSe<Ch>:]SEGMent<Seg>:BWIDth[:RESolution]:SELect:CONTrol OFF).
    
         [SENSe<Ch>:]BANDwidth|BWIDth[:RESolution]:SELect NORMal | HIGH
        """
        scpi_command = scpi_preprocess(":SENS{:}:BAND:SEL?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_count(self, cnum=1):
        """Defines the number of sweeps to be measured in single sweep mode
    
         (INITiate<Ch>:CONTinuous OFF).
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:COUN?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_sweep_n_points(self, cnum=1):
        """Defines the total number of measurement points per sweep (Number of Points)."""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_sweep_spacing(self, cnum=1):
        """Defines the frequency vs. time characteristics of a frequency sweep
    
         (Lin. Frequency or Log Frequency). The command has no effect on segmented
         frequency, power or time sweeps.
    
         [SENSe<Ch>:]SWEep:SPACing  LINear | LOGarithmic
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:SPAC?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_step(self, cnum=1):
        """Sets the distance between two consecutive sweep points.
    
         [SENSe<Ch>:]SWEep:STEP <step_size>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:STEP?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_time(self, cnum=1):
        """Sets the duration of the sweep (Sweep Time).
    
         Setting a duration disables the automatic calculation of the (minimum) sweep time;
         see [SENSe<Ch>:]SWEep:TIME:AUTO.
    
         [SENSe<Ch>:]SWEep:TIME <duration>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_time_auto(self, cnum=1):
        """When enabled, the (minimum) sweep duration is calculated internally
    
         using the other channel settings and zero delay ([SENSe<Ch>:]SWEep:DWELl).
    
         [SENSe<Ch>:]SWEep:TIME:AUTO <Boolean>
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME:AUTO?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_sweep_type(self, cnum=1):
        """Selects the sweep type,
    
         i.e. the sweep variable (frequency/power/time) and the position of the
         sweep points across the sweep range.
    
         [SENSe<Ch>:]SWEep:TYPE LINear | LOGarithmic | SEGMent | POWer | CW | POINt | PULSe | IAMPlitude | IPHase
         """
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_system_err_next(self):
        """no help available"""
        scpi_command = ":SYST:ERR:NEXT?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_system_error_all(self):
        """no help available"""
        scpi_command = ":SYST:ERR:ALL?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value
