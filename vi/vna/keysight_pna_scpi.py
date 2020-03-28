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

    def set_active_calset(self, cnum=1, calset_name="", onoff=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:ACT '{:}',{:}", cnum, calset_name, onoff)
        self.write(scpi_command)

    def set_averaging_count(self, cnum=1, avg_count=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN {:}", cnum, avg_count)
        self.write(scpi_command)

    def set_averaging_mode(self, cnum=1, avg_mode="SWEEP"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:MODE {:}", cnum, avg_mode)
        self.write(scpi_command)

    def set_averaging_state(self, cnum=1, onoff="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT {:}", cnum, onoff)
        self.write(scpi_command)

    def set_calset_data(self, cnum=1, eterm="", portA=1, portB=2, param="", eterm_data=None):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:DATA {:},{:},{:},{:},", cnum, eterm, portA, portB, param)
        self.write_values(scpi_command, eterm_data)

    def set_calset_description(self, cnum=1, description=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:DESC '{:}'", cnum, description)
        self.write(scpi_command)

    def set_calset_eterm(self, cnum=1, eterm="", eterm_data=None):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:ETER '{:}',", cnum, eterm)
        self.write_values(scpi_command, eterm_data)

    def set_calset_name(self, cnum=1, calset_name=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:NAME '{:}'", cnum, calset_name)
        self.write(scpi_command)

    def set_channel_correction_state(self, cnum=1, onoff="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:STAT {:}", cnum, onoff)
        self.write(scpi_command)

    def set_create_calset(self, cnum=1, calset_name=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:CRE '{:}'", cnum, calset_name)
        self.write(scpi_command)

    def set_create_meas(self, cnum=1, mname="", param=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEF:EXT '{:}','{:}'", cnum, mname, param)
        self.write(scpi_command)

    def set_data(self, cnum=1, fmt="SDATA", data=None):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA {:},", cnum, fmt)
        self.write_values(scpi_command, data)

    def set_delete_calset(self, cnum=1, calset_name=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:DEL '{:}'", cnum, calset_name)
        self.write(scpi_command)

    def set_delete_meas(self, cnum=1, mname=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL '{:}'", cnum, mname)
        self.write(scpi_command)

    def set_display_format(self, cnum=1, fmt="MLOG"):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:FORM {:}", cnum, fmt)
        self.write(scpi_command)

    def set_display_onoff(self, onoff="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":DISP:ENAB {:}", onoff)
        self.write(scpi_command)

    def set_display_trace(self, wnum="", tnum="", mname=""):
        """no help available"""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:FEED '{:}'", wnum, tnum, mname)
        self.write(scpi_command)

    def set_f_start(self, cnum=1, freq=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:START {:}", cnum, freq)
        self.write(scpi_command)

    def set_f_stop(self, cnum=1, freq=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP {:}", cnum, freq)
        self.write(scpi_command)

    def set_groups_count(self, cnum=1, groups_count=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:GRO:COUN {:}", cnum, groups_count)
        self.write(scpi_command)

    def set_meas_correction_state(self, cnum=1, onoff="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:CORR:STAT {:}", cnum, onoff)
        self.write(scpi_command)

    def set_save_calset(self, cnum=1, user='01'):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:SAVE {:}", cnum, user)
        self.write(scpi_command)

    def set_selected_meas(self, cnum=1, mname=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL '{:}'", cnum, mname)
        self.write(scpi_command)

    def set_selected_meas_by_number(self, cnum=1, mnum=""):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:MNUM {:}", cnum, mnum)
        self.write(scpi_command)

    def set_snp_format(self, fmt="RI"):
        """no help available"""
        scpi_command = scpi_preprocess(":MMEM:STOR:TRAC:FORM:SNP {:}", fmt)
        self.write(scpi_command)

    def set_sweep_mode(self, cnum=1, sweep_mode="CONT"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:MODE {:}", cnum, sweep_mode)
        self.write(scpi_command)

    def set_sweep_n_points(self, cnum=1, n_points=401):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN {:}", cnum, n_points)
        self.write(scpi_command)

    def set_sweep_type(self, cnum=1, sweep_type="LIN"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE {:}", cnum, sweep_type)
        self.write(scpi_command)

    def set_trigger_source(self, trigger_source="IMM"):
        """no help available"""
        scpi_command = scpi_preprocess(":TRIG:SOUR {:}", trigger_source)
        self.write(scpi_command)

    def query_active_calset(self, cnum=1, form="NAME"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:ACT? {:}", cnum, form)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_active_channel(self):
        """no help available"""
        scpi_command = ":SYST:ACT:CHAN?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_available_channels(self):
        """no help available"""
        scpi_command = ":SYST:CHAN:CAT?"
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='int')
        return value

    def query_averaging_count(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_averaging_mode(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:MODE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_averaging_state(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_calset_catalog(self, cnum=1, form="NAME"):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:CAT? {:}", cnum, form)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_calset_data(self, cnum=1, eterm="", portA=1, portB=2):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:DATA? {:},{:},{:}", cnum, eterm, portA, portB)
        return self.query_values(scpi_command)

    def query_calset_description(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:DESC?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_calset_eterm(self, cnum=1, eterm=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:ETER? '{:}'", cnum, eterm)
        return self.query_values(scpi_command)

    def query_calset_eterm_catalog(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:ETER:CAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_calset_name(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:NAME?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_channel_correction_state(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_data(self, cnum=1, fmt="SDATA"):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA? {:}", cnum, fmt)
        return self.query_values(scpi_command)

    def query_display_format(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:FORM?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_display_onoff(self):
        """no help available"""
        scpi_command = ":DISP:ENAB?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_f_start(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:START?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_f_stop(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_groups_count(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:GRO:COUN?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_meas_correction_state(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:CORR:STAT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_meas_name_from_number(self, mnum=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SYST:MEAS{:}:NAME?", mnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_meas_name_list(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:CAT:EXT?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='str')
        return value

    def query_meas_number_list(self, cnum=""):
        """no help available"""
        scpi_command = scpi_preprocess(":SYST:MEAS:CAT? {:}", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='int')
        return value

    def query_save_calset(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:CORR:CSET:SAVE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_selected_meas(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_selected_meas_by_number(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:PAR:MNUM?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_snp_data(self, cnum=1, ports=(1, 2)):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC{:}:DATA:SNP:PORT? '{:}'", cnum, ports)
        return self.query_values(scpi_command)

    def query_snp_format(self):
        """no help available"""
        scpi_command = ":MMEM:STOR:TRAC:FORM:SNP?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_data(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:X:VALUES?", cnum)
        return self.query_values(scpi_command)

    def query_sweep_mode(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:MODE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_n_points(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_sweep_time(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_sweep_type(self, cnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE?", cnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_trigger_source(self):
        """no help available"""
        scpi_command = ":TRIG:SOUR?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_window_trace_numbers(self, wnum=""):
        """no help available"""
        scpi_command = scpi_preprocess(":DISP:WIND{:}:CAT?", wnum)
        value = self.query(scpi_command)
        value = process_query(value, csv=True, strip_outer_quotes=True, returns='int')
        return value
