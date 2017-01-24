def to_string(value):
    if type(value) in (list, tuple):
        return ",".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)


def scpi_preprocess(command_string, *args):
    args_list = list(args)
    for i, arg in enumerate(args_list):
        args_list[i] = to_string(arg)
    return command_string.format(*args_list)


class SCPI(object):
    def __init__(self):
        self.resource = None

    def set_groups_count(self, cnum="", num=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:GRO:COUN {:}", cnum, num)
        self.resource.write('{:}'.format(scpi_command))

    def set_sweep_mode(self, cnum="", mode=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:MODE {:}", cnum, mode)
        self.resource.write('{:}'.format(scpi_command))

    def set_sweep_type(self, cnum="", sweep_type=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE {:}", cnum, sweep_type)
        self.resource.write('{:}'.format(scpi_command))

    def set_sweep_n_points(self, cnum="", num=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN {:}", cnum, num)
        self.resource.write('{:}'.format(scpi_command))

    def set_averaging_mode(self, cnum="", mode=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:MODE {:}", cnum, mode)
        self.resource.write('{:}'.format(scpi_command))

    def set_averaging_count(self, cnum="", num=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN {:}", cnum, num)
        self.resource.write('{:}'.format(scpi_command))

    def set_averaging_state(self, cnum="", onoff=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT {:}", cnum, onoff)
        self.resource.write('{:}'.format(scpi_command))

    def set_f_stop(self, cnum="", num=""):
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP {:}", cnum, num)
        self.resource.write('{:}'.format(scpi_command))

    def set_f_start(self, cnum="", num=""):
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:START {:}", cnum, num)
        self.resource.write('{:}'.format(scpi_command))

    def set_display_trace(self, wnum="", tnum="", mname=""):
        scpi_command = scpi_preprocess(":DISP:WIND{:}:TRAC{:}:FEED '{:}'", wnum, tnum, mname)
        self.resource.write('{:}'.format(scpi_command))

    def set_data(self, cnum="", fmt=""):
        scpi_command = scpi_preprocess(":CALC{:}:DATA {:}", cnum, fmt)
        self.resource.write('{:}'.format(scpi_command))

    def set_create_meas(self, cnum="", mname="", param=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEF:EXT '{:}','{:}'", cnum, mname, param)
        self.resource.write('{:}'.format(scpi_command))

    def set_selected_meas(self, cnum="", mname=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL '{:}'", cnum, mname)
        self.resource.write('{:}'.format(scpi_command))

    def set_delete_meas(self, cnum="", mmame=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:DEL '{:}'", cnum, mmame)
        self.resource.write('{:}'.format(scpi_command))

    def set_selected_meas_by_number(self, cnum="", mnum=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:MNUM {:}", cnum, mnum)
        self.resource.write('{:}'.format(scpi_command))

    def set_disp_format(self, cnum="", fmt=""):
        scpi_command = scpi_preprocess(":CALC{:}:FORM {:}", cnum, fmt)
        self.resource.write('{:}'.format(scpi_command))

    def set_trigger_source(self, trigger_source=""):
        scpi_command = scpi_preprocess(":TRIG:SOUR {:}", trigger_source)
        self.resource.write('{:}'.format(scpi_command))

    def query_groups_count(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:GRO:COUN?", cnum)
        value = self.resource.query(scpi_command)
        return int(value)

    def query_sweep_mode(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:MODE?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_sweep_type(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TYPE?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_sweep_n_points(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:POIN?", cnum)
        value = self.resource.query(scpi_command)
        return int(value)

    def query_sweep_time(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:SWE:TIME?", cnum)
        value = self.resource.query(scpi_command)
        return float(value)

    def query_averaging_mode(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:MODE?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_averaging_count(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:COUN?", cnum)
        value = self.resource.query(scpi_command)
        return int(value)

    def query_averaging_state(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:AVER:STAT?", cnum)
        value = self.resource.query(scpi_command)
        return bool(value)

    def query_f_stop(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:STOP?", cnum)
        value = self.resource.query(scpi_command)
        return float(value)

    def query_f_start(self, cnum=""):
        scpi_command = scpi_preprocess(":SENS{:}:FREQ:START?", cnum)
        value = self.resource.query(scpi_command)
        return float(value)

    def query_meas_name_list(self, cnum=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:CAT:EXT?", cnum)
        value = self.resource.query(scpi_command)
        return value.split(',')

    def query_selected_meas(self, cnum=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:SEL?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_selected_meas_by_number(self, cnum=""):
        scpi_command = scpi_preprocess(":CALC{:}:PAR:MNUM?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_disp_format(self, cnum=""):
        scpi_command = scpi_preprocess(":CALC{:}:FORM?", cnum)
        value = self.resource.query(scpi_command)
        return value

    def query_trigger_source(self):
        scpi_command = ":TRIG:SOUR?"
        value = self.resource.query(scpi_command)
        return value

    def query_meas_number_list(self, cnum=""):
        scpi_command = scpi_preprocess(":SYST:MEAS:CAT? {:}", cnum)
        value = self.resource.query(scpi_command)
        return list(map(int, value))

    def query_available_channels(self, cnum=""):
        scpi_command = scpi_preprocess(":SYST:CHAN:CAT? {:}", cnum)
        value = self.resource.query(scpi_command)
        return list(map(int, value))

    def query_active_channel(self):
        scpi_command = ":SYST:ACT:CHAN?"
        value = self.resource.query(scpi_command)
        return int(value)

    def query_meas_name_from_number(self, mnum=""):
        scpi_command = scpi_preprocess(":SYST:MEAS{:}:NAME?", mnum)
        value = self.resource.query(scpi_command)
        return value

    def query_data(self, cnum="", fmt=""):
        scpi_command = scpi_preprocess(":CALC{:}:DATA? {:}", cnum, fmt)
        return self.resource.query_values(scpi_command)

    def query_snp_data(self, cnum="", ports=""):
        scpi_command = scpi_preprocess(":CALC{:}:DATA:SNP:PORT? '{:}'", cnum, ports)
        return self.resource.query_values(scpi_command)
