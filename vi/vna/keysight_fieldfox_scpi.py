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

    def set_averaging_count(self, avg_count=""):
        """no help available"""
        scpi_command = scpi_preprocess(":AVER:COUN {:}", avg_count)
        self.write(scpi_command)

    def set_averaging_mode(self, avg_mode="SWEEP"):
        """no help available"""
        scpi_command = scpi_preprocess(":AVER:MODE {:}", avg_mode)
        self.write(scpi_command)

    def set_clear_averaging(self):
        """no help available"""
        scpi_command = ":AVER:CLE"
        self.write(scpi_command)

    def set_continuous_sweep(self, onoff="ON"):
        """no help available"""
        scpi_command = scpi_preprocess(":INIT:CONT {:}", onoff)
        self.write(scpi_command)

    def set_current_trace(self, trace=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC:PAR{:}:SEL", trace)
        self.write(scpi_command)

    def set_display_format(self, fmt="MLOG"):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC:FORM {:}", fmt)
        self.write(scpi_command)

    def set_f_start(self, freq=""):
        """no help available"""
        scpi_command = scpi_preprocess(":FREQ:START {:}", freq)
        self.write(scpi_command)

    def set_f_stop(self, freq=""):
        """no help available"""
        scpi_command = scpi_preprocess(":FREQ:STOP {:}", freq)
        self.write(scpi_command)

    def set_if_bandwidth(self, bandwidth=1000):
        """no help available"""
        scpi_command = scpi_preprocess(":BWID {:}", bandwidth)
        self.write(scpi_command)

    def set_instrument(self, instr="NA"):
        """no help available"""
        scpi_command = scpi_preprocess(":INST '{:}'", instr)
        self.write(scpi_command)

    def set_single_sweep(self):
        """no help available"""
        scpi_command = ":INIT"
        self.write(scpi_command)

    def set_sweep_n_points(self, n_points=401):
        """no help available"""
        scpi_command = scpi_preprocess(":SWE:POIN {:}", n_points)
        self.write(scpi_command)

    def set_trace_autoscale(self, tnum=1):
        """no help available"""
        scpi_command = scpi_preprocess(":DISP:WIND:TRAC{:}:Y:AUTO", tnum)
        self.write(scpi_command)

    def set_trace_count(self, num=4):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC:PAR:COUN {:}", num)
        self.write(scpi_command)

    def set_trace_display_config(self, config="D12_34"):
        """no help available"""
        scpi_command = scpi_preprocess(":DISP:WIND:SPL {:}", config)
        self.write(scpi_command)

    def set_trace_measurement(self, trace=1, meas='S11'):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC:PAR{:}:DEF {:}", trace, meas)
        self.write(scpi_command)

    def query_averaging_count(self):
        """no help available"""
        scpi_command = ":AVER:COUN?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_averaging_mode(self):
        """no help available"""
        scpi_command = ":AVER:MODE?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_continuous_sweep(self):
        """no help available"""
        scpi_command = ":INIT:CONT?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='bool')
        return value

    def query_current_trace_data(self):
        """no help available"""
        scpi_command = ":CALC:DATA:SDAT?"
        return self.query_values(scpi_command)

    def query_display_format(self):
        """no help available"""
        scpi_command = ":CALC:FORM?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_f_start(self):
        """no help available"""
        scpi_command = ":FREQ:START?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_f_stop(self):
        """no help available"""
        scpi_command = ":FREQ:STOP?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_if_bandwidth(self):
        """no help available"""
        scpi_command = ":BWID?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_instrument(self):
        """no help available"""
        scpi_command = ":INST?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_sweep_n_points(self):
        """no help available"""
        scpi_command = ":SWE:POIN?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_sweep_time(self):
        """no help available"""
        scpi_command = ":SWE:TIME?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='float')
        return value

    def query_trace_count(self):
        """no help available"""
        scpi_command = ":CALC:PAR:COUN?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='int')
        return value

    def query_trace_display_config(self):
        """no help available"""
        scpi_command = ":DISP:WIND:SPL?"
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value

    def query_trace_measurement(self, trace=1):
        """no help available"""
        scpi_command = scpi_preprocess(":CALC:PAR{:}:DEF?", trace)
        value = self.query(scpi_command)
        value = process_query(value, csv=False, strip_outer_quotes=True, returns='str')
        return value
