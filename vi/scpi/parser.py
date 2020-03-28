import yaml
import os.path
import re
import sys

kwarg_pattern = re.compile('<([a-zA-Z0-9_]+=?[a-zA-Z0-9_, \(\)\'\"]*)>')


def to_string(value):
    if type(value) in (list, tuple):
        return ",".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)


def indent(text, levels, pad="    "):
    padding = "".join([pad] * levels)
    return padding + text.replace("\n", "\n" + padding)


def isnumeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def process_kwarg_default(value):
    if value[0] + value[-1] in ("()", "[]"):
        return value  # default is a list or tuple, assume values were entered correctly
    elif value[0] + value[-1] in ('""', "''"):
        return value  # value is an explicit string, return as is
    elif isnumeric(value):
        return str(value)
    else:
        return '"{:}"'.format(value)  # treat as string, must have quotes to use as a kwarg default value


def parse_command_string(command_string):
    args = kwarg_pattern.findall(command_string)
    kwargs = list()
    for arg in args:
        if "=" in arg:
            kwarg, val = arg.split("=")
            val = process_kwarg_default(val)
        else:
            kwarg = arg
            val = '""'
        kwargs.append([kwarg, val])
    kwargs_string = "".join([', ' + kwarg + "=" + val for kwarg, val in kwargs])

    if len(args) > 0:
        command_base = kwarg_pattern.sub("{:}", command_string)
        args_string = ", ".join(kwarg for kwarg, val in kwargs)
        scpi_command = 'scpi_preprocess("{:}", {:})'.format(command_base, args_string)
    else:
        scpi_command = '"{:}"'.format(command_string)

    return kwargs_string, scpi_command


def parse_write_values_string(command_string):
    """
    parse the command string for the write_values scpi command which is a little different than the others

    Parameters
    ----------
    command_string : str
        the input string that will be parsed for keyword arguments
    """
    args = kwarg_pattern.findall(command_string)
    kwargs = list()
    for arg in args:
        if "=" in arg:
            kwarg, val = arg.split("=")
            val = process_kwarg_default(val)
        else:
            kwarg = arg
            val = '""'
        kwargs.append([kwarg, val])
    kwargs[-1][1] = "None"  # data_values will be set to None as default
    kwargs_string = "".join([', ' + kwarg + "=" + val for kwarg, val in kwargs])

    command_string = command_string.replace("<{:}>".format(args[-1]), "")
    command_base = kwarg_pattern.sub("{:}", command_string)
    args_string = ", ".join(kwarg for kwarg, val in kwargs[:-1])  # last arg is the data we pass in
    scpi_command = 'scpi_preprocess("{:}", {:})'.format(command_base, args_string)

    return kwargs_string, scpi_command, kwargs[-1][0]


def generate_set_string(command, command_root):
    command_string = " ".join((command_root, to_string(command["set"]))).strip()
    kwargs_string, scpi_command = parse_command_string(command_string)

    if 'help' not in command:
        command['help'] = 'no help available'
    command['help'] = command['help'].replace('\t', '    ')

    function_string = \
"""def set_{:s}(self{:}):
    \"\"\"{:s}\"\"\"
    scpi_command = {:}
    self.write(scpi_command)""".format(command['name'], kwargs_string, command['help'], scpi_command)

    return function_string


def generate_set_values_string(command, command_root):
    command_string = " ".join((command_root, to_string(command["set_values"]))).strip()
    kwargs_string, scpi_command, data_variable = parse_write_values_string(command_string)

    if 'help' not in command:
        command['help'] = 'no help available'
    command['help'] = command['help'].replace('\t', '    ')

    function_string = \
"""def set_{:s}(self{:}):
    \"\"\"{:s}\"\"\"
    scpi_command = {:}
    self.write_values(scpi_command, {:})""".format(
            command['name'], kwargs_string, command['help'], scpi_command, data_variable)

    return function_string


def generate_query_string(command, command_root):
    command_string = "? ".join((command_root, to_string(command["query"]))).strip()
    kwargs_string, scpi_command = parse_command_string(command_string)

    if 'help' not in command:
        command['help'] = 'no help available'
    command['help'] = command['help'].replace('\t', '    ')

    converter = command.get('returns', "str")
    valid_converters = ("int", "str", "float", "bool")
    if converter not in valid_converters:
        raise ValueError("""error in processing command {:}
        returns value '{:}' is invalid
        must be one of {:}
        """.format(command_string, converter, ", ".join(valid_converters)))

    pre_line = ""
    strip_outer_quotes = bool(command.get("strip_outer_quotes", True))
    csv = bool(command.get('csv', False))
    if csv or strip_outer_quotes or converter != "str":
        pre_line = \
            "\n    value = process_query(value, csv={:}, strip_outer_quotes={:}, returns='{:}')".format(
                csv, strip_outer_quotes, converter
            )

    function_string = \
"""def query_{:s}(self{:}):
    \"\"\"{:s}\"\"\"
    scpi_command = {:}
    value = self.query(scpi_command){:}
    return value""".format(command['name'], kwargs_string, command['help'], scpi_command, pre_line)

    return function_string


def generate_query_values_string(command, command_root):
    command_string = "? ".join((command_root, to_string(command["query_values"]))).strip()
    kwargs_string, scpi_command = parse_command_string(command_string)

    if 'help' not in command:
        command['help'] = 'no help available'
    command['help'] = command['help'].replace('\t', '    ')

    function_string = \
"""def query_{:s}(self{:}):
    \"\"\"{:s}\"\"\"
    scpi_command = {:}
    return self.query_values(scpi_command)""".format(
            command['name'], kwargs_string, command['help'], scpi_command)

    return function_string


def parse_branch(branch, set_strings=None, query_strings=None, query_value_strings=None, root=""):
    if set_strings is None:
        set_strings = list()
    if query_strings is None:
        query_strings = list()
    if query_value_strings is None:
        query_value_strings = list()

    for key, value in branch.items():
        command_root = root + ":" + key
        command = None
        branch = None

        try:
            if "name" in value.keys():
                command = value
            elif "command" in value.keys():
                command = value["command"]
                branch = value["branch"]
            else:
                branch = value
        except Exception as e:
            print(key, value)
            raise Exception(e)

        if command:
            if "set" in command.keys():
                set_strings.append(generate_set_string(command, command_root))
            if "set_values" in command.keys():
                set_strings.append(generate_set_values_string(command, command_root))
            if "query" in command.keys():
                query_strings.append(generate_query_string(command, command_root))
            if "query_values" in command.keys():
                query_strings.append(generate_query_values_string(command, command_root))

        if branch:
            parse_branch(branch, set_strings, query_strings, query_value_strings, command_root)

    return set_strings, query_strings

header_string = """import re

null_parameter = re.compile(",{2,}")  # detect optional null parameter as two consecutive commas, and remove
converters = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda x: bool(int(x)),
}"""

string_converter = """def to_string(value):
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
        return str(value)"""

scpi_preprocessor = """def scpi_preprocess(command_string, *args):
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = to_string(arg)
    cmd = command_string.format(*args)
    return null_parameter.sub(",", cmd)"""

query_processor = """def process_query(query, csv=False, strip_outer_quotes=True, returns="str"):
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

    return query"""

class_header = """class SCPI(object):
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
"""


def parse_yaml_file(driver_yaml_file):
    driver = os.path.splitext(driver_yaml_file)[0] + ".py"

    driver_template = None
    with open(driver_yaml_file, 'r', encoding='utf-8') as yaml_file:
        driver_template = yaml.load(yaml_file)

    sets, queries = parse_branch(driver_template["COMMAND_TREE"])

    driver_str = "\n\n\n".join((header_string, string_converter, scpi_preprocessor, query_processor)) + "\n\n\n"
    driver_str += class_header

    for s in sorted(sets, key=str.lower):
        driver_str += "\n" + indent(s, 1) + "\n"
    for q in sorted(queries, key=str.lower):
        driver_str += "\n" + indent(q, 1) + "\n"

    with open(driver, 'w', encoding='utf8') as scpi_driver:
        scpi_driver.write(driver_str)


if __name__ == "__main__":
    driver_yaml_file = os.path.abspath(sys.argv[1])
    parse_yaml_file(driver_yaml_file)
