import yaml
import os.path
import re
import sys

kwarg_pattern = re.compile('<(\w+)>', re.ASCII)


def to_string(value):
    if type(value) in (list, tuple):
        return ",".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)


def generate_set_string(command, command_root):
    command_string = " ".join((command_root, to_string(command["set"]))).strip()

    args = kwarg_pattern.findall(command_string)
    args_string = ", ".join(args)
    kwargs_string = "".join([', ' + kwarg + '=""' for kwarg in args])

    if len(args) > 0:
        command_base = kwarg_pattern.sub("{:}", command_string)
        scpi_command = "scpi_preprocess(\"{:}\", {:})".format(command_base, args_string)
    else:
        scpi_command = '"{:}"'.format(command_string)

    function_string = \
        """def set_{:s}(self{:}):
            scpi_command = {:}
            self.resource.write('{:}'.format(scpi_command))""".format(
            command['name'], kwargs_string, scpi_command, '{:}')

    return function_string


def indent(text, levels, pad="    "):
    padding = "".join([pad] * levels)
    return padding + text.replace("\n", "\n" + padding)


def generate_query_string(command, command_root):
    command_string = "? ".join((command_root, to_string(command["query"]))).strip()

    args = kwarg_pattern.findall(command_string)
    args_string = ", ".join(args)
    kwargs_string = "".join([', ' + kwarg + '=""' for kwarg in args])

    if len(args) > 0:
        command_base = kwarg_pattern.sub("{:}", command_string)
        scpi_command = "scpi_preprocess(\"{:}\", {:})".format(command_base, args_string)
    else:
        scpi_command = '"{:}"'.format(command_string)

    converter = command.get('returns', "str")
    valid_converters = ("int", "str", "float", "bool")
    if converter not in valid_converters:
        raise ValueError("""error in processing command {:}
        returns value '{:}' is invalid
        must be one of {:}
        """.format(command_string, converter, ", ".join(valid_converters)))

    is_csv = command.get('csv', False)
    if is_csv is True:
        if converter != "str":
            return_line = "return list(map({:}, value))".format(converter)
        else:
            return_line = "return value.split(',')"
    else:
        if converter != "str":
            return_line = "return {:}(value)".format(converter)
        else:
            return_line = "return value"

    function_string = \
        """def query_{:s}(self{:}):
            scpi_command = {:}
            value = self.resource.query(scpi_command)
            {:}""".format(command['name'], kwargs_string, scpi_command, return_line)

    return function_string


def generate_query_values_string(command, command_root):
    command_string = "? ".join((command_root, to_string(command["query_values"]))).strip()

    args = kwarg_pattern.findall(command_string)
    args_string = ", ".join(args)
    kwargs_string = "".join([', ' + kwarg + '=""' for kwarg in args])

    if len(args) > 0:
        command_base = kwarg_pattern.sub("{:}", command_string)
        scpi_command = "scpi_preprocess(\"{:}\", {:})".format(command_base, args_string)
    else:
        scpi_command = '"{:}"'.format(command_string)

    function_string = \
        """def query_{:s}(self{:}):
            scpi_command = {:}
            return self.resource.query_values(scpi_command)""".format(
            command['name'], kwargs_string, scpi_command)

    return function_string


def parse_branch(branch, set_strings=[], query_strings=[], query_value_strings=[], root=""):
    for key, value in branch.items():
        command_root = root + ":" + key
        command = None
        branch = None

        if "name" in value.keys():
            command = value
        elif "command" in value.keys():
            command = value["command"]
            branch = value["branch"]
        else:
            branch = value

        if command:
            if "set" in command.keys():
                set_strings.append(generate_set_string(command, command_root))
            if "query" in command.keys():
                query_strings.append(generate_query_string(command, command_root))
            if "query_values" in command.keys():
                query_value_strings.append(generate_query_values_string(command, command_root))

        if branch:
            parse_branch(branch, set_strings, query_strings, query_value_strings, command_root)

    return set_strings, query_strings, query_value_strings

string_converter = """def to_string(value):
    if type(value) in (list, tuple):
        return ",".join(map(str, value))
    elif value is None:
        return ""
    else:
        return str(value)"""

scpi_preprocessor = """def scpi_preprocess(command_string, *args):
    args_list = list(args)
    for i, arg in enumerate(args_list):
        args_list[i] = to_string(arg)
    return command_string.format(*args_list)"""

class_header = """class SCPI(object):
"""

driver_yaml_file = os.path.abspath(sys.argv[1])
driver_dir = os.path.dirname(driver_yaml_file)
driver = os.path.splitext(driver_yaml_file)[0] + ".py"

with open(driver_yaml_file, 'r') as yaml_file:
    driver_template = yaml.load(yaml_file)
sets, querys, arrays = parse_branch(driver_template)

driver_str = string_converter + "\n\n\n" + scpi_preprocessor + "\n\n\n"
driver_str += class_header

for s in sets:
    driver_str += indent(s, 1) + "\n\n"
for q in querys:
    driver_str += indent(q, 1) + "\n\n"
for a in arrays:
    driver_str += indent(a, 1) + "\n\n"

with open(driver, 'w') as scpi_driver:
    scpi_driver.write(driver_str)
