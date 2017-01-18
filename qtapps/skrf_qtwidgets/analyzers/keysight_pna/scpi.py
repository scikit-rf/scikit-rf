"""
scpi.py provides a small SCPI pre-processor that makes writing driver code a little bit more clear, and allows
the user to write different SCPI template strings that are suited to their instrument as they might be
slightly different from model to model.

TODO:
- Consider whether the pre-processor should be in the main analyzers folder rather than the keysight folder.
    ** I am wholly unfamiliar with other analyzer syntax, and don't know if they follow the same patterns, nor am
    ** I familiar with the SCPI standard enough to know if

*** USAGE ***:
SCPI commands have a command tree and an argument set separated by a space:
:COMMAND:TREE arg1,arg2

This structure is the same for query and write, though queries have a '?' attached to the end of the command tree
Some commands are read-only, some are write-only and some are read-write.  Not all accept arguments, but the call
structure is always the same.

preprocess_scpi(command, kwargs**):
    ...
    return scpi_command, scpi_args

command is a sensible dictionary key to look up a commonly used SCPI command:
e.g. "create_meas": ":CALCulate<cnum>:PARameter:DEFine:EXTended <'mname'>,<'param'>",

the arguments in angle brackets -- <kwarg>, or <'quoted_kwarg'> --represent the arguments needed to complete the SCPI
command but unknown until usage.  The arguments are supplied as keyword arguments to a function
and the preprocessor strips out the tags and replaces them with.  Some arguments are supplied in SCPI surrounded
by quotes.  In this case the quotes (either ', or ") are supplied in the template string and the preprocessor
will strip out the brackets and the kwarg name, and leave the quotes in place.

the template string provides everything the user needs to know about structuring the code
For example to query the instrument about the start frequency, you need a scpi command which is obtained
with:
>>>preprocess("start_frequency", cnum=1, num=1e7)
returns:
    [":SENSe1:FREQuency:STARt", "10000000.0"]

if no keyword argument is supplied in the command tree the variable simply is an empty string.  In the arguments
if no keyword argument is supplied, then the argument is simply omitted.  Currently there is no specification if a
variable is required or not.  It is up to the user to know whether or not the arguments are necessary.
>>>preprocess("start_frequency")
returns:
    [":SENSe:FREQuency:STARt" ""]
which can easily be converted into the valid query string: ":SENSe:FREQuency:STARt?" by simply adding a questionmark

usually this would be used with a helper function, e.g. write/query, that would convert this list into a proper
query command and then return the result:

def write(command, **kwargs):
    scpi_string = " ".join(preprocess_scpi(command, **kwargs)).strip()
#     resource.write(scpi_string)
    print(scpi_string)

def query(command, **kwargs):
    scpi_string = "? ".join(preprocess_scpi(command, **kwargs)).strip()
#     return resource.query(scpi_string)
    print(scpi_string)

***TEMPLATE STRING CONVENTION***
Whenver possible I try to adhere to the Keysight documentation for what to name keyword arguments:
It is important to check because the documentation can be a little inconsisten in places.  For example
numeric arguments are usually referenced with <num> while measurement numbers are <n> rather than <num> or

<num> --> argument supplied as a number
<cnum> --> channel number
<tnum> --> trace number
<wnum> --> window number
<n> --> numeric argument, for example, measurement number
<onoff> --> "on" or "off" string for parameters that are set to on or off, e.g. averaging
<char> --> argument supplied as a string (data formats for example, :FORM:DATA <char> where char="REAL, 64"
<'Mname'> --> measurement name, usually in quotes
<'param'> --> parameter, usually in quotes, e.g. "S11"
<'ports'> --> argument specifying a list of ports, usually in quotes.  the argument can be a list of ints, or a
              preformatted string, for example ports=(1,2) and ports="1,2" are both valid

below are some examples:

create a new S11 measurement
>>>write("create_meas", cnum=1, Mname="CH1_S11_1", param="S11")

what is the active channel:
>>>query("active_channel")

get snp data from ports 1 and 2
>>>query("snp_data", cnum=1, ports=(1,2))

set the start frequency to 1 GHz
>>>write("start_frequency", num=1e9)
"""

import re

cmd_pattern = re.compile('<(\w+)>', re.ASCII)
arg_pattern = re.compile('<([\'"]?\w+[\'"]?)>', re.ASCII)
kwarg_pattern = re.compile('[\'"]?(\w+)[\'"]?', re.ASCII)


def preprocess(command_set, command, **kwargs):
    base_command = command_set[command].split(" ")
    cmd = base_command[0]
    if len(base_command) > 1:
        args = base_command[1]
    else:
        args = ""

    split_command = cmd_pattern.split(cmd)
    for i in range(1, len(split_command), 2):
        split_command[i] = str(kwargs.get(split_command[i], ""))
    scpi_command = "".join(split_command)

    split_args = arg_pattern.findall(args)
    new_args = list()
    for arg in split_args:
        kwarg = kwarg_pattern.search(arg).group(1)
        value = kwargs.get(kwarg, "")
        if value:
            if type(value) in (list, tuple):
                value = ",".join(map(str, value))
            else:
                value = str(value)
            new_args.append(arg.replace(kwarg, value))
    scpi_args = ",".join(new_args)
    return scpi_command, scpi_args
