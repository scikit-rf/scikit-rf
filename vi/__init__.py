
'''
.. module:: skrf.vi
========================================================
virtual instruments (:mod:`skrf.vi`)
========================================================

This module holds Virtual Instruments.


.. automodule:: skrf.vi.vna
.. automodule:: skrf.vi.sa
.. automodule:: skrf.vi.stages


Creating A Driver 
+++++++++++++++++++


SCPI Commands
------------------

To learn about SCPI, you can read the `IVI website`_, or `Wikipedia`_.  For this tutorial you only need to know that
SCPI comamnds are provided in a tree structure where similar commands are grouped in branches of the tree.  All scpi
commands are either 'set' commands where the user writes something to instrument or 'query' commands where the user
is requesting data or information from the instrument and the syntax is like this:

.. _IVI website: http://www.ivifoundation.org/scpi/
.. _Wikipedia: https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments

::

    :COMMAND:TREE:BRANCH(?) arg1,arg2

Each branch on the command tree is separated by a colon.  Query commands provide a question mark at the end of the
command tree and arguments are provided as needed after the command.  Here is an example of commands grouped in a
branch:

::

    :CALC1:PAR:DEF:EXT 'my_meas','S11'  # create a new S11 measurement on channel 1
    :CALC2:PAR:CAT?  # return a catalog of existing measurements on channel2
    :CALC:PAR:SEL 'my_meas'  # select 'my_meas' on the default channel (channel1)

With this structure in mind, scikit-rf is adopting a tool where SCPI commands are described in a yaml file, which
allows for a very compact and universal way to describe the SCPI command tree.  The yaml file is then parsed in order
to generate a python script file that accesses the SCPI commands through object methods.  See the snippet below from
the keysight_pna_scpi.yaml file:

::

    COMMAND_TREE:
      CALC<cnum=1>:
        DATA:
          command: {name: data, set: "<fmt=SDATA>,<data=None>", query_values: <fmt=SDATA>}
          branch:
            SNP:
              PORT: {name: snp_data, query_values: "'<ports=(1, 2)>'"}
        PAR:
          CAT:
            EXT: {name: meas_name_list, query: "", csv: True}
          DEF:
            EXT: {name: create_meas, set: "'<mname>','<param>'"}
          DEL: {name: delete_meas, set: "'<mname>'"}
          SEL: {name: selected_meas, set: "'<mname>'", query: ""}
          MNUM: {name: selected_meas_by_number, set: <mnum>, query: "", returns: int}
        FORM: {name: display_format, set: <fmt=MLOG>, query: ""}

In the command tree, all nodes are either:

1. A new branch of the command tree
2. A command
3. Both a command and a branch, as in :CALC:DATA

In the case that a node is both a branch and a command we need to explicitly mark the mappings of each with 'branch'
and 'command'.  Otherwise, the presence of a mapping item 'name', which is a rational name for the command, indicates
that the node is a command.  Absent that, the node is assumed to be a branch and is further parsed for more nodes.

yaml Command Syntax
------------------------

For a node that is a command, there are only 2 requirements:

1. 'name' must be provided
2. at least one of 'set', 'query' or 'query_values' must be provided.

Optionally you may also specify the following options in the command for query processing:

3. 'returns' : default 'str'; specify the return type (int, float, str, bool) and the value returned from the functions will be converted into the appropriate python type.
4. 'csv' : defalt False; if the return value is a csv list, parse it as such and return a python list where the members are individually converted to the type specified by 'returns'
5. 'strip_outer_quotes' : default True; check if the return value is a string literal encapsulated in quotes and strip these out for proper parsing of the return value.

If a command is read or write only, then you specify only the query or set items.  If it is read/write, then you
specify both.  query_values is just a query command that returns a data set, and will then use the pyvisa
query_values convenience function that will automatically parse the returned data and place it in a list or numpy
array of floats.

The set/query items are merely the string of arguments that must be provided to the command.  For example, the
'create_meas' example above looked like this:

::

    :CALC1:PAR:DEF:EXT 'my_meas','S11'

If we placed this command in a python function and called it, it might look something like this:

::

    class SCPI:
        def __init__(self, pyvisa_resource):
            self.resource = pyvisa_resource

        def set_create_meas(self, cnum=1, mname="", param="")
            scpi_command = ":CALC{:}:PAR:DEF:EXT '{:}','{:}'".format(cnum, mname, param)
            self.resource.write(scpi_command)

    vna = SCPI(pyvisa_resource)
    vna.set_create_meas(1, "my_meas", "S11")

This illustrates what is going on with the yaml file.  If we isolate only the create_meas command in the command tree
the yaml mapping looks like this:

::

    CALC<cnum=1>:
      PAR:
        DEF:
          EXT: {name: create_meas, set: "'<mname>','<param>'"}

the python function ``set_create_meas`` is simply the write method of the create_meas SCPI command.  The 'set' item
in the yaml mapping is the string: ``"'<mname>','<param>'"`` which is the the required arguments for the SCPI command.
The function has three keyword arguments, cnum, mname and param which are enclosed in <> brackets.  If an argument
has a default parameter (int, float or string) then it can be specified with an '=' sign.

Using the SCPI functions to write a Driver
---------------------------------------------------

The parser outputs a python file with a SCPI class.  The current model for using this class is to initialize a VNA
class, which in turn initialized a pyvisa resource.  The VNA object then initializes a SCPI object and assigns it to
self.scpi.  See the abbreviated code from the keysight_pna.py PNA class __init__ method:

::

    def __init__(self, address=DEFAULT_VISA_ADDRESS, **kwargs):
        super(PNA, self).__init__(address, **kwargs)
        self.resource.timeout = kwargs.get("timeout", 2000)
        self.scpi = keysight_pna_scpi.SCPI(self.resource)

Subsequently, the low-level instrument commands are contained with the .scpi namespace and can be called in the
following manner inside vna object methods:

::

    f_start = self.scpi.query_f_start(channel)
    self.scpi.set_f_start(channel, f_start)




'''
__test__=False
all = ['vna','sa','stages']


