from traits.api import *
from traitsui.api import *#View, Item, ButtonEditor, Group, HSplit,
from traitsui.menu import *

import pylab as plb
import skrf as rf
import os
from plotTool import PlotTool

class Network2Spreadsheet(HasTraits):
    input_dir = Directory('')
    output_dir = Directory('')

    output_filename = String('output.xls')
    ntwk_dict = Property(depends_on = 'input_dir')
    available_ntwk_list = Property(depends_on = 'input_dir')
    active_ntwk_name = Enum(values='available_ntwk_list')
    active_ntwk = Property(depends_on = 'active_ntwk_name')

    export_network = Button('Export Network')
    export_all_networks = Button('Export All Networks')
    form = Enum('dB/deg',['dB/deg','mag/deg','real/imag'])

    @cached_property
    def _get_ntwk_dict(self):
        try:
            return rf.ran(self.input_dir)
        except(OSError):
            return {}

    def _get_available_ntwk_list(self):
        return  self.ntwk_dict.keys()

    def _get_active_ntwk(self):
        try:
            return self.ntwk_dict[self.active_ntwk_name]
        except:
            pass

    def _get_ports(self):
        try:
            return ['All']+range(self.active_ntwk.nports)
        except:
            return []

    def _export_network_fired(self):
        form_dict = {
            'dB/deg':'db',
            'mag/deg':'ma',
            'real/imag':'ri',
            }
        form = form_dict[self.form]
        file_name = os.path.join(self.output_dir, self.output_filename)

        rf.io.general.network_2_spreadsheet(
            self.active_ntwk,
            file_name = file_name,
            form = form )
        print(('{} --> {}'.format(self.active_ntwk_name,file_name)))

    def _export_all_networks_fired(self):
        form_dict = {
            'dB/deg':'db',
            'mag/deg':'ma',
            'real/imag':'ri',
            }
        form = form_dict[self.form]
        file_name = os.path.join(self.output_dir, self.output_filename)
        ns = rf.NS(self.ntwk_dict)
        rf.io.general.networkset_2_spreadsheet(
            ns,
            file_name=file_name,
            form = form
            )
        for k in [ '{} --> {}'.format(k.name,file_name) for k in ns.ntwk_set]:
            print(k)


    view = View(
        Group(
            Item('input_dir', label='Input Dir'),
            Item('output_dir', label='Output Dir'),
            Item('output_filename', label='Output Filename'),
            HSplit(
                Item('active_ntwk_name', show_label=False),
                Item('export_network',show_label=False),
                Item('export_all_networks',show_label=False),
                Item('form')

                ),
            ),
        resizable=True,
        width = 800, height = 300,
        title = 'Skrf Spreadsheet Exporter'
        )


if __name__  == '__main__':
    a = Network2Spreadsheet()
    a.configure_traits()
