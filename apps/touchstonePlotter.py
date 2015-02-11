from traits.api import *
from traitsui.api import *#View, Item, ButtonEditor, Group, HSplit,
from traitsui.menu import *

import pylab as plb
import skrf as rf

from plotTool import PlotTool

class TouchstonePlotter(HasTraits):
    cur_dir = Directory('')
    ntwk_dict = Property(depends_on = 'cur_dir')
    available_ntwk_list = Property(depends_on = 'cur_dir')
    active_ntwk_name = Enum(values='available_ntwk_list')
    active_ntwk = Property(depends_on = 'active_ntwk_name')

    ports = Property(depends_on = 'active_ntwk_name')
    m = Enum(values = 'ports')
    n = Enum(values = 'ports')
    param = Enum('s',['s','z','y','a'])
    form = Enum('Mag (dB)', ['Mag (dB)','Phase (deg)','Smith',
        'Mag (lin)', 'Phase (lin)','Real','Imag'])

    plot_button = Button('Plot ')

    plot_tool = Instance(PlotTool,())

    @cached_property
    def _get_ntwk_dict(self):
        try:
            return rf.ran(self.cur_dir)
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

    def _plot_button_fired(self):
        form_dict = {
            'Mag (dB)': 'db',
            'Mag (lin)': 'mag',
            'Phase (deg)': 'deg',
            'Phase Unwrapped (deg)': 'deg_unwrap',
            'Smith': 'smith',
            'Real': 're',
            'Imag': 'im',
            }
        form = form_dict[self.form]
        m = None if self.m == 'All' else self.m
        n = None if self.n == 'All' else self.n


        self.active_ntwk.__getattribute__(\
            'plot_{}_{}'.format(self.param, form))(m=m, n=n)


        plb.draw();plb.show()

    view = View(
        Item('cur_dir'),
        HSplit(
            Item('active_ntwk_name', show_label=False),
            Item('plot_button', show_label=False),
            Item('param', show_label=False),
            Item('form', show_label=False),
            Item('m'),
            Item('n'),

            ),
        '_',


        Item('plot_tool',style='custom',show_label=False),

        resizable=True,
        width = 800, height = 300,
        title = 'Skrf Touchstone plotter')


if __name__  == '__main__':
    a = TouchstonePlotter()
    a.configure_traits()
