

from traits.api import *
from traitsui.api import *#View, Item, ButtonEditor, Group, HSplit,
from traitsui.menu import *

import numpy as npy
import os
import pylab as plb
import skrf as rf




class PlotTool(HasTraits):
    save_dir = Directory()
    clf = Button('Clear')
    newfig = Button('New')
    close_all = Button('Close All')
    tight_layout = Button('Tight Layout')
    draw_show = Button('Draw/Show')
    save_all_figs = Button('Save All')
    plot_format =  Enum('png',['svg','eps','jpg','png'],label='Format')
    title = String('')
    apply_title = Button('Title')
    def _clf_fired(self):
        plb.clf()
        plb.draw()
        plb.show()

    def _newfig_fired(self):
        plb.figure()
        plb.draw()
        plb.show()

    def _close_all_fired(self):
        plb.close('all')

    def _tight_layout_fired(self):
        plb.tight_layout()
        plb.draw()
        plb.show()

    def _draw_show_fired(self):
        plb.draw()
        plb.show()

    def _save_all_figs_fired(self):
        rf.save_all_figs(dir=self.save_dir, format = [self.plot_format])
        plb.ion()

    def _apply_title_fired(self):
        plb.title(self.title)
        plb.draw();plb.show()

    view = View(
        Group(
            HSplit(
                Item('newfig', label='Figure\t\t'),
                Item('clf', show_label=False),
                Item('close_all', show_label=False),
                Item('tight_layout', show_label=False),
                Item('draw_show', show_label=False),
                ),
            HSplit(
                Item('apply_title', label='Formatting\t'),
                Item('title', show_label=False),
                ),
            HSplit(
                Item('save_all_figs', label='Save\t\t'),
                Item('plot_format'),
                Item('save_dir', label='Directory'),

                )
            )
        )
