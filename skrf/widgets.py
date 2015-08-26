from __future__ import division
from past.utils import old_div

from IPython.html.widgets import interact, interactive

from matplotlib.pyplot import *
from traits.api import *
from .network import Network
from .media import Media
from .constants import distance_dict

class TimeGate(HasTraits):
    ntwk =  Instance(Network)
    med = Instance(Media)
    distance_unit = Enum(list(distance_dict.keys()))
    gate_center = Float(1.0)
    gate_width= Float(.5)
    
    @property
    def deembeded_ntwk(self):
        line= self.med.line(self.gate_center, unit=self.distance_unit,
                            name=self.ntwk.name)
        
        return line.inv**self.ntwk
    @property
    def gated_ntwk(self):
        gw= self.gate_width * 1e-9
        gated =self.deembeded_ntwk.time_gate(-gw,gw)
        return gated 
            
    def _repr_html_(self):
        fig=figure(figsize=(8,4))
        def f(gate_center=0, gate_width=.1, distance_unit='cm'):
            
            self.distance_unit=distance_unit
            self.gate_center=gate_center
            self.gate_width=gate_width
            
            
            subplot(121)
            self.deembeded_ntwk.plot_s_db_time()
            axvline(old_div(-gate_width,2.),color='k')
            axvline(old_div(gate_width,2.),color='k')
            ylim(-100,20)
            
            
            
            subplot(122)
            self.deembeded_ntwk.plot_s_db()
            self.gated_ntwk.plot_s_db()
        
            #ylim(-10,00)
        
            #tight_layout()
        
        gate_max = self.ntwk.frequency.t.max()*1e9*2
        return interact(f, gate_center = (0,20,.1),
                              gate_width=(0,gate_max,.01),
                              )

    

