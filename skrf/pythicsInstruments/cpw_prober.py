#import skrf as rf
from skrf.vi.lifetimeProbeTester import LifeTimeProbeTester
from skrf import Network
import pdb
import multiprocessing
from time import sleep
import numpy as npy


class Private():
    def __init__(self):
        self.lpt = None
        self.logger = multiprocessing.get_logger()
        self.counter= 0
private = Private()




def connect(**kwargs):
    try:
        private.lpt = LifeTimeProbeTester()
        private.logger.info('prober connected.')
    except:
        private.logger.error('prober failed to connect')
def close(**kwargs):
    private.lpt.close()

def contact(**kwargs):
    private.lpt.contact()

def byebye(**kwargs):
    private.move_apart_fast(100)
    private.lpt.close()

def enable_settings(text_contact_force,text_step_increment,\
        text_position_upper_limit, text_stage_delay,**kwargs):
    private.lpt.contact_force = float(text_contact_force.value)
    private.lpt.step_increment = float(text_step_increment.value)*1e-3
    private.lpt.position_upper_limit = float(text_position_upper_limit.value)
    private.lpt.stage.delay = float(text_stage_delay.value)


def update_plot (chart, **kwargs):
    private.counter +=1
    lpt= private.lpt
    force  = lpt.read_loadcell()
    chart.append(npy.array([private.counter, force]))

def stop(timer, **kwargs):
    timer.stop()
def monitor( timer, chart, **kwargs):
    chart.span = 100
    if not timer.running:
        timer.start(interval = .3)
    else:
        timer.stop()
