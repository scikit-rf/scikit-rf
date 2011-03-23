#import mwavepy as mv
from mwavepy.virtualInstruments.lifetimeProbeTester import LifeTimeProbeTester
from mwavepy import Network 
import pdb
import multiprocessing
from time import sleep 
class Private():
	def __init__(self):
		self.lpt = None
		self.logger = multiprocessing.get_logger()
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
	

def update_plot (mpl_plot, **kwargs):
		lpt= private.lpt
		mpl_plot.clear()
		lpt.read_loadcell_and_stage_position()
		mpl_plot.plot(range(len(lpt.force_history)), lpt.force_history)
		mpl_plot.set_title('Force Monitor')
		mpl_plot.axis('tight')
		mpl_plot.set_ylabel('Force (mN)')
		mpl_plot.show()
		sleep(.1)
		
def monitor( timer, **kwargs):
	print (timer.running)
	if not timer.running:
		timer.start(interval = .1)
	else:
		timer.stop()
	
