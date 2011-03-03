#import mwavepy as mv
from mwavepy.virtualInstruments import vna as mv_vna

import pdb
import multiprocessing

class Private():
	def __init__(self):
		self.network = None
		self.vna = None
		self.ax = None
		self.logger = multiprocessing.get_logger()
		self.plot_format = 'Magnitude(dB)'
private = Private()

def load_vna(text_gpib_address,choice_vna_model,**kwargs):
	
	vna_class_dict = {\
		'HP8510C':mv_vna.HP8510C,\
		'HP8720':mv_vna.HP8720,\
		}
	#vna = text_gpib_address.value
	private.vna = vna_class_dict[choice_vna_model.value]\
			(int(text_gpib_address.value))


def get_one_port(m,n, mpl_plot, **kwargs):
	private.logger.info('Getting S%i%i'%(m,n))
	private.network = private.vna.__getattribute__('s%i%i'%(m,n))

def clear_plot():
	raise (NotImplementedError)

def update_plot(mpl_plot, radio_plot_type,**kwargs):
	raise(NotImplementedError)
	#FUTURE FIX: this will work once bug is fixed in pythics
	#private.ax = mpl_plot.get_axes()
	#private.network.plot_s_db(m-1,n-1, ax = private.ax)
	private.plot_format = radio_plot_type.value
	print private.plot_format
	mpl_plot.set_title('asdf')
	mpl_plot.show()
	
	
def get_two_port():
	raise(NotImplementedError)
	
def get_switch_terms():
	raise(NotImplementedError)

	
def get_s11(mpl_plot, **kwargs):
	return get_one_port(1,1,mpl_plot)
def get_s12(mpl_plot, **kwargs):
	return get_one_port(1,1,mpl_plot)
def get_s21(mpl_plot, **kwargs):
	return get_one_port(1,1,mpl_plot)
def get_s22(mpl_plot, **kwargs):
	return get_one_port(1,1,mpl_plot)

def open_file(file_dialog, file_dialog_result,**kwargs):
	file_dialog_result.value = file_dialog.open()
	
def new_file(file_dialog, file_dialog_result,**kwargs):
	file_dialog_result.value = file_dialog.save()

def save_file(file_dialog, file_dialog_result,**kwargs):
	if private.network is not None:
		private.network.write_touchstone(file_dialog_result.value)

