#import mwavepy as mv
from mwavepy.virtualInstruments import vna as mv_vna


class Private():
	def __init__(self):
		self.network = None
		self.vna = None
		self.ax = None
private = Private()

def load_vna(text_gpib_address,choice_vna_model,**kwargs):
	
	vna_class_dict = {\
		'HP8510C':mv_vna.HP8510C,\
		'HP8720':mv_vna.HP8720,\
		}
	#vna = text_gpib_address.value
	private.vna = vna_class_dict[choice_vna_model.value](int(text_gpib_address.value))

def get_s11(mpl_plot,text_status, **kwargs):
	text_status.value = 'Waiting..'
	private.network = private.vna.s11
	text_status.value = 'Plotting..'
	private.ax = mpl_plot.get_axes()
	private.network.plot_s_db(0,0, ax = private.ax)
	mpl_plot.set_title('asdf')
	mpl_plot.show()
	text_status.value = 'Done.'

def open_file(file_dialog, file_dialog_result,**kwargs):
	file_dialog_result.value = file_dialog.open()
	
def new_file(file_dialog, file_dialog_result,**kwargs):
	file_dialog_result.value = file_dialog.save()

def save_file(file_dialog, file_dialog_result,**kwargs):
	if private.network is not None:
		private.network.write_touchstone(file_dialog_result.value)
