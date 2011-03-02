#import mwavepy as mv
from mwavepy.virtualInstruments import vna as mv_vna


global vna, network

def load_vna(text_gpib_address,choice_vna_model,**kwargs):
	
	vna_class_dict = {\
		'HP8510C':mv_vna.HP8510C,\
		'HP8720':mv_vna.HP8720,\
		}
	#vna = text_gpib_address.value
	vna = vna_class_dict[choice_vna_model.value](int(text_gpib_address.value))

def get_s11(**kwargs):
	network = vna.s11

