import mwavepy as mv
from mwavepy.virtualInstruments import vna as mv_vna


global vna

def load_vna(text_gpib_address,**kwargs):
	vna = mv_vna.HP8510c(text_gpib_address)

