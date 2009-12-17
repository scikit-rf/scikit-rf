import multiprocessing
import mwavepy as m	
from mwavepy.pythics_instruments import vna

logger = multiprocessing.get_logger()

myvna = []
myntwk = []

def initializeVna(gpib_num_box, **kwargs):
	global myvna
	myvna = vna.hp8720c("GPIB::" + str(int(gpib_num_box.value)))
	myvna.setDataFormatAscii()
	
def getAllS(**kwargs):
	global myvna
	global myntwk

	
	freqVector = myvna.getFrequencyAxis() # this may not work on other vna's
	myntwk = myvna.getTwoPort(freqVector,50) # implicitly makes OPC? call
	myvna.putInContSweepMode()


def getCurrentS( **kwargs):
	global myvna
	global myntwk
	myntwk = m.onePort(myvna.getS(myvna.getFrequencyAxis()))
	myvna.putInContSweepMode()
	
def saveTouchtone(touchtone_file_picker,**kwargs ):
	myntwk.writeTouchtone(touchtone_file_picker.value)

