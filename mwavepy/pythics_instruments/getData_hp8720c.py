
import visa
import pythics.libinstrument
import mwavepy as m	
import numpy as n

from mwavepy.pythics_instruments import vna_hp8720c

global myntwk = m.twoPort()


def initialize(gpib_address_numbox):
	myvna = vna_hp8720c("GPIB::" + gpib_address_numbox.value)

def getData(get_freq_vector_box, s11_check_box.value,s21_check_box.value,s12_check_box.value,s22_check_box.value):
	# setup vna for correct data format
	myvna.setDataFormatAscii()
	mynva.putInHold() # do we need this?
	myvna.putInSingSweepMode()
	
	if get_freq_vector_box.value== True:
		freqVector = myvna.getFrequencyAxis() # this may not work on other vna's
	
	if s11_check_box.value == True:
		myvna.setS('s11') # this defaults to sending an OPC?, see help
		s11 = myvna.getS()
		
	if s21_check_box.value == True:
		myvna.setS('s11') # this defaults to sending an OPC?, see help
		s21 = myvna.getS()
	
	if s11_check_box.value == True:
		myvna.setS('s11') # this defaults to sending an OPC?, see help
		s12 = myvna.getS()
	
	if s22_check_box.value == True:
		myvna.setS('s11') # this defaults to sending an OPC?, see help
		s22 = myvna.getS()
	
	myntwk = m.twoPort(s11,s21,s12,s22)


def plotData():
	# see if we can re-use plotTouchtone


def saveData(touchtone_file_picker):








