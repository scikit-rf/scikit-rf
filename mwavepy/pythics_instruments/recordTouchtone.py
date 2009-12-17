import multiprocessing
import mwavepy as m	
from mwavepy.pythics_instruments import vna

logger = multiprocessing.get_logger()

myvna = []
myntwk = []

def initializeVna(logger_text_box,gpib_num_box, timeout_num_box,**kwargs):
	global myvna
	try:
		logger.info(str(timeout_num_box.value))
		myvna = vna.hp8510c("GPIB::" + str(int(gpib_num_box.value)),timeout=timeout_num_box.value)
		myvna.setDataFormatAscii()
		logger_text_box.value =  'loaded vna.\n' + logger_text_box.value
	except:
		logger_text_box.value ='ERROR: failed to load vna.\n'+ logger_text_box.value

def clearPlots(smith_plot, phase_plot,mag_plot,**kwargs):
	mag_plot.clear()
	phase_plot.clear()
	smith_plot.clear()
	mag_plot.show()
	phase_plot.show()
	smith_plot.show()
	
	
def getAllS(smith_plot, phase_plot,mag_plot,logger_text_box,timer_1, **kwargs):
	global myvna
	global myntwk
	
	# save all S's in to global twoPort myntwk
	logger_text_box.value = 'getting All S-parameters......\n' + logger_text_box.value
	freqVector = myvna.getFrequencyAxis() # this may not work on other vna's
	myntwk = myvna.getTwoPort(freqVector,50) # implicitly makes OPC? call
	myvna.putInContSweepMode()
	logger_text_box.value =  'done.\n' + logger_text_box.value
	
	
	
	for param in myntwk.s11,myntwk.s12,myntwk.s21,myntwk.s22:
		m.updatePlotDb(param, mag_plot)
		m.updatePlotPhase(param, phase_plot)
		m.updatePlotSmith(param, smith_plot)
	
	m.updateSmithChart(smith_plot)	
	mag_plot.legend(['S11','S12','S21','S22'])
	mag_plot.show()
	phase_plot.legend(['S11','S12','S21','S22'])
	phase_plot.show()
	smith_plot.legend(['S11','S12','S21','S22'])
	smith_plot.axis('equal')
	smith_plot.show()
	


    
def getCurrentS( smith_plot, phase_plot,mag_plot,logger_text_box,**kwargs):
	global myvna
	global myntwk
	logger_text_box.value = 'getting current  S-parameters......\n' + logger_text_box.value
	myntwk = m.onePort(myvna.getS(myvna.getFrequencyAxis()))
	myvna.putInContSweepMode()
	m.updatePlotDb(myntwk.s11, mag_plot)
	m.updatePlotPhase(myntwk.s11, phase_plot)
	m.updatePlotSmith(myntwk.s11, smith_plot)
	logger_text_box.value = 'done\n' + logger_text_box.value
	m.updateSmithChart(smith_plot)	
	mag_plot.show()
	phase_plot.show()
	smith_plot.axis('equal')
	smith_plot.show()
	

	
	
def saveTouchtone(logger_text_box, touchtone_file_picker, **kwargs ):
	try:
		myntwk.writeTouchtone(touchtone_file_picker.value)
		logger_text_box.value = 'Saved File: ' +touchtone_file_picker.value + '\n' + logger_text_box.value
	except:
		logger_text_box.value = 'ERROR: Failed to Save File: ' +touchtone_file_picker.value + '\n' + logger_text_box.value
def run_timer(timer_gauge, **kwargs):
    timer_gauge.pulse()
