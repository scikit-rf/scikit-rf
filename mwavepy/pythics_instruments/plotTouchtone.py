'''
this vi plots the contents of a touchtone file. 
it plots: magnitude (in dB), phase (in deg), and a smith chart


'''

import mwavepy as m	# all work is done in this class
import numpy as npy


def plot(mag_plot,phase_plot,smith_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, smith_radius_num_box, **kwargs):
	plotParamsDB(mag_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, **kwargs)
	plotParamsPhase(phase_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, **kwargs)
	plotParamsSmith(smith_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box,smith_radius_num_box, **kwargs)

def plotParamsDB(mag_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, **kwargs):
	ntwk = m.loadTouchtone(touchtoneFileName.value)
	
	legendList=[]
	if (s11_check_box.value == True ):
		m.updatePlotDb(ntwk.s11, mag_plot)
		legendList.append('S11')
	if (s12_check_box.value == True ):
		m.updatePlotDb(ntwk.s12, mag_plot)
		legendList.append('S12')
	if (s21_check_box.value == True ):
		m.updatePlotDb(ntwk.s21, mag_plot)
		legendList.append('S21')
	if (s22_check_box.value == True ):
		m.updatePlotDb(ntwk.s22, mag_plot)
		legendList.append('S22')
	
	if (legend_check_box.value == True):
		mag_plot.legend(legendList)
		
	mag_plot.show()

def plotParamsPhase(phase_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, **kwargs):
	
	ntwk = m.loadTouchtone(touchtoneFileName.value)
	
	legendList=[]
	if (s11_check_box.value == True ):
		m.updatePlotPhase(ntwk.s11, phase_plot)
		legendList.append('S11')
	if (s12_check_box.value == True ):
		m.updatePlotPhase(ntwk.s12, phase_plot)
		legendList.append('S12')
	if (s21_check_box.value == True ):
		m.updatePlotPhase(ntwk.s21, phase_plot)
		legendList.append('S21')
	if (s22_check_box.value == True ):
		m.updatePlotPhase(ntwk.s22, phase_plot)
		legendList.append('S22')
	
	if (legend_check_box.value == True):
		phase_plot.legend(legendList)
		
	phase_plot.show()
		
def plotParamsSmith(smith_plot, touchtoneFileName,s11_check_box,s12_check_box,s21_check_box,s22_check_box , legend_check_box, smith_radius_num_box, **kwargs):
	
	ntwk = m.loadTouchtone(touchtoneFileName.value)
	m.updateSmithChart(smith_plot, smith_radius_num_box.value)
	
	legendList=[]
	if (s11_check_box.value == True ):
		m.updatePlotSmith(ntwk.s11, smith_plot)
		legendList.append('S11')
	if (s12_check_box.value == True ):
		m.updatePlotSmith(ntwk.s12, smith_plot)
		legendList.append('S12')
	if (s21_check_box.value == True ):
		m.updatePlotSmith(ntwk.s21, smith_plot)
		legendList.append('S21')
	if (s22_check_box.value == True ):	
		m.updatePlotSmith(ntwk.s22, smith_plot)
		legendList.append('S22')
	
	if (legend_check_box.value == True):
		smith_plot.legend(legendList)
		
		
	smith_plot.axis([-smith_radius_num_box.value , smith_radius_num_box.value, -smith_radius_num_box.value, smith_radius_num_box.value])		
	smith_plot.axis('equal')
	smith_plot.show()		

def clear(mag_plot, phase_plot,smith_plot,**kwargs):
	mag_plot.clear()
	mag_plot.show()
	phase_plot.clear()
	phase_plot.show()
	smith_plot.clear()
	smith_plot.show()



