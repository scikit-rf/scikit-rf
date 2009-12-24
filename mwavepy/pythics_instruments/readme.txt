these files are pythics vi's which unitilize mwavepy. 
pythics is 'python instrument control system', and can be found here 
 http://code.google.com/p/pythics/

to run these vi's you need to install pythics, then run the app.py in pythics.

here is a summary of the pythics vi's, 
	plotTouchtone.html
		loads a touchtone file, and plots its contents in log mag, phase, and smith chart
		format. 

	vnaGetData.html
		grabs data from a VNA over a GPIB bus. 
		can plot results and save as image, or save to a touchstone file.
		supports: HP8720C, HP8510C	


here is a summary of the instruments:
	vna.py
		provides the classes hp8720c, and hp8510c.  
		for further details on features and methods belonging to these classes see the doc folder distributed with mwavepy. 


