here are instructions to integrate this program with gnome file-manager
(nautilus)
you need to:
	0)install skrf, and place plot_touchstone.py in your path
	1)add a MIME-type for touchstone files
	2)update the mime database
	3)add a custom command launcher in nautilus for touchsone files. 

0) see skrf website of INSTALL.txt

1) a file called skrf.xml, which should be in this directory, is a gnome
mime definitions and may be placed in /usr/share/mime/packages/ . 

2) then run,
	sudo update-mime-database /usr/share/mime

3) in the file-manager right click on  a touchstone file
	right click->properties->open with->add->use a custom command
	enter in 
		plot_touchstone.py %f
	
restart file-manager. 



xdg-icon-resource install --context mimetypes --size 48 skrf-icon.png skrf-touchstone
