#!/usr/bin/python
'''
script to generate docs. got this idea from this thread

http://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package
'''
from subprocess import call
import pkgutil
import mwavepy 

package = mwavepy
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
	call(['pydoc','-w',package.__name__+'.'+modname])
	if ispkg:
		for importer, modname2, ispkg in pkgutil.iter_modules([package.__path__[0]+'/'+modname] ):
			call(['pydoc','-w',package.__name__+'.'+modname+'.'+modname2])
			
