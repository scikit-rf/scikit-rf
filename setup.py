#!/usr/bin/env python

from distutils.core import setup



setup(name='mwavepy',
	version='1.0',
	description='Microwave Engineering Functions for python',
	author=['Alex Arsenovic','Lihan Chen'],
	author_email='arsenovic@virginia.edu',
	url='http://code.google.com/p/mwavepy/',
	packages=['mwavepy','mwavepy.transmissionLine','mwavepy.virtualInstruments'])
