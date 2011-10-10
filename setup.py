#!/usr/bin/env python

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages

setup(name='mwavepy',
	version='1.4',
	license='gpl',
	description='Object Oriented Microwave Engineering',
	author='Alex Arsenovic',
	author_email='arsenovic@virginia.edu',
	url='http://code.google.com/p/mwavepy/',
	packages=find_packages(),
	#install_requires = [
	#	'numpy',
	#	'scipy',
	#	'matplotlib',
	#	],
	)

#from distutils.core import setup
#setup(name='mwavepy',
	#version='1.0',
	#description='Microwave Engineering Functions for python',
	#author='Alex Arsenovic, Lihan Chen',
	#author_email='arsenovic@virginia.edu',
	#url='http://code.google.com/p/mwavepy/',
	#packages=['mwavepy','mwavepy.transmissionLine','mwavepy.virtualInstruments'])
