#!/usr/bin/env python

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages

setup(name='mwavepy',
	version='1.5',
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

