#!/usr/bin/env python

#import ez_setup
#ez_setup.use_setuptools()
from setuptools import setup, find_packages
from distutils.core import Extension

VERSION = '0.14.1'
LONG_DESCRIPTION = """
	sckit-rf is an open source approach to RF/Microwave engineering implemented in the Python programming language.
"""
setup(name='scikit-rf',
	version=VERSION,
	license='new BSD',
	description='Object Oriented Microwave Engineering',
	long_description=LONG_DESCRIPTION,
	author='Alex Arsenovic',
	author_email='arsenovic@virginia.edu',
	url='http://scikit-rf.org',
	packages=find_packages(),
	install_requires = [
		'ipython',
		'numpy',
		'scipy',
		'matplotlib',
		],
	#ext_modules=[Extension('skrf.src.connect', ['skrf/src/connect.c', ], export_symbols=['innerconnect_s','connect_s'])],
	package_dir={'skrf':'skrf'},
	include_package_data = True,
	#exclude_package_data = {'':'doc/*'},

	#package_data = {'skrf':['*tests*']}
	)

