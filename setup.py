#!/usr/bin/env python

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages

VERSION = '0.14'
LONG_DESCRIPTION = """
	sckit-rf is an open source approach to RF/Microwave engineering implemented in the Python programming language. 
"""
setup(name='scikit-rf',
	version=VERSION,
	license='gpl',
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
	package_dir={'skrf':'skrf'},
	include_package_data = True,
	#exclude_package_data = {'':'doc/*'},
	
	#package_data = {'skrf':['*tests*']}
	)

