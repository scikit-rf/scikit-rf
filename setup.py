#!/usr/bin/env python

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages



LONG_DESCRIPTION = """
	sckit-rf is an object-oriented approach to RF/Microwave engineering implemented in the Python programming language. It provides a general set of objects and features which can be used to construct solutions to specific problems. 
"""
setup(name='scikit-rf',
	version='0.1',
	license='gpl',
	description='Object Oriented Microwave Engineering',
	long_description=LONG_DESCRIPTION,
	author='Alex Arsenovic',
	author_email='arsenovic@virginia.edu',
	url='http://github.com/scikit-rf/scikit-rf/wiki',
	packages=find_packages(),
	#install_requires = [
	#	'numpy',
	#	'scipy',
	#	'matplotlib',
	#	],
	)

