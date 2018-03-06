#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.core import Extension

with open('skrf/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break

LONG_DESCRIPTION = """
	sckit-rf is an open source approach to RF/Microwave engineering implemented in the Python programming language.
"""
setup(name='scikit-rf',
	version=VERSION,
	license='new BSD',
	description='Object Oriented Microwave Engineering',
	long_description=LONG_DESCRIPTION,
	author='Alex Arsenovic',
	author_email='alexanderarsenovic@gmail.com',
	url='http://www.scikit-rf.org',
	packages=find_packages(),
	install_requires = [
		'numpy',
		'scipy',
        'pandas',
		'matplotlib',
        'ipython',
        'six',
        'future',
		],
	#ext_modules=[Extension('skrf.src.connect', ['skrf/src/connect.c', ], export_symbols=['innerconnect_s','connect_s'])],
	package_dir={'skrf':'skrf'},
	include_package_data = True,
    #data_files = [("", ["LICENSE.txt"])],
	#exclude_package_data = {'':'doc/*'},

	#package_data = {'skrf':['*tests*']}
	)

