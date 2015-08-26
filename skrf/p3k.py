"""
Python 3 compatibility tools.

"""
from __future__ import division, absolute_import, print_function
from builtins import str
from past.builtins import basestring

__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
           'asstr', 'open_latin1', 'long', 'basestring', 'sixu',
           'integer_types']

import sys

if sys.version_info[0] >= 3:
    import io

    long = int
    integer_types = (int,)
    basestring = str
    str = str
    bytes = bytes

    def asunicode(s):
        if isinstance(s, bytes):
            return s.decode('latin1')
        return str(s)

    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return str(s).encode('latin1')

    def asstr(s):
        if isinstance(s, bytes):
            return s.decode('latin1')
        return str(s)

    def isfileobj(f):
        return isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter))

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode, encoding='iso-8859-1')

    def sixu(s):
        return s

    strchar = 'U'


else:
    bytes = str
    long = int
    basestring = basestring
    str = str
    integer_types = (int, int)
    asbytes = str
    asstr = str
    strchar = 'S'

    def isfileobj(f):
        return isinstance(f, file)

    def asunicode(s):
        if isinstance(s, str):
            return s
        return str(s).decode('ascii')

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode)

    def sixu(s):
        return str(s, 'unicode_escape')


def getexception():
    return sys.exc_info()[1]

def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)

def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)
