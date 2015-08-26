
'''

.. currentmodule:: skrf.util
========================================
util (:mod:`skrf.util`)
========================================

Holds utility functions that are general conveniences.



General
------------
.. autosummary::
   :toctree: generated/

   now_string
   find_nearest
   find_nearest_index
   get_fid
   get_extn



'''

from . import mathFunctions as mf


import matplotlib as mpl
import warnings
import os, fnmatch
try:
	import cPickle as pickle
except ImportError:
    import pickle as pickle

import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime
import collections, pprint
from subprocess import Popen,PIPE
# globals


# other
def now_string():
    '''
    returns a unique sortable string, representing the current time

    nice for generating date-time stamps to be used in file-names,
    the companion function :func:`now_string_2_dt` can be used
    to read these string back into datetime objects.

    See Also
    ------------
    now_string_2_dt

    '''
    return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')

def now_string_2_dt(s):
    '''
    Converts the output of  :func:`now_string` to a datetime object.

    See Also
    -----------
    now_string

    '''
    return datetime(*[int(k) for k in s.split('.')])

def find_nearest(array,value):
    '''
    find nearest value in array.

    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in
    value : element of the array
        value to search for

    Returns
    --------
    found_value : an element of the array
        the value that is numerically closest to `value`

    '''
    idx=(npy.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    '''
    find nearest value in array.

    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in
    value : element of the array
        value to search for

    Returns
    --------
    found_index : int
        the index at which the  numerically closest element to `value`
        was found at


    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    '''
    return (npy.abs(array-value)).argmin()

def slice_domain(x,domain):
    '''
    Returns a slice object closest to the `domain` of `x`

    domain = x[slice_domain(x, (start, stop))]

    Parameters
    -----------
    vector : array-like
        an array of values
    domain : tuple
        tuple of (start,stop) values defining the domain over
        which to slice

    Examples
    -----------
    >>> x = linspace(0,10,101)
    >>> idx = slice_domain(x, (2,6))
    >>> x[idx]

    '''
    start = find_nearest_index(x, domain[0])
    stop = find_nearest_index(x, domain[1])
    return slice(start,stop+1)

# file IO

def get_fid(file, *args, **kwargs):
    '''
    Returns a file object, given a filename or file object

    Useful when you want to allow the arguments of a function to
    be either files or filenames

    Parameters
    -------------
    file : str or file-object
        file to open
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load
    '''
    if isinstance(file, basestring):
        return open(file, *args, **kwargs)
    else:
        return file

def get_extn(filename):
    '''
    Get the extension from a filename.

    The extension is defined as everything passed the last '.'.
    Returns None if it ain't got one

    Parameters
    ------------
    filename : string
        the filename

    Returns
    --------
    ext : string, None
        either the extension (not including '.') or None if there
        isn't one


    '''
    ext = os.path.splitext(filename)[-1]
    if len(ext)==0:
        return None
    else:
        return ext[1:]

def basename_noext(filename):
    '''
    gets the basename and strips extension
    '''
    return os.path.splitext(os.path.basename(filename))[0]


# git
def git_version( modname):
    '''
    Returns output 'git describe', executed in a module's root directory.
    '''
    mod = __import__(modname)
    mod_dir =os.path.split(mod.__file__)[0]
    p = Popen(['git', 'describe'], stdout = PIPE,stderr=PIPE, cwd =mod_dir )

    try:
        out,er = p.communicate()
    except(OSError):
        return None

    out = out.strip('\n')
    if out == '':
        return None
    return out


def stylely(rc_dict={}, style_file = 'skrf.mplstyle'):
    '''
    loads the rc-params from the specified file (file must be located in skrf/data)
    '''

    from skrf.data import pwd # delayed to solve circular import
    rc = mpl.rc_params_from_file(os.path.join(pwd, style_file))
    mpl.rcParams.update(rc)
    mpl.rcParams.update(rc_dict)


def dict_2_recarray(d, delim, dtype):
    '''
    Turns a dictionary of structured keys to a record array of objects

    This is useful if you save data-base like meta-data in the form
    or file-naming conventions, aka 'the poor-mans database'


    Examples
    -------------

    given a directory of networks like:

    >>> ls
    a1,0.0,0.0.s1p    a1,3.0,3.0.s1p    a2,3.0,-3.0.s1p   b1,-3.0,3.0.s1p
    ...

    you can sort based on the values or each field, after defining their
    type with `dtype`. The `values` field accesses the objects.


    >>>d =rf.ran('/tmp/' )
    >>>delim =','
    >>>dtype = [('name', object),('voltage',float),('current',float)]
    >>>ra = dict_2_recarray(d=rf.ran(dir), delim=delim, dtype =dtype)

    then you can sift like you do with numpy arrays

    >>>ra[ra['voltage']<3]['values']
    array([1-Port Network: 'a2,0.0,-3.0',  450-800 GHz, 101 pts, z0=[ 50.+0.j],
           1-Port Network: 'b1,0.0,3.0',  450-800 GHz, 101 pts, z0=[ 50.+0.j],
           1-Port Network: 'a1,0.0,-3.0',  450-800 GHz, 101 pts, z0=[ 50.+0.j],
    '''

    split_keys = [tuple(k.split(delim)+[d[k]]) for k in d.keys()]
    x = npy.array(split_keys, dtype=dtype+[('values',object)])
    return x



def findReplace(directory, find, replace, filePattern):
    '''
    Find/replace some txt in all files in a directory, recursively

    This was found in [1]_.

    Examples
    -----------
    findReplace("some_dir", "find this", "replace with this", "*.txt")
    .. [1] http://stackoverflow.com/questions/4205854/python-way-to-recursively-find-and-replace-string-in-text-files
    '''
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)


# general purpose objects

class HomoList(collections.Sequence):
    '''

    A Homogeneous Sequence

    Provides a class for a list-like object which contains
    homogeneous values. Attributes of the values can be accessed through
    the attributes of HomoList. Searching is done like numpy arrays.

    Initialized from a list  of all the same type

    >>> h = HomoDict([Foo(...), Foo(...)])

    The individual values of `h` can be access in identical fashion to
    Lists.

    >>> h[0]

    Assuming that `Foo` has property `prop`  and function `func` ...

    Access elements' properties:

    >>> h.prop

    Access elements' functions:

    >>> h.func()

    Searching:

    >>> h[h.prop == value]
    >>> h[h.prop < value]

    Multiple search:

    >>> h[set(h.prop==value1) & set( h.prop2==value2)]

    Combos:

    >>> h[h.prop==value].func()
    '''


    def __init__(self, list_):
        self.store = list(list_)

    def __eq__(self, value):
        return [k for k in range(len(self)) if self.store[k] == value ]

    def __ne__(self, value):
        return [k for k in range(len(self)) if self.store[k] != value ]

    def __gt__(self, value):
        return [k for k in range(len(self)) if self.store[k] > value ]

    def __ge__(self, value):
        return [k for k in range(len(self)) if self.store[k] >= value ]

    def __lt__(self, value):
        return [k for k in range(len(self)) if self.store[k] < value ]

    def __le__(self, value):
        return [k for k in range(len(self)) if self.store[k] <= value ]

    def __getattr__(self, name):
        return self.__class__(
            [k.__getattribute__(name) for k in self.store])

    def __getitem__(self, idx):
        try:
            return self.store[idx]
        except(TypeError):
            return self.__class__([self.store[k] for k in idx])


    def __call__(self, *args, **kwargs):
        return self.__class__(
            [k(*args,**kwargs) for k in self.store])

    def __setitem__(self, idx, value):
        self.store[idx] = value

    def __delitem__(self, idx):
        del self.store[idx]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        return pprint.pformat(self.store)

    def __repr__(self):
        return pprint.pformat(self.store)

class HomoDict(collections.MutableMapping):
    '''
    A Homogeneous Mutable Mapping

    Provides a class for a dictionary-like object which contains
    homogeneous values. Attributes of the values can be accessed through
    the attributes of HomoDict. Searching is done like numpy arrays.

    Initialized from a dictionary containing values of all the same type

    >>> h = HomoDict({'a':Foo(...),'b': Foo(...), 'c':Foo(..)})

    The individual values of `h` can be access in identical fashion to
    Dictionaries.

    >>> h['key']

    Assuming that `Foo` has property `prop`  and function `func` ...

    Access elements' properties:

    >>> h.prop

    Access elements' functions:

    >>> h.func()

    Searching:

    >>> h[h.prop == value]
    >>> h[h.prop < value]

    Multiple search:

    >>> h[set(h.prop==value1) & set( h.prop2==value2)]

    Combos:

    >>> h[h.prop==value].func()
    '''
    def __init__(self, dict_):
        self.store = dict(dict_)

    def __eq__(self, value):
        return [k for k in self.store if self.store[k] == value ]

    def __ne__(self, value):
        return [k for k in self.store if self.store[k] != value ]

    def __gt__(self, value):
        return [k for k in self.store if self.store[k] > value ]

    def __ge__(self, value):
        return [k for k in self.store if self.store[k] >= value ]

    def __lt__(self, value):
        return [k for k in self.store if self.store[k] < value ]

    def __le__(self, value):
        return [k for k in self.store if self.store[k] <= value ]

    def __getattr__(self, name):
        return self.__class__(
            {k: getattr(self.store[k],name) for k in self.store})

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.store[key]
        else:
            c =   self.__class__({k:self.store[k] for k in key})
        return c
        #if len(c) == 1:
        #    return c.store.values()[0]
        #else:
        #    return c

    def __call__(self, *args, **kwargs):
        return self.__class__(
            {k: self.store[k](*args, **kwargs) for k in self.store})

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        return pprint.pformat(self.store)

    def __repr__(self):
        return pprint.pformat(self.store)


    def copy(self):
        return HomoDict(self.store)


    def filter_nones(self):
        self.store =  {k:self.store[k] for k in self.store \
                        if self.store[k] is not None}

    def filter(self, **kwargs):
        '''
        Filter self based on kwargs

        This is equivalent to:

        >>> h = HomoDict(...)
        >>> for k in kwargs:
        >>>     h = h[k ==kwargs[k]]
        >>> return h

        prefixing the kwarg value with a '!' causes a not equal test (!=)

        Examples
        ----------
        >>> h = HomoDict(...)
        >>> h.filter(name='jean', age = '18', gender ='!female')

        '''
        a = self
        for k in kwargs:
            if kwargs[k][0] == '!':
                a = a[a.__getattr__(k) != kwargs[k][1:]]
            else:
                a = a[a.__getattr__(k) == kwargs[k]]
        return a
