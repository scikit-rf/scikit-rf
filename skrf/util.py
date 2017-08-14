"""

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

"""
from __future__ import print_function
import contextlib
import fnmatch
import json
import os
import warnings
import zipfile

import six.moves.cPickle as pickle

import numpy as npy
from datetime import datetime
import collections
import pprint
import re
from subprocess import Popen, PIPE
import sys
from functools import wraps

# globals

try:
    basestring
except NameError:
    basestring = (str, bytes)


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


def find_nearest(array, value):
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


def find_nearest_index(array, value):
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


def slice_domain(x, domain):
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
    file : str/unicode or file-object
        file to open
    \*args, \*\*kwargs : arguments and keyword arguments to `open()`

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


def has_duplicate_value(value, values, index):
    """
    convenience function to check if there is another value of the current index in the list

    Parameters
    ----------
    value :
        any value in a list
    values : Iterable
        the iterable containing the values
    index : int
        the index of the current item we are checking for

    Returns
    -------
    bool,int
        returns None if no duplicate found, or the index of the first found duplicate
    """

    for i, val in enumerate(values):
        if i == index:
            continue
        if value == val:
            return i
    return False


def unique_name(name, names, exclude=-1):
    """
    pass in a name and a list of names, and increment with _## as necessary to ensure a unique name

    Parameters
    ----------
    name : str
        the chosen name, to be modified if necessary
    names : list
        list of names (str)
    exclude : int
        the index of an item to be excluded from the search
    """
    if not has_duplicate_value(name, names, exclude):
        return name
    else:
        if re.match("_\d\d", name[-3:]):
            name_base = name[:-3]
            suffix = int(name[-2:])
        else:
            name_base = name
            suffix = 1

        for num in range(suffix, 100, 1):
            name = "{:s}_{:02d}".format(name_base, num)
            if has_duplicate_value(name, names, exclude) is False:
                break
    return name


def smooth(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.
    based on the function from the scipy cookbook
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    npy.hanning, npy.hamming, npy.bartlett, npy.blackman, npy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = npy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = npy.ones(window_len, 'd')
    else:
        w = eval('npy.' + window + '(window_len)')
    y = npy.convolve(w / w.sum(), s, mode='same')
    return y[window_len-1:-(window_len-1)]


class ProgressBar:
    """
    a progress bar based off of the notebook/ipython progress bar from PyMC.  Useful when waiting for long operations
    such as taking a large number of VNA measurements that may take a few minutes
    """
    def __init__(self, iterations, label="iterations"):
        self.iterations = iterations
        self.label = label
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iteration):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iteration + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s %s complete' % (elapsed_iter, self.iterations, self.label)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


@contextlib.contextmanager
def suppress_numpy_warnings(**kw):
    olderr = npy.seterr(**kw)
    yield
    npy.seterr(**olderr)

def suppress_warning_decorator(msg):
    def suppress_warnings_decorated(func):
        @wraps(func)
        def suppressed_func(*k, **kw):
            show_warnings = []
            with warnings.catch_warnings(record=True) as wlist:
                 res = func(*k, **kw)
                 for w in wlist:
                     if not w.message.args[0].startswith(msg):
                         show_warnings.append(w)
            for w in show_warnings:
                warnings.warn(w.message.args[0])
        return suppressed_func
    return suppress_warnings_decorated
