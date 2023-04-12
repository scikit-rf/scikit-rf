import re
import copy
from collections.abc import Iterable
import zipfile

import numpy as np
import skrf


def debug_counter(n=-1):
    count = 0
    while count != n:
        count += 1
        yield count


def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


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
        if re.match(r"_\d\d", name[-3:]):
            name_base = name[:-3]
            suffix = int(name[-2:])
        else:
            name_base = name
            suffix = 1

        for num in range(suffix, 100, 1):
            name = f"{name_base:s}_{num:02d}"
            if not has_duplicate_value(name, names, exclude):
                break
    return name


def trace_color_cycle(n=1000):
    """
    :type n: int
    :return:
    """

    # TODO: make this list longer
    lime_green = "#00FF00"
    cyan = "#00FFFF"
    magenta = "#FF00FF"
    yellow = "#FFFF00"
    pink = "#C04040"
    blue = "#0000FF"
    lavendar = "#FF40FF"
    turquoise = "#00FFFF"

    count = 0
    colors = [yellow, cyan, magenta, lime_green, pink, blue, lavendar, turquoise]
    num = len(colors)
    while count < n:
        yield colors[count % num]
        count += 1


def snp_string(ntwk, comments=None):
    """
    get the RI .snp touchstone file string, which we will use for saving with zip files
    *** ONLY SUPPORTS 1 and 2-port networks

    Parameters
    ----------
    ntwk : skrf.Network
        a one or two-port Network
    comments : str, list
        any comments to append to the top of the file

    Returns
    -------
    str
    """
    if type(comments) == str:
        lines = comments.splitlines()
    elif type(comments) == list:
        lines = comments
    elif type(comments) == tuple:
        lines = list(comments)
    elif comments:
        raise TypeError("Must provide either a string, or a list of strings")
    else:
        lines = None

    if lines:
        for i, line in enumerate(lines):
            if not line.startswith("!"):
                lines[i] = "! " + line
        s2p_comments = "\n".join(lines) + "\n"
    else:
        s2p_comments = ""

    if ntwk.nports == 1:
        S11 = ntwk.s[:, 0, 0]
        rows = np.vstack((ntwk.f, S11.real, S11.imag)).T.tolist()
    elif ntwk.nports == 2:
        S11 = ntwk.s[:, 0, 0]
        S21 = ntwk.s[:, 1, 0]
        S12 = ntwk.s[:, 0, 1]
        S22 = ntwk.s[:, 1, 1]
        rows = np.vstack(
            (ntwk.f, S11.real, S11.imag, S21.real, S21.imag, S12.real, S12.imag, S22.real, S22.imag)).T.tolist()
    else:
        raise ValueError("Network must be a two-port network")

    return s2p_comments + "# Hz S RI R 50\n" + \
           "\n".join(" ".join(f"{val:0.8g}" for val in row) for row in rows)


def network_from_zip(zipfid):
    """
    convenience function to read zipinfo file into a network

    Parameters
    ----------
    zipfid : zipfile.ZipExtFile

    Returns
    -------
    skrf.Network
    """
    ntwk = skrf.Network()
    ntwk.read_touchstone(zipfid)
    return ntwk


def read_zipped_touchstones(ziparchive):
    """
    similar to skrf.io.read_all_networks, which works for directories but only for touchstones in ziparchives

    Parameters
    ----------
    ziparchive : zipfile.ZipFile

    Returns
    -------
    dict
    """
    networks = dict()
    fnames = [f.filename for f in ziparchive.filelist]
    for fname in fnames:
        if fname[-4:].lower() in (".s1p", ".s2p", ".s3p", ".s4p"):
            with ziparchive.open(fname) as zfid:
                network = network_from_zip(zfid)
            networks[network.name] = network
    return networks
