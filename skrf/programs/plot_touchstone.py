#!/usr/bin/env python3
import optparse

import matplotlib.pyplot as plt

import skrf as rf


def main():
    ##for use with argparse
    #parser = argparse.ArgumentParser(description='Plots contents of
    # a touchstone file.')
    parser = optparse.OptionParser(\
            usage='plot_touchstone.py [options] file.s2p [file1.s2p]',\
            description='Plots contents of a touchstone file.')
    # for use with argparse
    #parser.add_option('touchstone_files', \
    #       metavar='touchstone_file', type=str, nargs='+', \
    #       help='a touchstone file')
    parser.add_option('-m',default=None, metavar='M',type=int,\
            help='first index of s-parameter to plot' )
    parser.add_option('-n',default=None, metavar='N',type=int,\
            help='second index of s-parameter to plot' )
    (options, args) = parser.parse_args()

    if options.m is not None:
        options.m -=1
    if options.n is not None:
        options.n -=1

    plt.figure(figsize=(8,6))
    ax_1 = plt.subplot(221)
    ax_2 = plt.subplot(222)
    ax_3 = plt.subplot(223)
    ax_4 = plt.subplot(224)  # noqa: F841
    for touchstone_filename in args:
        ntwk = rf.Network(touchstone_filename)
        ntwk.plot_s_db(ax = ax_1, m=options.m,n=options.n)
        ntwk.plot_s_deg(ax = ax_2, m=options.m,n=options.n)
        ntwk.plot_s_smith(ax = ax_3,m=options.m,n=options.n )


    plt.savefig("test.png")
    return (1)

if __name__ == "__main__":
    main()
