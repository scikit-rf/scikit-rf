#!/usr/bin/env python
import sys
import optparse
import pylab as plb

try:
    import skrf as rf
except (ImportError):
    print('IMPORT ERROR: skrf is not installed correctly. Check you path.')


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

    plb.figure(figsize=(8,6))
    ax_1 = plb.subplot(221)
    ax_2 = plb.subplot(222)
    ax_3 = plb.subplot(223)
    ax_4 = plb.subplot(224)

    for touchstone_filename in args:
        print(touchstone_filename)
        ntwk = rf.Network(touchstone_filename)
        ntwk.plot_s_db(ax = ax_1, m=options.m,n=options.n)
        ntwk.plot_s_deg(ax = ax_2, m=options.m,n=options.n)
        ntwk.plot_s_smith(ax = ax_3,m=options.m,n=options.n )


    plb.show()
    return (1)

if __name__ == "__main__":
    main()
