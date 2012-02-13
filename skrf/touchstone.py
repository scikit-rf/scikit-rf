

#     Copyright (C) 2008 Werner Hoch
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
.. module:: skrf.touchstone
========================================
touchstone (:mod:`skrf.touchstone`)
========================================


This module provides a class to represent touchstone files.

This module was written by Werner Hoch.

touchstone Class
------------------

.. autosummary::
        :toctree: generated/

        touchstone

contains touchstone class
'''

import numpy

class touchstone():
    """
    class to read touchstone s-parameter files
    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0
    http://www.eda-stds.org/ibis/adhoc/interconnect/touchstone_spec2_draft.pdf
    """
    def __init__(self,filename):
        ## file name of the touchstone data file
        self.filename = filename

        ## file format version
        self.version = '1.0'
        ## unit of the frequency (Hz, kHz, MHz, GHz)
        self.frequency_unit = None
        ## s-parameter type (S,Y,Z,G,H)
        self.parameter = None
        ## s-parameter format (MA, DB, RI)
        self.format = None
        ## reference resistance, global setup
        self.resistance = None
        ## reference impedance for each s-parameter
        self.reference = None

        ## numpy array of original sparameter data
        self.sparameters = None
        ## numpy array of original noise data
        self.noise = None

        ## kind of s-parameter data (s1p, s2p, s3p, s4p)
        self.rank = None

        self.load_file(filename)

    def load_file(self, filename):
        """
        Load the touchstone file into the interal data structures
        """
        f = open(filename)

        extention = filename.split('.')[-1].lower()
        #self.rank = {'s1p':1, 's2p':2, 's3p':3, 's4p':4}.get(extention, None)
        try:
            self.rank = int(extention[1:-1])
        except (ValueError):
            raise (ValueError("filename does not have a s-parameter extention. It has  [%s] instead. please, correct the extension to of form: 'sNp', where N is any integer." %(extention)))


        linenr = 0
        values = []
        while (1):
            linenr +=1
            line = f.readline()
            if not line:
                break

            # remove comment extentions '!'
            # this may even be the whole line if '!' is the first character
            # everything is case insensitive in touchstone files
            line = line.split('!',1)[0].strip().lower()
            if len(line) == 0:
                continue

            # grab the [version] string
            if line[:9] == '[version]':
                self.version = line.split()[1]
                continue

            # grab the [reference] string
            if line[:11] == '[reference]':
                self.reference = [ float(r) for r in line.split()[2:] ]
                continue

            # the option line
            if line[0] == '#':
                toks = line[1:].strip().split()
                # fill the option line with the missing defaults
                toks.extend(['ghz', 's', 'ma', 'r', '50'][len(toks):])
                self.frequency_unit = toks[0]
                self.parameter = toks[1]
                self.format = toks[2]
                self.resistance = toks[4]
                if self.frequency_unit not in ['hz', 'khz', 'mhz', 'ghz']:
                    print 'ERROR: illegal frequency_unit [%s]',  self.frequency_unit
                    # TODO: Raise
                if self.parameter not in 'syzgh':
                    print 'ERROR: illegal parameter value [%s]', self.parameter
                    # TODO: Raise
                if self.format not in ['ma', 'db', 'ri']:
                    print 'ERROR: illegal format value [%s]', self.format
                    # TODO: Raise

                continue

            # collect all values without taking care of there meaning
            # we're seperating them later
            values.extend([ float(v) for v in line.split() ])

        # let's do some postprocessing to the read values
        # for s2p parameters there may be noise parameters in the value list
        values = numpy.asarray(values)
        if self.rank == 2:
            # the first frequency value that is smaller than the last one is the
            # indicator for the start of the noise section
            # each set of the s-parameter section is 9 values long
            pos = numpy.where(numpy.sign(numpy.diff(values[::9])) == -1)
            if len(pos[0]) != 0:
                # we have noise data in the values
                pos = pos[0][0] + 1   # add 1 because diff reduced it by 1
                noise_values = values[pos*9:]
                values = values[:pos*9]
                self.noise = noise_values.reshape((-1,5))

        # reshape the values to match the rank
        self.sparameters = values.reshape((-1, 1 + 2*self.rank**2))
        # multiplier from the frequency unit
        self.frequency_mult = {'hz':1.0, 'khz':1e3,
                               'mhz':1e6, 'ghz':1e9}.get(self.frequency_unit)
        # set the reference to the resistance value if no [reference] is provided
        if not self.reference:
            self.reference = [self.resistance] * self.rank

    def get_format(self, format="ri"):
        """
        returns the file format string used for the given format.
        This is usefull to get some informations.
        """
        if format == 'orig':
            frequency = self.frequency_unit
            format = self.format
        else:
            frequency = 'hz'
        return "%s %s %s r %s" %(frequency, self.parameter,
                                 format, self.resistance)

    def get_sparameter_names(self, format="ri"):
        """
        generate a list of column names for the s-parameter data
        The names are different for each format.
        posible format parameters:
          ri, ma, db, orig  (where orig refers to one of the three others)
        returns a list of strings.
        """
        names = ['frequency']
        if format == 'orig':
            format = self.format
        ext1, ext2 = {'ri':('R','I'),'ma':('M','A'), 'db':('DB','A')}.get(format)
        for r1 in xrange(self.rank):
            for r2 in xrange(self.rank):
                names.append("S%i%i%s"%(r1+1,r2+1,ext1))
                names.append("S%i%i%s"%(r1+1,r2+1,ext2))
        return names

    def get_sparameter_data(self, format='ri'):
        """
        get the data of the sparameter with the given format.
        supported formats are:
          orig:  unmodified s-parameter data
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitute and angle (degree)
        Returns a list of numpy.arrays
        """
        ret = {}
        if format == 'orig':
            values = self.sparameters
        else:
            values = self.sparameters.copy()
            # use frequency in hz unit
            values[:,0] = values[:,0]*self.frequency_mult
            if (self.format == 'db') and (format == 'ma'):
                values[:,1::2] = 10**(values[:,1::2]/20.0)
            elif (self.format == 'db') and (format == 'ri'):
                v_complex = ((10**values[:,1::2]/20.0)
                             * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ma') and (format == 'db'):
                values[:,1::2] = 20*numpy.log10(values[:,1::2])
            elif (self.format == 'ma') and (format == 'ri'):
                v_complex = (values[:,1::2] * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ri') and (format == 'ma'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = numpy.absolute(v_complex)
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)
            elif (self.format == 'ri') and (format == 'db'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = 20*numpy.log10(numpy.absolute(v_complex))
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)

        for i,n in enumerate(self.get_sparameter_names(format=format)):
            ret[n] = values[:,i]
        return ret

    def get_sparameter_arrays(self):
        """
        returns the sparameters as a tuple of arrays, where the first element is
        the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the sparameters are complex number.
        usage:
          f,a = self.sgetparameter_arrays()
          s11 = a[:,0,0]
        """
        v = self.sparameters

        if self.format == 'ri':
            v_complex = v[:,1::2] + 1j* v[:,2::2]
        elif self.format == 'ma':
            v_complex = (v[:,1::2] * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))
        elif self.format == 'db':
            v_complex = ((10**(v[:,1::2]/20.0)) * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))

        # this return is tricky its do the stupid way the touchtone lines are in order like s11,s21, etc. because of this we need the transpose command, and axes specifier
        return (v[:,0] * self.frequency_mult,
                numpy.transpose(v_complex.reshape((-1, self.rank, self.rank)),axes=(0,2,1)))

    def get_noise_names(self):
        """
        TODO: NIY
        """
        TBD = 1


    def get_noise_data(self):
        """
        TODO: NIY
        """
        TBD = 1
        noise_frequency = noise_values[:,0]
        noise_minimum_figure = noise_values[:,1]
        noise_source_reflection = noise_values[:,2]
        noise_source_phase = noise_values[:,3]
        noise_normalized_resistance = noise_values[:,4]


#if __name__ == "__main__":
    #import sys
    #import pylab

    #filename = sys.argv[1]
    #t = touchstone(filename)
    #n = t.get_sparameter_names(format='orig')
    #d = t.get_sparameter_data(format='orig')
    #f = d['frequency']
    #pylab.subplot(211)
    #for i in range(1,len(n),2):
        #pylab.plot(f, d[n[i]], label=n[i])
    #pylab.legend(loc='best')
    #pylab.title('Touchstone S-parameter, File=[%s]'%filename)
    #pylab.grid()
    #pylab.subplot(212)
    #for i in range(2,len(n),2):
        #pylab.plot(f, d[n[i]], label=n[i])
    #pylab.legend(loc='best')
    #pylab.xlabel('frequency [%s]' %(t.frequency_unit))
    #pylab.grid()
    #pylab.show()
