#       generic.py
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.from media import Media
'''
Provides  generic parametric standards which dont depend on any
specific properties of a given media

The naming convention for these classes is
        Standard_UnknownQuantity
'''

from parametricStandard import ParametricStandard

class Parameterless(ParametricStandard):
    '''
            A parameterless standard.

            note:
            this is needed so that the calibration algorithm doesnt have to
            handle more than one class type for  standards
            '''
    def __init__(self, ideal_network):
        '''
        takes:
                ideal_network: a Network instance of the standard
        '''
        ParametricStandard.__init__(self, \
                function  = lambda: ideal_network)

class Line_UnknownLength(ParametricStandard):
    '''
    A matched delay line of unknown length

    initial guess for length should be given to constructor
    '''
    def __init__(self, media, d,**kwargs):
        '''
        takes:
                media: a skrf.Media type
                d: initial guess for line length [m]*
                **kwargs: passed to self.function

        *note:
                the Media.line function can take the kwarg 'unit', in case
                you want to specify the line length in electrical length

        '''
        ParametricStandard.__init__(self, \
                function = media.line,\
                parameters = {'d':d},\
                **kwargs\
                )

class DelayLoad_UnknownLength(ParametricStandard):
    '''
    A  Delayed Termination of unknown length, but known termination
    '''
    def __init__(self, media,d,Gamma0,**kwargs):
        '''
        takes:
                media: a skrf.Media type
                d: initial guess for distance to termination (in m)*
                Gamma0: complex reflection coefficient off load
                        [complex number of array]
                **kwargs: passed to self.function

        *note:
                the Media.line function can take the kwarg 'unit', in case
                you want to specify the line length in electrical length
        '''
        kwargs.update({'Gamma0':Gamma0})

        ParametricStandard.__init__(self, \
                function = media.delay_load,\
                parameters = {'d':d},\
                **kwargs\
                )

class DelayShort_UnknownLength(DelayLoad_UnknownLength):
    '''
    A delay short of unknown length

    '''
    def __init__(self, media,d,**kwargs):
        '''
        This calls DelayLoad_UnknownLength. see that class for more info.

        takes:
                media: a Media type
                d: initial guess for delay short physical length [m]
                **kwargs: passed to self.function
        '''
        DelayLoad_UnknownLength.__init__(self,\
                media= media,
                d=d,
                Gamma0=-1,
                **kwargs)

class DelayOpen_UnknownLength(DelayLoad_UnknownLength):
    '''
    A delay open of unknown length

    '''
    def __init__(self, media,d,**kwargs):
        '''
        This calls DelayLoad_UnknownLength. see that class for more info.

        takes:
                media: a Media type
                d: initial guess for delay short physical length [m]
                **kwargs: passed to self.function
        '''
        DelayLoad_UnknownLength.__init__(self,\
                media= media,
                d=d,
                Gamma0=1,
                **kwargs)

class DelayLoad_UnknownLoad(ParametricStandard):
    '''
    A  Delayed Load of unknown Load. Assumes load is frequency independent
    '''
    def __init__(self, media,d,Gamma0,**kwargs):
        '''
        takes:
                media: a Media type
                d: distance to termination, in m.
                Gamma0: initial guess for complex reflection coefficient
                        of load. [complex number]
                **kwargs: passed to self.function
        '''
        kwargs.update({'d':d})
        ParametricStandard.__init__(self, \
                function = media.delay_load,\
                parameters = {'Gamma0':Gamma0},\
                **kwargs\
                )

class DelayLoad_UnknownLength_UnknownLoad(ParametricStandard):
    '''
    A  Delayed load of unknown length or reflection coefficient.
    Assumes the load is frequency independent
    '''
    def __init__(self, media,d,Gamma0,**kwargs):
        '''
        takes:
                media: a Media type
                d: initial guess distance to termination, in m.
                Gamma0: initial guess for complex reflection coefficient
                        of load. [complex number]
                **kwargs: passed to self.function
        '''

        ParametricStandard.__init__(self, \
                function = media.delay_load,\
                parameters = {'d':d,'Gamma0':Gamma0},\
                **kwargs\
                )



class UnknownShuntCapacitance(ParametricStandard):
    '''
    A Network with unknown connector capacitance
    '''
    def __init__(self, media,C,ntwk,**kwargs):
        '''
        takes:
                media: a skrf.Media type
                C: initial guess at connector capacitance, in farads
                ntwk: network type, representing the standard after the
                        unknown capacitor
                **kwargs: passed to self.function

        *note:
                the Media.line function can take the kwarg 'unit', in case
                you want to specify the line length in electrical length
        '''
        def func(*args, **kwargs):
            return media.shunt_capacitor(*args, **kwargs)**ntwk

        ParametricStandard.__init__(self, \
                function = func,\
                parameters = {'C':C},\
                **kwargs\
                )
class UnknownShuntInductance(ParametricStandard):
    '''
    A Network with unknown connector inductance
    '''
    def __init__(self, media,L,ntwk,**kwargs):
        '''
        takes:
                media: a skrf.Media type
                L: initial guess at connector inductance, in henrys
                ntwk: network type, representing the standard after the
                        unknown capacitor
                **kwargs: passed to self.function

        *note:
                the Media.line function can take the kwarg 'unit', in case
                you want to specify the line length in electrical length
        '''
        def func(*args, **kwargs):
            return media.shunt_inductor(*args, **kwargs)**ntwk

        ParametricStandard.__init__(self, \
                function = func,\
                parameters = {'L':L},\
                **kwargs\
                )

class UnknownShuntCapacitanceInductance(ParametricStandard):
    '''
    A Network with unknown connector inductance and capacitance
    '''
    def __init__(self, media,C,L,ntwk,**kwargs):
        '''
        takes:
                media: a skrf.Media type
                C: initial guess at connector capacitance, in farads
                L: initial guess at connector inductance, in henrys
                ntwk: network type, representing the standard after the
                        unknown capacitor
                **kwargs: passed to self.function

        *note:
                the Media.line function can take the kwarg 'unit', in case
                you want to specify the line length in electrical length
        '''
        def func(L,C, **kwargs):
            return media.shunt_inductor(L, **kwargs)**\
                    media.shunt_capacitor(C, **kwargs)**ntwk

        ParametricStandard.__init__(self, \
                function = func,\
                parameters = {'C':C,'L':L},\
                **kwargs\
                )
