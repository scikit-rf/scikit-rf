

'''
.. module:: skrf.vi.sa
++++++++++++++++++++++++++++++++++++++++++++++++++++
Spectrum Analyzers  (:mod:`skrf.vi.sa`)
++++++++++++++++++++++++++++++++++++++++++++++++++++


.. autosummary::
    :toctree: generated/

    HP8500
'''


import numpy as npy


from ..frequency import Frequency
from ..network import Network
from .. import mathFunctions as mf

from . ivihack import Driver

class HP8500(Driver):
    '''
    HP8500's series Spectrum Analyzers

    Examples
    -----------

    Get trace, and store in a Network object

    >>> from skrf.vi.sa import HP
    >>> my_sa = HP() # default address is 18
    >>> trace = my_sa.get_ntwk()

    Activate single sweep mode, get a trace, return to continuous sweep

    >>> my_sa.single_sweep()
    >>> my_sa.sweep()
    >>> trace_a = my_sa.trace_a
    >>> my_sa.cont_sweep()
    '''
    def __init__(self, address=18, *args, **kwargs):
        '''
        Initializer

        Parameters
        --------------
        address : int
            GPIB address
        \*args, \*\*kwargs :
            passed to ``ivi.Driver.__init__``
        '''
        Driver.__init__(self,'GPIB::'+str(address),*args,**kwargs)

    @property
    def frequency(self):
        '''
        '''
        f = Frequency(self.f_start, self.f_stop, len(self.trace_a),'hz')
        f.unit = 'ghz'
        return f

    def get_ntwk(self, trace='a', goto_local=False, *args, **kwargs):
        '''
        Get a trace and return the data in a :class:`~skrf.network.Network` format

        This will save instrument stage to reg 1, activate single sweep
        mode, sweep, save data, then recal state from reg 1.

        Returning the data in a the form of a
        :class:`~skrf.network.Network` allows  all the plotting methods
        and IO functions of that class to be used. Not all the methods
        of Network make sense for this type of data (scalar), but we
        assume the user knows this.

        Parameters
        ------------
        trace : ['a', 'b']
            save trace 'a' or trace 'b'
        goto_local : Boolean
            Go to local mode after taking a sweep
        \*args,\*\*kwargs :
            passed to :func:`~skrf.network.Network.__init__`

        '''
        trace = trace.lower()
        if trace not in ['a','b']:
            raise ValueError('\'trace\' should be \'a\' or \'b\'')

        self.save_state()
        self.single_sweep()
        self.sweep()
        #TODO: ask if magnitude is in linear (LN) or log (LG) mode
        if trace== 'a':
            s = self.trace_a
        elif trace == 'b':
            s = self.trace_b
        self.recall_state()
        s = mf.db_2_magnitude(npy.array(s))
        freq = self.frequency
        n = Network(s=s, frequency=freq, z0=1, *args, **kwargs)

        if goto_local:
            self.goto_local()
        return n

    @property
    def f_start(self):
        '''
        starting frequency
        '''
        return float(self.ask('fa?'))

    @property
    def f_stop(self):
        '''
        stopping frequency
        '''
        return float(self.ask('fb?'))

    @property
    def trace_a(self):
        '''
        trace 'a'
        '''
        return self.ask_for_values("tra?")

    @property
    def trace_b(self):
        '''
        trace 'b'
        '''
        return self.ask_for_values("trb?")

    def sweep(self):
        '''
        trigger a sweep, return when done
        '''
        self.write('ts')
        return self.ask('done?')

    def single_sweep(self):
        '''
        Activate single sweep mode
        '''
        self.write('sngls')

    def cont_sweep(self):
        '''
        Activate continuous sweep mode
        '''
        self.write('conts')

    def goto_local(self):
        '''
        Switches from remote to local control
        '''
        pass#visa.vpp43.gpib_control_ren(self.vi,0)

    def save_state(self, reg_n=1):
        '''
        Save current state to a given register
        '''
        self.write('saves %i'%reg_n)

    def recall_state(self, reg_n=1):
        '''
        Recall current state to a given register
        '''
        self.write('rcls %i'%reg_n)

