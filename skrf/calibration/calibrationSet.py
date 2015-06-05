

'''
.. module:: skrf.calibration.calibrationSet
================================================================
calibrationSet (:mod:`skrf.calibration.calibrationSet`)
================================================================


Contains the CalibrationSet class, and supporting functions

CalibrationSet Class
===============

.. autosummary::
   :toctree: generated/

   CalibrationSet

'''
from itertools import product, combinations, permutations
from .calibration import Calibration
from ..networkSet import NetworkSet



def cartesian_product(ideals, measured_sets, *args, **kwargs):
    '''
    '''
    measured_lists = product(*[k[:] for k in measured_sets])
    return [Calibration(ideals = ideals, measured = measured,
        *args, **kwargs) for measured in measured_lists ]

def dot_product(ideals, measured_sets, *args, **kwargs):
    '''
    '''
    for measured_set in measured_sets:
        if len(measured_set) != len(measured_sets[0]):
            raise(IndexError('all measured NetworkSets must have same length for dot product combinatoric function'))

    cal_list = []
    for k in list(range(len(measured_sets[0]))):
        measured = [measured_set[k] for measured_set in measured_sets]
        cal_list.append(
            Calibration(ideals=ideals, measured= measured,
            *args,**kwargs)
            )

    return cal_list

class CalibrationSet(object):
    '''
    A set of Calibrations

    This is designed to support experimental uncertainty analysis [1]_.

    References
    -----------

    .. [1] A. Arsenovic, L. Chen, M. F. Bauwens, H. Li, N. S. Barker, and R. M. Weikle, "An Experimental Technique for Calibration Uncertainty Analysis," IEEE Transactions on Microwave Theory and Techniques, vol. 61, no. 1, pp. 263-269, 2013.

    '''

    def __init__(self, cal_class, ideals, measured_sets,*args, **kwargs):
        '''
        Parameters
        ----------
        cal_class : a Calibration class
            this is the class of calibration to use on the set. This
            argument is the actual class itself like OnePort, TRL, SOLT, etc

        ideals : list of Networks

        measured_set :  list of NetworkSets, or list of lists
            each element in this list should be a corresponding measured
            set to the ideals element of the same index. The sets
            themselves  can be anything list-like

        \\*args\\**kargs :
            passed to self.run(),

        '''
        self.cal_class = cal_class
        self.ideals = ideals
        self.measured_sets = measured_sets
        self.args = args
        self.kwargs = kwargs
        self.run(*args, **kwargs)

    def __getitem__(self, key):
        return self.cal_list[key]

    def apply_cal(self, raw_ntwk, *args, **kwargs):
        '''
        '''
        return NetworkSet([k.apply_cal(raw_ntwk) for k in self.cal_list],
            *args, **kwargs)

    def plot_uncertainty_per_standard(self):
        '''
        '''
        self.dankness('std_s','plot_s_mag')

    def dankness(self, prop, func, *args, **kwargs):
        '''
        '''
        try:
            [k.__getattribute__(prop).__getattribute__(func)\
                (*args, **kwargs) for k in self.measured_sets]
        except (TypeError):
            return [k.__getattribute__(prop).__getattribute__(func) \
                for k in self.measured_sets]

    def run(self):
        NotImplementedError('SubClass must implement this')

    @property
    def corrected_sets(self):
        '''
        The set of corrected networks, each is corrected by its corresponding
        element in the cal_list
        '''
        n_meas = len(self.cal_list[0].measured)
        mat = [k.caled_ntwks for k in self.cal_list]
        return [NetworkSet([k[l] for k in mat]) for l in range(n_meas)]



class Dot(CalibrationSet):

    def run(self, *args, **kwargs):
        ideals = self.ideals
        measured_sets = self.measured_sets
        if len(set(map(len, measured_sets))) !=1:
            raise(IndexError('all measured NetworkSets must have same length for dot product combinatoric function'))

        self.cal_list = []
        for k in range(len(measured_sets[0])):
            measured = [measured_set[k] for measured_set in measured_sets]
            cal = self.cal_class(ideals=ideals, measured= measured,
                                 *args,**kwargs)
            self.cal_list.append(cal)



