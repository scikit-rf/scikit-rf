import unittest

import skrf as rf
import numpy as npy
from skrf.calibration.calibrationSet import Dot
from skrf.calibration import OnePort, TRL, SOLT, EightTerm
from skrf.util import suppress_warning_decorator


class CalsetTest(object):
    @suppress_warning_decorator("No switch terms")
    def test_run(self):
        '''
        ensure cal_set can be generated
        '''
        self.calset.run()

    @suppress_warning_decorator("No switch terms")
    def test_correct_ntwk(self):
        '''
        ensure a network can be corrected
        '''
        ntwk = self.wg.random(n_ports = self.n_ports)
        self.calset.apply_cal(ntwk)

class DotOneport(unittest.TestCase,CalsetTest):
    '''
    

    '''
    def setUp(self):
        self.wg = rf.RectangularWaveguide(rf.F(75,100,11), a=100*rf.mil,z0=50)
        wg = self.wg
        self.n_ports = 1
        self.E = wg.random(n_ports =2, name = 'E')
        
        
        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.match( name='load'),
                ]
        measured_sets = []
        for ideal in ideals:
            ns = rf.NetworkSet([self.measure(ideal) for k in range(3)])
            measured_sets.append(ns)
        
        
        self.calset = Dot(cal_class = OnePort, 
                          ideals = ideals, 
                          measured_sets = measured_sets,
                          is_reciprocal = True)
            
    def measure(self, ntwk):
        out = self.E**ntwk
        out.name = ntwk.name
        return out

class DotEightTerm(unittest.TestCase, CalsetTest):
    @suppress_warning_decorator("invalid value encountered in multiply")
    @suppress_warning_decorator("divide by zero encountered in true_divide")
    def setUp(self):
        self.n_ports = 2
        self.wg = rf.RectangularWaveguide(rf.F(75,100,3), a=100*rf.mil,z0=50)
        wg= self.wg
        wg.frequency = rf.F.from_f([100])
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
            
        measured_sets = []
        for ideal in ideals:
            ns = rf.NetworkSet([self.measure(ideal) for k in range(3)])
            measured_sets.append(ns)
        
        self.calset = Dot(  cal_class = EightTerm,
                            ideals = ideals,
                            measured_sets = measured_sets,
                            switch_terms = (self.gamma_f, self.gamma_r)
                            )
        
        
 
        
    def measure(self,ntwk):
        out =  self.X**ntwk**self.Y
        out.name = ntwk.name
        return out
    
