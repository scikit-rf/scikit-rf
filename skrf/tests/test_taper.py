from builtins import object
import unittest
import os
import numpy as npy

import skrf as rf
from skrf import mil
from skrf.taper import *

from skrf.media import RectangularWaveguide

class TaperTestCase(object):
    
    def test_ntwk(self):
        self.taper.ntwk

class LinearTestCase(TaperTestCase, unittest.TestCase):
    def setUp(self):
        freq=rf.Frequency(75,110,101,'ghz')
        self.taper = Linear(
                   med= RectangularWaveguide,
                   med_kw = dict(frequency=freq),
                   param = 'a',
                   start = 10*mil, 
                   stop = 12*mil,
                   length = 1000*mil, 
                   n_sections = 10,
                   )
                   


class SmoothStepTestCase(TaperTestCase, unittest.TestCase):
    def setUp(self):
        freq=rf.Frequency(75,110,101,'ghz')
        self.taper = SmoothStep(
                   med= RectangularWaveguide,
                   med_kw = dict(frequency=freq),
                   param = 'a',
                   start = 10*mil, 
                   stop = 12*mil,
                   length = 1000*mil, 
                   n_sections = 10,
                   )

class ExponentialTestCase(TaperTestCase, unittest.TestCase):
    def setUp(self):
        freq=rf.Frequency(75,110,101,'ghz')
        self.taper = Exponential(
                   med= RectangularWaveguide,
                   med_kw = dict(frequency=freq),
                   param = 'a',
                   start = 10*mil, 
                   stop = 12*mil,
                   length = 1000*mil, 
                   n_sections = 10,
                   )

