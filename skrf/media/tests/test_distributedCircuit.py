import os
import tempfile
import unittest

import numpy as np
import pytest

from skrf.media import DistributedCircuit
from skrf.network import Frequency, Network


class MediaTestCase(unittest.TestCase):
    """

    """
    def setUp(self):
        """

        """
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
        fname = os.path.join(self.files_dir,\
                'distributedCircuit,line1mm.s2p')
        qucs_ntwk = Network(fname)
        self.a_media = DistributedCircuit(
                frequency = qucs_ntwk.frequency,
                R=1e5, G=1, L=1e-6, C=8e-12
                )

    def test_constructor(self):
        "Test various initializations"
        nb_f = 3
        freq = Frequency(1, 10, npoints=nb_f, unit='MHz')
        # C, L, R, G all scalars
        media = DistributedCircuit(frequency=freq, C=100e-12, L=50e-9, R=1, G=0)
        print(media)
        # all parameters are arrays
        media = DistributedCircuit(frequency=freq, C=[100e-12]*nb_f, L=[50e-9]*nb_f, R=[1]*nb_f, G=[0]*nb_f)
        print(media)
        media = DistributedCircuit(frequency=freq, C=100e-12, L=50e-9, R=np.linspace(1, 10, nb_f), G=0)
        print(media)
        with pytest.raises(ValueError):  # mixed arrays sizes
            media = DistributedCircuit(frequency=freq, C=[100e-12] * 1, L=[50e-9] * 2, R=[1] * nb_f, G=0)

    def test_line(self):
        """
        """
        fname = os.path.join(self.files_dir,\
                'distributedCircuit,line1mm.s2p')
        qucs_ntwk = Network(fname)

        a_media = DistributedCircuit(
            frequency = qucs_ntwk.frequency,
            R=1e5, G=1, L=1e-6, C=8e-12
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.line(1e-3,'m')\
                    **a_media.thru(z0=50)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


    def test_write_csv(self):
        with tempfile.TemporaryDirectory() as tempdir:
             fname = os.path.join(tempdir, 'out.csv')
             self.a_media.write_csv(fname)
             a_media = DistributedCircuit.from_csv(fname)
             self.assertEqual(a_media,self.a_media)