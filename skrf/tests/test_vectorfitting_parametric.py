import numpy as np
import unittest
import pytest
import tempfile
import os
from skrf.network import Network
from skrf.networkSet import NetworkSet
from skrf.vectorFitting import VectorFittingParametric
from pathlib import Path


class VectorFittingParametricTestCase(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).parent.parent.parent / 'doc/source/examples/vectorfitting_parametric'
        networks = [Network(folder / 'spiral_D80.s2p', params={'D': 80}),
                    Network(folder / 'spiral_D100.s2p', params={'D': 100}),
                    Network(folder / 'spiral_D120.s2p', params={'D': 120}),
                    Network(folder / 'spiral_D140.s2p', params={'D': 140}),
                    Network(folder / 'spiral_D160.s2p', params={'D': 160}),
                    Network(folder / 'spiral_D180.s2p', params={'D': 180}),
                    Network(folder / 'spiral_D200.s2p', params={'D': 200})]
        self.nwset = NetworkSet(networks)
        self.vf = VectorFittingParametric(self.nwset, 3, 0)
        self.vf.auto_fit(enforce_passivity=False)

    def test_error(self):
        for nw in self.nwset:
            vf_response = self.vf.get_model_response(nw.params, nw.f)
            error_rms = np.sqrt(np.mean(np.square(np.abs(vf_response - nw.s))))
            self.assertLess(error_rms, 0.01)

    def test_read_write_npz(self):
        # export (write) fitted parameters to .npz file in tmp directory
        with tempfile.TemporaryDirectory() as pathname:
            self.vf.write_npz(path=pathname, filename='spiral')

            # create a new vector fitting instance and import (read) those fitted parameters
            vf2 = VectorFittingParametric(None)
            vf2.read_npz(os.path.join(pathname, 'spiral.npz'))

        for nw in self.nwset:
            vf_response = self.vf.get_model_response(nw.params, nw.f)
            vf2_response = vf2.get_model_response(nw.params, nw.f)
            self.assertTrue(np.allclose(vf_response, vf2_response))
