import unittest

import numpy as np
from numpy import exp, linspace, log, sqrt
from numpy.testing import assert_allclose, assert_equal
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import iv

import skrf as rf
from skrf.media import CPW


class Taper1DTestCase(unittest.TestCase):
    @staticmethod
    def f(x: NDArray[np.float64], a: float = 1) -> NDArray[np.float64]:
        return a * x**2

    def setUp(self):
        self.frequency = rf.Frequency(1, 50, 101, "GHz")
        self.start = 10e-6
        self.stop = 530e-6
        self.n_sections = 30
        self.length = 1
        self.taper = rf.taper.Taper1D(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            f=self.f,
            length=self.length,
            length_unit="mm",
            param="w",
            f_is_normed=True,
            med_kw={"frequency": self.frequency, "s": 6e-6},
            f_kw={"a": 0.3},
        )

    def test_value_vector(self):
        expected_value_vector = self.f(linspace(0, 1, self.n_sections), 0.3) * (self.stop - self.start) + self.start
        assert_equal(self.taper.value_vector, expected_value_vector)

    def test_section_length(self):
        expected_section_length = self.length / self.n_sections
        assert_equal(self.taper.section_length, expected_section_length)

    def test_media_at(self):
        expected_impedances = CPW(frequency=self.frequency, w=123e-6, s=6e-6).z0
        assert_equal(self.taper.media_at(123e-6).z0, expected_impedances)

    def test_section_at(self):
        expected_impedances = np.repeat(CPW(frequency=self.frequency, w=123e-6, s=6e-6).z0[:, np.newaxis], 2, axis=1)
        assert_equal(self.taper.section_at(123e-6).z0, expected_impedances)


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(1, 50, 101, "GHz")
        self.start = 10
        self.stop = 100
        self.n_sections = 30
        self.length = 1
        self.taper = rf.taper.Linear(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            length=self.length,
            length_unit="mm",
            param="z0_override",
            med_kw={"frequency": self.frequency},
        )

    def test_value_vector(self):
        expected_value_vector = linspace(self.start, self.stop, self.n_sections)
        assert_allclose(self.taper.value_vector, expected_value_vector)

    def test_network(self):
        assert_equal(self.taper.network.z0[15], [10.0 + 0.0j, 100.0 + 0.0j])


class ExponentialTestCase(unittest.TestCase):
    def setUp(self):
        self.frequency = rf.Frequency(1, 50, 101, "GHz")
        self.start = 10e-6
        self.stop = 87e-6
        self.n_sections = 23
        self.length = 1
        self.taper = rf.taper.Exponential(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            length=self.length,
            length_unit="mm",
            param="s",
            med_kw={"frequency": self.frequency, "w": 10e-6},
        )

    def test_value_vector(self):
        expected_value_vector = self.start * exp(
            linspace(0, self.length, self.n_sections) / self.length * log(self.stop / self.start)
        )
        assert_equal(self.taper.value_vector, expected_value_vector)


class SmoothStepTestCase(unittest.TestCase):
    @staticmethod
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 3 * x**2 - 2 * x**3

    def setUp(self):
        self.frequency = rf.Frequency(1, 50, 101, "GHz")
        self.start = 11.7
        self.stop = 4.6
        self.n_sections = 30
        self.length = 0.3
        self.taper = rf.taper.SmoothStep(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            length=self.length,
            length_unit="mm",
            param="ep_r",
            med_kw={"frequency": self.frequency, "w": 10e-6, "s": 6e-6},
        )

    def test_value_vector(self):
        expected_value_vector = self.f(linspace(0, 1, self.n_sections)) * (self.stop - self.start) + self.start
        assert_equal(self.taper.value_vector, expected_value_vector)


class KlopfensteinTestCase(unittest.TestCase):
    @staticmethod
    def phi(z: NDArray[np.float64], a: float) -> NDArray[np.float64]:
        return np.array([quad(lambda y: iv(1, a * sqrt(1 - y**2)) / a / sqrt(1 - y**2), 0, zi)[0] for zi in z])

    @staticmethod
    def f(x: NDArray[np.float64], length: float, start: float, stop: float, rmax: float) -> NDArray[np.float64]:
        z = x - length / 2
        log_ratio = log(stop / start) / 2
        a = np.arccosh(1 / rmax)
        log_value = log(start * stop) / 2 + log_ratio / np.cosh(a) * (
            a * a * KlopfensteinTestCase.phi(2 * z / length, a)
            + np.heaviside(z - length / 2, 1)
            + np.heaviside(z + length / 2, 1)
            - 1
        )
        return exp(log_value)

    def setUp(self):
        self.frequency = rf.Frequency(1, 50, 101, "GHz")
        self.start = 250e-6
        self.stop = 10e-6
        self.n_sections = 10
        self.length = 0.5
        self.taper1 = rf.taper.Klopfenstein(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            length=self.length,
            length_unit="mm",
            param="w",
            med_kw={"frequency": self.frequency, "s": 6e-6},
        )
        self.taper2 = rf.taper.Klopfenstein(
            med=CPW,
            start=self.start,
            stop=self.stop,
            n_sections=self.n_sections,
            length=self.length,
            length_unit="mm",
            param="w",
            med_kw={"frequency": self.frequency, "s": 6e-6},
            f_kw={"rmax": 0.01},
        )

    def test_value_vector(self):
        expected_value_vector = self.f(
            linspace(0, self.length, self.n_sections), self.length, self.start, self.stop, 0.05
        )
        assert_equal(self.taper1.value_vector, expected_value_vector)

    def test_rmax_argument(self):
        expected_value_vector = self.f(
            linspace(0, self.length, self.n_sections), self.length, self.start, self.stop, 0.01
        )
        assert_equal(self.taper2.value_vector, expected_value_vector)
