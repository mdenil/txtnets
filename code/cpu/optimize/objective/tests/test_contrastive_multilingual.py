__author__ = 'mdenil'

import numpy as np
import unittest

import scipy.optimize

from cpu.optimize.objective.contrastive_multilingual import GaussianEnergy
from cpu.optimize.objective.contrastive_multilingual import ContrastiveHingeLoss
from cpu.optimize.objective.contrastive_multilingual import SquareExponentialLoss
from cpu.optimize.objective.contrastive_multilingual import SquareSquareMarginLoss


class TestGaussianEnergy(unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.d = 25

        self.objective = GaussianEnergy()
        self.x = np.random.standard_normal(size=(self.b, self.d))
        self.y = np.random.standard_normal(size=(self.b, self.d))

    def test_fprop(self):
        actual = self.objective.fprop(self.x, self.y)
        expected = 0.5 * np.sum((self.x - self.y)**2, axis=1, keepdims=True)

        self.assertTrue(np.allclose(actual, expected))

    def test_bprop_x(self):
        c = np.random.standard_normal(size=(self.b, 1))

        def func(x):
            x = x.reshape(self.x.shape)
            z = self.objective.fprop(x, self.y)
            return np.dot(z.T, c)

        def grad(x):
            x = x.reshape(self.x.shape)
            z, _ = self.objective.bprop(x, self.y, c)
            return z.ravel()

        actual = grad(self.x.ravel()).reshape(self.x.shape)
        expected = scipy.optimize.approx_fprime(self.x.ravel(), func, 1e-6).reshape(self.x.shape)

        self.assertTrue(np.allclose(actual, expected, atol=1e-5))

    def test_bprop_y(self):
        c = np.random.standard_normal(size=(self.b, 1))

        def func(y):
            y = y.reshape(self.x.shape)
            z = self.objective.fprop(self.x, y)
            return np.dot(c.T, z)

        def grad(y):
            y = y.reshape(self.x.shape)
            _, z = self.objective.bprop(self.x, y, c)
            return z.ravel()

        actual = grad(self.y.ravel()).reshape(self.y.shape)
        expected = scipy.optimize.approx_fprime(self.y.ravel(), func, 1e-6).reshape(self.y.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5))


class LossFunctionCommon(object):
    def _check_test_data(self):
        pass

    def test_bprop_x(self):
        self._check_test_data()

        def func(x):
            x = x.reshape(self.x.shape)
            z = self.objective.fprop(x, self.y)
            return z

        def grad(x):
            x = x.reshape(self.x.shape)
            z, _ = self.objective.bprop(x, self.y, 1.0)
            return z.ravel()

        actual = grad(self.x.ravel()).reshape(self.x.shape)
        expected = scipy.optimize.approx_fprime(self.x.ravel(), func, 1e-6).reshape(self.x.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5))

    def test_bprop_y(self):
        self._check_test_data()

        def func(y):
            y = y.reshape(self.x.shape)
            z = self.objective.fprop(self.x, y)
            return z

        def grad(y):
            y = y.reshape(self.x.shape)
            _, z = self.objective.bprop(self.x, y, 1.0)
            return z.ravel()

        actual = grad(self.y.ravel()).reshape(self.y.shape)
        expected = scipy.optimize.approx_fprime(self.y.ravel(), func, 1e-6).reshape(self.y.shape)

        self.assertTrue(np.allclose(actual, expected, atol=1e-5))


class TestContrastiveHingeLoss(LossFunctionCommon, unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.margin = 1.0

        self.objective = ContrastiveHingeLoss(margin=self.margin)
        self.x = np.random.standard_normal(size=(self.b, 1)) ** 2
        self.y = np.random.standard_normal(size=(self.b, 1)) ** 2

    def test_fprop(self):
        self._check_test_data()

        actual = self.objective.fprop(self.x, self.y)
        expected = np.mean(np.maximum(0.0, self.margin + self.x - self.y))

        self.assertTrue(np.allclose(actual, expected))

    def _check_test_data(self):
        # These are sanity checks for the test data.  If the test fails in this function
        # just re-run it.
        value = self.margin + self.x - self.y
        self.assertTrue(np.any(value > 0))
        self.assertTrue(np.any(value < 0))


class TestSquareExponentialLoss(LossFunctionCommon, unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.margin = 1.0

        self.objective = SquareExponentialLoss(margin=self.margin)
        self.x = np.random.standard_normal(size=(self.b, 1)) ** 2
        self.y = np.random.standard_normal(size=(self.b, 1)) ** 2


class TestSquareSquareMarginLoss(LossFunctionCommon, unittest.TestCase):
    def setUp(self):
        self.b = 100
        self.margin = 1.0

        self.objective = SquareSquareMarginLoss(margin=self.margin)
        self.x = np.random.standard_normal(size=(self.b, 1)) ** 2
        self.y = np.random.standard_normal(size=(self.b, 1)) ** 2


class TestContrastiveMultilingualEmbeddingObjective(unittest.TestCase):
    def setUp(self):
        pass

    def test_fprop(self):
        self.skipTest("WRITE ME")

    def test_bprop(self):
        self.skipTest("WRITE ME")