__author__ = 'mdenil'

import numpy as np
import sys

def _forward_difference(f, x_0, delta, eps, *args):
    return (f(x_0 + eps * delta, *args) - f(x_0, *args)) / eps

def _complex_autodiff(f, x_0, delta, eps, *args):
    return f(x_0 + eps * 1j*delta, *args).imag / eps

def fast_gradient_check(f, g, x_0, method='diff', eps=None, n_checks=10, args=()):
    if method == 'diff':
        eps = 1e-6 if eps is None else eps
        approx_g = _forward_difference

    if method == 'complex':
        eps = sys.float_info.min if eps is None else eps
        approx_g = _complex_autodiff

    max_relative_error = 0.0

    g_x0 = g(x_0, *args)
    for check in xrange(n_checks):
        delta = np.random.standard_normal(size=x_0.size)
        delta /= np.sqrt(np.dot(delta, delta))

        g_exact = np.dot(g_x0, delta)
        g_approx = approx_g(f, x_0, delta, eps, *args)

        # relative_error = np.abs(g_approx - g_exact) / np.abs(g_exact)
        relative_error = np.max(np.abs(g_approx - g_exact))

        max_relative_error = max(relative_error, max_relative_error)

    return max_relative_error


class ModelGradientChecker(object):
    def __init__(self, objective):
        self.objective = objective

    def _f(self, w, model):
        model.unpack(w)
        cost = self.objective.evaluate(model, return_grads=False)
        return cost

    def _g(self, w, model):
        model.unpack(w)
        cost, grads = self.objective.evaluate(model)
        return np.concatenate([g.ravel() for g in grads])

    def check(self, model):
        w = model.pack().copy()
        grad_check = fast_gradient_check(self._f, self._g, model.pack(), method='diff', args=(model,))
        model.unpack(w)
        return grad_check
