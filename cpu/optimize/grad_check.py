__author__ = 'mdenil'

import numpy as np
import sys

def _forward_difference(f, x_0, delta, eps):
    return (f(x_0 + eps * delta) - f(x_0)) / eps

def _complex_autodiff(f, x_0, delta, eps):
    return f(x_0 + eps * 1j*delta).imag / eps

def fast_gradient_check(f, g, x_0, method='diff', eps=None, n_checks=10):
    if method == 'diff':
        eps = 1e-8 if eps is None else eps
        approx_g = _forward_difference

    if method == 'complex':
        eps = sys.float_info.min if eps is None else eps
        approx_g = _complex_autodiff

    max_relative_error = 0.0

    g_x0 = g(x_0)
    for check in xrange(n_checks):
        delta = np.random.standard_normal(size=x_0.size)
        delta /= np.sqrt(np.dot(delta, delta))

        g_exact = np.dot(g_x0, delta)
        g_approx = approx_g(f, x_0, delta, eps)

        relative_error = np.abs(g_approx - g_exact) / np.abs(g_exact)

        max_relative_error = max(relative_error, max_relative_error)

    return max_relative_error


if __name__ == "__main__":
    A = np.random.standard_normal(size=(30, 30))
    A = A + A.T

    def f(x):
        return 0.5 * np.dot(x, np.dot(A, x))

    def g(x):
        return np.dot(A, x)

    print fast_gradient_check(f, g, np.random.standard_normal(size=A.shape[0]), method='complex')