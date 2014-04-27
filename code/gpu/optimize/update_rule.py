__author__ = 'mdenil'

import numpy as np
import pycuda.autoinit
import pycuda

# FIXME: this is an annoying hack to support GPU and CPU at the same time
def _zeros_like(x):
    if isinstance(x, pycuda.gpuarray.GPUArray):
        return pycuda.gpuarray.zeros_like(x)
    else:
        return np.zeros_like(x)

def _sqrt(x):
    if isinstance(x, pycuda.gpuarray.GPUArray):
        return pycuda.cumath.sqrt(x)
    else:
        return np.sqrt(x)


# FIXME: this should inherit from the generic update rule (I need to make the generic update rule)
class AdaGradUpdateRule(object):
    def __init__(self, model_template, gamma, stabilizer=1e-6):
        self.gamma = gamma
        self._stabilizer = stabilizer

        self.g2s = [_zeros_like(p) for p in model_template.params()]

    def updates(self, grads):
        dxs = []
        for g, g2 in zip(grads, self.g2s):
            g2 += g**2
            dx = - self.gamma / (self._stabilizer + _sqrt(g2)) * g
            dxs.append(dx)

        return dxs

    # FIXME: should be inherited from generic rule
    def pre_gradient_updates(self):
        return None