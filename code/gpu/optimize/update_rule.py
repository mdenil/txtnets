__author__ = 'mdenil'

import numpy as np
import pycuda.autoinit
import pycuda

import generic.optimize.update_rule as generic_rule


class GPUUpdateRuleUtilityMixin(object):
    def _zeros_like(self, x):
        if isinstance(x, pycuda.gpuarray.GPUArray):
            return pycuda.gpuarray.zeros_like(x)
        else:
            return np.zeros_like(x)

    def _sqrt(self, x):
        if isinstance(x, pycuda.gpuarray.GPUArray):
            return pycuda.cumath.sqrt(x)
        else:
            return np.sqrt(x)


class AdaGrad(GPUUpdateRuleUtilityMixin, generic_rule.AdaGrad):
    pass


class AdaDelta(GPUUpdateRuleUtilityMixin, generic_rule.AdaDelta):
    pass
