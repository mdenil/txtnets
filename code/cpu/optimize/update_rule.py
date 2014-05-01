__author__ = 'mdenil'

import numpy as np

import generic.optimize.update_rule as generic_rule


class CPUUpdateRuleUtilityMixin(object):
    def _zeros_like(self, x):
        return np.zeros_like(x)

    def _sqrt(self, x):
        return np.sqrt(x)


class Basic(CPUUpdateRuleUtilityMixin, generic_rule.Basic):
    pass


class Momentum(CPUUpdateRuleUtilityMixin, generic_rule.Momentum):
    pass


class NesterovAcceleratedGradient(CPUUpdateRuleUtilityMixin, generic_rule.NesterovAcceleratedGradient):
    pass


class AdaGrad(CPUUpdateRuleUtilityMixin, generic_rule.AdaGrad):
    pass


class AdaDelta(CPUUpdateRuleUtilityMixin, generic_rule.AdaDelta):
    pass