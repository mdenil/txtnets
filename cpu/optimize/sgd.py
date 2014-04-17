__author__ = 'mdenil'

import numpy as np

class SGD(object):
    def __init__(self, model, objective, update_rule, regularizer=None):
        self.model = model
        self.objective = objective
        self.update_rule = update_rule
        self.regularizer = regularizer

    def __iter__(self):
        return self

    def next(self):
        iteration_info = {}

        pre_gradient_updates = self.update_rule.pre_gradient_updates()

        if pre_gradient_updates:
            for p, g in zip(self.model.params(), pre_gradient_updates):
                p += g

        cost, grads = self.objective.evaluate(self.model)

        if self.regularizer:
            reg_cost, reg_grads = self.regularizer.regularize(self.model)

            cost += reg_cost
            iteration_info['reg_cost'] = reg_cost

            for g, rg in zip(grads, reg_grads):
                g += rg

        iteration_info['cost'] = cost
        iteration_info['grad_mean_abs_values'] = [np.mean(np.abs(g)) for g in grads]

        updates = self.update_rule.updates(grads)

        iteration_info['updates_mean_abs_values'] = [np.mean(np.abs(u)) for u in updates]

        for p, g in zip(self.model.params(), updates):
            p += g

        iteration_info['param_mean_abs_values'] = [np.mean(np.abs(p)) for p in self.model.params()]

        return iteration_info




