__author__ = 'mdenil'


class SGD(object):
    def __init__(self, model, objective, update_rule):
        self.model = model
        self.objective = objective
        self.update_rule = update_rule

    def __iter__(self):
        return self

    def next(self):
        iteration_info = {}

        pre_gradient_updates = self.update_rule.pre_gradient_updates()

        if pre_gradient_updates:
            for p, g in zip(self.model.params(), pre_gradient_updates):
                p += g

        cost, grads = self.objective.evaluate(self.model)

        iteration_info['cost'] = cost
        iteration_info['grad_mean_abs_values'] = [self._mean_abs(g) for g in grads]

        updates = self.update_rule.updates(grads)

        iteration_info['updates_mean_abs_values'] = [self._mean_abs(u) for u in updates]

        for p, g in zip(self.model.params(), updates):
            p += g

        iteration_info['param_mean_abs_values'] = [self._mean_abs(p) for p in self.model.params()]

        return iteration_info


    # Implement in subclasses
    def _mean_abs(self, x):
        raise NotImplementedError()
