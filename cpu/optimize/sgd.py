__author__ = 'mdenil'

import numpy as np

class SGD(object):
    def __init__(self, model, objective, update_rule):
        self.model = model
        self.objective = objective
        self.update_rule = update_rule

    def __iter__(self):
        return self

    def next(self):
        pre_gradient_updates = self.update_rule.pre_gradient_updates()

        for p, g in zip(self.model.params(), pre_gradient_updates):
            p += g

        cost, grads = self.objective.evaluate(self.model)
        updates = self.update_rule.updates(grads)

        for p, g in zip(self.model.params(), updates):
            p += g

        return {
            'cost': cost,
            'model': self.model,
        }





class UpdateRule(object):
    """
    An update rule connects gradient values to parameter updates.  Given gradients for the current model parameters it
    computes updates which tell the optimizer which direction to move.  This is where things like learning rates live.
    This is also the place to put negative signs if you want to _minimize_ the cost.

    Update rules can be stateful.  The updates function will be called exactly once per update.
    """

    def updates(self, grads):
        raise NotImplementedError()

    def pre_gradient_updates(self):
        return []


class BasicUpdateRule(UpdateRule):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def updates(self, grads):
        return [-self.learning_rate * g for g in grads]


class Momentum(UpdateRule):
    def __init__(self, momentum, epsilon, model_template):
        self.momentum = momentum
        self.epsilon = epsilon

        self.dxs = [np.zeros_like(p) for p in model_template.params()]

    def updates(self, grads):
        for g, dx in zip(grads, self.dxs):
            dx *= self.momentum
            dx -= self.epsilon * g
        return self.dxs

class NAG(UpdateRule):
    def __init__(self, momentum, epsilon, model_template):
        self.momentum = momentum
        self.epsilon = epsilon

        self.dxs = [np.zeros_like(p) for p in model_template.params()]

    def pre_gradient_updates(self):
        for dx in self.dxs:
            dx *= self.momentum
        return self.dxs

    def updates(self, grads):
        new_updates = []
        for g, dx in zip(grads, self.dxs):
            grad_adjustment = -self.epsilon * g
            new_updates.append(grad_adjustment)
            dx += grad_adjustment

        return new_updates

class AdaGradUpdateRule(UpdateRule):
    def __init__(self, model_template, eta, rho, epsilon):
        self.rho = rho
        self.epsilon = epsilon
        self.eta = eta

        self.g2s = [np.zeros_like(p) for p in model_template.params()]

    def updates(self, grads):
        dxs = []
        for g, g2 in zip(grads, self.g2s):
            g2 *= self.rho
            g2 += (1-self.rho) * g**2

            rms_g = np.sqrt(g2 + self.epsilon)
            dx = - self.eta / rms_g * g

            dxs.append(dx)

        return dxs

class AdaDeltaUpdateRule(UpdateRule):
    def __init__(self, model_template, rho, epsilon):
        self.rho = rho
        self.epsilon = epsilon

        self.g2s = [np.zeros_like(p) for p in model_template.params()]
        self.dx2s = [np.zeros_like(p) for p in model_template.params()]

    def updates(self, grads):
        dxs = []
        for g, g2, dx2 in zip(grads, self.g2s, self.dx2s):
            g2 *= self.rho
            g2 += (1-self.rho) * g**2

            rms_g = np.sqrt(g2 + self.epsilon)
            rms_dx = np.sqrt(dx2 + self.epsilon)
            dx = - rms_dx / rms_g * g

            dx2 *= self.rho
            dx2 += (1-self.rho) * dx**2

            dxs.append(dx)

        return dxs
