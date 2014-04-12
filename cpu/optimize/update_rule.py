__author__ = 'mdenil'

import numpy as np

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
        return None


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
    def __init__(self, model_template, gamma, stabilizer=1e-6):
        self.gamma = gamma
        self._stabilizer = stabilizer

        self.g2s = [np.zeros_like(p) for p in model_template.params()]

    def updates(self, grads):
        dxs = []
        for g, g2 in zip(grads, self.g2s):
            g2 += g**2
            dx = - self.gamma / (self._stabilizer + np.sqrt(g2)) * g
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
