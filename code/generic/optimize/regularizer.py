__author__ = 'mdenil'


class L2Regularizer(object):
    def __init__(self, lamb):
        self.lamb = lamb

    def cost(self, model):
        return sum(self.lamb * self._sum(p**2) for p in model.params())

    def grads(self, model):
        return [2.0 * self.lamb * p for p in model.params()]