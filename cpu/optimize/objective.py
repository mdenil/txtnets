__author__ = 'mdenil'



class CostMinimizationObjective(object):
    """
    The purpose of an objective is to connect a cost function and some data to a model.  The objective stores the cost
    function and data, and given a model it is able to compute the cost and the gradients of the model parameters with
    respect to the cost.
    """
    def __init__(self, cost, X, Y):
        self.cost = cost
        self.X = X
        self.Y = Y

    def evaluate(self, model):
        Y = model.fprop(self.X)
        cost = self.cost.fprop(Y, self.Y)
        delta = self.cost.bprop(Y, self.Y)
        grads = model.grads(self.X, delta)

        return cost, grads