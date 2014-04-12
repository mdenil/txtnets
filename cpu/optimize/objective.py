__author__ = 'mdenil'

import numpy as np

class CostMinimizationObjective(object):
    """
    The purpose of an objective is to connect a cost function and some data to a model.  The objective stores the cost
    function and data, and given a model it is able to compute the cost and the gradients of the model parameters with
    respect to the cost.
    """
    def __init__(self, cost, data_provider):
        self.cost = cost
        self.data_provider = data_provider

    def evaluate(self, model):
        X, Y, meta = self.data_provider.next_batch()

        Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)
        meta['space_below'] = meta['space_above']
        cost, meta, cost_state = self.cost.fprop(Y_hat, Y, meta=meta)
        delta, meta = self.cost.bprop(Y_hat, Y, meta=meta, fprop_state=cost_state)
        grads = model.grads(delta, meta=meta, fprop_state=model_state)

        return cost, grads
