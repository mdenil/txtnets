__author__ = 'mdenil'


class CostMinimizationObjective(object):
    """
    The purpose of an objective is to connect a cost function and some data to a model.  The objective stores the cost
    function and data, and given a model it is able to compute the cost and the gradients of the model parameters with
    respect to the cost.
    """
    def __init__(self, cost, data_provider, regularizer=None):
        self.cost = cost
        self.data_provider = data_provider
        self.regularizer = regularizer

    def evaluate(self, model, return_grads=True):
        X, Y, meta = self.data_provider.next_batch()

        Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)
        meta['space_below'] = meta['space_above']
        cost, meta, cost_state = self.cost.fprop(Y_hat, Y, meta=meta)

        if self.regularizer:
            cost += self.regularizer.cost(model)

        if not return_grads:
            return cost

        delta, meta = self.cost.bprop(Y_hat, Y, meta=meta, fprop_state=cost_state)
        grads = model.grads(delta, meta=meta, fprop_state=model_state)

        if self.regularizer:
            for g, rg in zip(grads, self.regularizer.grads(model)):
                g += rg

        return cost, grads


