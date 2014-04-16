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


class NoiseContrastiveObjective(object):
    def __init__(self, cost, data_provider, noise_model):

        self.cost = cost
        self.data_provider = data_provider
        self.noise_model = noise_model

    def evaluate(self, model):
        X_clean, meta_clean_orig = self.data_provider.next_batch()

        all_grads = []

        for _ in xrange(5):
            meta_clean = dict(meta_clean_orig)
            X_dirty, meta_dirty = self.noise_model.apply(X_clean, meta=dict(meta_clean))

            Y_hat_clean, meta_clean, model_state_clean = model.fprop(X_clean, meta=dict(meta_clean), return_state=True)
            Y_hat_dirty, meta_dirty, model_state_dirty = model.fprop(X_dirty, meta=dict(meta_dirty), return_state=True)

            meta_clean['space_below'] = meta_clean['space_above']
            meta_dirty['space_below'] = meta_dirty['space_above']

            cost, meta, cost_state = self.cost.fprop(Y_hat_clean, Y_hat_dirty, meta=dict(meta_clean)) # not allowed for meta_dirty to be different
            delta_clean, delta_dirty, meta = self.cost.bprop(Y_hat_clean, Y_hat_dirty, meta=dict(meta), fprop_state=cost_state)

            grads_clean = model.grads(delta_clean, meta=dict(meta), fprop_state=model_state_clean)
            grads_dirty = model.grads(delta_dirty, meta=dict(meta), fprop_state=model_state_dirty)

            # FIXME: abstract this
            grads = []
            for gc, gd in zip(grads_clean, grads_dirty):
                grads.append(gc + gd)
            all_grads.append(grads)

        grads = []
        for gs in zip(*all_grads):
            grads.append(sum(gs))

        return cost, grads
