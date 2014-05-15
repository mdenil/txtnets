__author__ = 'mdenil'

import numpy as np
import random
import generic.model.utils

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


def _parallel_shuffle_lists(*lists):
    pack = zip(*lists)
    random.shuffle(pack)
    return zip(*pack)


class ContrastiveMultilingualEmbeddingObjective(object):
    def __init__(self,
                 tagged_parallel_sequence_provider,
                 n_contrastive_samples,
                 margin):
        self.tagged_parallel_sequence_provider = tagged_parallel_sequence_provider
        self.n_contrastive_samples = n_contrastive_samples
        self.margin = margin

    def evaluate(self, tagged_model_collection):
        t1, t2 = random.choice(self.tagged_parallel_sequence_provider.tags)

        m1 = generic.model.utils.ModelEvaluator(
            tagged_model_collection.get_model(t1),
            desired_axes=('b', ('d', 'f', 'w')))
        m2 = generic.model.utils.ModelEvaluator(
            tagged_model_collection.get_model(t2),
            desired_axes=('b', ('d', 'f', 'w')))

        # energy_function = GaussianEnergy()
        # loss_function = ContrastiveHingeLoss(margin=self.margin)
        energy_function = self.__class__.Energy()
        loss_function = self.__class__.LossFunction(margin=self.margin)

        p_parallel = self.tagged_parallel_sequence_provider.get_provider((t1, t2))
        x1_clean, meta1_clean, x2_clean, meta2_clean = p_parallel.next_batch()

        total_loss = 0.0
        grads_1_total = [self._zeros_like(p) for p in m1.model.params()]
        grads_2_total = [self._zeros_like(p) for p in m2.model.params()]

        y1_clean = m1.fprop(x1_clean, meta1_clean)
        y2_clean = m2.fprop(x2_clean, meta2_clean)

        # energy and denerngy are specific
        # e_clean = energy_function.fprop(y1_clean, y2_clean)

        for _ in xrange(self.n_contrastive_samples):
            # new state containers
            m2_noise = generic.model.utils.ModelEvaluator(
                tagged_model_collection.get_model(t2),
                desired_axes=('b', ('d', 'f', 'w')))

            # meta1_noise = dict(meta1_clean)
            # x1_noise = x1_clean
            y1_noise = y1_clean

            meta2_noise = dict(meta2_clean)
            x2_noise, meta2_noise['lengths'] = _parallel_shuffle_lists(x2_clean, meta2_clean['lengths'])
            y2_noise = m2_noise.fprop(x2_noise, meta2_noise)

            e_clean = energy_function.fprop(y1_clean, y2_clean)
            e_noise = energy_function.fprop(y1_noise, y2_noise)

            loss = loss_function.fprop(e_clean, e_noise)
            total_loss += self._mean(loss)

            dloss_clean, dloss_noise = loss_function.bprop(e_clean, e_noise)

            denergy_1_clean, denergy_2_clean = energy_function.bprop(y1_clean, y2_clean, dloss_clean)
            denergy_1_noise, denergy_2_noise = energy_function.bprop(y1_noise, y2_noise, dloss_noise)

            grads_1_clean = m1.grads(denergy_1_clean)
            grads_1_noise = m1.grads(denergy_1_noise)

            grads_2_clean = m2.grads(denergy_2_clean)
            grads_2_noise = m2_noise.grads(denergy_2_noise)

            for g, g_clean, g_noise in zip(grads_1_total, grads_1_clean, grads_1_noise):
                g += g_clean
                g += g_noise

            for g, g_clean, g_noise in zip(grads_2_total, grads_2_clean, grads_2_noise):
                g += g_clean
                g += g_noise

        full_grads = tagged_model_collection.full_grads_from_tagged_grads({
            t1: grads_1_total,
            t2: grads_2_total,
        })

        return total_loss, full_grads