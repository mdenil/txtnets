__author__ = 'mdenil'

import numpy as np
import random

import generic.model.utils

from generic.optimize.objective import CostMinimizationObjective
from generic.optimize.objective import NoiseContrastiveObjective


def energy(x, y):
    return np.sum(0.5 * (x - y)**2, axis=1)


def denergy(x, y):
    dx = x
    dy = -y
    return dx, dy


class ContrastiveMultilingualEmbeddingObjective(object):
    def __init__(self,
                 tagged_parallel_sequence_provider,
                 tagged_contrastive_sequence_provider,
                 n_contrastive_samples,
                 margin):
        self.tagged_parallel_sequence_provider = tagged_parallel_sequence_provider
        self.tagged_contrastive_sequence_provider = tagged_contrastive_sequence_provider
        self.n_contrastive_samples = n_contrastive_samples
        self.margin = margin

    def evaluate(self, tagged_model_collection):
        # TODO: worry about the order of these
        t1, t2 = random.choice(self.tagged_parallel_sequence_provider.tags)

        m1 = generic.model.utils.ModelEvaluator(tagged_model_collection.get_model(t1))
        m2 = generic.model.utils.ModelEvaluator(tagged_model_collection.get_model(t2))

        # tagged_parallel_sequence_provider contains ParallelProvider's for each language pair
        p1, p2 = self.tagged_parallel_sequence_provider.get_provider((t1, t2)).providers

        total_cost = 0.0
        total_grads_1 = [np.zeros_like(p) for p in m1.model.params()]
        total_grads_2 = [np.zeros_like(p) for p in m2.model.params()]

        positive_embeddings_1 = m1.fprop(p1)
        positive_embeddings_2 = m2.fprop(p2)

        positive_energies = energy(positive_embeddings_1, positive_embeddings_2)
        denergy_pos_1, denergy_pos_2 = denergy(positive_embeddings_1, positive_embeddings_2)

        for _ in xrange(self.n_contrastive_samples):
            # new state containers
            m2_negative = generic.model.utils.ModelEvaluator(tagged_model_collection.get_model(t2))

            # tagged_contrastive_sequence_provider contains one provider for each language
            p2_negative = self.tagged_contrastive_sequence_provider.get_provider(t2)

            negative_embeddings_2 = m2_negative.fprop(p2_negative)
            negative_energies = energy(positive_embeddings_1, negative_embeddings_2)

            costs = np.maximum(0.0, self.margin + positive_energies - negative_energies)
            cost_switches = np.reshape(costs > 0.0, (-1, 1))

            total_cost += costs.sum()

            dmodel_pos_1 = m1.grads(cost_switches * denergy_pos_1)
            dmodel_pos_2 = m2.grads(cost_switches * denergy_pos_2)

            denergy_neg_1, denergy_neg_2 = denergy(positive_embeddings_1, negative_embeddings_2)
            dmodel_neg_1 = m1.grads(cost_switches * denergy_neg_1)
            dmodel_neg_2 = m2.grads(cost_switches * denergy_neg_2)

            for g, gpos, gneg in zip(total_grads_1, dmodel_pos_1, dmodel_neg_1):
                g += gpos
                g += gneg

            for g, gpos, gneg in zip(total_grads_2, dmodel_pos_2, dmodel_neg_2):
                g += gpos
                g += gneg

        full_grads = tagged_model_collection.full_grads_from_tagged_grads({
            t1: total_grads_1,
            t2: total_grads_2,
        })

        return total_cost, full_grads