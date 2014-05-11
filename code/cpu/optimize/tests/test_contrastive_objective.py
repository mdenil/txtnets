__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import unittest
import cpu.optimize.objective
import cpu.model.model
import cpu.optimize.data_provider
import generic.optimize.data_provider
import cpu.model.layer


class IdentityModel(cpu.model.layer.Layer):

    def fprop(self, X, meta, return_state=False):
        meta['space_above'] = meta['space_below']
        if return_state:
            return X, meta, {}
        else:
            return X

    def bprop(self, delta, meta, fprop_state):
        meta['space_below'] = meta['space_above']
        return delta, meta

    # These are the grads wrt X but I want these to be returned from the objective so I'm returning them here
    def grads(self, delta, meta, fprop_state):
        return [delta]


class ParallelMatrixProvider(object):
    def __init__(self, X1, X2):
        self.X1 = X1
        self.X2 = X2

    def next_batch(self):
        meta1 = {
            'space_below': cpu.space.CPUSpace.infer(self.X1, ('b', 'w'))
        }
        meta2 = {
            'space_below': cpu.space.CPUSpace.infer(self.X2, ('b', 'w'))
        }
        return self.X1, meta1, self.X2, meta2


class SingleMatrixProvider(object):
    def __init__(self, X):
        self.X = X

    def next_batch(self):
        meta = {
            'space_below': cpu.space.CPUSpace.infer(self.X, ('b', 'w'))
        }
        return self.X, meta


class ContrastiveMultilingualEmbeddingObjective(unittest.TestCase):
    def setUp(self):
        self.models = cpu.model.model.TaggedModelCollection({
            'en': IdentityModel(),
            'de': IdentityModel(),
        })

        X1 = np.random.standard_normal(size=(10, 5))
        X2 = np.random.standard_normal(size=(10, 5))

        self.tagged_parallel_provider = generic.optimize.data_provider.TaggedProviderCollection({
            ('en', 'de'): ParallelMatrixProvider(X1=X1, X2=X2)
        })

        self.contrastive_provider = generic.optimize.data_provider.TaggedProviderCollection({
            'en': SingleMatrixProvider(X1),
            'de': SingleMatrixProvider(X2),
        })

        self.contrastive_objective = cpu.optimize.objective.ContrastiveMultilingualEmbeddingObjective(
            tagged_parallel_sequence_provider=self.tagged_parallel_provider,
            tagged_contrastive_sequence_provider=self.contrastive_provider,
            n_contrastive_samples=1,
            margin=1.0)

    def test_me(self):
        print
        print self.contrastive_objective.evaluate(self.models)
        print self.contrastive_objective.evaluate(self.models)
        print self.contrastive_objective.evaluate(self.models)
        print self.contrastive_objective.evaluate(self.models)
        print self.contrastive_objective.evaluate(self.models)