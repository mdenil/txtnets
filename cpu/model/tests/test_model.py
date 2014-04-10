__author__ = 'mdenil'

import numpy as np
import scipy.optimize

from collections import OrderedDict

import unittest
from cpu import model
from cpu import space

class ModelBProp(unittest.TestCase):
    def setUp(self):
        embedding_dimension = 4
        n_feature_maps = 2
        kernel_width = 4
        pooling_size = 4
        batch_size = 3
        sentence_length = 10
        n_classes = 6

        self.X = np.random.standard_normal(size=(sentence_length, embedding_dimension, batch_size))
        self.Y = np.random.randint(n_classes, size=batch_size)
        self.Y = np.equal.outer(np.arange(n_classes), self.Y).astype(self.Y.dtype)

        self.meta = {
            'space_below': space.Space.infer(self.X, ['w', 'd', 'b']),
            'lengths': np.random.randint(1, sentence_length, size=batch_size),
            }

        self.model = model.model.CSM(
            layers=[
                model.transfer.SentenceConvolution(
                    n_feature_maps=n_feature_maps,
                    kernel_width=kernel_width,
                    n_input_dimensions=embedding_dimension),
                model.pooling.SumFolding(),
                model.pooling.KMaxPooling(k=pooling_size),
                model.transfer.Bias(
                    n_input_dims=embedding_dimension / 2,
                    n_feature_maps=n_feature_maps),
                model.nonlinearity.Tanh(),
                model.transfer.Softmax(
                    n_classes=n_classes,
                    n_input_dimensions=n_feature_maps*pooling_size*embedding_dimension / 2),
                ],
            )
        self.cost = model.cost.CrossEntropy()


    def test_bprop(self):
        def func(x):
            X = x.reshape(self.X.shape)
            Y, meta, fprop_state = self.model.fprop(X, meta=dict(self.meta), return_state=True)
            cost, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            return cost

        def grad(x):
            X = x.reshape(self.X.shape)

            Y, meta, fprop_state = self.model.fprop(X, meta=dict(self.meta), return_state=True)
            cost, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            delta, meta = self.cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
            delta, meta = self.model.bprop(delta, meta=dict(meta), fprop_state=fprop_state, return_state=True)

            delta, _ = meta['space_below'].transform(delta, self.meta['space_below'].axes)
            return delta.ravel()

        assert scipy.optimize.check_grad(func, grad, self.X.ravel()) < 1e-5


class Model(unittest.TestCase):
    def setUp(self):
        embedding_dimension = 4
        n_feature_maps = 1
        kernel_width = 4
        pooling_size = 2
        batch_size = 3
        sentence_length = 10
        n_classes = 6
        vocabulary_size = 50

        self.X = np.random.randint(vocabulary_size, size=(batch_size, sentence_length))
        self.Y = np.random.randint(n_classes, size=batch_size)
        self.Y = np.equal.outer(np.arange(n_classes), self.Y).astype(self.Y.dtype)

        self.meta = {
            'space_below': space.Space.infer(self.X, ['b', 'w']),
            'lengths': np.random.randint(1, sentence_length, size=batch_size),
        }

        self.model = model.model.CSM(
            layers=[
                model.embedding.WordEmbedding(
                    dimension=embedding_dimension,
                    vocabulary_size=vocabulary_size),

                model.transfer.SentenceConvolution(
                    n_feature_maps=n_feature_maps,
                    kernel_width=kernel_width,
                    n_input_dimensions=embedding_dimension),

                model.transfer.Bias(
                    n_input_dims=embedding_dimension,
                    n_feature_maps=n_feature_maps),

                model.pooling.SumFolding(),

                model.pooling.KMaxPooling(k=pooling_size),

                model.transfer.Bias(
                    n_input_dims=embedding_dimension / 2,
                    n_feature_maps=n_feature_maps),

                model.nonlinearity.Tanh(),

                model.transfer.Softmax(
                    n_classes=n_classes,
                    n_input_dimensions=n_feature_maps*pooling_size*embedding_dimension / 2),
                ],
            )
        self.cost = model.cost.CrossEntropy()

    def test_misc(self):
        Y, meta, fprop_state = self.model.fprop(self.X, self.meta, return_state=True)
        cost, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
        delta, meta = self.cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
        grads = self.model.grads(delta, meta=dict(meta), fprop_state=fprop_state)

        self.assertEqual(len(grads), len(self.model.params()))

        for p, g in zip(self.model.params(), grads):
            self.assertEqual(p.shape, g.shape)

    def test_grad(self):
        def func(w):
            self.model.unpack(w)

            Y, meta, fprop_state = self.model.fprop(self.X, meta=dict(self.meta), return_state=True)
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))

            return c

        def grad(w):
            self.model.unpack(w)

            Y, meta, fprop_state = self.model.fprop(self.X, meta=dict(self.meta), return_state=True)
            c, meta, cost_state = self.cost.fprop(Y, self.Y, meta=dict(meta))
            delta, meta = self.cost.bprop(Y, self.Y, meta=dict(meta), fprop_state=cost_state)
            grads = self.model.grads(delta, meta=dict(meta), fprop_state=fprop_state)

            return np.concatenate([g.ravel() for g in grads])

        assert scipy.optimize.check_grad(func, grad, self.model.pack()) < 1e-5


    def test_pack_unpack(self):
        from copy import deepcopy

        packed = self.model.pack()
        new_model = deepcopy(self.model)

        # verify unpacking noise makes the models different
        new_model.unpack(np.random.standard_normal(size=packed.shape))
        for p1, p2 in zip(self.model.params(), new_model.params()):
            assert p1 is not p2
            assert not np.all(p1 == p2)

        # verify unpacking the original model makes them equal again
        new_model.unpack(packed)
        for p1, p2 in zip(self.model.params(), new_model.params()):
            assert p1 is not p2
            assert np.all(p1 == p2)