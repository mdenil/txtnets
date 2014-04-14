__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import matplotlib.pyplot as plt

from cpu.model.model import CSM
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import SumFolding
from cpu.model.pooling import KMaxPooling
from cpu.model.nonlinearity import Tanh
from cpu.model.transfer import Softmax
from cpu.model.cost import CrossEntropy

from cpu.optimize.data_provider import MinibatchDataProvider
from cpu.optimize.data_provider import BatchDataProvider

from cpu.optimize.objective import CostMinimizationObjective

from cpu.optimize.update_rule import AdaGradUpdateRule
from cpu.optimize.update_rule import AdaDeltaUpdateRule
from cpu.optimize.update_rule import BasicUpdateRule
from cpu.optimize.update_rule import NAG
from cpu.optimize.update_rule import Momentum

from cpu.optimize.grad_check import fast_gradient_check

from cpu.optimize.sgd import SGD

np.random.seed(32423)

if __name__ == "__main__":
    data_file_name = os.path.join(os.environ['DATA'], "text8", "text8.encoded.npz")
    data = np.load(data_file_name)

    X_all = data['X']
    Y_all = data['Y']

    n_train = int(0.999 * X_all.shape[0])
    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train].astype(np.float).ravel()
    X_valid = X_all[n_train:]
    Y_valid = Y_all[n_train:].astype(np.float).ravel()


    batch_size = 50
    n_epochs = 1

    embedding_dimension = 26
    n_feature_maps = 5
    kernel_width = 7
    pooling_size = 7
    n_classes = Y_all.max() + 1
    vocabulary_size = n_classes

    Y_train = np.equal.outer(Y_train, np.arange(n_classes)).astype(Y_train.dtype)
    Y_valid = np.equal.outer(Y_valid, np.arange(n_classes)).astype(Y_valid.dtype)

    context_length = X_train.shape[1]
    lengths_train = context_length + np.zeros(shape=Y_train.shape)
    lengths_valid = context_length + np.zeros(shape=Y_valid.shape)


    train_data_provider = MinibatchDataProvider(
        X=X_train,
        Y=Y_train,
        lengths=lengths_train,
        batch_size=batch_size)

    validation_data_provider = BatchDataProvider(
        X=X_valid,
        Y=Y_valid,
        lengths=lengths_valid)

    model = CSM(
        layers=[
            WordEmbedding(
                dimension=embedding_dimension,
                vocabulary_size=vocabulary_size),

            SentenceConvolution(
                n_feature_maps=n_feature_maps,
                kernel_width=kernel_width,
                n_input_dimensions=embedding_dimension),

            SumFolding(),

            KMaxPooling(k=pooling_size),

            Bias(
                n_input_dims=embedding_dimension / 2,
                n_feature_maps=n_feature_maps),

            Tanh(),

            Softmax(
                n_classes=n_classes,
                n_input_dimensions=n_feature_maps*pooling_size*embedding_dimension / 2),
        ]
    )


    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=train_data_provider)

    update_rule = AdaGradUpdateRule(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(model=model, objective=objective, update_rule=update_rule)

    n_batches = train_data_provider.batches_per_epoch * n_epochs

    costs = []
    prev_weights = model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:
            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = model.fprop(X_valid, meta=meta_valid)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            print "B: {}, A: {}, C: {}, Param size: {}".format(batch_index, acc, np.exp(costs[-1]), np.mean(np.abs(model.pack())))