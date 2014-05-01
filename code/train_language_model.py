__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import cPickle as pickle
import matplotlib.pyplot as plt

from cpu.model.model import CSM
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import SumFolding
from cpu.model.pooling import MaxFolding
from cpu.model.pooling import KMaxPooling
from cpu.model.nonlinearity import Tanh
from cpu.model.nonlinearity import Relu
from cpu.model.transfer import Softmax
from cpu.model.cost import CrossEntropy

from cpu.optimize.data_provider import MinibatchDataProvider
from cpu.optimize.data_provider import BatchDataProvider

from cpu.optimize.objective import CostMinimizationObjective

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
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


    batch_size = 100
    n_epochs = 1

    embedding_dimension = 8
    n_feature_maps = 10
    kernel_width = 15
    pooling_size = 8
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

    # model = CSM(
    #     layers=[
    #         WordEmbedding(
    #             dimension=embedding_dimension,
    #             vocabulary_size=vocabulary_size),
    #
    #         SentenceConvolution(
    #             n_feature_maps=n_feature_maps,
    #             kernel_width=kernel_width,
    #             n_input_dimensions=embedding_dimension),
    #
    #         # KMaxPooling(k=pooling_size),
    #
    #         # TODO: make a bias that runs along the w dimension
    #         Bias(
    #             n_input_dims=embedding_dimension,
    #             n_feature_maps=n_feature_maps),
    #
    #         MaxFolding(),
    #         MaxFolding(),
    #         MaxFolding(),
    #
    #         Softmax(
    #             n_classes=n_classes,
    #             n_input_dimensions=n_feature_maps*(context_length + kernel_width - 1)*embedding_dimension / 8),
    #     ]
    # )


    model = CSM(
        layers=[
            WordEmbedding(
                dimension=embedding_dimension,
                vocabulary_size=vocabulary_size),

            SentenceConvolution(
                n_feature_maps=n_feature_maps,
                kernel_width=kernel_width,
                n_channels=1,
                n_input_dimensions=embedding_dimension),

            # KMaxPooling(k=pooling_size),

            # TODO: make a bias that runs along the w dimension
            Bias(
                n_input_dims=embedding_dimension,
                n_feature_maps=n_feature_maps),

            MaxFolding(),

            SentenceConvolution(
                n_feature_maps=3,
                kernel_width=5,
                n_channels=n_feature_maps,
                n_input_dimensions=embedding_dimension / 2),


            MaxFolding(),
            MaxFolding(),

            Softmax(
                n_classes=n_classes,
                n_input_dimensions=3*(context_length + kernel_width - 1 + 5 - 1)*embedding_dimension / 8),
            ]
    )


    print model

    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=train_data_provider)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=model)

    # update_rule = AdaDeltaUpdateRule(
    #     rho=0.0,
    #     epsilon=1e-6    ,
    #     model_template=model)

    # update_rule = Momentum(
    #     momentum=0.5,
    #     epsilon=0.05,
    #     model_template=model)

    # update_rule = NAG(
    #     momentum=0.95,
    #     epsilon=0.001,
    #     model_template=model)

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

            print "B: {}, A: {}, Entropy (bits): {}".format(batch_index, acc, costs[-1]*np.log2(np.exp(1)))

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model, model_file, protocol=-1)