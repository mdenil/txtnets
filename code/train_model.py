__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import scipy.io
import pyprind
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

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
from cpu.optimize.update_rule import Momentum

from cpu.optimize.grad_check import fast_gradient_check

from cpu.optimize.sgd import SGD

# np.random.seed(32423)

if __name__ == "__main__":
    data_file_name = "verify_forward_pass/data/SENT_vec_1_emb_ind_bin.mat"
    data = scipy.io.loadmat(data_file_name)


    batch_size = 40
    n_epochs = 1


    embedding_dimension = 42
    n_feature_maps = 5
    kernel_width = 6
    pooling_size = 4
    n_classes = 2
    vocabulary_size = int(data['size_vocab'])


    X_train = data['train'] - 1
    lengths_train = data['train_lbl'][:,1]
    Y_train = data['train_lbl'][:,0] - 1 # -1 to switch to zero based indexing
    Y_train = np.equal.outer(Y_train, np.arange(n_classes)).astype(np.float)
    assert np.all(np.sum(Y_train, axis=1) == 1)

    max_sentence_length = X_train.shape[1]

    X_valid = data['valid'] - 1
    lengths_valid = data['valid_lbl'][:,1]
    Y_valid = data['valid_lbl'][:,0] - 1
    Y_valid = np.equal.outer(Y_valid, np.arange(n_classes)).astype(np.float)
    assert np.all(np.sum(Y_valid, axis=1) == 1)

    validation_data_provider = BatchDataProvider(
        X=X_valid,
        Y=Y_valid,
        lengths=lengths_valid)

    ## BUILD THE MODEL

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

            SumFolding(),

            KMaxPooling(k=pooling_size*2),

            Bias(
                n_input_dims=embedding_dimension / 2,
                n_feature_maps=n_feature_maps),

            Tanh(),

            # Softmax(
            #     n_classes=n_classes,
            #     n_input_dimensions=420),

            SentenceConvolution(
                n_feature_maps=n_feature_maps,
                kernel_width=3,
                n_channels=n_feature_maps,
                n_input_dimensions=embedding_dimension/2),

            KMaxPooling(k=pooling_size),

            Bias(
                n_input_dims=embedding_dimension / 2,
                n_feature_maps=n_feature_maps),

            Tanh(),

            Softmax(
                n_classes=n_classes,
                n_input_dimensions=420),
            ],
        )

    print model

    # pre-initialize the vocabulary

    # model.layers[0].E[:-1,:] = data['vocab_emb'][:embedding_dimension, :].T
    # model.layers[0].E[-1,:] = 0.0

    # build the optimizer

    data_provider = MinibatchDataProvider(
        X=X_train,
        Y=Y_train,
        lengths=lengths_train,
        batch_size=batch_size)

    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=data_provider)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(model=model, objective=objective, update_rule=update_rule)

    n_batches = data_provider.batches_per_epoch * n_epochs

    # This is useful for checking gradients
    #
    # def func(w):
    #     model.unpack(w)
    #     objective = CostMinimizationObjective(cost=cost_function, data_provider=validation_data_provider)
    #     cost, grads = objective.evaluate(model)
    #     return cost
    #
    # def grad(w):
    #     model.unpack(w)
    #     objective = CostMinimizationObjective(cost=cost_function, data_provider=validation_data_provider)
    #     cost, grads = objective.evaluate(model)
    #
    #     return np.concatenate([g.ravel() for g in grads])
    #
    # print fast_gradient_check(func, grad, model.pack(), method='diff')

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