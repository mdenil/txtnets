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

np.random.seed(222)

def generate_simple_classification_problem(n_data):
    X_1 = np.random.standard_normal(size=(n_data/2, n_dims)) + 2*np.random.uniform(size=(1, n_dims)) + 3
    X_2 = np.random.standard_normal(size=(n_data/2, n_dims)) - 2*np.random.uniform(size=(1, n_dims)) + 3
    Y_1 = np.hstack([np.ones((n_data/2, 1)), np.zeros((n_data/2, 1))])
    Y_2 = np.hstack([np.zeros((n_data/2, 1)), np.ones((n_data/2, 1))])

    X = np.vstack([X_1, X_2])
    Y = np.vstack([Y_1, Y_2])

    return X, Y

def generate_dense_grid(min, max):
    x, y = np.meshgrid(np.linspace(min, max), np.linspace(min, max))
    X = np.vstack([x.ravel(), y.ravel()]).T
    return X, x, y


def build_model_and_objective(n_classes, n_input_dimensions, X, Y):
    model = CSM(
        layers=[
            Softmax(
                n_classes=n_classes,
                n_input_dimensions=n_input_dimensions),
        ],
        )

    lengths = np.zeros(X.shape[0])
    data_provider = BatchDataProvider(
        X=X,
        Y=Y,
        lengths=lengths)

    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=data_provider)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(model=model, objective=objective, update_rule=update_rule)

    return model, objective, optimizer, data_provider

def check_model_gradient(model, objective):
    def func(w):
        model.unpack(w)
        cost, grads = objective.evaluate(model)
        return cost

    def grad(w):
        model.unpack(w)
        cost, grads = objective.evaluate(model)
        return np.concatenate([g.ravel() for g in grads])

    packed = model.pack().copy()
    err = fast_gradient_check(func, grad, model.pack(), method='diff')
    model.unpack(packed)
    return err

def check_model_accuracy(model, data_provider):
    X, Y, meta = data_provider.next_batch()

    Y_hat = model.fprop(X, meta=meta)


    return np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1))

if __name__ == "__main__":

    n_data = 100
    n_dims = 2
    n_classes = 2

    X_train, Y_train = generate_simple_classification_problem(n_data)

    model, objective, optimizer, data_provider = build_model_and_objective(n_classes, n_dims, X_train, Y_train)



    print check_model_gradient(model, objective)
    print check_model_accuracy(model, data_provider)

    for batch_index, iteration_info in enumerate(optimizer):
        if batch_index % 10 == 0:
            print batch_index, check_model_accuracy(model, data_provider), iteration_info['cost']

        if batch_index == 200:
            break


    # X_grid, x_grid, y_grid = generate_dense_grid(X_train.min(), X_train.max())
    # grid_meta = {'lengths': np.zeros(X_grid.shape[0])}
    # Y_grid = model.fprop(X_grid, input_axes=['b', 'w'], meta=grid_meta)
    # Y_grid = np.argmax(Y_grid, axis=1).reshape(x_grid.shape)
    # plt.pcolor(x_grid, y_grid, Y_grid)
    #
    # plt.scatter(X_train[:,0], X_train[:,1], s=50, c=np.argmax(Y_train, axis=1))
    #
    # plt.show()
