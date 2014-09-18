__author__ = 'mdenil'

import numpy as np

import time

import pycuda.autoinit
import pycuda

import gpu.utils

from gpu import space

from gpu.model.model import CSM


from gpu.model.transport import DeviceToHost
from gpu.model.transport import HostToDevice

from gpu.model.cost import CrossEntropy
from gpu.model.transfer import Softmax
from gpu.optimize.update_rule import AdaGradUpdateRule
from gpu.optimize.data_provider import BatchDataProvider

# from cpu.model.cost import CrossEntropy
# from cpu.model.transfer import Softmax
# from cpu.optimize.update_rule import AdaGradUpdateRule
# from cpu.optimize.data_provider import BatchDataProvider

from gpu.optimize.objective import CostMinimizationObjective
from gpu.optimize.sgd import SGD


# from cpu.optimize.grad_check import fast_gradient_check

np.random.seed(222)

def generate_simple_classification_problem(n_data, n_dims, n_classes):
    X_1 = np.random.standard_normal(size=(n_data/2, n_dims)) + 2*np.random.uniform(size=(1, n_dims)) + 3
    X_2 = np.random.standard_normal(size=(n_data/2, n_dims)) - 2*np.random.uniform(size=(1, n_dims)) + 3
    Y_1 = np.hstack([np.ones((n_data/2, 1)), np.zeros((n_data/2, 1))])
    Y_2 = np.hstack([np.zeros((n_data/2, 1)), np.ones((n_data/2, 1))])

    X = np.vstack([X_1, X_2])
    Y = np.vstack([Y_1, Y_2])

    if n_classes > 2:
        Y = np.hstack([Y, np.zeros((n_data, n_classes-2))])

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    X = gpu.utils.cpu_to_gpu(X.astype(np.float32))
    Y = gpu.utils.cpu_to_gpu(Y.astype(np.float32))

    return X, Y


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

    update_rule = AdaGradUpdateRule(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(model=model, objective=objective, update_rule=update_rule)

    return model, objective, optimizer, data_provider


def check_model_accuracy(model, data_provider):
    X, Y, meta = data_provider.next_batch()

    Y_hat = model.fprop(X, meta=meta)


    if isinstance(Y_hat, pycuda.gpuarray.GPUArray):
        return np.mean(np.argmax(Y_hat.get(), axis=1) == np.argmax(Y.get(), axis=1))
    else:
        return np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1))


def run():
    n_data = 10000
    n_dims = 50
    n_classes = 2

    X_train, Y_train = generate_simple_classification_problem(n_data, n_dims, n_classes)

    model, objective, optimizer, data_provider = build_model_and_objective(n_classes, n_dims, X_train, Y_train)

    # print check_model_gradient(model, objective)
    print check_model_accuracy(model, data_provider)

    start = time.time()

    for batch_index, iteration_info in enumerate(optimizer):
        if batch_index % 10 == 0:
            print batch_index, check_model_accuracy(model, data_provider), iteration_info['cost']

        if batch_index == 100:
            break

    end = time.time()

    print "Time in optimizer:", (end - start)

if __name__ == "__main__":
    run()