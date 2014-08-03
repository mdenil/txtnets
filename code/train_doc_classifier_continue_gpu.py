__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import time
import random
import simplejson as json
import cPickle as pickle
import argparse

from gpu.model.model import CSM
from gpu.model.encoding import DictionaryEncoding
from gpu.model.embedding import WordEmbedding
from gpu.model.transfer import SentenceConvolution
from gpu.model.transfer import Bias
from gpu.model.transfer import Linear
from gpu.model.pooling import KMaxPooling
from gpu.model.transfer import ReshapeForDocuments
from gpu.model.nonlinearity import Tanh
from gpu.model.transfer import Softmax
from gpu.model.cost import CrossEntropy
from gpu.model.dropout import Dropout
from gpu.optimize.objective import CostMinimizationObjective
from gpu.optimize.regularizer import L2Regularizer
from gpu.optimize.update_rule import AdaGrad
from gpu.optimize.sgd import SGD
from gpu.optimize.data_provider import LabelledDocumentMinibatchProvider

import gpu.model.host_device_component_mapping

import gpu.model.dropout


def maybe_get(x):
    return x


if __name__ == "__main__":
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    parser = argparse.ArgumentParser("Tell me where to start from.")
    parser.add_argument("--model", help="load this model to start")
    parser.add_argument("--lr", type=float, help="new learning rate")
    parser.add_argument("--out_folder")
    args = parser.parse_args()

    data_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(data_dir, "stanfordmovie.train.sentences.clean.projected.json")) as data_file:
        data = json.load(data_file)
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(data_dir, "stanfordmovie.train.sentences.clean.dictionary.encoding.json")) as encoding_file:
        encoding = json.load(encoding_file)

    print len(encoding)

    n_validation = 500
    batch_size = 25

    train_data_provider = LabelledDocumentMinibatchProvider(
        X=X[:-n_validation],
        Y=Y[:-n_validation],
        batch_size=batch_size,
        padding='PADDING',
        fixed_n_sentences=30,
        fixed_n_words=50)

    print train_data_provider.batches_per_epoch

    validation_data_provider = LabelledDocumentMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=batch_size,
        padding='PADDING',
        fixed_n_sentences=30,
        fixed_n_words=50)

    with open(args.model) as model_file:
        model = gpu.model.host_device_component_mapping.move_to_gpu(pickle.load(model_file))

    print model


    cost_function = CrossEntropy()

    regularizer = L2Regularizer(lamb=1e-4)

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=float(args.lr),
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    n_epochs = 1
    n_batches = train_data_provider.batches_per_epoch * n_epochs

    time_start = time.time()

    best_acc = -1.0

    progress = []
    costs = []
    prev_weights = model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:

            model_valid = gpu.model.dropout.remove_dropout(model)
            Y_hat = []
            Y_valid = []
            for _ in xrange(validation_data_provider.batches_per_epoch):
                X_valid_batch, Y_valid_batch, meta_valid = validation_data_provider.next_batch()
                Y_valid.append(Y_valid_batch.get())
                Y_hat.append(model_valid.fprop(X_valid_batch, meta=meta_valid).get())
            Y_valid = np.concatenate(Y_valid, axis=0)
            Y_hat = np.concatenate(Y_hat, axis=0)
            worst_normalization_error = np.max(np.abs(Y_hat.sum(axis=1) - 1))

            # This is really slow:
            #grad_check = gradient_checker.check(model)
            grad_check = "skipped"

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            # if acc > best_acc:
            #     best_acc = acc
            #     with open("model_best.pkl", 'w') as model_file:
            #         pickle.dump(model.move_to_cpu(), model_file, protocol=-1)

            with open(os.path.join(args.out_folder, "model_{:06}.pkl".format(batch_index)), 'w') as model_file:
                    pickle.dump(model.move_to_cpu(), model_file, protocol=-1)


            time_now = time.time()

            examples_per_hr = (batch_index * batch_size) / (time_now - time_start) * 3600

            print "B: {}, A: {}, C: {}, Prop1: {}, EPH: {}, WNE: {}, best: {}".format(
                batch_index,
                acc,
                iteration_info['cost'],
                np.argmax(Y_hat, axis=1).mean(),
                examples_per_hr,
                worst_normalization_error,
                best_acc)

            progress.append({
                'batch': batch_index,
                'validation_accuracy': acc,
                'best_validation_accuracy': best_acc,
                'cost': iteration_info['cost'],
                'examples_per_hr': examples_per_hr,
            })

            with open("progress.pkl", 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        # if batch_index == 1000:
        #     break

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model.move_to_cpu(), model_file, protocol=-1)

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)