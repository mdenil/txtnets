__author__ = 'mdenil'

import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle

from cpu.model.cost import CrossEntropy
from cpu.optimize.objective import CostMinimizationObjective
from cpu.optimize.regularizer import L2Regularizer
from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.sgd import SGD
from cpu.optimize.data_provider import LabelledDocumentMinibatchProvider

import cpu.model.dropout

import experiment_config

def run():

    with open("{{train_data_json}}") as data_file:
        data = json.load(data_file)
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open("{{train_encoding_json}}") as encoding_file:
        encoding = json.load(encoding_file)

    n_validation = {{n_validation}}
    batch_size = {{batch_size}}

    train_data_provider = LabelledDocumentMinibatchProvider(
        X=X[:-n_validation],
        Y=Y[:-n_validation],
        batch_size=batch_size,
        padding='PADDING',
        fixed_n_sentences={{fixed_n_sentences}},
        fixed_n_words={{fixed_n_words}})

    print train_data_provider.batches_per_epoch

    validation_data_provider = LabelledDocumentMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=batch_size,
        padding='PADDING',
        fixed_n_sentences={{fixed_n_sentences}},
        fixed_n_words={{fixed_n_words}})

    model = experiment_config.get_model(encoding)

    print model

    cost_function = CrossEntropy()

    regularizer = L2Regularizer(lamb={{regularizer}})

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma={{adagrad_gamma}},
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    n_epochs = {{n_epochs}}
    n_batches = train_data_provider.batches_per_epoch * n_epochs

    time_start = time.time()

    best_acc = -1.0

    progress = []

    for batch_index, iteration_info in enumerate(optimizer):
        if batch_index % {{validation_frequency}} == 0:

            model_nodropout = cpu.model.dropout.remove_dropout(model)
            Y_hat = []
            Y_valid = []
            for _ in xrange(validation_data_provider.batches_per_epoch):
                X_valid_batch, Y_valid_batch, meta_valid = validation_data_provider.next_batch()
                X_valid_batch = X_valid_batch
                Y_valid_batch = Y_valid_batch
                Y_valid.append(Y_valid_batch)
                Y_hat.append(model_nodropout.fprop(X_valid_batch, meta=meta_valid))
            Y_valid = np.concatenate(Y_valid, axis=0)
            Y_hat = np.concatenate(Y_hat, axis=0)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            if acc > best_acc:
                best_acc = acc
                with open(os.path.join("{{job_dir}}", "model_best.pkl"), 'w') as model_file:
                    pickle.dump(model, model_file, protocol=-1)

            if batch_index % {{save_frequency}} == 0:
                with open(os.path.join("{{job_dir}}", "model_{:05}.pkl".format(batch_index)), 'w') as model_file:
                    pickle.dump(model, model_file, protocol=-1)

            print "B: {}, A: {}, C: {}, Prop1: {}, Param size: {}, best: {}".format(
                batch_index,
                acc,
                iteration_info['cost'],
                np.argmax(Y_hat, axis=1).mean(),
                np.mean(np.abs(model.pack())),
                best_acc)

            time_now = time.time()

            examples_per_hr = (batch_index * batch_size) / (time_now - time_start) * 3600

            progress.append({
                'batch': batch_index,
                'validation_accuracy': acc,
                'best_validation_accuracy': best_acc,
                'cost': iteration_info['cost'],
                'examples_per_hr': examples_per_hr,
            })

            with open(os.path.join("{{job_dir}}", "progress.pkl"), 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        if batch_index >= n_batches:
            break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)


if __name__ == "__main__":
    run()