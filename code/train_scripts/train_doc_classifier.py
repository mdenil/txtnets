__author__ = 'albandemiraj, mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import time
import random
import simplejson as json
import cPickle as pickle

from cpu.model.model import CSM
from cpu.model.encoding import DictionaryEncoding
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import KMaxPooling
from cpu.model.transfer import ReshapeForDocuments
from cpu.model.nonlinearity import Tanh
from cpu.model.transfer import Softmax
from cpu.model.dropout import Dropout
from cpu.model.cost import CrossEntropy
from cpu.optimize.objective import CostMinimizationObjective
from cpu.optimize.regularizer import L2Regularizer
from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.sgd import SGD
from cpu.optimize.data_provider import LabelledDocumentMinibatchProvider


def maybe_get(x):
    return x


if __name__ == "__main__":
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

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
        fixed_n_sentences=15,
        fixed_n_words=50)

    print train_data_provider.batches_per_epoch

    validation_data_provider = LabelledDocumentMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=batch_size,
        padding='PADDING',
        fixed_n_sentences=15,
        fixed_n_words=50)


    model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=encoding),

            WordEmbedding(
                dimension=20,
                vocabulary_size=len(encoding),
                padding=encoding['PADDING']),

            Dropout(('b', 'w', 'f'), 0.2),

            SentenceConvolution(
                n_feature_maps=10,
                kernel_width=15,
                n_channels=20,
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps=10),

            KMaxPooling(k=7, k_dynamic=0.5),

            Tanh(),

            SentenceConvolution(
                n_feature_maps=30,
                kernel_width=9,
                n_channels=10,
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps=30),

            KMaxPooling(k=5),

            Tanh(),

            ReshapeForDocuments(),

            SentenceConvolution(
                n_feature_maps=20,
                kernel_width=11,
                n_channels=30*5,
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps=20),

            KMaxPooling(k=5),

            Tanh(),

            Dropout(('b', 'd', 'f', 'w'), 0.5),

            Softmax(
                n_classes=2,
                n_input_dimensions=100),
            ]
    )



    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding(
    #             dimension=28,
    #             vocabulary_size=len(encoding),
    #             padding=encoding['PADDING']),
    #
    #         SentenceConvolution(
    #             n_feature_maps=6,
    #             kernel_width=7,
    #             n_channels=28,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=6),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4, k_dynamic=0.5),
    #
    #         Tanh(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=14,
    #             kernel_width=5,
    #             n_channels=6,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=14),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4),
    #
    #         Tanh(),
    #
    #         ReshapeForDocuments(),
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=3,
    #             n_channels=56,
    #             n_input_dimensions=1),
    #
    #         # DocumentConvolution(
    #         #     n_feature_maps=10,
    #         #     kernel_width=3,
    #         #     n_channels=56,
    #         #     n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=10),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=2),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=20),
    #         ]
    # )

    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding(
    #             dimension=20,
    #             vocabulary_size=len(encoding),
    #             padding=encoding['PADDING']),
    #
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=7,
    #             n_channels=20,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=10
    #         ),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4, k_dynamic=0.75),
    #
    #         Tanh(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=15,
    #             kernel_width=5,
    #             n_channels=10,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=15),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4, k_dynamic=0.5),
    #
    #         Tanh(),
    #
    #         # ReshapeForDocuments(),
    #         # SentenceConvolution(
    #         #     n_feature_maps=15,
    #         #     kernel_width=3,
    #         #     n_channels=345,
    #         #     n_input_dimensions=1),
    #
    #         DocumentConvolution(
    #             n_feature_maps=15,
    #             kernel_width=3,
    #             n_channels=345,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=15),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=60),
    #         ]
    # )


    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding(
    #             dimension=20,
    #             vocabulary_size=len(encoding),
    #             padding=encoding['PADDING']),
    #
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=7,
    #             n_channels=20,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=10
    #         ),
    #
    #         KMaxPooling(k=4, k_dynamic=0.75),
    #
    #         Tanh(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=15,
    #             kernel_width=5,
    #             n_channels=10,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=15),
    #
    #         KMaxPooling(k=4, k_dynamic=0.5),
    #
    #         Tanh(),
    #
    #         ReshapeForDocuments(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=15,
    #             kernel_width=3,
    #             n_channels=345,
    #             n_input_dimensions=1),
    #
    #         Bias(
    #             n_input_dims=1,
    #             n_feature_maps=15),
    #
    #         KMaxPooling(k=4),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=60),
    #         ]
    # )

    print model


    cost_function = CrossEntropy()

    regularizer = L2Regularizer(lamb=1e-4)

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=0.01,
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

            Y_hat = []
            Y_valid = []
            for _ in xrange(validation_data_provider.batches_per_epoch):
                X_valid_batch, Y_valid_batch, meta_valid = validation_data_provider.next_batch()
                X_valid_batch = maybe_get(X_valid_batch)
                Y_valid_batch = maybe_get(Y_valid_batch)
                Y_valid.append(Y_valid_batch)
                Y_hat.append(maybe_get(model.fprop(X_valid_batch, meta=meta_valid)))
            Y_valid = np.concatenate(Y_valid, axis=0)
            Y_hat = np.concatenate(Y_hat, axis=0)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            # This is really slow:
            #grad_check = gradient_checker.check(model)
            grad_check = "skipped"

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            if acc > best_acc:
                best_acc = acc
                with open(os.path.expanduser("~/model_best.pkl"), 'w') as model_file:
                    pickle.dump(model, model_file, protocol=-1)

            current = dict()
            current['B']=batch_index
            current['A']=acc
            current['C']=costs[-1]
            current['Prop']=np.argmax(Y_hat, axis=1).mean()
            current['Params']=np.mean(np.abs(model.pack()))
            current['G']=grad_check

            progress.append(current)
            print current
            with open(os.path.expanduser("~/progress.pkl"), 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        # if batch_index == 100:
        #     break

        if batch_index % 100 == 0:
            with open(os.path.expanduser("~/model.pkl"), 'w') as model_file:
                pickle.dump(model, model_file, protocol=-1)

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)
