__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import time
import gzip
import random
import simplejson as json
import cPickle as pickle
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer

from collections import OrderedDict

from cpu.model.model import CSM
from cpu.model.encoding import DictionaryEncoding
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import SumFolding
from cpu.model.pooling import MaxFolding
from cpu.model.pooling import KMaxPooling
from cpu.model.nonlinearity import Tanh
from cpu.model.nonlinearity import Relu
from cpu.model.transfer import Softmax
from cpu.model.transfer import Linear

from cpu import space
from cpu.model import layer

from cpu.model.cost import CrossEntropy
from cpu.model.cost import LargeMarginCost

from cpu.optimize.data_provider import MinibatchDataProvider
from cpu.optimize.data_provider import BatchDataProvider
from cpu.optimize.data_provider import PaddedSequenceMinibatchProvider

from cpu.optimize.objective import CostMinimizationObjective
from cpu.optimize.objective import NoiseContrastiveObjective

from cpu.optimize.regularizer import L2Regularizer

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
from cpu.optimize.update_rule import Momentum

from cpu.optimize.data_provider import LabelledSequenceMinibatchProvider

from cpu.optimize.grad_check import ModelGradientChecker

from cpu.optimize.sgd import SGD


if __name__ == "__main__":
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    # tweets_dir = os.path.join("data", "tweets")
    # # with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.json.gz")) as data_file:
    # with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.json.gz")) as data_file:
    #     data = json.loads(data_file.read())
    #     X, Y = map(list, zip(*data))
    #
    #     # shuffle
    #     combined = zip(X, Y)
    #     random.shuffle(combined)
    #     X, Y = map(list, zip(*combined))
    #
    #     Y = [ [":)", ":("].index(y) for y in Y ]
    #
    # # with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.encoding.encoding.json")) as alphabet_file:
    # # with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.encoding.encoding.json")) as alphabet_file:
    # with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.dictionary.encoding.json")) as alphabet_file:
    #     encoding = json.loads(alphabet_file.read())


    tweets_dir = os.path.join("../data", "sentiment140")

    # with open(os.path.join(tweets_dir, "sentiment140.train.json")) as data_file:
    with open(os.path.join(tweets_dir, "sentiment140.train.clean.json")) as data_file:
        data = json.loads(data_file.read())
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]


    # with open(os.path.join(tweets_dir, "sentiment140.train.encoding.encoding.json")) as alphabet_file:
    with open(os.path.join(tweets_dir, "sentiment140.train.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    print len(alphabet)

    # lists of characters.
    # X = [list(x) for x in X]


    # lists of words
    # replace unknowns with an unknown character
    tokenizer = WordPunctTokenizer()
    new_X = []
    for x in X:
        new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    X = new_X



    train_data_provider = LabelledSequenceMinibatchProvider(
        X=X[:-500],
        Y=Y[:-500],
        batch_size=100,
        padding='PADDING')

    print train_data_provider.batches_per_epoch

    n_validation = 500
    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=n_validation,
        padding='PADDING')


    # ~70% after 300 batches of 100, regularizer L2=1e-4 on tweets100k
    #
    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding( # really a character embedding
    #                        dimension=32,
    #                        vocabulary_size=len(encoding)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=10,
    #             n_channels=1,
    #             n_input_dimensions=32),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=7),
    #
    #         Bias(
    #             n_input_dims=16,
    #             n_feature_maps=5),
    #
    #         Tanh(),
    #
    #         MaxFolding(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=280),
    #         ]
    # )

    # Approximately Nal's model
    #
    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding(
    #             dimension=12,
    #             vocabulary_size=len(encoding)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=6,
    #             kernel_width=7,
    #             n_channels=1,
    #             n_input_dimensions=12),
    #
    #         Bias(
    #             n_input_dims=12,
    #             n_feature_maps=6),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=4, k_dynamic=0.5),
    #
    #         Tanh(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=14,
    #             kernel_width=5,
    #             n_channels=6,
    #             n_input_dimensions=6),
    #
    #         Bias(
    #             n_input_dims=6,
    #             n_feature_maps=14),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=4),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=168),
    #         ]
    # )

    # model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=encoding),
    #
    #         WordEmbedding(
    #             dimension=24,
    #             vocabulary_size=len(encoding)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=10,
    #             n_channels=1,
    #             n_input_dimensions=24),
    #
    #         Bias(
    #             n_input_dims=24,
    #             n_feature_maps=10),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=15, k_dynamic=0.5),
    #
    #         Tanh(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=5,
    #             n_channels=10,
    #             n_input_dimensions=12),
    #
    #         Bias(
    #             n_input_dims=12,
    #             n_feature_maps=10),
    #
    #         KMaxPooling(k=4),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=480),
    #         ]
    # )

    tweet_model = CSM(
        layers=[
            # cpu.model.encoding.
            DictionaryEncoding(vocabulary=alphabet),

            # cpu.model.embedding.
            WordEmbedding(
                dimension=28,
                vocabulary_size=len(alphabet)),

            # HostToDevice(),

            SentenceConvolution(
                n_feature_maps=6,
                kernel_width=7,
                n_channels=1,
                n_input_dimensions=28),

            Bias(
                n_input_dims=28,
                n_feature_maps=6),

            SumFolding(),

            KMaxPooling(k=4, k_dynamic=0.5),

            Tanh(),

            SentenceConvolution(
                n_feature_maps=14,
                kernel_width=5,
                n_channels=6,
                n_input_dimensions=14),

            Bias(
                n_input_dims=14,
                n_feature_maps=14),

            SumFolding(),

            KMaxPooling(k=4),

            Tanh(),

            Softmax(
                n_classes=2,
                n_input_dimensions=392),
            ]
    )



    print tweet_model


    cost_function = CrossEntropy()

    regularizer = L2Regularizer(lamb=1e-4)

    objective = CostMinimizationObjective(cost=cost_function, data_provider=train_data_provider, regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=0.05,
        model_template=tweet_model)

    optimizer = SGD(
        model=tweet_model,
        objective=objective,
        update_rule=update_rule)


    gradient_checker = ModelGradientChecker(
        CostMinimizationObjective(cost=cost_function, data_provider=validation_data_provider, regularizer=regularizer))


    n_epochs = 1
    n_batches = train_data_provider.batches_per_epoch * n_epochs

    time_start = time.time()

    costs = []
    prev_weights = tweet_model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:
            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = tweet_model.fprop(X_valid, meta=meta_valid)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            # This is really slow:
            #grad_check = gradient_checker.check(model)
            grad_check = "skipped"

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            print "B: {}, A: {}, C: {}, Prop1: {}, Param size: {}, g: {}".format(
                batch_index,
                acc, costs[-1],
                np.argmax(Y_hat, axis=1).mean(),
                np.mean(np.abs(tweet_model.pack())),
                grad_check)

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(tweet_model, model_file, protocol=-1)

        if batch_index == 100:
            break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)