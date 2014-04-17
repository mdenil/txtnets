__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
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

from cpu.optimize.update_rule import AdaGradUpdateRule
from cpu.optimize.update_rule import AdaDeltaUpdateRule
from cpu.optimize.update_rule import BasicUpdateRule
from cpu.optimize.update_rule import NAG
from cpu.optimize.update_rule import Momentum

from cpu.optimize.data_provider import LabelledSequenceMinibatchProvider

from cpu.optimize.grad_check import fast_gradient_check

from cpu.optimize.sgd import SGD







if __name__ == "__main__":
    random.seed(32423)
    np.set_printoptions(linewidth=100)

    tweets_dir = os.path.join("data", "tweets")


    with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.json.gz")) as data_file:
    # with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.json.gz")) as data_file:
    # with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.json.gz")) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))

        # shuffle
        combined = zip(X, Y)
        random.shuffle(combined)
        X, Y = map(list, zip(*combined))

        Y = [ [":)", ":("].index(y) for y in Y ]

    with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.alphabet.encoding.json")) as alphabet_file:
    # with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.alphabet.encoding.json")) as alphabet_file:
    # with open(os.path.join(tweets_dir, "tweets_100k.english.balanced.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())


    # lists of characters.
    X = [list(x) for x in X]

    # print X[:100]

    # lists of words
    # replace unknowns with an unknown character
    # tokenizer = WordPunctTokenizer()
    # new_X = []
    # for x in X:
    #     new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    # X = new_X


    train_data_provider = LabelledSequenceMinibatchProvider(
        X=X[:-500],
        Y=Y[:-500],
        batch_size=50,
        padding='PADDING')

    print train_data_provider.batches_per_epoch

    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-500:],
        Y=Y[-500:],
        batch_size=500,
        padding='PADDING')


    # ~70% after 300 epochs with batches of 100, regularizer L2=1e-4
    #
    # tweet_model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=alphabet),
    #
    #         WordEmbedding( # really a character embedding
    #                        dimension=32,
    #                        vocabulary_size=len(alphabet)),
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



    # tweet_model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=alphabet),
    #
    #         WordEmbedding( # really a character embedding
    #                        dimension=32*4,
    #                        vocabulary_size=len(alphabet)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=10,
    #             n_channels=1,
    #             n_input_dimensions=32*4),
    #
    #         Relu(),
    #         SumFolding(),
    #         SumFolding(),
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



    # ~70% after 500 epochs with batches of 50, regularizer L2=1e-4
    #
    # tweet_model = CSM(
    #     layers=[
    #         Encoding(encoding=alphabet),
    #
    #         WordEmbedding( # really a character embedding
    #                        dimension=32,
    #                        vocabulary_size=len(alphabet)),
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
    #         # MaxFolding(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=5,
    #             n_channels=5,
    #             n_input_dimensions=16),
    #
    #         # SumFolding(),
    #
    #         KMaxPooling(k=4),
    #
    #         Bias(
    #             n_input_dims=16,
    #             n_feature_maps=5),
    #
    #         Tanh(),
    #
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=320),
    #         ]
    # )


    # tweet_model = CSM(
    #     layers=[
    #         Encoding(encoding=alphabet),
    #
    #         WordEmbedding( # really a character embedding
    #                        dimension=32,
    #                        vocabulary_size=len(alphabet)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=10,
    #             kernel_width=5,
    #             n_channels=1,
    #             n_input_dimensions=32),
    #
    #         Relu(),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=15),
    #
    #         # Bias(
    #         #     n_input_dims=16,
    #         #     n_feature_maps=10),
    #         #
    #         # Tanh(),
    #
    #         # MaxFolding(),
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=3,
    #             n_channels=10,
    #             n_input_dimensions=16),
    #
    #         SumFolding(),
    #
    #         KMaxPooling(k=7),
    #
    #         Tanh(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=280),
    #         ]
    # )

    # tweet_model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=alphabet),
    #
    #         WordEmbedding(
    #             dimension=32*4,
    #             vocabulary_size=len(alphabet)),
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=10,
    #             n_channels=1,
    #             n_input_dimensions=32*4),
    #
    #         Relu(),
    #         SumFolding(),
    #         SumFolding(),
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
    #
    #         SentenceConvolution(
    #             n_feature_maps=5,
    #             kernel_width=5,
    #             n_channels=5,
    #             n_input_dimensions=16),
    #
    #         KMaxPooling(k=4),
    #
    #         Bias(
    #             n_input_dims=16,
    #             n_feature_maps=5),
    #
    #         Tanh(),
    #
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=320),
    #         ]
    # )

    tweet_model = CSM(
        layers=[

            DictionaryEncoding(vocabulary=alphabet),


            WordEmbedding(
                dimension=42,
                vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=6,
                n_channels=1,
                n_input_dimensions=42),

            SumFolding(),

            KMaxPooling(k=4),

            Bias(
                n_input_dims=21,
                n_feature_maps=5),

            Tanh(),

            Softmax(
                n_classes=2,
                n_input_dimensions=420),
        ]
    )


    print tweet_model

    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=train_data_provider)

    update_rule = AdaGradUpdateRule(
        gamma=0.05,
        model_template=tweet_model)

    regularizer = L2Regularizer(lamb=1e-4)

    optimizer = SGD(
        model=tweet_model,
        objective=objective,
        update_rule=update_rule,
        regularizer=regularizer)


    n_epochs = 1
    n_batches = train_data_provider.batches_per_epoch * n_epochs

    costs = []
    prev_weights = tweet_model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:
            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = tweet_model.fprop(X_valid, meta=meta_valid)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            print "B: {}, A: {}, C: {}, Prop1: {}, Param size: {}".format(
                batch_index,
                acc, costs[-1],
                np.argmax(Y_hat, axis=1).mean(),
                np.mean(np.abs(tweet_model.pack())))


        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(tweet_model, model_file, protocol=-1)