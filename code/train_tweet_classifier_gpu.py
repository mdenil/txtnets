__author__ = 'mdenil'


import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer

from gpu.model.model import CSM
from gpu.model.encoding import DictionaryEncoding
from gpu.model.embedding import WordEmbedding
from gpu.model.transfer import SentenceConvolution
from gpu.model.transfer import Bias
from gpu.model.pooling import SumFolding
from gpu.model.pooling import MaxFolding
from gpu.model.pooling import KMaxPooling
from gpu.model.nonlinearity import Tanh
from gpu.model.transfer import Softmax

# It might be nice to keep the word embedding dictionaries on the host, they're kind of big
from gpu.model.transport import HostToDevice
import cpu.model.encoding
import cpu.model.embedding

from gpu.model.cost import CrossEntropy

from gpu.optimize.sgd import SGD
from gpu.optimize.objective import CostMinimizationObjective
from gpu.optimize.regularizer import L2Regularizer
from gpu.optimize.update_rule import AdaGrad
from gpu.optimize.data_provider import LabelledSequenceMinibatchProvider

from cpu.optimize.grad_check import ModelGradientChecker


def run():
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    tweets_dir = os.path.join("../data", "sentiment140")

    with open(os.path.join(tweets_dir, "sentiment140.train.clean.json")) as data_file:
        data = json.loads(data_file.read())
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(tweets_dir, "sentiment140.train.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    print len(alphabet)

    # X = X[:1000]
    # Y = Y[:1000]

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
        fixed_length=50,
        padding='PADDING')

    print train_data_provider.batches_per_epoch

    n_validation = 500
    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=n_validation,
        fixed_length=50,
        padding='PADDING')


    # tweet_model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=alphabet),
    #
    #         WordEmbedding(
    #             dimension=32,
    #             vocabulary_size=len(alphabet)),
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
    #         SumFolding(),
    #
    #         Softmax(
    #             n_classes=2,
    #             n_input_dimensions=280),
    #         ]
    #     )

    # Approximately Nal's model
    #
    # tweet_model = CSM(
    #     layers=[
    #         DictionaryEncoding(vocabulary=alphabet),
    #
    #         WordEmbedding(
    #             dimension=12,
    #             vocabulary_size=len(alphabet)),
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

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=0.05,
        model_template=tweet_model)

    optimizer = SGD(
        model=tweet_model,
        objective=objective,
        update_rule=update_rule)

    gradient_checker = ModelGradientChecker(
        CostMinimizationObjective(
            cost=cost_function,
            data_provider=validation_data_provider,
            regularizer=regularizer))

    time_start = time.time()

    costs = []
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:
            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = tweet_model.fprop(X_valid, meta=meta_valid)

            Y_hat = Y_hat.get()
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            # grad_check = gradient_checker.check(tweet_model)
            grad_check = "skipped"

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid.get(), axis=1))

            print "B: {}, A: {}, C: {}, Prop1: {}, Param size: {}, g: {}".format(
                batch_index,
                acc, costs[-1],
                np.argmax(Y_hat, axis=1).mean(),
                np.mean(np.abs(tweet_model.pack())),
                grad_check)

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(tweet_model.move_to_cpu(), model_file, protocol=-1)

        # if batch_index % 1000 == 0 and batch_index > 0:
        #     with open("model_optimization.pkl", 'w') as model_file:
        #         pickle.dump(optimizer, model_file, protocol=-1)

        if batch_index == 30000:
            break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)


if __name__ == "__main__":
    run()