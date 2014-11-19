from __future__ import print_function

__author__ = 'mdenil'

import numpy as np
import os
import gensim
import subprocess
import random
import json
import cPickle as pickle

from cpu.model.model import CSM
from cpu.model.encoding import DictionaryEncoding
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import Sum
from cpu.model.transfer import Softmax
from cpu.optimize.data_provider import LabelledDocumentMinibatchProvider
from cpu.optimize.data_provider import LabelledSequenceMinibatchProvider
from cpu.optimize.data_provider import TransformedLabelledDataProvider
from cpu.model.cost import CrossEntropy
from cpu.optimize.regularizer import L2Regularizer
from cpu.optimize.objective import CostMinimizationObjective
from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.sgd import SGD


def get(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.get()


# Gensim training is slow as balls
class Word2Vec(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self):
        subprocess.check_call(self._make_command())

    def _make_command(self):
        command = ['word2vec']
        for arg, value in self.kwargs.iteritems():
            command.append("-{}".format(arg))
            command.append(str(value))
        return command

def txtnets_model_from_gensim_word2vec(gensim_model):
    # build vocabulary mapping
    encoding = {}
    for index, word in enumerate(gensim_model.index2word):
        encoding[word] = index
    encoding['PADDING'] = len(encoding)

    vocabulary_size = len(encoding)
    embedding_dim = gensim_model.syn0.shape[1]

    E = np.concatenate([
        gensim_model.syn0,
        np.zeros((1, embedding_dim))
        ],
        axis=0)

    txtnets_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=encoding),

            WordEmbedding(
                vocabulary_size=vocabulary_size,
                dimension=embedding_dim,
                padding=encoding['PADDING'],
                E=E,
            )
        ]
    )

    return txtnets_model


def main():
    random.seed(34532)
    np.random.seed(675)
    np.set_printoptions(linewidth=100)

    data_dir = os.path.join("/users/mdenil/code/txtnets/txtnets_deployed/data", "stanfordmovie")


    trainer = Word2Vec(
        train=os.path.join(data_dir, "stanfordmovie.train.sentences.clean.projected.txt"),
        output="stanford-movie-vectors.bin",
        cbow=1,
        size=300,
        window=8,
        negative=25,
        hs=0,
        sample=1e-4,
        threads=20,
        binary=1,
        iter=15,
        min_count=1)

    trainer.train()

    gensim_model = gensim.models.Word2Vec.load_word2vec_format(
        "/users/mdenil/code/txtnets/txtnets_deployed/code/stanford-movie-vectors.bin",
        binary=True)

    # print(gensim_model.most_similar(["refund"]))
    # print(gensim_model.most_similar(["amazing"]))

    embedding_model = txtnets_model_from_gensim_word2vec(gensim_model)

    with open(os.path.join(data_dir, "stanfordmovie.train.sentences.clean.projected.flat.json")) as data_file:
        data = json.load(data_file)

    random.shuffle(data)
    X, Y = map(list, zip(*data))
    Y = [[":)", ":("].index(y) for y in Y]

    batch_size = 100
    n_validation = 500

    train_data_provider = LabelledSequenceMinibatchProvider(
        X=X[:-n_validation],
        Y=Y[:-n_validation],
        batch_size=batch_size,
        padding='PADDING')

    transformed_train_data_provider = TransformedLabelledDataProvider(
        data_source=train_data_provider,
        transformer=embedding_model)

    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-n_validation:],
        Y=Y[-n_validation:],
        batch_size=batch_size,
        padding='PADDING')

    transformed_validation_data_provider = TransformedLabelledDataProvider(
        data_source=validation_data_provider,
        transformer=embedding_model)

    logistic_regression = CSM(
        layers=[
            Sum(axes=['w']),

            Softmax(
                n_input_dimensions=gensim_model.syn0.shape[1],
                n_classes=2)
        ]
    )

    cost_function = CrossEntropy()
    regularizer = L2Regularizer(lamb=1e-4)
    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=transformed_train_data_provider,
        regularizer=regularizer)
    update_rule = AdaGrad(
        gamma=0.1,
        model_template=logistic_regression)

    optimizer = SGD(
        model=logistic_regression,
        objective=objective,
        update_rule=update_rule)


    for batch_index, iteration_info in enumerate(optimizer):
        if batch_index % 100 == 0:
            # print(iteration_info['cost'])

            Y_hat = []
            Y_valid = []
            for _ in xrange(transformed_validation_data_provider.batches_per_epoch):
                X_valid_batch, Y_valid_batch, meta_valid = transformed_validation_data_provider.next_batch()
                Y_valid.append(get(Y_valid_batch))
                Y_hat.append(get(logistic_regression.fprop(X_valid_batch, meta=meta_valid)))
            Y_valid = np.concatenate(Y_valid, axis=0)
            Y_hat = np.concatenate(Y_hat, axis=0)

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            print("B: {}, A: {}, C: {}".format(
                batch_index,
                acc,
                iteration_info['cost']))

            with open("model_w2vec_logreg.pkl", 'w') as model_file:
                pickle.dump(embedding_model.move_to_cpu(), model_file, protocol=-1)
                pickle.dump(logistic_regression.move_to_cpu(), model_file, protocol=-1)


if __name__ == "__main__":
    main()