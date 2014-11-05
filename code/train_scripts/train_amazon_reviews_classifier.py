from __future__ import print_function

__author__ = 'mdenil'

import numpy as np
import pyprind
import os
import time
import random
import simplejson as json
import cPickle as pickle

from gpu.model.model import CSM
from gpu.model.encoding import DictionaryEncoding
from gpu.model.embedding import WordEmbedding
from gpu.model.transfer import SentenceConvolution
from gpu.model.transfer import Bias
from gpu.model.pooling import KMaxPooling
from gpu.model.transfer import ReshapeForDocuments
from gpu.model.transfer import Linear
from gpu.model.nonlinearity import Tanh
from gpu.model.transfer import Softmax
from gpu.model.dropout import Dropout
from gpu.model.cost import CrossEntropy
from gpu.model.cost import SquaredError
from gpu.optimize.objective import CostMinimizationObjective
from gpu.optimize.regularizer import L2Regularizer
from gpu.optimize.update_rule import AdaGrad
from gpu.optimize.sgd import SGD
from gpu.optimize.data_provider import ShardedLabelledDocumentMinibatchProvider


def load_word2vec_embeddings(file_name, encoding):
    with open(file_name) as in_file:
        header = in_file.readline()
        vocabulary_size, dimension = map(int, header.split()[:2])

        num_loaded = 0

        # -2 for PADDING and UNKNOWN, which get initialized to zero
        assert vocabulary_size == len(encoding) - 2
        lut = np.zeros((len(encoding), dimension))

        for line in in_file:
            line = line.split()
            word = line[0]

            if word not in encoding:
                print ("No encoding for '{}'".format(word))
                continue

            vec = np.asarray(map(float, line[1:]))
            assert vec.size == dimension

            lut[encoding[word]] = vec

            num_loaded += 1

    assert num_loaded == vocabulary_size

    return lut


def main():
    random.seed(435)
    np.random.seed(23421)
    np.set_printoptions(linewidth=100)

    data_dir = os.path.join("/data/mulga/mdenil/amazon-reviews", "shards")

    batch_size = 100

    with open(os.path.join(data_dir, "dictionary.sentences.clean.encoding.json")) as encoding_file:
        encoding = json.load(encoding_file)

    print(len(encoding))


    # pretrained_lut = load_word2vec_embeddings(
    #     os.path.join("/data/brown/mdenil/amazon-reviews/word2vec-embeddings", "word-embeddings-30.txt"),
    #     encoding)


    train_data_provider = ShardedLabelledDocumentMinibatchProvider(
        shard_dir=os.path.join(data_dir, "train"),
        shard_pattern="shard_[0-9]*.sentences.clean.projected.json.gz",
        batch_size=batch_size,
        padding='PADDING',
        n_labels=5,
        # n_labels=2,
        fixed_n_sentences=15,
        fixed_n_words=25)

    validation_data_provider = ShardedLabelledDocumentMinibatchProvider(
        shard_dir=os.path.join(data_dir, "test"),
        shard_pattern="shard_[0-9]*.sentences.clean.projected.json.gz",
        batch_size=batch_size,
        padding='PADDING',
        n_labels=5,
        # n_labels=2,
        fixed_n_sentences=15,
        fixed_n_words=25)

    model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=encoding),

            WordEmbedding(
                dimension=30,
                vocabulary_size=len(encoding),
                padding=encoding['PADDING']),

            # WordEmbedding(
            #     dimension=pretrained_lut.shape[1],
            #     vocabulary_size=len(encoding),
            #     padding=encoding['PADDING'],
            #     E=pretrained_lut),

            # Dropout(('b', 'w', 'f'), 0.2),

            SentenceConvolution(
                n_feature_maps=10,
                kernel_width=3,
                n_channels=30,
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps=10),

            # KMaxPooling(k=7, k_dynamic=0.5),
            #
            # Tanh(),
            #
            # SentenceConvolution(
            #     n_feature_maps=30,
            #     kernel_width=3,
            #     n_channels=10,
            #     n_input_dimensions=1),
            #
            # Bias(
            #     n_input_dims=1,
            #     n_feature_maps=30),

            KMaxPooling(k=5),

            Tanh(),

            ReshapeForDocuments(),

            SentenceConvolution(
                n_feature_maps=20,
                kernel_width=3,
                n_channels=50,
                n_input_dimensions=1),

            Bias(
                n_input_dims=1,
                n_feature_maps=20),

            KMaxPooling(k=5),

            Tanh(),

            # Dropout(('b', 'd', 'f', 'w'), 0.5),

            # Softmax(
            #     # n_classes=2,
            #     n_classes=5,
            #     n_input_dimensions=100),

            Linear(
                n_input=100,
                n_output=1)
            ]
    )

    print(model)


    # cost_function = CrossEntropy()
    cost_function = SquaredError()

    regularizer = L2Regularizer(lamb=1e-5)

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    n_epochs = 1
    # n_batches = train_data_provider.batches_per_epoch * n_epochs

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
            for _ in xrange(1):
                X_valid_batch, Y_valid_batch, meta_valid = validation_data_provider.next_batch()
                Y_valid.append(Y_valid_batch)
                Y_hat.append(model.fprop(X_valid_batch, meta=meta_valid))
            Y_valid = Y_valid[0].get()
            Y_hat = Y_hat[0].get()
            # Y_valid = np.concatenate(Y_valid, axis=0)
            # Y_hat = np.concatenate(Y_hat, axis=0)
            # assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            # acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))
            acc = np.mean(np.abs(Y_valid - Y_hat))

            # if acc > best_acc:
            #     best_acc = acc
            # with open("/home/mdenil/model.pkl", 'w') as model_file:
            #     pickle.dump(model, model_file, protocol=-1)

            current = dict()
            current['B']=batch_index
            current['A']=acc
            current['C']=costs[-1].get()
            current['Prop']=np.argmax(Y_hat, axis=1).mean()
            current['Params']=np.mean(np.abs(model.pack()))

            progress.append(current)
            print(current)
            with open("progress.pkl", 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        # if batch_index == 100:
        #     break

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model, model_file, protocol=-1)

    time_end = time.time()

    print("Time elapsed: {}s".format(time_end - time_start))


if __name__ == "__main__":
    main()