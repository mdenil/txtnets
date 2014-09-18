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

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
from cpu.optimize.update_rule import Momentum

from cpu.optimize.data_provider import LabelledSequenceMinibatchProvider

from cpu.optimize.grad_check import fast_gradient_check

from cpu.optimize.sgd import SGD



def optimize_and_save(model, alphabet, n_batches, data_file_name, chars_or_words, result_file_name):

    print result_file_name

    with gzip.open(data_file_name) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))

        # shuffle
        combined = zip(X, Y)
        random.shuffle(combined)
        X, Y = map(list, zip(*combined))

        # map labels to something useful
        Y = [ [":)", ":("].index(y) for y in Y ]

    if chars_or_words == 'chars':
        X = [list(x) for x in X]
    elif chars_or_words == 'words':
        # replace unknowns with an unknown character
        tokenizer = WordPunctTokenizer()
        new_X = []
        for x in X:
            new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
        X = new_X
    else:
        raise ValueError("I don't know what that means :(")


    train_data_provider = LabelledSequenceMinibatchProvider(
        X=X[:-500],
        Y=Y[:-500],
        batch_size=50,
        padding='PADDING')

    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-500:],
        Y=Y[-500:],
        batch_size=500,
        padding='PADDING')



    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(
        cost=cost_function,
        data_provider=train_data_provider)

    update_rule = AdaGrad(
        gamma=0.05,
        model_template=model)

    regularizer = L2Regularizer(lamb=1e-4)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule,
        regularizer=regularizer)

    print model

    monitor_info = []
    iteration_info = []
    for batch_index, info in enumerate(optimizer):
        iteration_info.append(info)

        if batch_index % 10 == 0:
            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = model.fprop(X_valid, meta=meta_valid)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))
            prop_1 = np.argmax(Y_hat, axis=1).mean()

            monitor_info.append({
                'batch_index': batch_index,
                'acc': acc,
                'prop_1': prop_1,
            })

            print "B: {}, A: {}, C: {}, Prop1: {}, Param size: {}".format(
                batch_index,
                acc, info['cost'],
                prop_1,
                np.mean(np.abs(model.pack())))

        if batch_index == n_batches - 1:
            break

    result = {
        'model': model,
        'iteration_info': iteration_info,
        'monitor_info': monitor_info,
        }

    with open(result_file_name, 'w') as result_file:
        pickle.dump(result, result_file, protocol=-1)



def model_one_layer_small_embedding(alphabet):
    return CSM(
        layers=[
            DictionaryEncoding(vocabulary=alphabet),

            WordEmbedding(
                dimension=32,
                vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32),

            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),

            MaxFolding(),

            Softmax(
                n_classes=2,
                n_input_dimensions=280),
            ]
    )

def model_one_layer_large_embedding(alphabet):
    return CSM(
        layers=[
            DictionaryEncoding(vocabulary=alphabet),

            WordEmbedding(
               dimension=32*4,
               vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32*4),

            Relu(),
            SumFolding(),
            SumFolding(),
            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),

            MaxFolding(),

            Softmax(
                n_classes=2,
                n_input_dimensions=280),
            ]
    )


def model_two_layer_small_embedding(alphabet):
    return CSM(
        layers=[
            DictionaryEncoding(vocabulary=alphabet),

            WordEmbedding(
                dimension=32,
                vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32),

            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),


            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=5,
                n_channels=5,
                n_input_dimensions=16),

            KMaxPooling(k=4),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),


            Softmax(
                n_classes=2,
                n_input_dimensions=320),
            ]
    )


def model_two_layer_large_embedding(alphabet):
    return CSM(
        layers=[
            DictionaryEncoding(vocabulary=alphabet),

            WordEmbedding(
                dimension=32*4,
                vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32*4),

            Relu(),
            SumFolding(),
            SumFolding(),
            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),


            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=5,
                n_channels=5,
                n_input_dimensions=16),

            KMaxPooling(k=4),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),


            Softmax(
                n_classes=2,
                n_input_dimensions=320),
            ]
    )

def model_one_layer_variant_2(alphabet):
    return CSM(
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


if __name__ == "__main__":
    np.set_printoptions(linewidth=100)
    data_dir = os.path.join("data", "tweets")



    with open(os.path.join(data_dir, "tweets_100k.english.balanced.alphabet.encoding.json")) as alphabet_file:
        character_alphabet = json.loads(alphabet_file.read())

    with open(os.path.join(data_dir, "tweets_100k.english.balanced.clean.alphabet.encoding.json")) as alphabet_file:
        clean_character_alphabet = json.loads(alphabet_file.read())

    with open(os.path.join(data_dir, "tweets_100k.english.balanced.clean.dictionary.encoding.json")) as alphabet_file:
        clean_word_alphabet = json.loads(alphabet_file.read())

    n_batches = 1

    # one layer small embedding

    optimize_and_save(
        model=model_one_layer_small_embedding(character_alphabet),
        alphabet=character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.json.gz"),
        result_file_name="models/model_one_layer_small_embedding_chars.pkl")

    optimize_and_save(
        model=model_one_layer_small_embedding(clean_character_alphabet),
        alphabet=clean_character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_small_embedding_clean_chars.pkl")

    optimize_and_save(
        model=model_one_layer_small_embedding(clean_word_alphabet),
        alphabet=clean_word_alphabet,
        chars_or_words='words',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_small_embedding_clean_words.pkl")

    # one layer large embedding

    optimize_and_save(
        model=model_one_layer_large_embedding(character_alphabet),
        alphabet=character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.json.gz"),
        result_file_name="models/model_one_layer_large_embedding_chars.pkl")

    optimize_and_save(
        model=model_one_layer_large_embedding(clean_character_alphabet),
        alphabet=clean_character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_large_embedding_clean_chars.pkl")

    optimize_and_save(
        model=model_one_layer_large_embedding(clean_word_alphabet),
        alphabet=clean_word_alphabet,
        chars_or_words='words',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_large_embedding_clean_words.pkl")

    # model_two_layer_small_embedding

    optimize_and_save(
        model=model_two_layer_small_embedding(character_alphabet),
        alphabet=character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.json.gz"),
        result_file_name="models/model_two_layer_small_embedding_chars.pkl")

    optimize_and_save(
        model=model_two_layer_small_embedding(clean_character_alphabet),
        alphabet=clean_character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_two_layer_small_embedding_clean_chars.pkl")

    optimize_and_save(
        model=model_two_layer_small_embedding(clean_word_alphabet),
        alphabet=clean_word_alphabet,
        chars_or_words='words',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_two_layer_small_embedding_clean_words.pkl")


    # model_two_layer_large_embedding

    optimize_and_save(
        model=model_two_layer_large_embedding(character_alphabet),
        alphabet=character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.json.gz"),
        result_file_name="models/model_two_layer_large_embedding_chars.pkl")

    optimize_and_save(
        model=model_two_layer_large_embedding(clean_character_alphabet),
        alphabet=clean_character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_two_layer_large_embedding_clean_chars.pkl")

    optimize_and_save(
        model=model_two_layer_large_embedding(clean_word_alphabet),
        alphabet=clean_word_alphabet,
        chars_or_words='words',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_two_layer_large_embedding_clean_words.pkl")

    # model_one_layer_variant_2

    optimize_and_save(
        model=model_one_layer_variant_2(character_alphabet),
        alphabet=character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.json.gz"),
        result_file_name="models/model_one_layer_variant_2_chars.pkl")

    optimize_and_save(
        model=model_one_layer_variant_2(clean_character_alphabet),
        alphabet=clean_character_alphabet,
        chars_or_words='chars',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_variant_2_clean_chars.pkl")

    optimize_and_save(
        model=model_one_layer_variant_2(clean_word_alphabet),
        alphabet=clean_word_alphabet,
        chars_or_words='words',
        n_batches=n_batches,
        data_file_name=os.path.join(data_dir, "tweets_100k.english.balanced.clean.json.gz"),
        result_file_name="models/model_one_layer_variant_2_clean_words.pkl")