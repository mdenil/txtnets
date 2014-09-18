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

from collections import OrderedDict

from cpu.model.model import CSM
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

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
from cpu.optimize.update_rule import Momentum

from cpu.optimize.grad_check import fast_gradient_check

from cpu.optimize.sgd import SGD

class LabelledSequenceMinibatchProvider(object):
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._batch_index = -1
        self.batches_per_epoch = len(X) / batch_size

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]
        Y_batch = self.Y[batch_start:batch_end]

        Y_batch = [ [":)", ":("].index(y) for y in Y_batch ]
        Y_batch = np.equal.outer(Y_batch, [0, 1]).astype(np.float)


        lengths_batch = np.asarray(map(len, X_batch))
        max_length_batch = int(lengths_batch.max())

        meta = {
            'lengths': lengths_batch,
            'space_below': space.Space(axes=['b', 'w'], extent=OrderedDict([('b', len(X_batch)), ('w', max_length_batch)]))
        }

        return X_batch, Y_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        combined = zip(self.X, self.Y)
        random.shuffle(combined)
        self.X, self.Y = map(list, zip(*combined))

class UnlabelledSequenceBatchDataProvider(object):
    def __init__(self, X, padding):
        self._lengths = np.asarray(map(len, X))
        self._max_length = self._lengths.max()
        self.padding = padding

        self.X = np.vstack([np.atleast_2d(self._add_padding(x)) for x in X])

        self.batch_size = len(X)
        self.batches_per_epoch = 1

    def next_batch(self):
        meta = {
            'lengths': self._lengths.copy(),
            'space_below': space.Space.infer(self.X, axes=['b', 'w'])
        }

        return self.X, meta

    def _add_padding(self, x):
        return x + [self.padding] * (self._max_length - len(x))



class WordFromCharacterEmbedding(layer.Layer):
    def __init__(self,
                 embedding_model,
                 alphabet_encoding,
                 ):

        self.embedding_model = embedding_model
        self.alphabet_encoding = alphabet_encoding

    def fprop(self, X, meta):
        fprop_state = {}

        sentence_lengths = meta['lengths']

        fprop_state['sentence_lengths'] = sentence_lengths

        # print sentence_lengths

        # print "X", X

        # flatten sentences
        X_words = []
        for sentence in X:
            X_words.extend(sentence)

        # map each character to dictionary index, ignore sentence breaks, but preserve word breaks
        X_encoded = []
        for word in X_words:
            X_encoded.append(self._encode(word))

        # print X_encoded

        data_provider = UnlabelledSequenceBatchDataProvider(
            X=X_encoded,
            padding=self.alphabet_encoding['PADDING'])

        X_word_batch, meta_word_batch = data_provider.next_batch()

        # print X_word_batch.shape

        Y, meta_word_batch, fprop_state_word_batch = self.embedding_model.fprop(X_word_batch, meta=dict(meta_word_batch), return_state=True)

        fprop_state['embedding_fprop_state'] = fprop_state_word_batch

        # print Y.shape
        # print meta_word_batch['space_above']



        Y, working_space = meta_word_batch['space_above'].transform(Y, ['b','wdf'])
        fprop_state['backprop_space'] = working_space
        working_space = working_space.without_axes('w')
        working_space = working_space.with_extent(d=Y.shape[1])

        # print Y.shape
        # print working_space

        # unpack to padd sentences
        w = max(sentence_lengths)
        b = len(sentence_lengths)
        d = working_space.get_extent('d')
        X_sentences = np.zeros((b, w, d))

        # print X_sentences.shape

        base = 0
        for X_sentence, sentence_length in zip(X_sentences, sentence_lengths):
            X_sentence[:sentence_length, :] = Y[base:base+sentence_length]
            base += sentence_length

        # print X_sentences

        X_sentences_space = space.Space.infer(X_sentences, ['b', 'w', 'd'])

        meta['space_above'] = X_sentences_space

        # print "X", X_sentences.shape, X_sentences_space


        return X_sentences, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        raise NotImplementedError

    def grads(self, delta, meta, fprop_state):



        delta_below = np.zeros((np.sum(fprop_state['sentence_lengths']), meta['space_above'].get_extent('d')))
        delta_below_space = fprop_state['backprop_space']


        delta_above, delta_above_space = meta['space_above'].transform(delta, ['bw', 'df'])

        # print delta_above.shape, delta_above_space
        # print delta_below.shape, delta_below_space

        # pack up words for the character model
        base_below = 0
        base_above = 0
        step_above = meta['space_above'].get_extent('w')
        for sentence_length in fprop_state['sentence_lengths']:
            delta_below[base_below:base_below+sentence_length] = delta_above[base_above:base_above+sentence_length]
            base_below += sentence_length
            base_above += step_above

        meta_embedding_model = {
            'lengths': fprop_state['sentence_lengths'],
            'space_above': delta_below_space,
        }
        return self.embedding_model.grads(delta_below, meta=meta_embedding_model, fprop_state=fprop_state['embedding_fprop_state'])

    def _encode(self, x):
        return [self.alphabet_encoding['START']] + [self.alphabet_encoding[c] for c in x] + [self.alphabet_encoding['END']]

    def params(self):
        return self.embedding_model.params()

    def __repr__(self):
        return "{}(\nembedding_model={}\n)".format(
            self.__class__.__name__,
            self.embedding_model)





if __name__ == "__main__":
    random.seed(32423)
    np.set_printoptions(linewidth=100)

    tweets_dir = os.path.join("data", "tweets")


    with gzip.open(os.path.join(tweets_dir, "tweets_100k.english.balanced.json.gz")) as data_file:
        data = [json.loads(line) for line in data_file]
        X, Y = map(list, zip(*data))

        # shuffle
        combined = zip(X, Y)
        random.shuffle(combined)
        X, Y = map(list, zip(*combined))

    with open(os.path.join(tweets_dir, "tweets_100k.english.alphabet.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    # This model expects lists of words.
    X = [x.split(" ") for x in X]

    train_data_provider = LabelledSequenceMinibatchProvider(
        X=X[:-500],
        Y=Y[:-500],
        batch_size=100)

    print train_data_provider.batches_per_epoch

    validation_data_provider = LabelledSequenceMinibatchProvider(
        X=X[-500:],
        Y=Y[-500:],
        batch_size=500)


    word_embedding_model = CSM(
        layers=[
            WordEmbedding( # really a character embedding
                dimension=16,
                vocabulary_size=len(alphabet)
            ),

            SentenceConvolution(
                n_feature_maps=10,
                kernel_width=5,
                n_channels=1,
                n_input_dimensions=16),

            SumFolding(),

            KMaxPooling(k=2),
            MaxFolding(),

            Tanh(),
        ]
    )

    word_embedding = WordFromCharacterEmbedding(
        embedding_model=word_embedding_model,
        alphabet_encoding=alphabet)

    # print word_embedding.fprop(X, meta)

    tweet_model = CSM(
        layers=[
            word_embedding,

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=80),

            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=40,
                n_feature_maps=5),

            Tanh(),

            # Linear(
            #     n_input=1400,
            #     n_output=500),
            #
            # Tanh(),

            MaxFolding(),

            Softmax(
                n_classes=2,
                n_input_dimensions=700),
        ]
    )

    print tweet_model


    # X, Y, meta = train_data_provider.next_batch()
    # Y, meta, fprop_state = model.fprop(X, meta, return_state=True)

    # print meta['lengths']
    # print Y.shape, meta['space_above']

    # print [p.shape for p in model.params()]

    cost_function = CrossEntropy()

    objective = CostMinimizationObjective(cost=cost_function, data_provider=train_data_provider)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=tweet_model)

    optimizer = SGD(model=tweet_model, objective=objective, update_rule=update_rule)


    n_epochs = 1
    n_batches = train_data_provider.batches_per_epoch * n_epochs

    costs = []
    prev_weights = tweet_model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 10 == 0:
            # print costs[-1], iteration_info['param_mean_abs_values']

            X_valid, Y_valid, meta_valid = validation_data_provider.next_batch()

            Y_hat = tweet_model.fprop(X_valid, meta=meta_valid)
            assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

            # print Y_hat[:5]

            acc = np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y_valid, axis=1))

            print "B: {}, A: {}, C: {}, Param size: {}".format(batch_index, acc, costs[-1], np.mean(np.abs(tweet_model.pack())))


        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(tweet_model, model_file, protocol=-1)