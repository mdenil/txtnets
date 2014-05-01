__author__ = 'mdenil'

import numpy as np
import scipy.optimize
import pyprind
import os
import random
import simplejson as json
import cPickle as pickle
import matplotlib.pyplot as plt

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


class ModelEvaluator(object):
    def __init__(self, alphabet_encoding):
        self.alphabet_encoding = alphabet_encoding

        #self.test_words = ["cat", "feline", "car", "truck", "tuck"]
        #self.test_words = ['cat', 'CAT', 'egg', 'eggplant', 'brontosaurus']
        self.test_words = ['pokemon', 'bigger', 'better', 'faster', 'stronger']

        _encoded_test_words = map(self._encode, self.test_words)
        self.data_provider = PaddedSequenceMinibatchProvider(
            X=_encoded_test_words,
            padding=self.alphabet_encoding['PADDING'],
            batch_size=len(_encoded_test_words),
            shuffle=False)

    def _encode(self, word):
        encoded_word = [self.alphabet_encoding[c] for c in word]
        encoded_word = [self.alphabet_encoding['START']] + encoded_word + [self.alphabet_encoding['END']]
        return encoded_word

    def evaluate(self, model):
        X, meta = self.data_provider.next_batch()

        Y_hat, meta, _ = model.fprop(X, meta=meta, num_layers=-1, return_state=True)
        Y_hat, _ = meta['space_above'].transform(Y_hat, ['b', 'dwf'])

        # print Y_hat

        # compute cosine distance between rows of Y
        Y_hat_norms = np.sqrt(np.sum(Y_hat**2, axis=1, keepdims=True))

        distances = np.dot(Y_hat, Y_hat.T) / (Y_hat_norms * Y_hat_norms.T)

        print self.test_words
        print distances


class RandomAlphabetCorruption(object):
    def __init__(self, alphabet_encoding):
        self._noise_values = [ v for k,v in alphabet_encoding.iteritems() if k not in ['START', 'END', 'PADDING']]

    def apply(self, X, meta):
        X_dirty = X.copy()

        # choose locations to corrupt, don't choose first or last character because those are the padding
        for idx, x in enumerate(X_dirty):
            times_to_corrupt = int(1+0.5*meta['lengths'][idx])
            for _ in xrange(times_to_corrupt):
                corrupt_at = np.random.randint(1, meta['lengths'][idx]-1)
                x[corrupt_at] = np.random.choice(self._noise_values)



        return X_dirty, meta



def load_json(file_name):
    with open(file_name) as f:
        return json.loads(f.read())


if __name__ == "__main__":
    np.set_printoptions(linewidth=100)
    data = load_json(os.path.join(os.environ['DATA'], "words", "words.encoded.json"))
    alphabet = load_json(os.path.join(os.environ['DATA'], "words", "words.alphabet.encoding.json"))

    train_data_provider = PaddedSequenceMinibatchProvider(
        X=data,
        padding=alphabet['PADDING'],
        batch_size=100)

    embedding_dimension = 8
    vocabulary_size = len(alphabet)
    n_feature_maps = 8
    kernel_width = 5
    pooling_size = 2

    n_epochs = 1

    model = CSM(
        layers=[
            WordEmbedding(
                dimension=embedding_dimension,
                vocabulary_size=len(alphabet)),

            SentenceConvolution(
                n_feature_maps=n_feature_maps,
                kernel_width=kernel_width,
                n_channels=1,
                n_input_dimensions=embedding_dimension),

            SumFolding(),

            KMaxPooling(k=pooling_size),

            # Bias(
            #     n_input_dims=embedding_dimension / 2,
            #     n_feature_maps=n_feature_maps),

            Linear(
                n_input=n_feature_maps*pooling_size*embedding_dimension / 2,
                n_output=64
            ),

            Tanh(),

            Linear(
                n_output=1,
                n_input=64),
        ]
    )

    print model

    cost_function = LargeMarginCost(0.1)
    noise_model = RandomAlphabetCorruption(alphabet)

    objective = NoiseContrastiveObjective(
        cost=cost_function,
        data_provider=train_data_provider,
        noise_model=noise_model)

    update_rule = AdaGrad(
        gamma=0.1,
        model_template=model)

    optimizer = SGD(model=model, objective=objective, update_rule=update_rule)

    evaluator = ModelEvaluator(alphabet)

    n_batches = train_data_provider.batches_per_epoch * n_epochs

    costs = []
    prev_weights = model.pack()
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        if batch_index % 50 == 0:
            print costs[-1], iteration_info['param_mean_abs_values']

            evaluator.evaluate(model)

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model, model_file, protocol=-1)