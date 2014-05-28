__author__ = 'albandemiraj'

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

from cpu.optimize.regularizer import L2Regularizer

from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.update_rule import AdaDelta
from cpu.optimize.update_rule import Basic
from cpu.optimize.update_rule import NesterovAcceleratedGradient
from cpu.optimize.update_rule import Momentum

from cpu.optimize.data_provider import LabelledSequenceMinibatchProvider

from cpu.optimize.grad_check import ModelGradientChecker

from cpu.optimize.sgd import SGD
from generic.model.transfer import DocumentConvolution
from generic.optimize.data_provider import LabelledDocumentMinibatchProvider

if __name__ == "__main__":
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    # LOADING
    tweets_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(tweets_dir, "stanfordmovie.test.clean.json")) as data_file:
        data = json.loads(data_file.read())
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(tweets_dir, "stanfordmovie.test.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=8,
        padding='PADDING')

    model_file = "model_updated_save2.pkl"
    with open(model_file) as model_file:
        trained_model = pickle.load(model_file)

    # PRINT USEFUL INFORMATION
    print evaluation_data_provider.batches_per_epoch
    print trained_model

    time_start = time.time()

    #EVALUATING
    full_Y_valid = []
    full_Y_hat = []
    for batch_index in xrange(0, evaluation_data_provider.batches_per_epoch-1):
        X_valid, Y_valid, meta_valid = evaluation_data_provider.next_batch()
        Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
        assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

        full_Y_valid = np.concatenate((full_Y_valid, Y_valid), axis=0)
        full_Y_hat = np.concatenate((full_Y_hat, Y_hat), axis=0)

        if batch_index % 100 == 0:
            acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
            print 'Accuracy so far: '+str(acc)

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)