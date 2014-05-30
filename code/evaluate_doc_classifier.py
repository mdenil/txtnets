__author__ = 'albandemiraj'

import numpy as np
import scipy.optimize
import pyprind
import os
import time
import random
import simplejson as json
import cPickle as pickle

from generic.optimize.data_provider import LabelledDocumentMinibatchProvider

if __name__ == "__main__":
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    # LOADING
    tweets_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(tweets_dir, "stanfordmovie.test.sentences.clean.projected.json")) as data_file:
        data = json.loads(data_file.read())
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(tweets_dir, "stanfordmovie.train.sentences.clean.dictionary.encoding.json")) as \
            alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    # tokenizer = WordPunctTokenizer()
    # new_X = []
    # for x in X:
    #     new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    # X = new_X


    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=50,
        padding='PADDING',
        fixed_n_sentences=15,
        fixed_n_words=50)

    model_file = "/home/mdenil/code/txtnets/txtnets_deployed/results/test_job_launcher/00000075/model_best.pkl"
    with open(model_file) as model_file:
        trained_model = pickle.load(model_file)

    # PRINT USEFUL INFORMATION
    print evaluation_data_provider.batches_per_epoch
    print trained_model

    time_start = time.time()

    #EVALUATING
    X_valid, full_Y_valid, meta_valid = evaluation_data_provider.next_batch()
    full_Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
    assert np.all(np.abs(full_Y_hat.sum(axis=1) - 1) < 1e-6)

    for batch_index in xrange(0, evaluation_data_provider.batches_per_epoch-2):
        X_valid, Y_valid, meta_valid = evaluation_data_provider.next_batch()
        Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
        assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

        full_Y_valid = np.concatenate((full_Y_valid, Y_valid), axis=0)
        full_Y_hat = np.concatenate((full_Y_hat, Y_hat), axis=0)

        if batch_index % 100 == 0:
            acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
            print 'Batch: '+str(batch_index)+'/'+str(evaluation_data_provider.batches_per_epoch)+';  Accuracy so far: '+str(acc)

    time_end = time.time()

    acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
    print 'FINAL ACCURACY: '+str(acc)
    print "Time elapsed: {}s".format(time_end - time_start)