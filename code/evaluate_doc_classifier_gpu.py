__author__ = 'mdenil'

import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle

import gpu.model.dropout

from gpu.optimize.data_provider import LabelledDocumentMinibatchProvider

def run():
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    # LOADING
    data_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(data_dir, "stanfordmovie.train.sentences.clean.projected.json")) as data_file:
        data = json.load(data_file)
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(data_dir, "stanfordmovie.train.sentences.clean.dictionary.encoding.json")) as encoding_file:
        encoding = json.load(encoding_file)

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=1,
        padding='PADDING',
        fixed_n_sentences=30,
        fixed_n_words=50)

    model_file = "model_best.pkl"
    with open(model_file) as model_file:
        trained_model = gpu.model.dropout.remove_dropout(pickle.load(model_file))

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


if __name__ == "__main__":
    run()