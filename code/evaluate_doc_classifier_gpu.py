__author__ = 'mdenil'

import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle
import pyprind
import argparse

import gpu.model.dropout
import gpu.model.host_device_component_mapping

from gpu.optimize.data_provider import LabelledDocumentMinibatchProvider

def run():
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="model_best.pkl")
    args = parser.parse_args()

    # LOADING
    data_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(data_dir, "stanfordmovie.test.sentences.clean.projected.json")) as data_file:
        data = json.load(data_file)
        random.shuffle(data)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=50,
        padding='PADDING',
        fixed_n_sentences=30,
        fixed_n_words=50)

    model_file_path = args.model_file
    with open(model_file_path) as model_file:
            trained_model = gpu.model.dropout.remove_dropout(
                gpu.model.host_device_component_mapping.move_to_gpu(
                    pickle.load(model_file)))

    # PRINT USEFUL INFORMATION
    print evaluation_data_provider.batches_per_epoch
    print trained_model

    progress_bar = pyprind.ProgBar(evaluation_data_provider.batches_per_epoch)

    ys = []
    y_hats = []
    for batch_index in xrange(evaluation_data_provider.batches_per_epoch):
        x_batch, y_batch, meta_batch = evaluation_data_provider.next_batch()
        y_hat_batch = trained_model.fprop(x_batch, meta=meta_batch)

        ys.append(y_batch.get())
        y_hats.append(y_hat_batch.get())

        progress_bar.update()

    y = np.concatenate(ys, axis=0)
    y_hat = np.concatenate(y_hats, axis=0)

    print y.shape
    print y_hat.shape

    acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))

    print acc


if __name__ == "__main__":
    run()