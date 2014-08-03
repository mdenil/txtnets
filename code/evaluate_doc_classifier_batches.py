__author__ = 'mdenil'

import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle
import argparse
import re
import pyprind

import gpu.model.dropout
import gpu.model.host_device_component_mapping

from gpu.optimize.data_provider import LabelledDocumentMinibatchProvider

def run():
    # random.seed(435)
    # np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batches_dir")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--min_batch", type=int, default=0)
    parser.add_argument("--training", type=bool, default=False)
    args = parser.parse_args()

    is_batch_file_name = re.compile("^model_[0-9]+.pkl$")

    model_file_names = [f for f in os.listdir(args.batches_dir) if is_batch_file_name.match(f)]
    # evaluate the models in random order to get even coverage of the curve
    random.shuffle(model_file_names)
    if args.training:
        results_file_names = [f.replace(".pkl", "_train-results.pkl") for f in model_file_names]
    else:
        results_file_names = [f.replace(".pkl", "_test-results.pkl") for f in model_file_names]

    model_file_paths = [os.path.join(args.batches_dir, f) for f in model_file_names]
    results_file_paths = [os.path.join(args.batches_dir, f) for f in results_file_names]

    model_file_paths, results_file_paths = zip(*filter(
        lambda (n, p): not os.path.exists(p),
        zip(model_file_paths, results_file_paths)))


    # LOADING
    data_dir = os.path.join("../data", "stanfordmovie")

    if args.training:
        data_file = "stanfordmovie.train.sentences.clean.projected.json"
    else:
        data_file = "stanfordmovie.test.sentences.clean.projected.json"

    print "Evaluating on {}".format(data_file)

    with open(os.path.join(data_dir, data_file)) as data_file:
        data = json.load(data_file)
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    progress_bar = pyprind.ProgBar(len(model_file_paths) * len(X) / args.batch_size)

    for model_file_path, results_file_path in zip(model_file_paths, results_file_paths):
        if os.path.exists(results_file_path):
            progress_bar.update(args.batch_size)
            continue

        batch_index = int(re.match(".*model_([0-9]+)", model_file_path).group(1))
        if batch_index < args.min_batch:
            progress_bar.update(args.batch_size)
            continue

        # Re make this every time just to be absolutely sure we're doing everything in the same order for each model
        evaluation_data_provider = LabelledDocumentMinibatchProvider(
            X=X,
            Y=Y,
            batch_size=args.batch_size,
            padding='PADDING',
            fixed_n_sentences=30,
            fixed_n_words=50,
            shuffle=False)


        with open(model_file_path) as model_file:
            trained_model = gpu.model.dropout.remove_dropout(
                gpu.model.host_device_component_mapping.move_to_gpu(
                    pickle.load(model_file)))

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

        acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))

        with open(results_file_path, 'w') as results_file:
            pickle.dump({
                'acc': acc,
                'y_hat': y_hat,
                'y': 'y',
            }, results_file, protocol=-1)

        print model_file_path, acc


if __name__ == "__main__":
    run()