__author__ = 'albandemiraj'




__author__ = 'albandemiraj'


__author__ = 'albandemiraj'
import cPickle as pickle
import os
import sh
import ruffus
import psutil
from cpu.model.dropout import remove_dropout
import simplejson as json
import numpy as np
from generic.optimize.data_provider import LabelledDocumentMinibatchProvider
import random
import time
import argparse



#progress_dir = "../results"
output_dir = "../output"

# trained_models = []
# model_files = []
def run():
    parser = argparse.ArgumentParser(
    description="Collect cluster data.")
    parser.add_argument("--model_file", help="where the results subset of folders is")
    parser.add_argument("--data_dir", default='../data/stanfordmovie')
    parser.add_argument("--dataset_name", default='stanfordmovie')
    parser.add_argument("--batch_size", default=5)

    args = parser.parse_args()

    model_file = args.model_file
    output_eval = model_file.replace('.pkl', '_evaluation.pkl')
    output_predict = 'yhat_'+output_eval
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    batch_size = args.batch_size

    #PREPARING DIRECTORIES
    with open(os.path.join(data_dir, dataset_name+".test.sentences.clean.projected.json")) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=batch_size,
        padding='PADDING')

    with open(model_file) as model_file:
        trained_model = pickle.load(model_file)

    # EVALUATING
    time_start = time.time()

    #EVALUATING
    progress = []

    X_valid, full_Y_valid, meta_valid = evaluation_data_provider.next_batch()
    full_Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
    assert np.all(np.abs(full_Y_hat.sum(axis=1) - 1) < 1e-6)

    for batch_index in xrange(0, evaluation_data_provider.batches_per_epoch-2):
        X_valid, Y_valid, meta_valid = evaluation_data_provider.next_batch()
        Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
        assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

        full_Y_valid = np.concatenate((full_Y_valid, Y_valid), axis=0)
        full_Y_hat = np.concatenate((full_Y_hat, Y_hat), axis=0)

        #EVERY 100BATCHES
        if batch_index % 100 == 0:
            acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
            current = dict()
            current['B'] = batch_index
            current['A'] = acc
            current['T'] = time.time()-time_start

            progress.append(current)

            with open(output_eval, 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        #FINAL EVALUATION
        acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
        current = dict()
        current['B'] = batch_index
        current['A'] = acc
        current['T'] = time.time()-time_start

        progress.append(current)

        with open(output_eval, 'w') as progress_file:
            pickle.dump(progress, progress_file, protocol=-1)

        with open(output_predict, 'w') as progress_file:
            pickle.dump(full_Y_hat, progress_file, protocol=-1)

if __name__ == "__main__":
    run()