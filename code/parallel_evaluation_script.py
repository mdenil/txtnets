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


progress_dir = "../results"
output_dir = "../output"

trained_models = ['00000003',
                  '00000025',
                  '00000037',
                  '00000054',
                  '00000055',
                  '00000070',
                  '00000075',
                  '00000097',
                  '00000102',
                  '00000105',
                  '00000109']


model_files = ['model_00000003.pkl',
                'model_00000025.pkl',
                'model_00000037.pkl',
                'model_00000054.pkl',
                'model_00000055.pkl',
                'model_00000070.pkl',
                'model_00000075.pkl',
                'model_00000097.pkl',
                'model_00000102.pkl',
                'model_00000105.pkl',
                'model_00000109.pkl']

@ruffus.originate(model_files)
def copy_models(output_file):
    for result_folder in trained_models:
        sh.cp(os.path.join(progress_dir, result_folder, 'model_best.pkl'), 'model_'+result_folder+'.pkl')


@ruffus.transform(copy_models, ruffus.suffix(".pkl"), ".nodropout.pkl")
def no_dropout(input_file_name, output_file_name):
    with open(input_file_name) as model_file:
        trained_model = pickle.load(model_file)

    no_dropout = remove_dropout(trained_model)

    with open(output_file_name, 'w') as model_file:
                pickle.dump(no_dropout, model_file, protocol=-1)


@ruffus.transform(no_dropout, ruffus.suffix(".pkl"), ".evaluation.pkl")
def evaluate(input_file_name, output_file_name):
    data_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(data_dir, "stanfordmovie.test.sentences.clean.projected.json")) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=1,
        padding='PADDING',
        fixed_n_sentences=15,
        fixed_n_words=50)

    model_file = input_file_name
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

            with open(output_file_name, 'w') as progress_file:
                pickle.dump(progress, progress_file, protocol=-1)

        #FINAL EVALUATION
        acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
        current = dict()
        current['B'] = batch_index
        current['A'] = acc
        current['T'] = time.time()-time_start

        progress.append(current)

        with open(output_file_name, 'w') as progress_file:
            pickle.dump(progress, progress_file, protocol=-1)

        with open('yhat_'+output_file_name, 'w') as progress_file:
            pickle.dump(full_Y_hat, progress_file, protocol=-1)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sh.cd(output_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)