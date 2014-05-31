__author__ = 'albandemiraj'

import cPickle as pickle
import os
import simplejson as json
import numpy as np
from generic.optimize.data_provider import LabelledDocumentMinibatchProvider
import time


def evaluate_model(model_file, test_file, batch_size, output_dir=None, job_id=None):
    #CHECKING IF WE ARE WORKING ON A TEST SET
    assert('test' in test_file)

    #INITIALIZING THE FILENAMES
    if job_id:
        output_eval = model_file.rsplit('/')[::-1][0].replace('.pkl', '_'+model_file.rsplit('/')[::-1][1]+'_evaluation.pkl')
        output_predict = model_file.rsplit('/')[::-1][0].replace('.pkl', '_'+model_file.rsplit('/')[::-1][1]+'_yhat.pkl')
    else:
        output_eval = model_file.rsplit('/')[::-1][0].replace('.pkl', '_evaluation.pkl')
        output_predict = model_file.rsplit('/')[::-1][0].replace('.pkl', '_yhat.pkl')


    #LOADING THE DATA
    with open(test_file) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    evaluation_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=batch_size,
        padding='PADDING')

    with open(model_file) as file:
        trained_model = pickle.load(file)


    # EVALUATING
    time_start = time.time()

    progress = []

    X_valid, full_Y_valid, meta_valid = evaluation_data_provider.next_batch()
    full_Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
    assert np.all(np.abs(full_Y_hat.sum(axis=1) - 1) < 1e-6)

    for batch_index in xrange(0, evaluation_data_provider.batches_per_epoch-2):
        X_valid, Y_valid, meta_valid = evaluation_data_provider.next_batch()
        Y_hat = trained_model.fprop(X_valid, meta=meta_valid)
        assert np.all(np.abs(Y_hat.sum(axis=1) - 1) < 1e-6)

        #Putting it togather
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

            write_data(model_file, output_dir, output_eval, output_predict, full_Y_hat, progress, job_id)

    #FINAL EVALUATION
    acc = np.mean(np.argmax(full_Y_hat, axis=1) == np.argmax(full_Y_valid, axis=1))
    current = dict()
    current['B'] = evaluation_data_provider.batches_per_epoch
    current['A'] = acc
    current['T'] = time.time()-time_start

    progress.append(current)
    write_data(model_file, output_dir, output_eval, output_predict, full_Y_hat, progress, job_id)

def write_data(model_file, output_dir, output_eval, output_predict, full_Y_hat, progress, job_id):
    if output_dir:
        with open(os.path.join(output_dir,output_eval), 'w') as progress_file:
            pickle.dump(progress, progress_file, protocol=-1)

        with open(os.path.join(output_dir,output_predict), 'w') as progress_file:
            pickle.dump(full_Y_hat, progress_file, protocol=-1)
    else:
        if job_id:
            file = model_file.replace('.pkl', '_'+model_file.rsplit('/')[::-1][1]+'.pkl')
        else:
            file = model_file

        with open(str(file).replace('.pkl', '_evaluation.pkl'), 'w') as progress_file:
            pickle.dump(progress, progress_file, protocol=-1)
            progress_file.close()

        with open(str(file).replace('.pkl', '_yhat.pkl'), 'w') as progress_file:
            pickle.dump(full_Y_hat, progress_file, protocol=-1)
            progress_file.close()
