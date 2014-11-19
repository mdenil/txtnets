from __future__ import print_function

__author__ = 'mdenil'

import numpy as np
import random
import os
import json
import pyprind
import cPickle as pickle
import argparse

from cpu.optimize.data_provider import LabelledDocumentMinibatchProvider
from cpu.model.cost import CrossEntropy

import cpu.space


def max_error_label(y_hat):
    y_error_idx = np.argmin(y_hat, axis=1)
    y_error = np.equal.outer(y_error_idx, np.arange(y_hat.shape[1]))
    return y_error.astype(np.float32)

def combiner_matrix(lengths):
    C = np.zeros((sum(lengths), len(lengths)))
    cum_lengths = np.cumsum([0] + lengths)
    for i, s, e in zip(range(len(cum_lengths)), cum_lengths, cum_lengths[1:]):
        C[s:e,i] = 1
    return C

def get_sentence_importance_scores(embedding_model, logistic_regression, x):
    objective = CrossEntropy()

    x_combined = [w for s in x for w in s]

    meta_combined = {
        'lengths': np.asarray([len(x_combined)]),
        'space_below': cpu.space.CPUSpace(
            axes=('b', 'w'),
            extents={'b': 1, 'w': len(x_combined)})
        }

    x_combined = np.asarray(x_combined).reshape((1, -1))

    embeddings, embeddings_meta, embeddings_state = embedding_model.fprop(
        x_combined, meta=dict(meta_combined), return_state=True)
    embeddings_meta['space_below'] = embeddings_meta['space_above']
    y_hat, y_hat_meta, log_reg_state = logistic_regression.fprop(
        embeddings, meta=dict(embeddings_meta), return_state=True)
    y_hat_meta['space_below'] = y_hat_meta['space_above']

    loss, loss_meta, loss_state = objective.fprop(
        y_hat, max_error_label(y_hat), meta=dict(y_hat_meta))

    delta, delta_meta = objective.bprop(
        y_hat, max_error_label(y_hat), meta=dict(loss_meta), fprop_state=loss_state)

    delta = logistic_regression.bprop(
        delta, meta=dict(delta_meta), fprop_state=log_reg_state)

    C = combiner_matrix(map(len, x))

    sentence_delta = np.dot(delta, C)
    sentence_embedding = np.dot(embeddings, C)

    # normalize for cosine distance
    sentence_delta /= np.sqrt(np.sum(sentence_delta**2, axis=1, keepdims=True))
    sentence_embedding /= np.sqrt(np.sum(sentence_embedding**2, axis=1, keepdims=True))

    sentence_importance_scores = np.abs(np.sum(sentence_delta * sentence_embedding, axis=0))

    return sentence_importance_scores


def main():
    random.seed(665243)
    np.random.seed(61734)
    np.set_printoptions(linewidth=100)

    parser = argparse.ArgumentParser(description="Create summaries from w2vec model.")
    parser.add_argument('--size', type=int, help="number of sentences to keep")
    args = parser.parse_args()

    data_dir = os.path.join("/users/mdenil/code/txtnets/txtnets_deployed/data", "stanfordmovie")


    with open("model_w2vec_logreg.pkl") as model_file:
        embedding_model = pickle.load(model_file)
        logistic_regression = pickle.load(model_file)


    with open(os.path.join(data_dir, "stanfordmovie.test.sentences.clean.projected.json")) as data_file:
        data = json.load(data_file)

    # random.shuffle(data)
    X, Y = map(list, zip(*data))
    Y = [[":)", ":("].index(y) for y in Y]

    objective = CrossEntropy()

    test_data_provider = LabelledDocumentMinibatchProvider(
        X=X,
        Y=Y,
        batch_size=1,
        padding=None,
        shuffle=False)

    prog_bar = pyprind.ProgBar(test_data_provider.batches_per_epoch)

    summaries = []

    for _ in range(test_data_provider.batches_per_epoch):
        x_batch, y_batch, meta_batch = test_data_provider.next_batch()
        label = [":)", ":("][int(y_batch[0,1])]

        sentence_importance_scores = get_sentence_importance_scores(
            embedding_model, logistic_regression, x_batch)

        most_important_sentence_indexes = np.argsort(sentence_importance_scores)
        most_important_sentence_indexes = most_important_sentence_indexes[:args.size]
        most_important_sentence_indexes.sort()

        summary = []
        for i in most_important_sentence_indexes:
            summary.append(x_batch[i])

        summaries.append([summary, label])

        prog_bar.update()


    with open("summaries_{}.json".format(args.size), 'w') as summaries_file:
        json.dump(summaries, summaries_file)
        summaries_file.write("\n")

if __name__ == "__main__":
    main()
