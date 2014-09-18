from __future__ import division

__author__ = 'mdenil'

import numpy as np
import os
import simplejson as json
import cPickle as pickle
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def load_data(file_name):
    data_dir = os.path.join("../data", "stanfordmovie")

    with open(os.path.join(data_dir, file_name)) as data_file:
        raw_data = json.load(data_file)
        train_x, train_y = map(list, zip(*raw_data))
        # train_x, train_y = map(list, zip(*raw_data[:100]))

        data = []
        for sentences, label in zip(train_x, train_y):
            words = [w for s in sentences for w in s]
            data.append((FreqDist(words), label))

    return data


def load_train_data():
    return load_data("stanfordmovie.train.sentences.clean.json")


def load_test_data():
    return load_data("stanfordmovie.test.sentences.clean.json")


def load_unsup_data():
    return load_data("stanfordmovie.unsup.sentences.clean.json")


def get_linear_svc():
    # 0.87632
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2, k=1000)),
        ('svm', LinearSVC()),
    ])
    return SklearnClassifier(pipeline)


# def get_random_forest():
#     pipeline = Pipeline([
#         ('rf', RandomForestClassifier(n_estimators=10)),
#     ])
#     return SklearnClassifier(pipeline)


def run():
    classifiers = [
        get_linear_svc(),
        # get_random_forest(),
        ]

    train_data = load_train_data()
    unsup_data = load_unsup_data()

    y_hats = []
    for classifier in classifiers:
        classifier.train(train_data)
        y_hat = classifier.batch_classify([x for x,y in unsup_data])
        y_hats.append(y_hat)


    data_dir = os.path.join("../data", "stanfordmovie")
    with open(os.path.join(data_dir, "stanfordmovie.unsup.sentences.clean.projected.json")) as data_file:
        projected_unsup_data = json.load(data_file)

    unsup_x, unsup_y = zip(*projected_unsup_data)
    assert len(unsup_x) == len(y_hats[0])
    projected_unsup_data = zip(unsup_x, y_hats[0])

    with open('stanfordmovie.unsup.sentences.clean.projected.labelled.json', 'w') as unsup_file:
        json.dump(projected_unsup_data, unsup_file)


if __name__ == "__main__":
    run()