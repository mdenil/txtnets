from __future__ import print_function
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
# from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def load_data(file_name):
    with open(file_name) as data_file:
        raw_data = json.load(data_file)
        train_x, train_y = map(list, zip(*raw_data))

        data = []
        for sentences, label in zip(train_x, train_y):
            words = [w for s in sentences for w in s]
            data.append((FreqDist(words), label))

    return data


def load_train_data():
    return load_data("/users/mdenil/code/txtnets/txtnets_deployed/data/stanfordmovie/stanfordmovie.train.sentences.clean.json")

def load_test_data():
    return load_data("/users/mdenil/code/txtnets/txtnets_deployed/data/stanfordmovie/stanfordmovie.test.sentences.clean.json")

def load_unsup_data():
    return load_data("/users/mdenil/code/txtnets/txtnets_deployed/data/stanfordmovie/stanfordmovie.unsup.sentences.clean.json")


def get_naive_bayes():
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('nb', MultinomialNB()),
    ])
    return SklearnClassifier(pipeline)

`
def run():
    train_data = load_train_data()
    test_data = load_test_data()

    ys = np.asarray([y for x,y in test_data])


    naive_bayes = get_naive_bayes()
    naive_bayes.train(train_data)


    y_hats = np.asarray(naive_bayes.classify_many([x for x,y in test_data]))
    print("At 100%, acc:", np.mean(ys == y_hats))

    for k in [5, 4, 3, 2]:
        summary_data = load_data("summaries_{}.json".format(k))
        y_hats = np.asarray(naive_bayes.classify_many([x for x,y in summary_data]))
        print("Pick {}:".format(k), np.mean(ys == y_hats))


if __name__ == "__main__":
    run()