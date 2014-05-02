__author__ = 'mdenil'


import numpy as np
import os
import argparse
import simplejson as json
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer

import cpu.optimize.data_provider


data_dir = "../data/sentiment140"

def run():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained network on the sentiment140 test set")
    parser.add_argument("--model_file", help="pickle file to load the model from")
    args = parser.parse_args()

    with open(args.model_file) as model_file:
        model = pickle.load(model_file)

    print model

    with open(os.path.join(data_dir, "sentiment140.test.clean.json")) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    with open(os.path.join(data_dir, "sentiment140.train.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    tokenizer = WordPunctTokenizer()
    new_X = []
    for x in X:
        new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    X = new_X

    data_provider = cpu.optimize.data_provider.LabelledSequenceBatchProvider(
        X=X, Y=Y, padding='PADDING')

    X, Y, meta = data_provider.next_batch()

    Y_hat = model.fprop(X, meta=meta)

    Y_hat = np.argmax(Y_hat, axis=1)
    Y = np.argmax(Y, axis=1)

    print "Acc: {}".format(np.mean(Y_hat == Y))


if __name__ == "__main__":
    run()