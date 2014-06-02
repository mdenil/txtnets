__author__ = 'albandemiraj'




__author__ = 'albandemiraj'

import numpy as np
import os
import argparse
import simplejson as json
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer
from cpu.model.cost import CrossEntropy

import cpu.optimize.data_provider
import random
from generic.optimize.data_provider import LabelledDocumentMinibatchProvider
import operator
import io
from collections import defaultdict
import cpu.space
from cpu.model.dropout import remove_dropout
import itertools

data_dir = "../data"
visual_dir = "../visualisations"
model_dir = "../models"

def shuffle(x, y=None):
    if y:
        combined = zip(x,y)
        random.shuffle(combined)
        x, y = map(list, zip(*combined))
        return x, y
    else:
        random.shuffle(x)
        return x

def run():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained network on the sentiment140 test set")
    parser.add_argument("--model_file", help="pickle file to load the model from")
    parser.add_argument("--output_file", help="html file to write the output to")
    parser.add_argument("--dataset", help="name of the dataset to visualise")
    args = parser.parse_args()

    #Preparing filenames
    model_file = args.model_file
    test_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.sentences.clean.projected.json')
    vocab_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.sentences.clean.dictionary.encoding.json')
    output_file = model_file+'.summary.html'

    #Opening model file
    with open(model_file) as model_file:
        model = pickle.load(model_file)
    model = remove_dropout(model)

    print model
    print len(model.layers)

    #Opening data file to visualise
    with open(test_file) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y_original = Y
        Y = [[":)", ":("].index(y) for y in Y]

    #Loading vocabulary
    with open(vocab_file) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())


    data_provider = LabelledDocumentMinibatchProvider(
        X=X, Y=Y, padding='PADDING', batch_size=8)

    output_file=open(output_file, 'w+')

    for _ in xrange(20):
        X, Y, meta = data_provider.next_batch()
        Y_hat, Y_inverted, delta, meta = get_model_output(model, X, Y, meta)
        print_visualisation(X, meta, Y_hat, Y_inverted, delta, output_file)

def get_model_output(model, X, Y, meta):
    #Initializing the data provided

    #Define the cost function
    cEntr = CrossEntropy()

    Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)

    #Create a Y that maximizes the error of the model
    Y_inverted = enforce_error(Y_hat)

    #FINDING THE INDEX WHERE TO STOP BACKPROP
    break_index = 0
    for index, layer in zip(itertools.count(), model.layers):
        if layer.__class__.__name__=='ReshapeForDocuments':
            break_index=index
            break


    #Bookkeep the spaces and BPROP to get the deltas
    meta['space_below'] = meta['space_above']
    cost, meta, cost_state = cEntr.fprop(Y_hat, Y_inverted, meta=meta)
    delta, meta = cEntr.bprop(Y_hat, Y_inverted, meta=meta, fprop_state=cost_state)
    delta, meta = model.bprop(delta, meta=meta, fprop_state=model_state, return_state=True, num_layers=break_index)

    print np.shape(delta)
    print np.shape(np.array(X))

    #RESHAPING DELTA
    delta, space = meta['space_below'].transform(delta, ('b', ('d','f','w',)))
    delta = delta.sum(axis=space.axes.index(('d','f','w')))
    space = space.without_axes(('d','f','w'))
    delta, space = space.transform(delta, (('b','s'),'c'))
    s_extent = meta['padded_sentence_length']
    space = space.with_extents(**{'s':s_extent, 'b':space.get_extent('b')//s_extent})
    delta, space = space.transform(delta, ('b','s'))

    return Y_hat, Y_inverted, delta, meta

def enforce_error(Y_hat):
    Y_inverted = np.ones_like(Y_hat)
    for i in xrange(len(Y_hat)):
        for j in xrange(len(Y_hat[1])):
            if Y_hat[i][j]>0.5:
                Y_inverted[i][j]=0
    return Y_inverted

def print_visualisation(X, meta, Y_hat, Y_inverted, delta, output_file):
    #REFACTORING DELTA
    original_delta=delta
    delta = np.absolute(delta)
    delta = np.sqrt(delta)

    assert (len(np.ravel(delta))==len(X))
    #Colour = 00FF00
    X = np.array(X)
    X_space = cpu.space.CPUSpace.infer(X, ('b','w'))
    X_space = X_space.with_axes('s')
    s_extent = meta['padded_sentence_length']
    X_space = X_space.with_extents(**{'s':s_extent, 'b':X_space.get_extent('b')//s_extent})
    X_space = X_space.transposed((('b','s'),'w'))
    X, X_space = X_space.transform(X, ('b','s','w'))

    #COMPLETE SPLIT
    output_file.write('<div align="center">')
    output_file.write('<table with="100%" border="1">')
    output_file.write('<tr>')
    output_file.write('<td width="50%">')
    output_file.write('ORIGINAL REVIEW')
    output_file.write('</td>')
    output_file.write('<td width="50%">')
    output_file.write('SUMMARIZED REVIEW')
    output_file.write('</td>')
    output_file.write('</tr>')

    for i in xrange(len(X)):
        output_file.write('<tr>')

        #FULL REVIEW
        important_sentence = (np.argsort(delta[i])[::-1])[:max(1, int(len(X[i])/5))]
        print str(len(important_sentence))+"/"+str(max(1, int(len(X[i])/5)))

        output_file.write('<td>')
        for j in xrange(len(X[i])):
            if j in important_sentence:
                if (original_delta[i][j]>0 and Y_hat[i][0]>0.5) or (original_delta[i][j]<0 and Y_hat[i][0]<0.5):
                    output_file.write('<span style="background-color: #00FF00">')
                else:
                    output_file.write('<span style="background-color: #C80000">')
            for k in xrange(len(X[i][j])):
                if unicode(X[i][j][k]) == u'PADDING':
                    continue
                try:
                    output_file.write(str(X[i][j][k]))
                except:
                    continue
                output_file.write(' ')
            if j in important_sentence:
                output_file.write('</span>')
        output_file.write('</td>')


        #SUMMARIZED REVIEW
        output_file.write('<td>')
        important_sentence = (np.argsort(delta[i])[::-1])[:max(1, (len(X[i])/5))]
        for j in xrange(len(X[i])):
            if j in important_sentence:
                for k in xrange(len(X[i][j])):
                    if unicode(X[i][j][k]) == u'PADDING':
                        continue
                    try:
                        output_file.write(str(X[i][j][k]))
                    except:
                        continue
                    output_file.write(' ')

        output_file.write('</td>')



        output_file.write('</tr>')
    output_file.write('</div>')
    # DONE

if __name__ == "__main__":
    run()