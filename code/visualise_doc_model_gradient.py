__author__ = 'albandemiraj'

import numpy as np
import argparse
import simplejson as json
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer
from cpu.model.cost import CrossEntropy
import cpu.optimize.data_provider
from generic.optimize.data_provider import LabelledDocumentMinibatchProvider
from math import ceil
import cpu.space
from cpu.model.dropout import remove_dropout
import itertools
import os
import re
import simplejson as json
import random

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
    original_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.sentences.json')
    output_file = model_file+'.summary_grad.html'

    #Opening model file
    with open(model_file) as model_file:
        model = pickle.load(model_file)
    model = remove_dropout(model)

    print model
    print len(model.layers)

    #OPENING PROJECTED DATA
    with open(test_file) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y_original = Y
        Y = [[":)", ":("].index(y) for y in Y]

    data_provider = LabelledDocumentMinibatchProvider(
        X=X, Y=Y, padding='PADDING', batch_size=8)
    #DONE

    #OPENING ORIGINAL DATA
    tokenizer = WordPunctTokenizer()
    X_ = []
    with open(original_file) as input_file:
        for sentences, label in json.loads(input_file.read()):
            cleaned_sentences = []
            for sentence in sentences:
                cleaned_sentence = " ".join(map(clean_word, sentence.split()))
                cleaned_sentence = tokenizer.tokenize(cleaned_sentence)
                cleaned_sentences.append(cleaned_sentence)

            X_.append([cleaned_sentences])

    X_ =  np.ravel(X_)

    original_text_provider = LabelledDocumentMinibatchProvider(
        X=X_, Y=Y, padding='PADDING', batch_size=8)
    #DONE

    #START HTML FILE STRUCTURE
    output_file=open(output_file, 'w+')
    output_file.write('<div align="center">')
    output_file.write('<table with="100%" border="1">')
    output_file.write('<tr>')
    output_file.write('<td width="4%">')
    output_file.write('INDEX')
    output_file.write('</td>')
    output_file.write('<td width="48%">')
    output_file.write('ORIGINAL REVIEW')
    output_file.write('</td>')
    output_file.write('<td width="48%">')
    output_file.write('SUMMARIZED REVIEW')
    output_file.write('</td>')
    output_file.write('</tr>')
    #DONE INITIALIZING HTML

    for batch_index in xrange(20):
        X, Y, meta = data_provider.next_batch()
        Y_hat, Y_inverted, delta, meta = get_model_output(model, X, Y, meta)

        X_, _, _ = original_text_provider.next_batch()
        print_visualisation(X_, meta, Y_hat, Y_inverted, delta, output_file, batch_index, data_provider.batch_size)

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

    #RESHAPING DELTA
    delta, space = meta['space_below'].transform(delta, ('b', ('d','f','w',)))
    delta, space = space.transform(delta, ('b','f','w'))
    delta, space = space.transform(delta, (('b','w'),'f'))

    #Reshaping input data
    print meta.keys()
    Y_ = meta['Y']
    Y_space_ = meta['Y_space']
    Y_, Y_space_ = Y_space_.transform(Y_, (('b','w'),'f'))

    delta = np.sum(Y_ * delta, axis=1)
    space = space.without_axes('f')
    delta, space = space.transform(delta, ('b','w'))


    #delta = delta.sum(axis=space.axes.index(('d','f','w')))
    #space = space.without_axes(('d','f','w'))
    # delta, space = space.transform(delta, (('b','s'),'c'))
    # s_extent = meta['padded_sentence_length']
    # space = space.with_extents(**{'s':s_extent, 'b':space.get_extent('b')//s_extent})
    # delta, space = space.transform(delta, ('b','s'))

    return Y_hat, Y_inverted, delta, meta

def enforce_error(Y_hat):
    Y_inverted = np.ones_like(Y_hat)
    for i in xrange(len(Y_hat)):
        for j in xrange(len(Y_hat[1])):
            if Y_hat[i][j]>0.5:
                Y_inverted[i][j]=0
    return Y_inverted

def print_visualisation(X, meta, Y_hat, Y_inverted, delta, output_file, batch_index, batch_size):
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
    for i in xrange(len(X)):
        output_file.write('<tr>')

        #INDEX
        output_file.write('<td>')
        output_file.write(str(batch_index*batch_size+i))
        output_file.write('</td>')

        n_sentences = meta['lengths2'][i]

        #FULL REVIEW
        important_sentence = (np.argsort(delta[i][0:meta['lengths2'][i]])[::-1])[:max(1, ceil(n_sentences/4))]
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

def clean_word(word):
    # word = word.lower()
    # word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
    # word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
    # word = re.sub(r'(\S\S)\1+', r'\1\1', word)

    word = word.replace("n't", " nt")  # <===MAYBE TAKE THIS OFF
    # word = word.replace('"', '')
    # word = word.replace('(', '')
    # word = word.replace(')', '')
    # word = word.replace('[', '')
    # word = word.replace(']', '')
    # word = word.replace('.', ' .')
    # word = word.replace(',', ' ,')
    # word = word.replace("'", "")

    word = word.encode('ascii', 'ignore')

    if re.match(r'[^A-Za-z0-9]*@', word):
        word = 'GENERIC_USER'

    if re.search(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
        word = 'GENERIC_HTTP'

    return word.encode('ascii', 'ignore')


if __name__ == "__main__":
    run()