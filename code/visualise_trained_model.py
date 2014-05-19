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


data_dir = "../data/sentiment140"

def run():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained network on the sentiment140 test set")
    parser.add_argument("--model_file", help="pickle file to load the model from")
    args = parser.parse_args()

    with open(args.model_file) as model_file:
        model = pickle.load(model_file)

    print model

    test_modified = "sentiment140.test.clean2.json"

    with open(os.path.join(data_dir, test_modified)) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y_original = Y
        Y = [[":)", ":("].index(y) for y in Y]

    #added shuffling
    # combined = zip(X, Y)
    # random.shuffle(combined)
    # X, Y = map(list, zip(*combined))
    #end


    # X=X[:50]
    # Y=Y[:50]

    with open(os.path.join(data_dir, "sentiment140.train.clean.dictionary.encoding.json")) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    tokenizer = WordPunctTokenizer()
    new_X = []
    for x in X:
        new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    X = new_X


    data_provider = cpu.optimize.data_provider.LabelledSequenceBatchProvider(
        X=X, Y=Y, padding='PADDING')


    #HERE IT IS
    #Redifine how to evaluate and regularize
    cEntr = CrossEntropy()

    #get next batch
    X, Y, meta = data_provider.next_batch()
    X = np.vstack([np.atleast_2d(x) for x in X])

    Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)

    Y_inverted = np.ones_like(Y_hat)
    for i in xrange(len(Y_hat)):
        for j in xrange(len(Y_hat[1])):
            if Y_hat[i][j]>0.5:
                Y_inverted[i][j]=0

    #print Y_hat[5:]
    #print Y_inverted[5:]


    meta['space_below'] = meta['space_above']
    cost, meta, cost_state = cEntr.fprop(Y_hat, Y_inverted, meta=meta)

    delta, meta = cEntr.bprop(Y_hat, Y_inverted, meta=meta, fprop_state=cost_state)
    delta, meta = model.bprop(delta, meta=meta, fprop_state=model_state, return_state=True, num_layers=-1)

    delta, space = meta['space_below'].transform(delta, ('b', 'w'))


    #trying to pick the ones where the actuar is :(
    # combined = zip(X, Y, Y_hat, Y_inverted)
    # new_combined = []
    # for tuple in combined:
    #     x, y, z, e = tuple
    #     if z[0]<0.5:
    #         new_combined.append(tuple)
    # X, Y, Y_hat, Y_inverted = map(list, zip(*new_combined))
    #end


    abs_delta = np.absolute(delta)
    abs_delta = np.sqrt(abs_delta)
    abs_delta = np.sqrt(abs_delta)

    #outputfile
    output_file=open('./model_0padding.html', 'w+')

    #ranking and proceeding

    print >> output_file, '<div align="center">'
    for i in xrange(len(X)):
        min = np.min(abs_delta[i])
        max = np.max(abs_delta[i])
        windows_size = (max-min)/5 #defining 5 levels of importance
        for j in xrange(len(X[i])):
            # if X[i][j]=='PADDING':
            #    break
            # try: X[i][j] = str(X[i][j])
            # except: continue

            if abs_delta[i][j] > min+(windows_size*4):
                if Y_inverted[i][0]==1:
                    delta[i][j]=-delta[i][j]
                if delta[i][j]>0:
                    print >> output_file, '<span style="background-color: #21610B">'+str(X[i][j])+'</span>'
                else:
                    print >> output_file, '<span style="background-color: #B40404">'+str(X[i][j])+'</span>'

            elif abs_delta[i][j] > min+(windows_size*3):
                if Y_inverted[i][0]==1:
                    delta[i][j]=-delta[i][j]
                if delta[i][j]>0:
                    print >> output_file, '<span style="background-color: #31B404">'+str(X[i][j])+'</span>'
                else:
                    print >> output_file, '<span style="background-color: #FE2E2E">'+str(X[i][j])+'</span>'

            elif abs_delta[i][j] > min+(windows_size*2):
                if Y_inverted[i][0]==1:
                    delta[i][j]=-delta[i][j]
                if delta[i][j]>0:
                    print >> output_file, '<span style="background-color: #00FF00">'+str(X[i][j])+'</span>'
                else:
                    print >> output_file, '<span style="background-color: #FA5858">'+str(X[i][j])+'</span>'

            elif abs_delta[i][j] > min+(windows_size*1):
                if Y_inverted[i][0]==1:
                    delta[i][j]=-delta[i][j]
                if delta[i][j]>0:
                    print >> output_file, '<span style="background-color: #81F781">'+str(X[i][j])+'</span>'
                else:
                    print >> output_file, '<span style="background-color: #F78181">'+str(X[i][j])+'</span>'

            elif abs_delta[i][j] >= min:
                if Y_inverted[i][0]==1:
                    delta[i][j]=-delta[i][j]
                if delta[i][j]>0:
                    print >> output_file, '<span style="background-color: White">'+str(X[i][j])+'</span>'
                else:
                    print >> output_file, '<span style="background-color: White">'+str(X[i][j])+'</span>'
        smiley = '&#128522'
        smileyE = '&#128522'
        if Y[i][1]==1:
            smiley = '&#128542'
        if Y_hat[i][1]>0.5:
            smileyE = '&#128542'
        print >> output_file, 'THE SUM:' + str(sum(delta[i]))
        print >> output_file, '<b>         (EXP:' +smileyE+'/'+smiley+':ACT'+')</b>'
        print >> output_file, '</br>'
        print >> output_file, '</br>'
    print >> output_file, '</div>'
    #DONE

if __name__ == "__main__":
    run()