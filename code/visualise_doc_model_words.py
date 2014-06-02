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
import operator
from collections import defaultdict
from cpu.model.dropout import remove_dropout

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
    parser = argparse.ArgumentParser(description="Evaluate a trained network on the sentiment140 test set")
    parser.add_argument("--model_file", help="pickle file to load the model from")
    parser.add_argument("--output_file", help="html file to write the output to")
    parser.add_argument("--dataset", default='', help="name of the dataset to visualise")
    args = parser.parse_args()

    #Preparing filenames
    model_file = args.model_file
    test_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.sentences.clean.projected.json')
    vocab_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.sentences.clean.dictionary.encoding.json')
    output_file = model_file+'.html'

    #Opening model file
    with open(model_file) as model_file:
        model = pickle.load(model_file)
        model = remove_dropout(model)

    print model

    #Opening data file to visualise
    with open(test_file) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y = [[":)", ":("].index(y) for y in Y]

    data_provider = cpu.optimize.data_provider.LabelledDocumentMinibatchProvider(
        X=X, Y=Y, batch_size=10, padding='PADDING')

    output_file=open(output_file, 'w+')
    for i in xrange (0,200):
        X_, Y_, meta = data_provider.next_batch()
        Y_hat, Y_inverted, delta, meta = get_model_output(model, X_, Y_, meta)
        print_visualisation(X_, Y_hat, Y_inverted, delta, output_file, meta)

def get_model_output(model, X, Y, meta):
    #Initializing the data provided

    #Define the cost function
    cEntr = CrossEntropy()


    Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)

    #Create a Y that maximizes the error of the model
    Y_inverted = enforce_error(Y_hat)

    #Bookkeep the spaces and BPROP to get the deltas
    meta['space_below'] = meta['space_above']
    cost, meta, cost_state = cEntr.fprop(Y_hat, Y_inverted, meta=meta)
    delta, meta = cEntr.bprop(Y_hat, Y_inverted, meta=meta, fprop_state=cost_state)
    delta, meta = model.bprop(delta, meta=meta, fprop_state=model_state, return_state=True, num_layers=-1)
    delta, space = meta['space_below'].transform(delta, ('b', 'w'))
    s_extent = meta['padded_sentence_length']
    space = space.with_axes('s')
    space = space.transposed((('b','s'),'w'))
    space = space.with_extents(**{'s':s_extent, 'b':space.get_extent('b')/s_extent})
    delta, space = space.transform(delta, ('b','s','w'))

    return Y_hat, Y_inverted, delta, meta

def enforce_error(Y_hat):
    Y_inverted = np.ones_like(Y_hat)
    for i in xrange(len(Y_hat)):
        for j in xrange(len(Y_hat[1])):
            if Y_hat[i][j]>0.5:
                Y_inverted[i][j]=0
    return Y_inverted

def print_visualisation(X, Y_hat, Y_inverted, delta, output_file, meta):
    X = np.array(X)
    print 'HERE: '+str(np.shape(X))
    X_space = cpu.space.CPUSpace.infer(X, ('b','w'))
    X_space = X_space.with_axes('s')
    s_extent = meta['padded_sentence_length']
    X_space = X_space.with_extents(**{'s':s_extent, 'b':X_space.get_extent('b')//s_extent})
    X_space = X_space.transposed((('b','s'),'w'))
    X, X_space = X_space.transform(X, ('b','s','w'))
    #'--------------------------------------------------------------------------------------'

    abs_delta = np.absolute(delta)
    abs_delta = np.sqrt(abs_delta)

    print >> output_file, '<div align="center">'
    for b in xrange(len(X)):
        for s in xrange(len(X[b])):
            if X[b][s][0]=='PADDING':
                continue
            min = np.min(abs_delta[b][s])
            max = np.max(abs_delta[b][s])
            for w in xrange(len(X[b][s])):
                windows_size = (max-min)/5 #defining 5 levels of importance
                # Don't print PADDING
                if X[b][s][w]=='PADDING' or X[b][s][w]=='GENERIC_SYMBOL':
                   continue
                if abs_delta[b][s][w] > min+(windows_size*4):
                    if Y_inverted[b][0]==1:
                        delta[b][s][w]=-delta[b][s][w]
                    if delta[b][s][w]>0:
                        print >> output_file, '<span style="background-color: #21610B">'+str(X[b][s][w])+'</span>'
                    else:
                        print >> output_file, '<span style="background-color: #B40404">'+str(X[b][s][w])+'</span>'

                elif abs_delta[b][s][w] > min+(windows_size*3):
                    if Y_inverted[b][0]==1:
                        delta[b][s][w]=-delta[b][s][w]
                    if delta[b][s][w]>0:
                        print >> output_file, '<span style="background-color: #31B404">'+str(X[b][s][w])+'</span>'
                    else:
                        print >> output_file, '<span style="background-color: #FE2E2E">'+str(X[b][s][w])+'</span>'

                elif abs_delta[b][s][w] > min+(windows_size*2):
                    if Y_inverted[b][0]==1:
                        delta[b][s][w]=-delta[b][s][w]
                    if delta[b][s][w]>0:
                        print >> output_file, '<span style="background-color: #00FF00">'+str(X[b][s][w])+'</span>'
                    else:
                        print >> output_file, '<span style="background-color: #FA5858">'+str(X[b][s][w])+'</span>'

                elif abs_delta[b][s][w] > min+(windows_size*2):
                    if Y_inverted[b][0]==1:
                        delta[b][s][w]=-delta[b][s][w]
                    if delta[b][s][w]>0:
                        print >> output_file, '<span style="background-color: #81F781">'+str(X[b][s][w])+'</span>'
                    else:
                        print >> output_file, '<span style="background-color: #F78181">'+str(X[b][s][w])+'</span>'

                elif abs_delta[b][s][w] >= min:
                    if Y_inverted[b][0]==1:
                        delta[b][s][w]=-delta[b][s][w]
                    if delta[b][s][w]>0:
                        print >> output_file, '<span style="background-color: White">'+str(X[b][s][w])+'</span>'
                    else:
                        print >> output_file, '<span style="background-color: White">'+str(X[b][s][w])+'</span>'
        smiley = '&#128522'
        smileyE = '&#128522'
        if Y_hat[b][1]<=0.5:
            smiley = '&#128542'
        if Y_hat[b][1]>0.5:
            smileyE = '&#128542'
        print >> output_file, '<b>         (EXP:' +smileyE+'/'+smiley+':ACT'+')</b>'
        print >> output_file, '</br>'
        print >> output_file, '</br>'
        print >> output_file, '------------------------------------------------------------------'
        print >> output_file, '</br>'
    print >> output_file, '</div>'

    #DONE

def extract_lexicon(X, Y_hat, Y_inverted, delta):
    positive= defaultdict(lambda: 0)
    negative = defaultdict(lambda: 0)

    abs_delta = np.absolute(delta)
    abs_delta = np.sqrt(abs_delta)


    for i in xrange(len(X)):
        min = np.min(abs_delta[i])
        max = np.max(abs_delta[i])
        windows_size = (max-min)/5 #defining 5 levels of importance

        for j in xrange(len(X[i])):
            if len(X[i][j])<3:
                break

            # #The ones that make it to the top windows
            # if abs_delta[i][j] > min+(windows_size*3):
            #     if Y_inverted[i][0]==1:
            #         delta[i][j]=-delta[i][j]
            #     if delta[i][j]>0:
            #         positive[X[i][j]]+=1
            #     else:
            #         negative[X[i][j]]+=1



            #The highest avarage
            if Y_inverted[i][0]==1:
                delta[i][j]=-delta[i][j]
            if delta[i][j]>0:
                if positive.get(X[i][j])!=None:
                    (val, count)=positive[X[i][j]]
                    val = val+delta[i][j]
                    count += 1
                    positive[X[i][j]]=(val,count)
                else:
                    positive[X[i][j]] = (delta[i][j],1)

            else:
                if negative.get(X[i][j])!=None:
                    (val, count)=negative[X[i][j]]
                    val = val+delta[i][j]
                    count += 1
                    negative[X[i][j]]=(val,count)
                else:
                    negative[X[i][j]] = (delta[i][j],1)

    for word in positive:
        (val, count)=positive[word]
        val = val/count
        positive[word]=(val,0)

    for word in negative:
        (val, count)=negative[word]
        val = val/count
        negative[word]=(val,0)

    positive = sorted(positive.items(), key=operator.itemgetter(1), reverse=True)
    negative = sorted(negative.items(), key=operator.itemgetter(1), reverse=True)

    for i in xrange(0,19):
        print positive[i]
    print '-------------------'
    for i in xrange(0,19):
        print negative[i]

if __name__ == "__main__":
    run()