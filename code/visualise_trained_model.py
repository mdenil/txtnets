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
    model_file = os.path.join(model_dir, args.model_file)
    test_file = os.path.join(data_dir, args.dataset, args.dataset + '.test.clean.json')
    vocab_file = os.path.join(data_dir, args.dataset, args.dataset + '.train.clean.dictionary.encoding.json')
    output_file = os.path.join(visual_dir, args.output_file)

    #Opening model file
    with open(model_file) as model_file:
        model = pickle.load(model_file)

    print model

    #Opening data file to visualise
    with open(test_file) as data_file:
        data = json.loads(data_file.read())
        X, Y = map(list, zip(*data))
        Y_original = Y
        Y = [[":)", ":("].index(y) for y in Y]

    #Loading vocabulary
    with open(vocab_file) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    #Tokenizing the data
    tokenizer = WordPunctTokenizer()
    new_X = []
    for x in X:
        new_X.append([w if w in alphabet else 'UNKNOWN' for w in tokenizer.tokenize(x)])
    X = new_X

    Y_hat, Y_inverted, delta = get_model_output(model, X, Y)
    #print_visualisation(X, Y_hat, Y_inverted, delta, output_file)
    extract_lexicon(X, Y_hat, Y_inverted, delta)

def get_model_output(model, X,Y):
    #Initializing the data provided
    data_provider = cpu.optimize.data_provider.LabelledSequenceBatchProvider(
        X=X, Y=Y, padding='PADDING')


    #Define the cost function
    cEntr = CrossEntropy()

    #Get data and use the model to Predict
    X, Y, meta = data_provider.next_batch()
    Y_hat, meta, model_state = model.fprop(X, meta=meta, return_state=True)

    #Create a Y that maximizes the error of the model
    Y_inverted = enforce_error(Y_hat)

    #Bookkeep the spaces and BPROP to get the deltas
    meta['space_below'] = meta['space_above']
    cost, meta, cost_state = cEntr.fprop(Y_hat, Y_inverted, meta=meta)
    delta, meta = cEntr.bprop(Y_hat, Y_inverted, meta=meta, fprop_state=cost_state)
    delta, meta = model.bprop(delta, meta=meta, fprop_state=model_state, return_state=True, num_layers=-1)
    delta, space = meta['space_below'].transform(delta, ('b', 'w'))

    return Y_hat, Y_inverted, delta

def enforce_error(Y_hat):
    Y_inverted = np.ones_like(Y_hat)
    for i in xrange(len(Y_hat)):
        for j in xrange(len(Y_hat[1])):
            if Y_hat[i][j]>0.5:
                Y_inverted[i][j]=0
    return Y_inverted

def print_visualisation(X, Y_hat, Y_inverted, delta, output_file):
    abs_delta = np.absolute(delta)
    abs_delta = np.sqrt(abs_delta)

    output_file=open(output_file, 'w+')

    print >> output_file, '<div align="center">'
    for i in xrange(len(X)):
        min = np.min(abs_delta[i])
        max = np.max(abs_delta[i])
        windows_size = (max-min)/5 #defining 5 levels of importance
        for j in xrange(len(X[i])):
            # Don't print PADDING
            if X[i][j]=='PADDING':
               break
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
        if Y_hat[i][1]<=0.5:
            smiley = '&#128542'
        if Y_hat[i][1]>0.5:
            smileyE = '&#128542'
        print >> output_file, '<b>         (EXP:' +smileyE+'/'+smiley+':ACT'+')</b>'
        print >> output_file, '</br>'
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