__author__ = 'albandemiraj'

import numpy as np
import pandas as pd
import os
import sh
import re
import ruffus
import psutil
import simplejson as json
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
import random

# http://help.sentiment140.com/for-students

data_dir = "../data"
stanfordmovie_dir = os.path.join(data_dir, "stanfordmovie")

@ruffus.originate(["aclImdb_v1.tar.gz"])
def download_data(output_file):
    sh.wget("-O", output_file, "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")


@ruffus.split(download_data, ["aclImdb"])
def extract_data(input_file_name, output_file_name):
    sh.tar("-xzvf", input_file_name)


@ruffus.split(
    extract_data,
    ["stanfordmovie.train.json", "stanfordmovie.test.json", "stanfordmovie.unsup.json"])
def create_jsons(input_file_name, output_file_names):
    #TODO: 1. Make filename dynamic
    #---------------------
    #WORKING ON TRAIN FILE
    pos_dir = os.path.join('aclImdb', "train", "pos")
    neg_dir = os.path.join('aclImdb', "train", "neg")

    reviews=[]

    #Work on positive reviews and add them to reviews list
    for file in os.listdir(pos_dir):
        if(file=='.DS_Store'): continue
        with open(os.path.join(pos_dir,file)) as f:
            for line in f:
                reviews.append([line, ':)'])

    #Work on negative reviews and add them to reviews list
    for file in os.listdir(neg_dir):
        if(file=='.DS_Store'): continue
        with open(os.path.join(neg_dir,file)) as f:
            for line in f:
                reviews.append([line, ':('])

    random.shuffle(reviews)

    with open("stanfordmovie.train.json", 'w') as file:
        file.write("{}\n".format(json.dumps(reviews)))

    #------------------------
    #WORKING ON THE TEST FILE
    pos_dir = os.path.join('aclImdb', "test", "pos")
    neg_dir = os.path.join('aclImdb', "test", "neg")

    reviews=[]

    #Work on positive reviews and add them to reviews list
    for file in os.listdir(pos_dir):
        if(file=='.DS_Store'): continue
        with open(os.path.join(pos_dir,file)) as f:
            for line in f:
                reviews.append([line, ':)'])

    #Work on negative reviews and add them to reviews list
    for file in os.listdir(neg_dir):
        if(file=='.DS_Store'): continue
        with open(os.path.join(neg_dir,file)) as f:
            for line in f:
                reviews.append([line, ':('])

    random.shuffle(reviews)

    with open("stanfordmovie.test.json", 'w') as file:
        file.write("{}\n".format(json.dumps(reviews)))

    #--------------------------------
    #WORKING ON THE UNSUPERVISED FILE
    unsup_dir = os.path.join('aclImdb', "train", "unsup")

    reviews=[]

    #Work on positive reviews and add them to reviews list
    for file in os.listdir(unsup_dir):
        if(file=='.DS_Store'): continue
        with open(os.path.join(unsup_dir,file)) as f:
            for line in f:
                reviews.append([line, ':|'])

    random.shuffle(reviews)

    with open("stanfordmovie.unsup.json", 'w') as file:
        file.write("{}\n".format(json.dumps(reviews)))


@ruffus.transform(create_jsons, ruffus.suffix(".json"), ".clean.json")
def clean_data(input_file_name, output_file_name):
    #TODO: 1. Cleaning procedure may still need improvement with some html tags
    def clean_word(word):
        word = word.lower() #lowercase is not ideal, TODO:
        word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
        word = re.sub(r'(\S)\1+', r'\1\1', word) #normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)

        word = word.replace("n't", " nt") #<===MAYBE TAKE THIS OFF
        word = word.replace('"', '')
        word = word.replace('<br', '').replace('/>','')
        word = word.replace('(', '')
        word = word.replace(')', '')
        word = word.replace('[', '')
        word = word.replace(']', '')
        word = word.replace('.', ' .')
        word = word.replace(',', ' ,')
        word = word.replace("'", "")

        #if word.startswith('@'): # this misses "@dudebro: (quote included)
        if re.match(r'[^A-Za-z0-9]*@', word):
            #word = 'GENERICUSER' #all other words are lowercase
            word = 'U'
        elif word.startswith('#'):
            #word = 'GENERICHASHTAG'
            word = 'H'
        elif re.search('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
            #word = 'GENERICHTTP'
            word = 'L'
        return word

    data = []
    with open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            text = " ".join(map(clean_word, text.split()))
            data.append([text, label])

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(data)))


@ruffus.transform(
    clean_data,
    ruffus.suffix(".json"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    with open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            tokenizer = WordPunctTokenizer()
            dictionary.update(tokenizer.tokenize(text))

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 5)) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(dictionary)))

@ruffus.merge(build_word_dictionary, "stanfordmovie.full.clean.dicionary.json")
def combine_dictionary(input_file_names, output_file_name):
    combined = []
    for input_file_name in input_file_names:
        with open(input_file_name) as input_file:
            print 'HERE: '+str(input_file)
            data = json.loads(input_file.read())
            combined += data

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(combined)))

@ruffus.transform([build_word_dictionary, combine_dictionary], ruffus.suffix(".json"), ".encoding.json")
def encode_alphabet(input_file, output_file):
    alphabet = dict()
    with open(input_file) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    with open(output_file, 'w') as alphabet_dictionary:
        alphabet_dictionary.write(json.dumps(alphabet))


if __name__ == "__main__":
    if not os.path.exists(stanfordmovie_dir):
        os.makedirs(stanfordmovie_dir)

    sh.cd(stanfordmovie_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)
