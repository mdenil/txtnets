__author__ = 'mdenil'

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

# http://help.sentiment140.com/for-students

data_dir = "../data"
sentiment_dir = os.path.join(data_dir, "sentiment140_2")

@ruffus.originate(["trainingandtestdata.zip"])
def download_data(output_file):
    sh.wget("-O", output_file, "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")


@ruffus.split(
    download_data,
    ["sentiment140.test.csv", "sentiment140.train.csv"])
def extract_data(input_file_name, output_file_names):
    sh.unzip("-o", input_file_name)

    # This also updates timestamps.  Ruffus doesn't recognize these files as complete results unles the
    # timestamp is up to date.
    sh.mv("testdata.manual.2009.06.14.csv", "sentiment140.test.csv")
    sh.mv("training.1600000.processed.noemoticon.csv", "sentiment140.train.csv")

    # Re-encode the files as utf8.  They look like utf8 already (e.g. file thinks they're utf8)
    # but they are actually encoded as latin1.  This doesn't make a difference for the test data
    # (the utf8 and latin1 encoded test data are identical files) but the train data has some
    # byte sequences that are invalid utf8 and this makes simplejson really upset.
    for output_file in output_file_names:
        sh.mv(output_file, "temp")
        sh.iconv("-f", "latin1", "-t", "utf8", "temp", _out=output_file)
        sh.rm("temp")


@ruffus.transform(extract_data, ruffus.suffix(".csv"), ".json")
def reformat_data(input_file_name, output_file_name):
    df = pd.io.parsers.read_csv(
        input_file_name,
        names=["polarity", "id", "date", "query", "user", "text"],
        encoding='utf8')

    # drop columns we don't care about
    df = df[["text", "polarity"]]

    # remove neutral class
    df = df[df.polarity != 2]
    assert all((df.polarity == 4) | (df.polarity == 0))

    # re-map polarity to smilies
    df.polarity = df.polarity.apply(lambda x: ':)' if x == 4 else ':(')

    print "Positive proportion:",  np.sum(df.polarity == ':)') / float(len(df.polarity))

    # smash everything to ascii
    df.text = df.text.apply(lambda x: x.encode('ascii', 'ignore').lower())

    with open(output_file_name, 'w') as output_file:
        output_file.write(u"{}\n".format(json.dumps(zip(df.text, df.polarity))))



@ruffus.transform(reformat_data, ruffus.suffix(".json"), ".clean.json")
def clean_data(input_file_name, output_file_name):
    def clean_word(word):
        word = word.lower() #lowercase is not ideal, TODO:
        word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
        word = re.sub(r'(\S)\1+', r'\1\1', word) #normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)
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


@ruffus.transform([reformat_data, clean_data], ruffus.suffix(".json"), ".alphabet.json")
def build_alphabet(input_file_name, output_file_name):
    alphabet = set()
    with open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            alphabet = alphabet.union(text)

    alphabet = list(sorted(alphabet)) + ['START', 'END', 'UNKNOWN', 'PADDING']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(alphabet)))


@ruffus.transform(
    [reformat_data, clean_data],
    ruffus.suffix(".json"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    with open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            tokenizer = WordPunctTokenizer()
            dictionary.update(tokenizer.tokenize(text))

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 3)) + ['PADDING', 'UNKNOWN']
    # dictionary = list(sorted(w for w,c in dictionary.most_common(3000))) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(dictionary)))


@ruffus.transform([build_alphabet, build_word_dictionary], ruffus.suffix(".json"), ".encoding.json")
def encode_alphabet(input_file, output_file):
    alphabet = dict()
    with open(input_file) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    with open(output_file, 'w') as alphabet_dictionary:
        alphabet_dictionary.write(json.dumps(alphabet))

if __name__ == "__main__":
    if not os.path.exists(sentiment_dir):
        os.makedirs(sentiment_dir)

    sh.cd(sentiment_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)
