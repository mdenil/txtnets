__author__ = 'mdenil'

import re
import os
import sh
import ruffus
import psutil
import subprocess
import codecs
import simplejson as json
import string
from collections import Counter

# This script requires you have a perl interpreter in your path.

# http://www.statmt.org/europarl/

data_dir = "../data"
europarl_dir = os.path.join(data_dir, "europarlv7")

@ruffus.originate(["tools/*"])
def download_tools(output_file_name):
    sh.wget("http://www.statmt.org/europarl/v7/tools.tgz")
    sh.tar("xvf", "tools.tgz")

@ruffus.originate([
    "de-en.tgz", "fr-en.tgz",
    ])
def download_data(output_file_name):
    url = "http://www.statmt.org/europarl/v7/{}".format(output_file_name)
    sh.wget("-O", output_file_name, url)


@ruffus.split(
    download_data,
    ruffus.regex("(.*)-(.*)\.tgz$"),
    [r"europarl-v7.\1-\2.\1", r"europarl-v7.\1-\2.\2"])
def extract_language_pair(input_file_name, output_file_names):
    sh.tar("xvf", input_file_name)
    for output_file_name in output_file_names:
        sh.touch(output_file_name)


@ruffus.transform(
    [extract_language_pair, download_tools],
    ruffus.regex(r"^(europarl-v7\...-..).(..)"),
    r"\1.\2.tokens")
def tokenize_sentence_file(input_file_name, output_file_name):
    lang = input_file_name[-2:]
    subprocess.call("perl tools/tokenizer.perl -l {lang} < {input} > {output}".format(
        lang=lang,
        input=input_file_name,
        output=output_file_name),
        shell=True)


@ruffus.transform(tokenize_sentence_file, ruffus.suffix(".tokens"), ".tokens.clean.json")
def clean_tokens(input_file_name, output_file_name):
    def clean_word(word):
        word = word.lower()
        word = re.sub(r'(\S)\1+', r'\1\1', word) # normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)
        ws = set(word)
        if any(d in ws for d in string.digits):
            word = 'NUMBER'
        return word

    data = []
    with open(input_file_name) as input_file:
        for text in input_file:
            tokens = map(clean_word, text.strip().split(" "))
            data.append(tokens)

    with codecs.open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file)


@ruffus.transform(
    [clean_tokens],
    ruffus.suffix(".json"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    with open(input_file_name) as input_file:
        for tokens in json.load(input_file):
            dictionary.update(tokens)

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 10)) + ['PADDING', 'UNKNOWN']

    with codecs.open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(dictionary, output_file)
        output_file.write(u"\n")


@ruffus.transform([build_word_dictionary], ruffus.suffix(".json"), ".encoding.json")
def encode_dictionary(input_file, output_file):
    alphabet = dict()
    with open(input_file) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    with open(output_file, 'w') as alphabet_dictionary:
        alphabet_dictionary.write(json.dumps(alphabet))


if __name__ == "__main__":
    if not os.path.exists(europarl_dir):
        os.makedirs(europarl_dir)

    sh.cd(europarl_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)