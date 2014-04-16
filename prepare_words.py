__author__ = 'mdenil'

import numpy as np

import sh
import ruffus
import os
import random
import pyprind
import gzip
import simplejson as json


data_dir = os.environ['DATA']
words_dir = os.path.join(data_dir, "words")

# /usr/share/dict/words is a text file full of words on most unix systems

@ruffus.follows(ruffus.mkdir(words_dir))
@ruffus.originate(os.path.join(words_dir, "words.txt"))
def get_words(output_file):
    sh.cp("/usr/share/dict/words", output_file)
    sh.chmod("u+w", output_file)

@ruffus.transform(get_words, ruffus.suffix(".txt"), ".alphabet.json")
def build_alphabet_dictionary(input_file, output_file):
    characters = set()
    with open(input_file) as f:
        for line in f:
            characters = characters.union(line.rstrip())

    alphabet = list(sorted(characters)) + ['PADDING', 'START', 'END']

    with open(output_file, 'w') as f:
        f.write(json.dumps(alphabet))

@ruffus.transform(build_alphabet_dictionary, ruffus.suffix(".alphabet.json"), ".alphabet.encoding.json")
def encode_alphabet_dictionary(input_file, output_file):
    alphabet = dict()
    with open(input_file) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    with open(output_file, 'w') as alphabet_dictionary:
        alphabet_dictionary.write(json.dumps(alphabet))


@ruffus.merge([get_words, encode_alphabet_dictionary], os.path.join(words_dir, "words.encoded.json"))
def encode_word_dictionary(input_files, output_file):
    word_file_name, alphabet_file_name = input_files

    with open(alphabet_file_name) as alphabet_file:
        alphabet = json.loads(alphabet_file.read())

    with open(word_file_name) as word_file:
        words = [word.rstrip() for word in word_file]

    progress_bar = pyprind.ProgPercent(len(words))

    encoded_words = []
    for word in words:
        encoded_word = [alphabet[c] for c in word]
        encoded_word = [alphabet['START']] + encoded_word + [alphabet['END']]

        encoded_words.append(encoded_word)

        progress_bar.update()

    with open(output_file, 'w') as f:
        f.write(json.dumps(encoded_words))

if __name__ == "__main__":
    ruffus.pipeline_run(verbose=5)