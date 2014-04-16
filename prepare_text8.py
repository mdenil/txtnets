__author__ = 'mdenil'

import numpy as np

import sh
import ruffus
import os
import random
import simplejson as json
import pyprind
import gzip

data_dir = os.environ['DATA']
text8_dir = os.path.join(data_dir, "text8")


N_TRAIN_CHAR_FRAGMENTS = 100000
CHAR_FRAGMENTS_CONTEXT_LENGTH = 50

@ruffus.follows(ruffus.mkdir(text8_dir))
@ruffus.originate(os.path.join(text8_dir, "text8.zip"))
def download_text8(output_file):
    sh.wget("-O", output_file, "http://mattmahoney.net/dc/text8.zip")

@ruffus.transform(download_text8, ruffus.suffix(".zip"), ".txt")
def extract_text8(input_file, output_file):
    sh.cd(text8_dir)
    sh.unzip(input_file)
    print sh.ls()
    sh.mv("text8", output_file)

@ruffus.transform(extract_text8, ruffus.suffix(".txt"), ".alphabet.json")
def build_alpabet_dictionary(input_file, output_file):
    characters = set()
    with open(input_file) as f:
        for line in f:
            characters = characters.union(line)

    alphabet = list(sorted(characters)) + ['PADDING', 'START', 'END']

    with open(output_file, 'w') as f:
        f.write(json.dumps(alphabet))

@ruffus.transform(extract_text8, ruffus.suffix(".txt"), ".char.fragments.gz")
def build_character_sequence_dataset(input_file, output_file):
    with open(input_file) as input:
        input_data = input.read()

    print "Preparing data"
    progress_bar = pyprind.ProgPercent(N_TRAIN_CHAR_FRAGMENTS)

    outfile = gzip.open(output_file, 'w')
    for _ in xrange(N_TRAIN_CHAR_FRAGMENTS):
        base = random.randint(0, len(input_data) - CHAR_FRAGMENTS_CONTEXT_LENGTH - 1)
        x = input_data[base:base+CHAR_FRAGMENTS_CONTEXT_LENGTH]
        y = input_data[base+CHAR_FRAGMENTS_CONTEXT_LENGTH]

        outfile.write("{}\t{}\n".format(x, y))
        progress_bar.update()

    outfile.close()

@ruffus.merge([build_character_sequence_dataset, build_alpabet_dictionary], os.path.join(text8_dir, "text8.char.fragments.encoded.npz"))
def encode_character_sequence_dataset(input_files, output_file):
    data_file_name, alphabet_file_name = input_files

    alphabet = dict()
    with open(alphabet_file_name) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    progress_bar = pyprind.ProgPercent(N_TRAIN_CHAR_FRAGMENTS)

    xs = []
    ys = []
    with gzip.open(data_file_name) as data_file:
        for line in data_file:
            x, y = line.rstrip('\n').split('\t')
            x = np.atleast_2d(np.asarray(list(alphabet[c] for c in x)))
            y = alphabet[y]

            xs.append(x)
            ys.append(y)

            progress_bar.update()

    X = np.vstack(xs)
    Y = np.vstack(ys)

    np.savez(output_file, X=X, Y=Y)

@ruffus.transform(extract_text8, ruffus.suffix(".txt"), ".words.txt.gz")
def build_word_dictionary(input_file, output_file):
    words = set()
    with open(input_file) as f:
        for line in f:
            words = words.union(line.split(" "))

    words = list(w for w in sorted(words) if len(w) > 0)

    progress_bar = pyprind.ProgPercent(len(words))

    with gzip.open(output_file, 'w') as f:
        for word in words:
            f.write("{}\n".format(word))
            progress_bar.update()

@ruffus.merge([build_word_dictionary, build_alpabet_dictionary], os.path.join(text8_dir, "text8.words.encoded.npz"))
def encode_word_dictionary(input_files, output_file):
    word_file_name, alphabet_file_name = input_files

    alphabet = dict()
    with open(alphabet_file_name) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    with gzip.open(word_file_name) as word_file:
        words = [word.rstrip() for word in word_file]

    progress_bar = pyprind.ProgPercent(len(words))

    max_word_length = max(len(w) for w in words)

    encoded_words = []
    for word in words:
        encoded_word = [alphabet[c] for c in word]
        encoded_word = encoded_word + [alphabet['PADDING']] * (max_word_length - len(encoded_word))
        encoded_word = [alphabet['START']] + encoded_word + [alphabet['END']]

        encoded_words.append(np.atleast_2d(np.asarray(encoded_word)))

        progress_bar.update()

    X = np.vstack(encoded_words)

    np.savez(output_file, X=X)


if __name__ == "__main__":
    ruffus.pipeline_run(verbose=5)