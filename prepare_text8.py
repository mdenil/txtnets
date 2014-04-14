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


N_TRAIN = 1000000
CONTEXT_LENGTH = 50

@ruffus.follows(ruffus.mkdir(text8_dir))
@ruffus.originate(os.path.join(text8_dir, "text8.txt"))
def download_text8(output_file):
    print "Downloading text8"
    sh.wget("-O", output_file, "http://mattmahoney.net/dc/text8.zip")
    sh.unzip(input_file)
    sh.mv("text8", output_file)

@ruffus.transform(download_text8, ruffus.suffix(".txt"), ".alphabet.json")
def build_alpabet_dictionary(input_file, output_file):
    characters = set()
    with open(input_file) as f:
        for line in f:
            characters = characters.union(line)

    alphabet = list(sorted(characters))

    with open(output_file, 'w') as f:
        f.write(json.dumps(alphabet))

@ruffus.transform(download_text8, ruffus.suffix(".txt"), ".fragments.gz")
def build_dataset(input_file, output_file):
    with open(input_file) as input:
        input_data = input.read()

    print "Preparing data"
    progress_bar = pyprind.ProgPercent(N_TRAIN)

    outfile = gzip.open(output_file, 'w')
    for _ in xrange(N_TRAIN):
        base = random.randint(0, len(input_data) - CONTEXT_LENGTH - 1)
        x = input_data[base:base+CONTEXT_LENGTH]
        y = input_data[base+CONTEXT_LENGTH]

        outfile.write("{}\t{}\n".format(x, y))
        progress_bar.update()

    outfile.close()

@ruffus.merge([build_dataset, build_alpabet_dictionary], os.path.join(text8_dir, "text8.encoded.npz"))
def encode_dataset(input_files, output_file):
    data_file_name, alphabet_file_name = input_files

    alphabet = dict()
    with open(alphabet_file_name) as alphabet_file:
        for index, char in enumerate(json.loads(alphabet_file.read())):
            alphabet[char] = index

    progress_bar = pyprind.ProgPercent(N_TRAIN)

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

if __name__ == "__main__":
    ruffus.pipeline_run(verbose=5)