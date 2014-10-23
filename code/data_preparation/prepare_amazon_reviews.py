__author__ = 'mdenil'

import numpy as np
import os
import sh
import re
import gzip
import ruffus
import psutil
import simplejson as json
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

# throwing away less than 280 words with less than occurences keeps 99%
# of occurrences of words that appear at least 5 times
# word_frequency_cutoff = 280

# this cutoff gives a vocabulary with ~20k words, which is ~97% of word occurences
word_frequency_cutoff = 2049

data_dir = "/data/brown/mdenil"
amazon_review_dir = os.path.join(data_dir, "amazon-reviews/shards")

shard_files = [ "shard_{:06}.json.gz".format(i) for i in xrange(18) ]

@ruffus.transform(shard_files, ruffus.suffix(".json.gz"), ".sentences.json.gz")
def split_into_sentences(input_file_name, output_file_name):
    tokenizer = PunktSentenceTokenizer()

    with gzip.open(input_file_name) as input_file:
        with gzip.open(output_file_name, 'w') as sentence_file:
            for line in input_file:
                labelled_review = json.loads(line)
                tokenized_text = tokenizer.tokenize(labelled_review['text'])
                json.dump([tokenized_text, labelled_review['score']], sentence_file)
                sentence_file.write("\n")


@ruffus.transform(split_into_sentences, ruffus.suffix(".json.gz"), ".clean.json.gz")
def clean_data(input_file_name, output_file_name):
    def clean_word(word):
        word = word.lower()
        word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
        word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)

        word = word.encode('ascii', 'ignore')

        if re.search(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
            word = 'GENERIC_HTTP'

        return word.encode('ascii', 'ignore')

    tokenizer = WordPunctTokenizer()

    with gzip.open(input_file_name) as input_file:
        with gzip.open(output_file_name, 'w') as output_file:
            for line in input_file:
                sentences, score = json.loads(line)
                cleaned_sentences = []
                for sentence in sentences:
                    cleaned_sentence = " ".join(map(clean_word, sentence.split()))
                    cleaned_sentences.append(tokenizer.tokenize(cleaned_sentence))

                json.dump([cleaned_sentences, score], output_file)
                output_file.write("\n")

@ruffus.merge(
    clean_data, "processed_text.txt")
def build_w2v_input_file(input_file_names, output_file_name):
    with open(output_file_name, 'w') as output_file:
        for input_file_name in input_file_names:
            with gzip.open(input_file_name) as input_file:
                for line in input_file:
                    sentences, score = json.loads(line)

                    for sentence in sentences:
                        output_file.write(" ".join(sentence))
                        output_file.write("\n")

@ruffus.merge(
    clean_data, "dictionary.sentences.clean.json")
def build_word_dictionary(input_file_names, output_file_name):
    dictionary = Counter()

    for input_file_name in input_file_names:
        with gzip.open(input_file_name) as input_file:
            for line in input_file:
                sentences, score = json.loads(line)
                for sentence in sentences:
                    dictionary.update(sentence)

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= word_frequency_cutoff)) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        json.dump(dictionary, output_file)
        output_file.write("\n")


@ruffus.merge(
    clean_data, "dictionary.sentences.clean.txt")
def build_w2v_vocabulary_file(input_file_names, output_file_name):
    dictionary = Counter()

    for input_file_name in input_file_names:
        with gzip.open(input_file_name) as input_file:
            for line in input_file:
                sentences, score = json.loads(line)
                for sentence in sentences:
                    dictionary.update(sentence)

    dictionary_list = list(sorted(w for w in dictionary if dictionary[w] >= word_frequency_cutoff))

    with open(output_file_name, 'w') as dictionary_file:
        for word in dictionary_list:
            dictionary_file.write("{} {}\n".format(word, dictionary[word]))


@ruffus.follows(build_word_dictionary)
@ruffus.transform(
    clean_data,
    ruffus.suffix(".json.gz"),
    ruffus.add_inputs("dictionary.sentences.clean.json"),
    ".projected.json.gz")
def project_sentences(input_file_names, output_file_name):
    review_file_name, dictionary_file_name = input_file_names

    with open(dictionary_file_name) as dictionary_file:
        dictionary = set(json.load(dictionary_file))

    def project_sentence(s):
        return [w if w in dictionary else "UNKNOWN" for w in s]

    with gzip.open(review_file_name) as review_file:
        with gzip.open(output_file_name, 'w') as output_file:
            for line in review_file:
                sentences, score = json.loads(line)
                projected_sentences = map(project_sentence, sentences)

                output_file.write(json.dumps([projected_sentences, score]))
                output_file.write("\n")


@ruffus.transform(build_word_dictionary, ruffus.suffix(".json"), ".encoding.json")
def encode_dictionary(input_file_name, output_file_name):
    encoding = {}
    with open(input_file_name) as input_file:
        for index, word in enumerate(json.load(input_file)):
            encoding[word] = index

    with open(output_file_name, 'w') as output_file:
        json.dump(encoding, output_file)
        output_file.write("\n")


if __name__ == "__main__":
    sh.cd(amazon_review_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)
