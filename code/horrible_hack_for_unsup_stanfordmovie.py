__author__ = 'mdenil'

import numpy as np
import os
import sh
import re
import psutil
import simplejson as json
from collections import Counter
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import random
import string


data_dir = "../data"
stanfordmovie_dir = os.path.join(data_dir, "stanfordmovie")




# @ruffus.transform(create_jsons, ruffus.suffix(".json"), ".sentences.json")
def split_into_sentences(input_file_name, output_file_name):
    with open(input_file_name) as input_file:
        labelled_reviews = json.load(input_file)

    tokenizer = PunktSentenceTokenizer()

    tokenized_labelled_reviews = []

    for text, label in labelled_reviews:
        tokenized_labelled_reviews.append([tokenizer.tokenize(text), label])

    with open(output_file_name, 'w') as sentence_file:
        json.dump(tokenized_labelled_reviews, sentence_file)
        sentence_file.write("\n")


# @ruffus.transform(split_into_sentences, ruffus.suffix(".json"), ".clean.json")
def clean_data(input_file_name, output_file_name):
    digits = set(string.digits)

    #TODO: 1. Cleaning procedure may still need improvement with some html tags
    def clean_word(word):
        word = word.lower()
        word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
        word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)

        word = word.replace("n't", " nt")  # <===MAYBE TAKE THIS OFF
        word = word.replace('"', '')
        word = word.replace('(', '')
        word = word.replace(')', '')
        word = word.replace('[', '')
        word = word.replace(']', '')
        word = word.replace('.', ' .')
        word = word.replace(',', ' ,')
        word = word.replace("'", "")

        word = word.encode('ascii', 'ignore')

        if re.match(r'[^A-Za-z0-9]*@', word):
            word = 'GENERIC_USER'

        if re.search(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
            word = 'GENERIC_HTTP'

        return word.encode('ascii', 'ignore')

    def clean_token(word):
        letter_set = set(word)

        if letter_set.isdisjoint(string.ascii_letters):
            # order matters here, although I'm not sure the current order is best

            if letter_set <= digits:
                word = 'GENERIC_NUMBER'

            elif '?' in letter_set:
                word = '?'

            elif '!' in letter_set:
                word = '!'

            elif '.' in letter_set:
                word = '.'

            else:
                word = 'GENERIC_SYMBOL'

        elif re.match("^\d+st$|^\d+th$|^\d+s$|\d+mm|^[\d:]+pm$|^[\d:]+am$", word):
            word = "GENERIC_NUMBER"

        return word

    tokenizer = WordPunctTokenizer()
    data = []
    with open(input_file_name) as input_file:
        for sentences, label in json.loads(input_file.read()):
            cleaned_sentences = []
            for sentence in sentences:
                cleaned_sentence = " ".join(map(clean_word, sentence.split()))
                cleaned_sentence = map(clean_token, tokenizer.tokenize(cleaned_sentence))
                cleaned_sentences.append(cleaned_sentence)

            data.append([cleaned_sentences, label])

    with open(output_file_name, 'w') as output_file:
        json.dump(data, output_file)
        output_file.write("\n")


# @ruffus.transform(
#     clean_data,
#     ruffus.suffix(".json"), ".dictionary.json")
# def build_word_dictionary(input_file_name, output_file_name):
#     dictionary = Counter()
#     with open(input_file_name) as input_file:
#         for sentences, label in json.loads(input_file.read()):
#             for words in sentences:
#                 dictionary.update(words)
#
#     dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 5)) + ['PADDING', 'UNKNOWN']
#
#     with open(output_file_name, 'w') as output_file:
#         json.dump(dictionary, output_file)
#         output_file.write("\n")


# @ruffus.follows(build_word_dictionary)
# @ruffus.transform(
#     clean_data,
#     ruffus.suffix(".json"),
#     ruffus.add_inputs(r"\1.dictionary.json"),
#     ".projected.json")
def project_sentences(input_file_names, output_file_name):
    review_file_name, dictionary_file_name = input_file_names

    with open(review_file_name) as review_file:
        reviews = json.load(review_file)

    dictionary_file_name = dictionary_file_name.replace('test', 'train')
    dictionary_file_name = dictionary_file_name.replace('unsup', 'train')


    with open(dictionary_file_name) as dictionary_file:
        dictionary = json.load(dictionary_file)

    def project_sentence(s):
        return [w if w in dictionary else "UNKNOWN" for w in s]

    projected_reviews = []
    for sentences, label in reviews:
        projected_sentences = map(project_sentence, sentences)
        projected_reviews.append([projected_sentences, label])

    with open(output_file_name, 'w') as output_file:
        json.dump(projected_reviews, output_file)
        output_file.write("\n")


# @ruffus.transform([build_word_dictionary], ruffus.suffix(".json"), ".encoding.json")
# def encode_dictionary(input_file_name, output_file_name):
#     encoding = dict()
#     with open(input_file_name) as input_file:
#         for index, char in enumerate(json.load(input_file)):
#             encoding[char] = index
#
#     with open(output_file_name, 'w') as output_file:
#         json.dump(encoding, output_file)
#         output_file.write("\n")


if __name__ == "__main__":
    sh.cd(stanfordmovie_dir)
    # split_into_sentences("stanfordmovie.unsup.json", "stanfordmovie.unsup.sentences.json")
    # clean_data("stanfordmovie.unsup.sentences.json", "stanfordmovie.unsup.sentences.clean.json")
    project_sentences(
        ["stanfordmovie.unsup.sentences.clean.json", "stanfordmovie.train.sentences.clean.dictionary.json"],
        "stanfordmovie.unsup.sentences.clean.projected.json")