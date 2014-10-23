__author__ = 'mdenil'

import re
import os
import sh
import ruffus
import psutil
import codecs
import simplejson as json
import string
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

# http://www.cs.jhu.edu/~mdredze/datasets/sentiment/

# These files are in "pseudo-xml" which means that they kinda look like XML but beautifulsoup does
# not understand them.

data_dir = "../data"
amazon_dir = os.path.join(data_dir, "amazon_sentiment")

@ruffus.originate(["amazon_sentiment.tar.gz"])
def download_data(output_file_name):
    sh.wget("-O", output_file_name, "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz")


def get_tag_content(tag, text):
    open_tag = "<{}>".format(tag)
    end_tag = "</{}>".format(tag)

    tag_start = text.find(open_tag)
    if tag_start == -1:
        return None

    tag_end = text.find(end_tag, tag_start)
    if tag_end == -1:
        return None

    tag_start += len(open_tag)

    return repr(text[tag_start:tag_end].strip())


def get_reviews(review_file_name, **extras):
    with open(review_file_name) as review_file:
        text = review_file.read()

    # [1:] because the first element ends up as an empty string
    reviews = re.split("<review>", text)[1:]

    review_infos = []
    for review in reviews:
        review_info = {
            'product_type': get_tag_content("product_type", review),
            'rating': get_tag_content("rating", review),
            'review_text': get_tag_content("review_text", review),
        }
        review_info.update(extras)
        review_infos.append(review_info)

    return review_infos


@ruffus.transform([download_data], ruffus.suffix(".tar.gz"), ".json")
def extract_reviews(input_file_name, output_file_name):
    # extracts to folder "sorted_data"
    sh.tar("xvf", input_file_name)

    reviews = []
    raw_dir = "sorted_data"
    categories = [name for name in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, name))]
    for category in categories:
        positive_file_name = os.path.join(raw_dir, category, "positive.review")
        negative_file_name = os.path.join(raw_dir, category, "negative.review")

        positive_reviews = get_reviews(positive_file_name, label=':)', categroy=category)
        negative_reviews = get_reviews(negative_file_name, label=':(', categroy=category)

        reviews.extend(positive_reviews)
        reviews.extend(negative_reviews)

    # This folder is really big, and we still have the compressed version of this anyway so there's
    # no need to keep it around.
    sh.rm("-rf", raw_dir)

    with open(output_file_name, 'wb') as output_file:
        json.dump(reviews, output_file)
        output_file.write(u"\n")


@ruffus.transform(extract_reviews, ruffus.suffix(".json"), ".sentences.json")
def split_into_sentences(input_file_name, output_file_name):
    with open(input_file_name) as input_file:
        labelled_reviews = json.load(input_file)

    tokenizer = PunktSentenceTokenizer()

    tokenized_labelled_reviews = []

    for review in labelled_reviews:
        text = review['review_text']
        label = review['label']
        tokenized_labelled_reviews.append([tokenizer.tokenize(text), label])

    with open(output_file_name, 'w') as sentence_file:
        json.dump(tokenized_labelled_reviews, sentence_file)
        sentence_file.write("\n")


@ruffus.transform(split_into_sentences, ruffus.suffix(".json"), ".clean.json")
def clean_data(input_file_name, output_file_name):
    def clean_word(word):
        word = word.encode('ascii', 'ignore')
        word = word.lower()
        word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)

        if re.search(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
            word = 'GENERIC_HTTP'

        return word

    tokenizer = WordPunctTokenizer()
    data = []
    with open(input_file_name) as input_file:
        for sentences, label in json.load(input_file):
            cleaned_sentences = []
            for sentence in sentences:
                cleaned_sentence = " ".join(map(clean_word, sentence.split()))
                cleaned_sentence = tokenizer.tokenize(cleaned_sentence)
                cleaned_sentences.append(cleaned_sentence)

            data.append([cleaned_sentences, label])

    with codecs.open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file)


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
#
#
# @ruffus.follows(build_word_dictionary)
# @ruffus.transform(
#     clean_data,
#     ruffus.suffix(".json"),
#     ruffus.add_inputs(r"\1.dictionary.json"),
#     ".projected.json")
# def project_sentences(input_file_names, output_file_name):
#     review_file_name, dictionary_file_name = input_file_names
#
#     with open(review_file_name) as review_file:
#         reviews = json.load(review_file)
#
#     dictionary_file_name = dictionary_file_name.replace('test', 'train')
#     dictionary_file_name = dictionary_file_name.replace('unsup', 'train')
#
#
#     with open(dictionary_file_name) as dictionary_file:
#         dictionary = json.load(dictionary_file)
#
#     def project_sentence(s):
#         return [w if w in dictionary else "UNKNOWN" for w in s]
#
#     projected_reviews = []
#     for sentences, label in reviews:
#         projected_sentences = map(project_sentence, sentences)
#         projected_reviews.append([projected_sentences, label])
#
#     with open(output_file_name, 'w') as output_file:
#         json.dump(projected_reviews, output_file)
#         output_file.write("\n")
#
#
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
    if not os.path.exists(amazon_dir):
        os.makedirs(amazon_dir)

    sh.cd(amazon_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)