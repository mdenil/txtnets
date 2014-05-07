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

        positive_reviews = get_reviews(positive_file_name, disposition='positive', categroy=category)
        negative_reviews = get_reviews(negative_file_name, disposition='negative', categroy=category)

        reviews.extend(positive_reviews)
        reviews.extend(negative_reviews)

    # This folder is really big, and we still have the compressed version of this anyway so there's
    # no need to keep it around.
    sh.rm("-rf", raw_dir)

    with open(output_file_name, 'wb') as output_file:
        json.dump(reviews, output_file)
        output_file.write(u"\n")


@ruffus.transform(extract_reviews, ruffus.suffix(".json"), ".clean.json")
def clean_reviews(input_file_name, output_file_name):
    def clean_word(word):
        word = word.lower()
        word = re.sub(r'(\S)\1+', r'\1\1', word) #normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)
        if set(word) <= set(string.digits):
            word = 'NUMBER'
        return word

    tokenizer = WordPunctTokenizer()
    data = []
    with open(input_file_name) as input_file:
        for record in json.loads(input_file.read()):
            record['review_text'] = " ".join(map(
                clean_word, tokenizer.tokenize(record['review_text'])))
            # record['review_text'] = " ".join(map(clean_word, record['review_text'].split()))
            data.append(record)

    with codecs.open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file)


@ruffus.transform(
    [clean_reviews],
    ruffus.suffix(".json"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    tokenizer = WordPunctTokenizer()
    with open(input_file_name) as input_file:
        for review_info in json.loads(input_file.read()):
            text = review_info['review_text']
            dictionary.update(tokenizer.tokenize(text))

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 5)) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        json.dump(dictionary, output_file)
        output_file.write(u"\n")


if __name__ == "__main__":
    if not os.path.exists(amazon_dir):
        os.makedirs(amazon_dir)

    sh.cd(amazon_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)