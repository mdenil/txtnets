__author__ = 'mdenil'

import os
import re
import cld2
import gzip
import ruffus
import simplejson as json
from collections import Counter
from nltk.tokenize import WordPunctTokenizer

data_dir = os.environ['DATA']
tweets_dir = os.path.join(data_dir, "tweets")

SMILIES = set([">:]", ":-)", ":)", ":]", ":3", "=]", "8)", "=)", ">:D", ":D", "8D", "xD", "XD", "=3", "8-)", ":-))", ">:[", ":-(", ":(", ":-c", ":c", ":-P", ":P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-b", ":b", ">:o", ">:O", ":-O", ":O", "o_O", "o_0", "o.O", "D:", "D8", "v.v", ":-/", ":-.", ":/", ":\\", "=/", "=\\", ":S", ":|", ":-|", ":<", ":-[", ":[", ">.>", "<.<", ">.<", "D:<", ">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", ">:P", ">:\\", ">:/", ":$", ">:)", ">;)"])
HAPPY_SMILIES = set([">:]", ":-)", ":)", ":]", ":3", "=]", "8)", "=)", ">:D", ":D", "8D", "xD", "XD", "=3", "8-)", ":-))", ":-P", ":P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-b", ":b", ">:o", ">:O", ":-O", ":O", ">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", ">:P", ">:)", ">;)"])
UNHAPPY_SMILIES = list(set(SMILIES) - set(HAPPY_SMILIES))



def detect_language(text):
    # details is 3x (langName, langCode, percent, score)
    lang_is_reliable, _, lang_details = cld2.detect(text)
    lang_details = lang_details[0] # take only the first lang detected
    lang_name, lang_code, lang_percent, lang_score = lang_details

    return lang_name, lang_code, lang_score, lang_is_reliable


@ruffus.transform(os.path.join(tweets_dir, "tweets_100k.json.gz"), ruffus.suffix(".json.gz"), ".english.json.gz")
def extract_english_tweets(input_file, output_file):
    tokenizer = WordPunctTokenizer()

    n_happy = 0
    n_sad = 0

    labelled_tweets = []
    with gzip.open(input_file) as input:
        for line in input:
            tweet_info = json.loads(line)

            if 'limit' in tweet_info:
                continue

            # TODO: care about unicode
            #text = tweet_info['text'].encode('utf-8')
            text = tweet_info['text'].encode('ascii', 'ignore').lower()

            lang_name, lang_code, lang_score, lang_is_reliable = detect_language(text)

            if not (lang_code == 'en' and lang_is_reliable):
                continue

            words = text.split()
            words = [word for word in words if not any(s in word for s in SMILIES)]

            if len(words) == 0:
                continue

            is_happy = any(sm in text for sm in HAPPY_SMILIES)
            is_sad = any(sm in text for sm in UNHAPPY_SMILIES)

            if is_happy and is_sad:
                continue
            if not (is_happy or is_sad):
                continue

            if is_happy:
                label = ":)"
                n_happy += 1
            if is_sad:
                label = ":("
                n_sad += 1

            text = " ".join(words)

            labelled_tweets.append([text, label])

        print "Created a dataset with english {} tweets. {} are happy and {} are sad".format(n_happy + n_sad, n_happy, n_sad)

    with gzip.open(output_file, 'w') as output:
        output.write(u"{}\n".format(json.dumps(labelled_tweets)))


@ruffus.transform(extract_english_tweets, ruffus.suffix(".json.gz"), ".balanced.json.gz")
def extract_balanced_dataset(input_file_name, output_file_name):
    happy = []
    sad = []
    with gzip.open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            if label == ":)":
                happy.append(text)
            elif label == ":(":
                sad.append(text)
            else:
                raise ValueError("You're not allowed to have fancy labels >:|")

    smaller_size = min(len(happy), len(sad))

    happy = happy[:smaller_size]
    sad = sad[:smaller_size]

    happy_labels = [":)"] * len(happy)
    sad_labels = [":("] * len(sad)

    happy = zip(happy, happy_labels)
    sad = zip(sad, sad_labels)


    with gzip.open(output_file_name, 'w') as output_file:
        output_file.write(u"{}\n".format(json.dumps(happy + sad)))

@ruffus.transform(extract_balanced_dataset, ruffus.suffix(".json.gz"), ".clean.json.gz")
def clean_words(input_file_name, output_file_name):
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
    with gzip.open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            text = " ".join(map(clean_word, text.split()))
            data.append([text, label])

    with gzip.open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(data)))



@ruffus.transform(
    [extract_balanced_dataset, clean_words],
    ruffus.suffix(".json.gz"), ".alphabet.json")
def build_alphabet(input_file_name, output_file_name):
    alphabet = set()
    with gzip.open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line

            alphabet = alphabet.union(text)

    alphabet = list(sorted(alphabet)) + ['START', 'END', 'UNKNOWN', 'PADDING']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(alphabet)))

@ruffus.transform(
    [extract_balanced_dataset, clean_words],
    ruffus.suffix(".json.gz"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    with gzip.open(input_file_name) as input_file:
        for line in json.loads(input_file.read()):
            text, label = line
            # dictionary.update(text.split())
            tokenizer = WordPunctTokenizer()
            dictionary.update(tokenizer.tokenize(text))

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 3)) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(dictionary)))


@ruffus.transform(
    [build_alphabet,
     build_word_dictionary],
    ruffus.suffix(".json"), ".encoding.json")
def build_encoding(input_file_name, output_file_name):
    with open(input_file_name) as input_file:
        alphabet = json.loads(input_file.read())

    encoding = dict()
    for idx, char in enumerate(alphabet):
        encoding[char] = idx

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(encoding)))



if __name__ == "__main__":
    ruffus.pipeline_run(verbose=3)