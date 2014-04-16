__author__ = 'mdenil'

import os
import cld2
import gzip
import ruffus
import simplejson as json
from nltk.tokenize import WordPunctTokenizer

data_dir = os.environ['DATA']
tweets_dir = os.path.join(data_dir, "tweets")

SMILIES = set([">:]", ":-)", ":)", ":]", ":3", "=]", "8)", "=)", ">:D", ":D", "8D", "xD", "XD", "=3", "8-)", ":-))", ">:[", ":-(", ":(", ":-c", ":c", ":-P", ":P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-b", ":b", ">:o", ">:O", ":-O", ":O", "o_O", "o_0", "o.O", "D:", "D8", "v.v", ":-/", ":-.", ":/", ":\\", "=/", "=\\", ":S", ":|", ":-|", ":<", ":-[", ":[", ">.>", "<.<", ">.<", "D:<", ">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", ">:P", ">:\\", ">:/", ":$", ">:)", ">;)"])
HAPPY_SMILIES = set([">:]", ":-)", ":)", ":]", ":3", "=]", "8)", "=)", ">:D", ":D", "8D", "xD", "XD", "=3", "8-)", ":-))", ":-P", ":P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-b", ":b", ">:o", ">:O", ":-O", ":O", ">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", ">:P", ">:)", ">;)"])
UNHAPPY_SMILIES = list(set(SMILIES) - set(HAPPY_SMILIES))

# Taking this out...
# Right now this data set is cheating because it leaves the labels in the training data, but this should at least
# let me see if the character level features can capture useful info.


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


            labelled_tweets.append([words, label])

        print "Created a dataset with english {} tweets.  {} are happy and {} are sad".format(n_happy + n_sad, n_happy, n_sad)

    with gzip.open(output_file, 'w') as output:
        for tweet in labelled_tweets:
            output.write(u"{}\n".format(json.dumps(tweet)))


@ruffus.transform(extract_english_tweets, ruffus.suffix(".json.gz"), ".balanced.json.gz")
def extract_balanced_dataset(input_file_name, output_file_name):
    happy = []
    sad = []
    with gzip.open(input_file_name) as input_file:
        for line in input_file:
            words, label = json.loads(line)
            if label == ":)":
                happy.append(words)
            elif label == ":(":
                sad.append(words)
            else:
                raise ValueError("You're not allowed to have fancy labels >:|")

    smaller_size = min(len(happy), len(sad))

    happy = happy[:smaller_size]
    sad = sad[:smaller_size]

    happy_labels = [":)"] * len(happy)
    sad_labels = [":("] * len(sad)

    happy = zip(happy, happy_labels)
    sad = zip(sad, sad_labels)

    with gzip.open(output_file_name, 'w') as output:
        for tweet in happy:
            output.write(u"{}\n".format(json.dumps(tweet)))

        for tweet in sad:
            output.write(u"{}\n".format(json.dumps(tweet)))

@ruffus.transform(extract_english_tweets, ruffus.suffix(".json.gz"), ".alphabet.json")
def build_alphabet(input_file_name, output_file_name):
    alphabet = set()
    with gzip.open(input_file_name) as input_file:
        for line in input_file:
            words, label = json.loads(line)

            for word in words:
                alphabet = alphabet.union(word)

    alphabet = list(sorted(alphabet)) + ['START', 'END', 'PADDING']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(alphabet)))

@ruffus.transform(build_alphabet, ruffus.suffix(".alphabet.json"), ".alphabet.encoding.json")
def build_alphabet_encoding(input_file_name, output_file_name):
    with open(input_file_name) as input_file:
        alphabet = json.loads(input_file.read())

    alphabet_encoding = dict()
    for idx, char in enumerate(alphabet):
        alphabet_encoding[char] = idx

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(alphabet_encoding)))



if __name__ == "__main__":
    ruffus.pipeline_run(verbose=3)