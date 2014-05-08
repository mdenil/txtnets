__author__ = 'mdenil'


import numpy as np
import os
import time
import random
import simplejson as json
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer

from cpu.model.model import CSM
from cpu.model.encoding import DictionaryEncoding
from cpu.model.embedding import WordEmbedding
from cpu.model.transfer import SentenceConvolution
from cpu.model.transfer import Bias
from cpu.model.pooling import SumFolding
from cpu.model.pooling import MaxFolding
from cpu.model.pooling import KMaxPooling
from cpu.model.nonlinearity import Tanh
from cpu.model.transfer import Softmax
from cpu.model.model import TaggedModelCollection

from cpu.optimize.objective import ContrastiveMultilingualEmbeddingObjective

from cpu.optimize.sgd import SGD
from cpu.optimize.objective import CostMinimizationObjective
from cpu.optimize.regularizer import L2Regularizer
from cpu.optimize.update_rule import AdaGrad
from cpu.optimize.data_provider import PaddedSequenceMinibatchProvider
from generic.optimize.data_provider import ParallelProvider
from generic.optimize.data_provider import TaggedProviderCollection


def replace_unknowns(data, dictionary, unknown):
    new_data = []
    for sentence in data:
        new_data.append(
            [word if word in dictionary else unknown for word in sentence]
        )
    return new_data


def run():
    random.seed(435)
    np.random.seed(2342)
    np.set_printoptions(linewidth=100)

    data_dir = os.path.join("../data", "europarlv7")

    with open(os.path.join(data_dir, "europarl-v7.de-en.en.tokens.clean.json")) as data_file:
        english_data = json.load(data_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.en.tokens.clean.dictionary.encoding.json")) as dictionary_file:
        english_dictionary = json.load(dictionary_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.de.tokens.clean.json")) as data_file:
        german_data = json.load(data_file)

    with open(os.path.join(data_dir, "europarl-v7.de-en.de.tokens.clean.dictionary.encoding.json")) as dictionary_file:
        german_dictionary = json.load(dictionary_file)

    english_data = replace_unknowns(english_data, english_dictionary, 'UNKNOWN')
    german_data = replace_unknowns(german_data, german_dictionary, 'UNKNOWN')

    parallel_en_de_provider = ParallelProvider(
        PaddedSequenceMinibatchProvider(
            X=list(english_data),
            batch_size=100,
            # fixed_length=50,
            padding='PADDING'),
        PaddedSequenceMinibatchProvider(
            X=list(german_data),
            batch_size=100,
            # fixed_length=50,
            padding='PADDING'),
    )

    multilingual_parallel_provider = TaggedProviderCollection({
        ('en', 'de'): parallel_en_de_provider
    })

    contrastive_sequence_provider = TaggedProviderCollection({
        'en': PaddedSequenceMinibatchProvider(
            X=list(english_data),
            batch_size=100,
            # fixed_length=50,
            padding='PADDING'),
        'de': PaddedSequenceMinibatchProvider(
            X=list(german_data),
            batch_size=100,
            # fixed_length=50,
            padding='PADDING'),
    })


    english_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=english_dictionary),

            WordEmbedding(
                dimension=32,
                vocabulary_size=len(english_dictionary)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32),

            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),

            SumFolding(),
            ]
        )

    german_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=german_dictionary),

            WordEmbedding(
                dimension=32,
                vocabulary_size=len(german_dictionary)),

            SentenceConvolution(
                n_feature_maps=5,
                kernel_width=10,
                n_channels=1,
                n_input_dimensions=32),

            SumFolding(),

            KMaxPooling(k=7),

            Bias(
                n_input_dims=16,
                n_feature_maps=5),

            Tanh(),

            SumFolding(),

            ]
        )

    print english_model
    print german_model

    model = TaggedModelCollection({
        'en': english_model,
        'de': german_model,
    })

    # regularizer = L2Regularizer(lamb=1e-4)

    objective = ContrastiveMultilingualEmbeddingObjective(
        tagged_parallel_sequence_provider=multilingual_parallel_provider,
        tagged_contrastive_sequence_provider=contrastive_sequence_provider,
        n_contrastive_samples=5,
        margin=5.0)

    # objective = CostMinimizationObjective(
    #     cost=cost_function,
    #     data_provider=train_data_provider,
    #     regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=0.05,
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    time_start = time.time()

    costs = []
    for batch_index, iteration_info in enumerate(optimizer):
        costs.append(iteration_info['cost'])

        print costs[-1]

        if batch_index % 10 == 0:
            print "B: {}, C: {}, Param size: {}".format(
                batch_index,
                costs[-1],
                np.mean(np.abs(model.pack())))

        if batch_index % 100 == 0:
            with open("model.pkl", 'w') as model_file:
                pickle.dump(model, model_file, protocol=-1)

        # if batch_index % 1000 == 0 and batch_index > 0:
        #     with open("model_optimization.pkl", 'w') as model_file:
        #         pickle.dump(optimizer, model_file, protocol=-1)

        if batch_index == 300:
            break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)


if __name__ == "__main__":
    run()