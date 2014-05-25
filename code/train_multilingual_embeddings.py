__author__ = 'mdenil'

import numpy as np
import os
import time
import simplejson as json
import cPickle as pickle
import argparse

import generic.model.utils
from gpu.model.model import CSM
from gpu.model.encoding import DictionaryEncoding
from gpu.model.embedding import WordEmbedding
from gpu.model.transfer import AxisReduction
from gpu.model.model import TaggedModelCollection
from gpu.optimize.objective.contrastive_multilingual import ContrastiveMultilingualEmbeddingObjective
from gpu.optimize.sgd import SGD
from gpu.optimize.regularizer import L2Regularizer
from gpu.optimize.update_rule import AdaGrad
from gpu.optimize.data_provider import PaddedParallelSequenceMinibatchProvider
from generic.optimize.data_provider import TaggedProviderCollection


def squared_distances(x, y):
    x2 = np.sum(x**2, axis=1, keepdims=True)
    y2 = np.sum(y**2, axis=1, keepdims=True)

    # (x - y) (x - y)' = xx' - 2 xy' + yy'
    return x2 + y2.T - 2*np.dot(x, y.T)


def replace_unknowns(data, dictionary, unknown):
    new_data = []
    for sentence in data:
        new_data.append(
            [word if word in dictionary else unknown for word in sentence]
        )
    return new_data


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--margin", type=float, default=40.0)
    parser.add_argument("--embedding_dimension", type=int, default=40)
    parser.add_argument("--regularizer", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=99999999)
    parser.add_argument("--save_file_name", type=str, default="model.pkl")
    parser.add_argument("--results_dir", type=str, default=".")
    args = parser.parse_args()

    # random.seed(435)
    # np.random.seed(2342)
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

    # english_data = english_data[:10000]
    # german_data = german_data[:10000]

    english_data = replace_unknowns(english_data, english_dictionary, 'UNKNOWN')
    german_data = replace_unknowns(german_data, german_dictionary, 'UNKNOWN')

    assert len(english_data) == len(german_data)
    print len(english_data) / args.batch_size

    parallel_en_de_provider = PaddedParallelSequenceMinibatchProvider(
        X1=list(english_data),
        X2=list(german_data),
        batch_size=args.batch_size,
        padding='PADDING',
    )

    multilingual_parallel_provider = TaggedProviderCollection({
        ('en', 'de'): parallel_en_de_provider
    })

    X_en_valid, meta_en_valid, X_de_valid, meta_de_valid = parallel_en_de_provider.next_batch()


    english_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=english_dictionary),

            WordEmbedding(
                dimension=args.embedding_dimension,
                vocabulary_size=len(english_dictionary),
                padding=english_dictionary['PADDING']),

            AxisReduction(axis='w'),

            # SentenceConvolution(
            #     n_feature_maps=15,
            #     kernel_width=10,
            #     n_channels=1,
            #     n_input_dimensions=12),
            #
            # SumFolding(),
            #
            # KMaxPooling(k=17),
            #
            # Bias(
            #     n_input_dims=6,
            #     n_feature_maps=15),
            #
            # Tanh(),
            ]
        )

    german_model = CSM(
        layers=[
            DictionaryEncoding(vocabulary=german_dictionary),

            WordEmbedding(
                dimension=args.embedding_dimension,
                vocabulary_size=len(german_dictionary),
                padding=german_dictionary['PADDING']),

            AxisReduction(axis='w'),

            # SentenceConvolution(
            #     n_feature_maps=15,
            #     kernel_width=10,
            #     n_channels=1,
            #     n_input_dimensions=12),
            #
            # SumFolding(),
            #
            # KMaxPooling(k=17),
            #
            # Bias(
            #     n_input_dims=6,
            #     n_feature_maps=15),
            #
            # Tanh(),

            ]
        )

    print english_model
    print german_model

    model = TaggedModelCollection({
        'en': english_model,
        'de': german_model,
    })

    regularizer = L2Regularizer(lamb=args.regularizer)

    objective = ContrastiveMultilingualEmbeddingObjective(
        tagged_parallel_sequence_provider=multilingual_parallel_provider,
        n_contrastive_samples=args.n_samples,
        margin=args.margin,
        regularizer=regularizer)

    update_rule = AdaGrad(
        gamma=args.learning_rate,
        model_template=model)

    optimizer = SGD(
        model=model,
        objective=objective,
        update_rule=update_rule)

    time_start = time.time()

    losses = []
    for batch_index, iteration_info in enumerate(optimizer):
        losses.append(iteration_info['cost'])

        if batch_index % 50 == 0:
            # This epoch count will be inaccurate when I move to multilingual
            epoch = (batch_index // parallel_en_de_provider.batches_per_epoch) + 1

            m_en = generic.model.utils.ModelEvaluator(
                model.get_model('en'),
                desired_axes=('b', ('d', 'f', 'w')))
            m_de = generic.model.utils.ModelEvaluator(
                model.get_model('de'),
                desired_axes=('b', ('d', 'f', 'w')))

            Y_en_valid = m_en.fprop(X_en_valid, meta_en_valid).get()
            Y_de_valid = m_de.fprop(X_de_valid, meta_de_valid).get()

            # Y_en_valid = m_en.fprop(X_en_valid, meta_en_valid)
            # Y_de_valid = m_de.fprop(X_de_valid, meta_de_valid)

            Y_sqd = squared_distances(Y_en_valid, Y_de_valid)
            correct = np.sum(np.diag(Y_sqd))
            incorrect = np.sum(Y_sqd) - correct

            correct /= Y_sqd.shape[0]
            incorrect /= Y_sqd.size - Y_sqd.shape[0]

            # print correct,  incorrect

            time_now = time.time()
            examples_per_hr = (batch_index * args.batch_size) / (time_now - time_start) * 3600

            print "B: {}, E: {}, L: {}, C: {}, I: {}, I-C: {}, EPH: {}, PS: {}".format(
                batch_index,
                epoch,
                losses[-1],
                correct,
                incorrect,
                incorrect - correct,
                examples_per_hr,
                np.mean(np.abs(model.pack())))

            if epoch > args.max_epochs:
                break

        if batch_index % 100 == 0:
            with open(os.path.join(args.results_dir, args.save_file_name), 'w') as model_file:
                 pickle.dump(model.move_to_cpu(), model_file, protocol=-1)

        # if batch_index % 1000 == 0 and batch_index > 0:
        #     with open("model_optimization.pkl", 'w') as model_file:
        #         pickle.dump(optimizer, model_file, protocol=-1)

        # if batch_index == 500:
        #     break

    time_end = time.time()

    print "Time elapsed: {}s".format(time_end - time_start)


if __name__ == "__main__":
    # batch_size = 100
    # learning_rate = 0.05
    # n_contrastive_samples = 50
    # margin = 40
    # embedding_dimension = 40
    run()