__author__ = 'mdenil, albandemiraj'

import numpy as np

import random
import os
import re
import gzip
import simplejson as json
from collections import OrderedDict

import cpu.space


class LabelledSequenceMinibatchProvider(object):
    def __init__(self, X, Y, batch_size, padding, shuffle=True, fixed_length=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.padding = padding
        self.fixed_length = fixed_length
        self.shuffle = shuffle

        self._batch_index = -1
        self.batches_per_epoch = len(X) / batch_size
        self.max_label_value = np.max(Y)

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]
        Y_batch = self.Y[batch_start:batch_end]

        Y_batch = np.equal.outer(Y_batch, np.arange(self.max_label_value+1)).astype(np.float)

        lengths_batch = np.asarray(map(len, X_batch))

        if self.fixed_length:
            max_length_batch = self.fixed_length
            lengths_batch = np.minimum(lengths_batch, self.fixed_length)
        else:
            max_length_batch = int(lengths_batch.max())

        X_batch = [self._pad_or_truncate(x, max_length_batch) for x in X_batch]

        meta = {
            'lengths': lengths_batch,
            'space_below': cpu.space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X_batch)), ('w', max_length_batch)]))
        }

        return X_batch, Y_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        combined = zip(self.X, self.Y)
        random.shuffle(combined)
        self.X, self.Y = map(list, zip(*combined))

    def _pad_or_truncate(self, x, max_length):
        if max_length > len(x):
            return x + [self.padding] * (max_length - len(x))
        else:
            return x[:max_length]


class LabelledSequenceBatchProvider(LabelledSequenceMinibatchProvider):
    def __init__(self, X, Y, padding):
        super(LabelledSequenceBatchProvider, self).__init__(
            X, Y, len(X), padding, shuffle=False, fixed_length=False)


class SequenceMinibatchProvider(object):
    def __init__(self, X, batch_size, padding, shuffle=True, fixed_length=False):
        self.X = X
        self.batch_size = batch_size
        self.padding = padding
        self.fixed_length = fixed_length
        self.shuffle = shuffle

        self._batch_index = -1
        self.batches_per_epoch = len(X) / batch_size

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]

        lengths_batch = np.asarray(map(len, X_batch))

        if self.fixed_length:
            max_length_batch = self.fixed_length
            lengths_batch = np.minimum(lengths_batch, self.fixed_length)
        else:
            max_length_batch = int(lengths_batch.max())

        X_batch = [self._pad_or_truncate(x, max_length_batch) for x in X_batch]

        meta = {
            'lengths': lengths_batch,
            'space_below': cpu.space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X_batch)), ('w', max_length_batch)]))
        }

        return X_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        random.shuffle(self.X)

    def _pad_or_truncate(self, x, max_length):
        if max_length > len(x):
            return x + [self.padding] * (max_length - len(x))
        else:
            return x[:max_length]


class TaggedProviderCollection(object):
    def __init__(self, tagged_providers):
        self.tagged_providers = tagged_providers

    @property
    def tags(self):
        return self.tagged_providers.keys()

    def get_provider(self, tag):
        return self.tagged_providers[tag]

    def next_batch(self, tag):
        return self.tagged_providers[tag].next_batch()


class PaddedParallelSequenceMinibatchProvider(object):
    def __init__(self, X1, X2, batch_size, padding, shuffle=True):
        assert len(X1) == len(X2)
        self.X = zip(X1, X2)
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = shuffle

        self._batch_index = -1 # will be incremeted to 0 when next_batch is called
        self.batches_per_epoch = len(self.X) / self.batch_size

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]

        X1_batch, X2_batch = zip(*X_batch)

        lengths_batch_1 = np.asarray(map(len, X1_batch))
        max_length_batch_1 = lengths_batch_1.max()
        X1_batch = [self._add_padding(x, max_length_batch_1) for x in X1_batch]

        lengths_batch_2 = np.asarray(map(len, X2_batch))
        max_length_batch_2 = lengths_batch_2.max()
        X2_batch = [self._add_padding(x, max_length_batch_2) for x in X2_batch]

        meta1 = {
            'lengths': lengths_batch_1,
            'space_below': cpu.space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X1_batch)), ('w', max_length_batch_1)]))
        }

        meta2 = {
            'lengths': lengths_batch_2,
            'space_below': cpu.space.CPUSpace(
                axes=['b', 'w'],
                extents=OrderedDict([('b', len(X2_batch)), ('w', max_length_batch_2)]))
        }

        return X1_batch, meta1, X2_batch, meta2

    def _add_padding(self, x, length):
        return x + [self.padding] * (length - len(x))

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        random.shuffle(self.X)


class LabelledDocumentMinibatchProvider(object):
    def __init__(self, X, Y, batch_size, padding, shuffle=True, fixed_n_words=False, fixed_n_sentences=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.padding = padding
        self.fixed_n_words = fixed_n_words
        self.fixed_n_sentences = fixed_n_sentences
        self.shuffle = shuffle

        self._batch_index = -1
        self.batches_per_epoch = len(X) / batch_size
        self.max_label_value = np.max(Y)

    def next_batch(self):
        self._prepare_for_next_batch()

        batch_start = self._batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X_batch = self.X[batch_start:batch_end]
        Y_batch = self.Y[batch_start:batch_end]

        Y_batch = np.equal.outer(Y_batch, np.arange(self.max_label_value+1)).astype(np.float32)
        assert not np.any(np.isnan(Y_batch))

        # Just making sure we have the right dimensions
        dimension_b = len(X_batch)

        #Defining number of SENTENCES in a DOCUMENT
        document_lengths = np.asarray(map(len, X_batch))

        if self.fixed_n_sentences:
            max_n_sentences = self.fixed_n_sentences
            document_lengths = np.minimum(document_lengths, self.fixed_n_sentences)
        else:
            max_n_sentences = int(document_lengths.max())

        # Padding or truncationg the number of SENTENCES
        X_batch = [self._pad_or_truncate_document(x, max_n_sentences) for x in X_batch]

        #NOW we have the information we need, so we can go back and work in 2d
        X_batch = np.array(X_batch)
        X_batch = X_batch.ravel()
        X_batch = X_batch.tolist()

        #Defining number of WORDS in a SENTENCE
        sentence_lengths = np.asarray(map(len, X_batch))

        if self.fixed_n_words:
            max_n_words = self.fixed_n_words
            sentence_lengths = np.minimum(sentence_lengths, self.fixed_n_words)
        else:
            max_n_words = int(sentence_lengths.max())

        # Padding or truncationg the number of WORDS
        X_batch = [self._pad_or_truncate_sentences(x, max_n_words) for x in X_batch]

        meta = {
            'lengths': sentence_lengths,
            'lengths2': document_lengths,
            'padded_sentence_length':  max_n_sentences,
            'space_below': cpu.space.CPUSpace(
                axes=('b', 'w'),
                extents={
                    'b': dimension_b * max_n_sentences,
                    'w': max_n_words,
                }),
        }

        return X_batch, Y_batch, meta

    def _prepare_for_next_batch(self):
        self._batch_index = (self._batch_index + 1) % self.batches_per_epoch

        if self._batch_index == 0 and self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        combined = zip(self.X, self.Y)
        random.shuffle(combined)
        self.X, self.Y = map(list, zip(*combined))

    def _pad_or_truncate_sentences(self, x, max_length):
        if self.padding:
            if max_length > len(x):
                return x + [self.padding] * (max_length - len(x))
            else:
                return x[:max_length]
        else:
            return x

    def _pad_or_truncate_document(self, x, max_length):
        if self.padding:
            if max_length > len(x):
                return x + [['PADDING']] * (max_length - len(x))
            else:
                #return x[:max_length]
                return x[0:max_length/2]+ x[max_length/2+len(x)-max_length:len(x)]
        else:
            return x

class ShardedLabelledDocumentMinibatchProvider(object):
    def __init__(
            self,
            shard_dir,
            shard_pattern,
            batch_size,
            padding,
            n_labels,
            shuffle=True,
            fixed_n_words=False,
            fixed_n_sentences=False):

        self.shard_dir = shard_dir
        self.shard_pattern = shard_pattern
        self.batch_size = batch_size
        self.padding = padding
        self.fixed_n_words = fixed_n_words
        self.fixed_n_sentences = fixed_n_sentences
        self.shuffle = shuffle
        self.n_labels = n_labels

        self.X_shard = None
        self.Y_shard = None

        self._example_index = 0
        self._shard_index = 0
        self._shard_file_names = self._load_shard_file_names()
        self._load_next_shard()

    def _load_shard_file_names(self):
        shard_file_names = []

        for file_name in os.listdir(self.shard_dir):
            if re.match(self.shard_pattern, file_name):
                shard_file_names.append(
                    os.path.join(self.shard_dir, file_name))

        if self.shuffle:
            random.shuffle(shard_file_names)
        else:
            shard_file_names.sort()

        return shard_file_names

    @property
    def n_shards(self):
        return len(self._shard_file_names)

    @property
    def current_shard_size(self):
        return len(self.X_shard)

    def _load_next_shard(self):
        self._example_index = 0
        self._shard_index += 1

        if self._shard_index >= self.n_shards:
            self._shard_index = 0
            if self.shuffle:
                random.shuffle(self._shard_file_names)

        self.X_shard, self.Y_shard = self._load_shard(
            self._shard_file_names[self._shard_index])

    def _load_shard(self, file_name):
        examples = []

        with gzip.open(file_name) as shard_file:
            for line in shard_file:
                examples.append(json.loads(line))

        if self.shuffle:
            random.shuffle(examples)

        return map(list, zip(*examples))

    def next_batch(self):
        X_batch, Y_batch = self._prepare_next_batch()


        Y_batch = np.asarray(Y_batch).reshape((-1,1)).astype(np.float32)

        # Y_batch = np.equal.outer(
        #     Y_batch, np.arange(self.n_labels)).astype(np.float32)
        # HACK: make the task binary
        # Y_batch = np.equal.outer(
        #     np.asarray(Y_batch) > 2.5, np.arange(self.n_labels)).astype(np.float32)

        assert not np.any(np.isnan(Y_batch))

        n_documents = len(X_batch)

        # number of sentences in each document
        document_lengths = np.asarray(map(len, X_batch))

        if self.fixed_n_sentences:
            max_n_sentences = self.fixed_n_sentences
            document_lengths = np.minimum(
                document_lengths, self.fixed_n_sentences)
        else:
            max_n_sentences = int(document_lengths.max())

        # pad documents
        X_batch = [
            self._pad_or_truncate_document(x, max_n_sentences)
            for x in X_batch
        ]

        # flatten sentences to words
        X_batch = [w for s in X_batch for w in s]

        sentence_lengths = np.asarray(map(len, X_batch))

        if self.fixed_n_words:
            max_n_words = self.fixed_n_words
            sentence_lengths = np.minimum(sentence_lengths, self.fixed_n_words)
        else:
            max_n_words = int(sentence_lengths.max())

        # pad sentences
        X_batch = [
            self._pad_or_truncate_sentence(x, max_n_words)
            for x in X_batch
        ]

        meta = {
            'lengths': sentence_lengths,
            'lengths2': document_lengths,
            'padded_sentence_length': max_n_sentences,
            'space_below': cpu.space.CPUSpace(
                axes=('b', 'w'),
                extents={
                    'b': n_documents * max_n_sentences,
                    'w': max_n_words,
                }
            )
        }

        return X_batch, Y_batch, meta


    def _pad_or_truncate_document(self, x, max_length):
        if max_length > len(x):
            return x + [[self.padding]] * (max_length - len(x))
        else:
            return x[:max_length]

    def _pad_or_truncate_sentence(self, x, max_length):
        if max_length > len(x):
            return x + [self.padding] * (max_length - len(x))
        else:
            return x[:max_length]

    def _prepare_next_batch(self):
        X_batch = []
        Y_batch = []

        while len(X_batch) < self.batch_size:
            x, y = self._next_example()
            X_batch.append(x)
            Y_batch.append(y)

        return X_batch, Y_batch


    def _next_example(self):
        if self._example_index >= self.current_shard_size:
            self._load_next_shard()

        x = self.X_shard[self._example_index]
        y = self.Y_shard[self._example_index]

        self._example_index += 1

        return x, y



class TransformedLabelledDataProvider(object):
    def __init__(self, data_source, transformer):
        self.data_source = data_source
        self.transformer = transformer

    def next_batch(self):
        x_batch, y_batch, meta = self.data_source.next_batch()
        x_batch, meta, fprop_state = self.transformer.fprop(
            x_batch, meta=dict(meta), return_state=True)

        meta['space_below'] = meta['space_above']

        return x_batch, y_batch, meta

    @property
    def batches_per_epoch(self):
        return self.data_source.batches_per_epoch
