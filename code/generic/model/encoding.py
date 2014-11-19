__author__ = 'mdenil'


class DictionaryEncoding(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def fprop(self, X, meta):
        X = [self._encode(x) for x in X]

        X, X_space = self._fprop(X, meta)

        meta['space_above'] = X_space

        fprop_state = {}

        return X, meta, fprop_state

    def _encode(self, x):
        # return [self.vocabulary[c] for c in x]
        return [self.vocabulary.get(c, self.vocabulary['UNKNOWN']) for c in x]

    def __repr__(self):
        return "{}(vocabulary_size={})".format(
            self.__class__.__name__,
            len(self.vocabulary))