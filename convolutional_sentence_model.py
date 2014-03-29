__author__ = 'mdenil'

import numpy as np
import scipy.fftpack

class WordEmbedding(object):
    def __init__(self,
                 dimension,
                 vocabulary_size,
                 ):

        self.dimension = dimension
        self.vocabulary_size = vocabulary_size

        self.E = 0.05 * np.random.standard_normal(size=(self.vocabulary_size, self.dimension))

        self.input_axes = ['b', 'w']
        self.output_axes = ['b', 'w', 'd']

    def fprop(self, X):
        b, w = X.shape

        # ravel() unwinds in C order, (row major).  The result is each sentence appears consecutavely.
        # There is one word per row, and sentence are stacked vertically.
        X = self.E[X.ravel()]

        return np.reshape(X, (b, w, self.dimension))

    def __repr__(self):
        return "{}(dim={}, vocab_size={})".format(
            self.__class__.__name__,
            self.dimension,
            self.vocabulary_size)


class SentenceConvolution(object):
    def __init__(self,
                 n_feature_maps,
                 kernel_width,
                 n_input_dimensions,
                 ):

        self.n_feature_maps = n_feature_maps
        self.kernel_width = kernel_width
        self.n_input_dimensions = n_input_dimensions

        self.W = 0.05 * np.random.standard_normal(size=(self.n_feature_maps * self.n_input_dimensions, self.kernel_width))

        self.input_axes = ['b', 'd', 'w']
        self.output_axes = ['b', 'f', 'd', 'w']

    def fprop(self, X):
        b, d, w = X.shape

        assert self.n_input_dimensions == d
        f = self.n_feature_maps

        X = np.reshape(
            np.transpose(
                np.concatenate([X[np.newaxis]] * f), # f b d w
                (1, 0, 2, 3)
            ),
            (b * f * d, w)
        )

        # This part is probably wrong, the shapes are right though

        X_padding_size = self.kernel_width - 1
        X = np.hstack([X, np.zeros((X.shape[0], X_padding_size))])

        K_padding_size = X.shape[1] - self.kernel_width
        K = np.hstack([self.W, np.zeros((self.W.shape[0], K_padding_size))])
        K = np.vstack([K] * b)

        X = scipy.fftpack.fft(X, axis=1)
        K = scipy.fftpack.fft(K, axis=1)

        X = scipy.fftpack.ifft(X*K)

        representation_length = w + self.kernel_width - 1

        return np.reshape(
            X,
            (b, f, d, representation_length))

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)


class SumFolding(object):
    def __init__(self):
        self.input_axes = ['d', 'b', 'f', 'w']
        self.output_axes = ['d', 'b', 'f', 'w']

    def fprop(self, X):
        d, b, f, w = X.shape

        assert ( d % 2 == 0 )
        folded_size = d / 2

        X = X[:folded_size] + X[folded_size:]

        X = np.reshape(
            X, (folded_size, b, f, w)
        )

        return X

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)

class KMaxPooling(object):
    def __init__(self, k):
        self.k = k
        self.input_axes = ['d', 'b', 'f', 'w']
        self.output_axes = ['d', 'b', 'f', 'w']

    def fprop(self, X):
        d, b, f, w = X.shape

        X = np.reshape(
            X,
            (d * b * f, w)
        )

        k_max_indexes = np.argsort(X, axis=1)
        k_max_indexes = k_max_indexes[:,-self.k:]
        k_max_indexes.sort(axis=1)

        rows = np.vstack([np.arange(d * b * f)] * self.k).T

        X = X[rows, k_max_indexes]

        X = np.reshape(
            X,
            (d, b, f, self.k)
        )

        return X

    def __repr__(self):
        return "{}(k={})".format(
            self.__class__.__name__,
            self.k)

class Bias(object):
    def __init__(self, n_input_dims, n_feature_maps):
        self.n_input_dims = n_input_dims
        self.n_feature_maps = n_feature_maps

        self.b = np.zeros((n_input_dims, n_feature_maps))

        self.input_axes = ['b', 'w', 'd', 'f']
        self.output_axes = ['b', 'w', 'd', 'f']

    def fprop(self, X):
        b, w, d, f = X.shape

        assert self.n_input_dims == d
        assert self.n_feature_maps == f

        X += self.b

        return X

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)

class Relu(object):
    def __init__(self):
        self.input_axes = ['b', 'w', 'f', 'd']
        self.output_axes = ['b', 'w', 'f', 'd']

    def fprop(self, X):
        return np.maximum(0, X)

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)

class CSM(object):
    def __init__(self,
                 input_axes,
                 layers,
                 ):
        self.input_axes = input_axes
        self.layers = layers

    def fprop(self, X):
        current_axes = self.input_axes
        for layer in self.layers:
            X = self._permute_data(X, current_axes, layer.input_axes)
            X = layer.fprop(X)
            current_axes = layer.output_axes

        return X

    def _permute_data(self, X, current_axes, desired_axes):
        """
        Axis types:

        b: batch_size, index over data
        w: sentence_length, index over words
        f: n_feature_maps, index over feature maps
        d: n_input_dimensions, index over dimensions of the input in the non-sentence direction
        """
        if current_axes == desired_axes:
            return X

        assert set(current_axes) == set(desired_axes)

        X = np.transpose(X, [current_axes.index(d) for d in desired_axes])

        return X

    @property
    def output_axes(self):
        if len(self.layers) == 0:
            return None
        return self.layers[-1].output_axes

    def __repr__(self):
        return "\n".join([
            "CSM {",
            "\n".join(l.__repr__() for l in self.layers),
            "}"
            ])

if __name__ == "__main__":
    import scipy.io

    data_file_name = "cnn-sm-gpu-kmax/SENT_vec_1_emb_ind_bin.mat"

    data = scipy.io.loadmat(data_file_name)

    batch_size = 40

    mini_batch = data['train'][:batch_size] - 1 # -1 to switch to zero based indexing
    max_sentence_length = mini_batch.shape[1]

    vocabulary_size = int(data['size_vocab'])

    csm = CSM(
        input_axes=['b', 'w'],
        layers=[
            WordEmbedding(dimension=42, vocabulary_size=vocabulary_size),
            SentenceConvolution(n_feature_maps=5, kernel_width=6, n_input_dimensions=42),
            SumFolding(),
            KMaxPooling(k=4),
            Bias(n_input_dims=21, n_feature_maps=5),
            Relu(),
        ],
    )

    print csm.fprop(mini_batch).shape, csm.output_axes

    print csm
