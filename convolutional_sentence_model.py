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
        """X is a matrix of indexes into the vocabulary.  It has shape batch_size x sentence_length.
         One row per sentence, words in the sentence run from left to right.

        Returns: A batch_size x sentence_length x embedding_dimension matrix of word embeddings"""

        batch_size, sentence_length = X.shape

        # ravel() unwinds in C order, (row major).  The result is each sentence appears consecutavely.
        # There is one word per row, and sentence are stacked vertically.
        embeddings = self.E[X.ravel()]

        return np.reshape(embeddings, (batch_size, sentence_length, self.dimension))

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

        self.input_axes = ['b', 'w', 'd']
        self.output_axes = ['b', 'f', 'd', 'w']

    def fprop(self, X):
        batch_size, sentence_length, n_input_dimensions = X.shape

        assert self.n_input_dimensions == n_input_dimensions

        X = np.reshape(
            np.transpose(
                np.reshape(
                    np.hstack([X]*self.n_feature_maps),
                    (batch_size, sentence_length, self.n_feature_maps, self.n_input_dimensions)
                ),
                (1,0,2,3)
            ),
            (batch_size * self.n_feature_maps * self.n_input_dimensions, sentence_length)
        )

        # This part is probably wrong, the shapes are right though

        X_padding_size = self.kernel_width - 1
        X = np.hstack([X, np.zeros((X.shape[0], X_padding_size))])

        K_padding_size = X.shape[1] - self.kernel_width
        K = np.hstack([self.W, np.zeros((self.W.shape[0], K_padding_size))])
        K = np.vstack([K]*batch_size)

        X = scipy.fftpack.fft(X, axis=1)
        K = scipy.fftpack.fft(K, axis=1)

        representation_length = sentence_length + self.kernel_width - 1

        return np.reshape(
            scipy.fftpack.ifft(X*K),
            (batch_size, self.n_feature_maps, self.n_input_dimensions, representation_length))

    def __repr__(self):
        return "{}(W={})".format(
            self.__class__.__name__,
            self.W.shape)

class SumFolding(object):
    def __init__(self):
        self.input_axes = ['b', 'f', 'd', 'w']
        self.output_axes = ['d', 'b', 'f', 'w']

    def fprop(self, X):
        batch_size, n_feature_maps, n_input_dimensions, representation_length = X.shape

        X = np.reshape(
            np.transpose(X, (3, 0, 1, 2)),
            (n_input_dimensions, batch_size * n_feature_maps * representation_length)
        )

        assert ( n_input_dimensions % 2 == 0 )
        folded_size = n_input_dimensions / 2

        X = X[:folded_size] + X[folded_size:]

        X = np.reshape(
            X, (folded_size, batch_size, n_feature_maps, representation_length)
        )

        return X

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)

class KMaxPooling(object):
    def __init__(self, k):
        self.k = k
        self.input_axes = ['d', 'b', 'f', 'w']
        self.output_axes = ['w', 'd', 'b', 'f']

    def fprop(self, X):
        folded_size, batch_size, n_feature_maps, representation_length = X.shape

        X = np.reshape(X, (folded_size * batch_size * n_feature_maps, representation_length))

        k_max_indexes = np.argsort(X, axis=1)
        k_max_indexes = k_max_indexes[:,-self.k:]
        k_max_indexes.sort(axis=1)

        rows = np.vstack([np.arange(folded_size * batch_size * n_feature_maps)] * self.k).T

        X = X[rows, k_max_indexes]

        X = np.reshape(
            X,
            (folded_size, batch_size, n_feature_maps, self.k)
        )

        X = np.transpose(X, (3,0,1,2)) # FIXME: remove this

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

        self.input_axes = ['w', 'd', 'b', 'f']
        self.output_axes = ['d', 'b', 'w', 'f']

    def fprop(self, X):
        n_input_dims, folded_size, batch_size, n_feature_maps = X.shape

        assert self.n_input_dims == n_input_dims
        assert self.n_feature_maps == n_feature_maps

        X = np.transpose(X, (1, 2, 0, 3))

        X += self.b

        return X

    def __repr__(self):
        return "{}(n_input_dims={}, n_feature_maps={})".format(
            self.__class__.__name__,
            self.n_input_dims,
            self.n_feature_maps)

class Relu(object):
    def __init__(self):
        self.input_axes = ['b', 'f', 'w', 'd']
        self.output_axes = ['b', 'f', 'w', 'd']

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
        print "Permuting {} -> {}".format(current_axes, desired_axes)
        return np.transpose(
            X,
            [current_axes.index(d) for d in desired_axes])

    @property
    def output_axes(self):
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
            Bias(n_input_dims=4, n_feature_maps=5),
            Relu(),
        ],
    )

    #print csm

    print csm.fprop(mini_batch).shape, csm.output_axes




    exit(0)

    embedding = WordEmbedding(
        dimension=42, # 1
        vocabulary_size=vocabulary_size,
    )

    mini_batch_embeddings = embedding.fprop(mini_batch)
    print "Embedding shape:", mini_batch_embeddings.shape

    layer1_conv = SentenceConvolution(
        n_feature_maps=5, # 3
        kernel_width=6, # 4
        n_input_dimensions=42, # 1
    )

    M_1 = layer1_conv.fprop(mini_batch_embeddings)

    print "First level feature maps shape:", M_1.shape

    folding = SumFolding()

    M_1_folded = folding.fprop(M_1)

    print "Folded first level feature maps shape:", M_1_folded.shape

    k_max_pooling = KMaxPooling(k=4)

    M_1_pooled = k_max_pooling.fprop(M_1_folded)


    bias = Bias(
        n_input_dims=4,
        n_feature_maps=5,
    )

    M_1_pooled_biased = bias.fprop(M_1_pooled)

    relu = Relu()

    M_1_pooled_biased_relu = relu.fprop(M_1_pooled_biased)

