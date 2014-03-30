__author__ = 'mdenil'

import numpy as np
import pyprind

from cpu import model


def load_testing_model(file_name):
    model_data = scipy.io.loadmat(file_name)

    CR_E = np.ascontiguousarray(np.transpose(model_data['CR_E']))
    # I want to convolve along rows because that makes sense for C-ordered arrays
    # the matlab code also convolves along rows, so I need to not transpose the convolution filters
    CR_1 = np.ascontiguousarray(model_data['CR_1'])
    CR_1_b = np.ascontiguousarray(np.transpose(model_data['CR_1_b']))

    embedding = model.embedding.WordEmbedding(
        dimension=CR_E.shape[1],
        vocabulary_size=CR_E.shape[0])
    assert CR_E.shape == embedding.E.shape
    embedding.E = CR_E

    conv = model.transfer.SentenceConvolution(
        n_feature_maps=5,
        kernel_width=6,
        n_input_dimensions=42)
    assert conv.W.shape == CR_1.shape
    conv.W = CR_1

    bias = model.transfer.Bias(
        n_input_dims=21,
        n_feature_maps=5)
    bias.b = CR_1_b.reshape(bias.b.shape)

    csm = model.model.CSM(
        input_axes=['b', 'w'],
        layers=[
            embedding,
            conv,
            model.pooling.SumFolding(),
            model.pooling.KMaxPooling(k=4),
            bias,
            model.nonlinearity.Tanh(),
            ],
        )

    return csm


if __name__ == "__main__":
    import scipy.io

    data_file_name = "verify_forward_pass/data/SENT_vec_1_emb_ind_bin.mat"
    data = scipy.io.loadmat(data_file_name)

    embedding_dim = 42
    batch_size = 40
    vocabulary_size = int(data['size_vocab'])
    max_epochs = 1

    train = data['train'] - 1
    train_sentence_lengths = data['train_lbl'][:,1]

    max_sentence_length = data['train'].shape[1]

    csm = load_testing_model("verify_forward_pass/data/debugging_model_params.mat")

    n_batches_per_epoch = int(data['train'].shape[0] / batch_size)

    matlab_results = scipy.io.loadmat("verify_forward_pass/data/batch_results_first_layer.mat")['batch_results']

    progress_bar = pyprind.ProgPercent(n_batches_per_epoch)

    for batch_index in xrange(n_batches_per_epoch):

        minibatch = train[batch_index*batch_size:(batch_index+1)*batch_size]

        meta = {'lengths': train_sentence_lengths[batch_index*batch_size:(batch_index+1)*batch_size]}

        out = csm.fprop(minibatch, meta)

        if not np.allclose(out, matlab_results[batch_index]):
            print "\nBatch {}: Max abs err={}. There are {} errors larger than 1e-2.".format(
                batch_index,
                np.max(np.abs(out - matlab_results[batch_index])),
                np.sum(np.abs(out - matlab_results[batch_index]) > 1e-2))

        progress_bar.update()



