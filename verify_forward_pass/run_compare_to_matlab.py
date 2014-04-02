__author__ = 'mdenil'

import numpy as np
import pyprind
import scipy.io
import scipy.optimize

from cpu import model


def load_testing_model(file_name):
    model_data = scipy.io.loadmat(file_name)

    CR_E = np.ascontiguousarray(np.transpose(model_data['CR_E']))
    # I want to convolve along rows because that makes sense for C-ordered arrays
    # the matlab code also convolves along rows, so I need to not transpose the convolution filters
    CR_1 = np.ascontiguousarray(model_data['CR_1'])
    CR_1_b = np.ascontiguousarray(np.transpose(model_data['CR_1_b']))
    CR_Z = np.ascontiguousarray(model_data['CR_Z'])

    embedding = model.embedding.WordEmbedding(
        dimension=CR_E.shape[1],
        vocabulary_size=CR_E.shape[0])
    assert CR_E.shape == embedding.E.shape
    embedding.E = CR_E

    conv = model.transfer.SentenceConvolution(
        n_feature_maps=5,
        kernel_width=2,
        n_input_dimensions=42)
    assert conv.W.shape == CR_1.shape
    conv.W = CR_1

    bias = model.transfer.Bias(
        n_input_dims=21,
        n_feature_maps=5)
    bias.b = CR_1_b.reshape(bias.b.shape)

    softmax = model.transfer.Softmax(
        n_classes=6,
        n_input_dimensions=420)
    softmax.W = CR_Z[:,:-1]
    softmax.b = CR_Z[:,-1].reshape((-1,1))

    csm = model.model.CSM(
        input_axes=['b', 'w'],
        layers=[
            embedding,
            conv,
            model.pooling.SumFolding(),
            model.pooling.KMaxPooling(k=7),
            bias,
            model.nonlinearity.Tanh(),
            softmax,
            ],
        )

    return csm

def run():
    tol = 1e-6

    data_file_name = "verify_forward_pass/data/SENT_vec_1_emb_ind_bin.mat"
    data = scipy.io.loadmat(data_file_name)
    embedding_dim = 42
    batch_size = 40
    vocabulary_size = int(data['size_vocab'])
    max_epochs = 1
    train = data['train'] - 1
    train_sentence_lengths = data['train_lbl'][:,1]
    train_labels = data['train_lbl'][:,0] - 1 # -1 to switch to zero based indexing
    max_sentence_length = data['train'].shape[1]
    csm = load_testing_model("verify_forward_pass/data/debugging_model_params.mat")
    n_batches_per_epoch = int(data['train'].shape[0] / batch_size)
    matlab_results = scipy.io.loadmat("verify_forward_pass/data/batch_results_first_layer.mat")['batch_results']
    progress_bar = pyprind.ProgPercent(n_batches_per_epoch)
    total_errs_big = 0
    total_errs_small = 0
    total_checked = 0

    cost_function = model.cost.CrossEntropy()

    for batch_index in xrange(n_batches_per_epoch):

        minibatch = train[batch_index*batch_size:(batch_index+1)*batch_size]
        minibatch_labels = train_labels[batch_index*batch_size:(batch_index+1)*batch_size]
        # don't keep these as bools because -True == False for numpy bools (not python bools)
        minibatch_labels = np.equal.outer(minibatch_labels, np.arange(6))#.astype(np.float64)

        meta = {'lengths': train_sentence_lengths[batch_index*batch_size:(batch_index+1)*batch_size]}

        out = csm.fprop(minibatch, meta)

        if not np.allclose(out, matlab_results[batch_index]):
            n_new_errs_small = np.sum(np.abs(out - matlab_results[batch_index]) > tol)
            n_new_errs_big = np.sum(np.abs(out - matlab_results[batch_index]) > np.sqrt(tol))
            total_errs_small += n_new_errs_small
            total_errs_big += n_new_errs_big
            # print "\nFailed batch {}. Max abs err={}.  There are {} errors larger than {}.".format(
            #     batch_index,
            #     np.max(np.abs(out - matlab_results[batch_index])),
            #     n_new_errs,
            #     tol)
        total_checked += out.size

        # out = model.tools.permute_axes(out, current_axes=csm.output_axes, desired_axes=cost_function.input_axes)

        # here: out == Z

        # cost, _ = cost_function.fprop(X=out, Y=minibatch_labels)
        #
        # print scipy.optimize.check_grad(lambda x: cost_function.fprop(X=x.reshape(40,6), Y=minibatch_labels)[0].ravel(),
        #                           lambda x: cost_function.bprop(X=x.reshape(40,6), Y=minibatch_labels)[0].ravel(),
        #                           out.ravel())
        #
        # b1, _ = cost_function.bprop(X=out, Y=minibatch_labels)
        #
        # softmax_layer = csm.layers[-1]
        #
        # b1_a = model.tools.permute_axes(-b1, current_axes=cost_function.input_axes, desired_axes=softmax_layer.output_axes)
        #
        # b2, _ = softmax_layer.bprop(b1_a)

        progress_bar.update()

    print "Total errs > {}: {} ({} %)".format(tol, total_errs_small, float(total_errs_small) / total_checked * 100.0)
    print "Total errs > {}: {} ({} %)".format(np.sqrt(tol), total_errs_big, float(total_errs_big) / total_checked * 100.0)


if __name__ == "__main__":
    run()