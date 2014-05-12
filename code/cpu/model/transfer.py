__author__ = 'mdenil'

import numpy as np

from cpu import space
from cpu import conv
from cpu.model import layer

import generic.model.transfer

class Linear(generic.model.transfer.Linear, layer.Layer):

    def _fprop(self, X):
        Y = np.dot(X, self.W)
        Y_space = space.CPUSpace.infer(Y, ('b', 'd'))
        return Y, Y_space

    def _bprop(self, delta):
        return np.dot(delta, self.W.T)

    def _grads(self, X, delta):
        return [np.dot(X.T, delta)]


class Softmax(generic.model.transfer.Softmax, layer.Layer):

    def _fprop(self, X, X_space):
        Y = np.exp(np.dot(X, self.W) + self.b)
        Y /= np.sum(Y, axis=1, keepdims=True)
        return Y

    def _bprop(self, delta, Y):
        return np.dot(delta * Y * (1-Y), self.W.T)

    def _grads(self, delta, X, Y):
        delta = delta * Y * (1-Y)
        grad_W = np.dot(X.T, delta)
        grad_b = delta.sum(axis=0).reshape(self.b.shape)
        return [grad_W, grad_b]


class SentenceConvolution(generic.model.transfer.SentenceConvolution, layer.Layer):
    def __init__(self, *args, **kwargs):
        super(SentenceConvolution, self).__init__(*args, **kwargs)

    def _fprop(self, X, X_space):
        K, _ = self._kernel_space.broadcast(np.fliplr(self.W), b=X_space.get_extent('b'))
        X = conv.fftconv1d(X, K, n_threads=self.n_threads)

        X_space = X_space.with_extents(w=X.shape[1])

        X, X_space = X_space.transform(X, (('d', 'b', 'f'), 'c', 'w'))
        X = X.sum(axis=X_space.axes.index('c'))
        X_space = X_space.without_axes('c')

        return X, X_space

    def _bprop(self, delta, delta_space):
        K, _ = self._kernel_space.broadcast(self.W, b=delta_space.get_extent('b'))

        delta = conv.fftconv1d(delta, K, n_threads=self.n_threads, mode='valid')
        delta_space = delta_space.with_extents(w=delta.shape[1])

        delta, delta_space = delta_space.transform(delta, delta_space.folded_axes)
        delta = delta.sum(axis=delta_space.axes.index('f'))
        delta_space = delta_space.without_axes('f')
        delta_space = delta_space.rename_axes(c='f')

        return delta, delta_space

    def _grads(self, delta, delta_space, X):
        grad_W = conv.fftconv1d(np.fliplr(delta), X, n_threads=self.n_threads, mode='valid')
        grad_W_space = delta_space.with_extents(w=grad_W.shape[1])

        grad_W, grad_W_space = grad_W_space.transform(grad_W, grad_W_space.folded_axes)

        grad_W = grad_W.sum(axis=grad_W_space.axes.index('b'))
        grad_W_space = grad_W_space.without_axes('b')
        grad_W, grad_W_space = grad_W_space.transform(grad_W, [('d', 'b', 'f', 'c'), 'w'])

        return [grad_W]


class Bias(generic.model.transfer.Bias, layer.Layer):

    def _fprop(self, X, X_space):
        return X + self.b[:, np.newaxis, :, np.newaxis]

    # bprop is a no-op

    def _grads(self, delta, delta_space):
        grad_b = delta_space.fold(delta)
        grad_b_space = delta_space.folded()
        grad_b = grad_b.sum(axis=grad_b_space.axes.index('b'))
        grad_b_space = grad_b_space.without_axes('b')
        grad_b = grad_b.sum(axis=grad_b_space.axes.index('w'))
        grad_b_space = grad_b_space.without_axes('w')
        grad_b, grad_b_space = grad_b_space.transform(grad_b, ('d', 'f'))
        return [grad_b]


class AxisReduction(layer.Layer):
    def __init__(self, axis):
        super(AxisReduction, self).__init__()
        self.axis = axis

    def fprop(self, X, meta):
        working_space = meta['space_below']

        X = working_space.fold(X)
        working_space = working_space.folded()

        if not self.axis in working_space.axes:
            raise ValueError("Cannot reduce along axis={} because it does not appear in axes={}".format(
                self.axis, working_space.axes))

        target_axis_index= working_space.axes.index(self.axis)
        fprop_state = {
            'expanded_size': X.shape[target_axis_index]
        }

        X = np.sum(X, axis=target_axis_index)
        working_space = working_space.without_axes(self.axis)

        meta['space_above'] = working_space

        return X, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']

        delta = working_space.fold(delta)
        working_space = working_space.folded()

        if self.axis in working_space.axes:
            if working_space.get_extent(self.axis) != 1:
                raise ValueError("Cannot introduce axis={}, already present in axes={}".format(
                    self.axis, working_space.axes))
            else:
                new_axes = working_space.axes
        else:
            new_axes = working_space.axes = (self.axis,)

        delta, working_space = working_space.transform(
            delta,
            new_axes,
            **{self.axis: fprop_state['expanded_size']})

        meta['space_below'] = working_space

        return delta, meta