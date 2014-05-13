__author__ = 'mdenil'

from generic.space import Space
from generic.space import _canonical_axes_description
from generic.space import _fold_axes

import cpu.space

import gpu.utils


class GPUSpace(Space):
    def __init__(self, axes, extents=None):
        super(GPUSpace, self).__init__(axes, extents)

    @classmethod
    def from_cpu(cls, X, cpu_space):
        X = gpu.utils.cpu_to_gpu(X)
        gpu_space = GPUSpace(cpu_space.axes, cpu_space.extents)
        return X, gpu_space

    def to_cpu(self, X):
        X = gpu.utils.gpu_to_cpu(X)
        cpu_space = cpu.space.CPUSpace(self.axes, self.extents)
        return X, cpu_space

    def fold(self, X):
        Y = X.reshape(self.folded_shape)
        return Y

    def unfold(self, X):
        return X.reshape(self.shape)

    def transpose(self, X, new_axes):
        self.check_compatible_shape(X)

        new_axes = _canonical_axes_description(new_axes)
        new_folded_axes = _fold_axes(new_axes)

        permutation = [self.folded_axes.index(axis) for axis in new_folded_axes]

        # check if this permutation only moves length 1 axes, if so we can just shuffle metadata
        new_space = self.transposed(new_axes)

        self_real_shape = tuple(s for s in self.folded_shape if s != 1)
        new_real_shape = tuple(s for s in new_space.folded_shape if s != 1)

        if self_real_shape == new_real_shape:
            X = new_space.unfold(X)
        else:
            X = self.fold(X)
            X = gpu.utils.transpose(X, permutation)
            new_space = self.transposed(new_axes)
            X = new_space.unfold(X)

        return X, new_space

    def broadcast(self, X, **replicas):
        self.check_compatible_shape(X)

        expanded_space = self.with_axes(replicas.keys())

        X = expanded_space.fold(X)
        for axis, times in replicas.iteritems():
            expanded_space = expanded_space.with_extents(**{axis: expanded_space.get_extent(axis) * times})

        X = gpu.utils.broadcast(X, expanded_space.folded_shape)
        X = expanded_space.unfold(X)

        return X, expanded_space

    # def broadcast(self, X, **replicas):
    #     self.check_compatible_shape(X)
    #
    #     expanded_space = self.with_axes(replicas.keys())
    #
    #     X = expanded_space.fold(X)
    #     for axis, times in replicas.iteritems():
    #         # TODO: this is kind of inefficient because stack permutes the axes twice internally
    #         X = gpu.utils.stack(X, times=times, axis=expanded_space.folded_axes.index(axis))
    #         expanded_space = expanded_space.with_extents(**{axis: expanded_space.get_extent(axis) * times})
    #     X = expanded_space.unfold(X)
    #
    #     return X, expanded_space
