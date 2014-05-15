__author__ = 'mdenil'

import numpy as np

import generic.model.model

import cpu.space
import cpu.model.layer


class CSM(generic.model.model.CSM, cpu.model.layer.Layer):
    Space = cpu.space.CPUSpace
    NDArray = np.ndarray

    def move_to_cpu(self):
        """
        This is a no-op here to make the interface between CPU and GPU models the same.
        """
        return self


class TaggedModelCollection(generic.model.model.TaggedModelCollection):
    def _zeros_like(self, x):
        return np.zeros_like(x)

    def pack(self):
        packed = []
        for m in self.tagged_models.itervalues():
            packed.append(m.pack())
        return np.concatenate(packed)

    def unpack(self, values):
        start = 0
        for model in self.tagged_models.itervalues():
            for param in model.params():
                end = start + param.size
                # assign to values in the array referenced by param
                param.ravel()[:] = values[start:end]
                start = end

    def move_to_cpu(self):
        """
        This is a no-op here to make the interface between CPU and GPU models the same.
        """
        return self