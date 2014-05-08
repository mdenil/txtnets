__author__ = 'mdenil'

import numpy as np
from collections import OrderedDict

import generic.model.model

import cpu.space
import cpu.model.layer


class CSM(generic.model.model.CSM, cpu.model.layer.Layer):
    Space = cpu.space.CPUSpace
    NDArray = np.ndarray


# Models need to have an order in there so grads will work
class TaggedModelCollection(object):
    def __init__(self, tagged_models):
        self.tagged_models = OrderedDict(tagged_models)

    def get_model(self, tag):
        return self.tagged_models[tag]

    def params(self):
        params = []
        for m in self.tagged_models.itervalues():
            params.extend(m.params())
        return params

    def full_grads_from_tagged_grads(self, tagged_grads):
        grads = []

        for tag, model in self.tagged_models.iteritems():
            if tag in tagged_grads:
                grads.extend(tagged_grads[tag])
            else:
                grads.extend(map(np.zeros_like, model.params()))

        return grads

    def pack(self):
        packed = []
        for m in self.tagged_models.itervalues():
            packed.append(m.pack())
        return np.concatenate(packed)
