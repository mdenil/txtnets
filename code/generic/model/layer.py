__author__ = 'mdenil'


class NoOpLayer(object):
    def fprop(self, X, meta):
        meta['space_above'] = meta['space_below']
        return X, meta, {}

    def bprop(self, delta, meta, fprop_state):
        meta['space_below'] = meta['space_above']
        return delta, meta