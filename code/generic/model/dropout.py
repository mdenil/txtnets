__author__ = 'mdenil'


class Dropout(object):
    def __init__(self, axes, dropout_rate):
        self.axes = axes
        self.dropout_rate = dropout_rate

    def fprop(self, X, meta):
        X = X
        X_space = meta['space_below']

        mask, mask_space = self._get_mask(X_space.get_extents(self.axes))

        extents = X_space.extents
        for ax in self.axes:
            extents.pop(ax)

        mask, mask_space = mask_space.transform(mask, X_space.axes, **extents)

        Y = X * mask

        meta['space_above'] = X_space

        fprop_state = {
            'mask': mask,
            'mask_space': mask_space,
        }

        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        delta_space = meta['space_above']
        mask = fprop_state['mask']
        mask_space = fprop_state['mask_space']

        mask, mask_space = mask_space.transform(mask, delta_space.axes)

        delta *= mask

        meta['space_below'] = meta['space_above']

        return delta, meta

    def __repr__(self):
        return "{}(p={}, axes={})".format(
            self.__class__.__name__,
            self.dropout_rate,
            self.axes)
