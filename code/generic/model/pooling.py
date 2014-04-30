__author__ = 'mdenil'


class SumFolding(object):
    def fprop(self, X, meta):
        working_space = meta['space_below']

        d = working_space.get_extent('d')
        assert d % 2 == 0

        X, working_space = working_space.transform(X, ['d', ('b', 'f', 'w')])

        X = self._fprop(X)
        working_space = working_space.with_extents(d=X.shape[0])

        meta['space_above'] = working_space
        return X, meta, {}

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']

        delta, working_space = working_space.broadcast(delta, d=2)

        meta['space_below'] = working_space
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)


class MaxFolding(object):
    def fprop(self, X, meta):
        working_space = meta['space_below']

        d = working_space.get_extent('d')
        assert d % 2 == 0

        X, working_space = working_space.transform(X, ('d', ('b', 'f', 'w')))

        Y, switches = self._fprop(X)
        switches_space = working_space
        working_space = working_space.with_extents(d=Y.shape[0])

        fprop_state = {
            'switches': switches,
            'switches_space': switches_space,
        }

        meta['space_above'] = working_space
        return Y, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):
        working_space = meta['space_above']
        switches = fprop_state['switches']
        switches_space = fprop_state['switches_space']

        delta, working_space = working_space.broadcast(delta, d=2)
        switches, switches_space = switches_space.transform(switches, working_space.axes)

        delta *= switches

        meta['space_below'] = working_space
        return delta, meta

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)