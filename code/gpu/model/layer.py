__author__ = 'mdenil'


class Layer(object):
    def __init__(self, *args, **kwargs):
        pass

    def params(self, *args, **kwargs):
        return []

    def grads(self, *args, **kwargs):
        return []

    def pack(self):
        raise NotImplementedError()

    def unpack(self, values):
        raise NotImplementedError()

    def __repr__(self):
        return "{}()".format(
            self.__class__.__name__)
