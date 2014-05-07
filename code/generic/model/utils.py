__author__ = 'mdenil'


class ModelEvaluator(object):
    """
    A utility class for holding the state of a model evaluation.  State management for fprop_state is wrapped up in
    this class so you don't need to keep track of the boilerplate yourself.  The saved state always comes from the
    most recent evaluation of fprop.

    You can put the same model into multiple ModelEvaluators and states will be saved separately.
    """
    def __init__(self, model):
        self.model = model
        self.state = None
        self.meta = None

    def fprop(self, data_provider):
        X, meta = data_provider.next_batch()
        Y, self.meta, self.state = self.model.fprop(X, meta=dict(meta), return_state=True)
        return Y

    def grads(self, delta):
        return self.model.grads(delta, meta=dict(self.meta), fprop_state=dict(self.state))

    def bprop(self, delta):
        return self.model.bprop(delta, meta=dict(self.meta), fprop_state=dict(self.state))

    def clear_state(self):
        self.state = None
        self.meta = None