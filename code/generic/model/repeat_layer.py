__author__ = 'mdenil'


class RepeatLayer(object):
    def __init__(self, model, stop_condition):
        self.model = model
        self.stop_condition = stop_condition

    def fprop(self, X, meta):

        fprop_states = []

        while not self.stop_condition(X, meta):
            X, meta, fprop_state = self.model.fprop(
                X,
                meta=dict(meta),
                return_state=True)

            fprop_states.append(fprop_state)

        fprop_state = {
            'fprop_states': fprop_states,
        }

        return X, meta, fprop_state

    def bprop(self, delta, meta, fprop_state):

        fprop_states = fprop_state['fprop_states']

        for state in reversed(fprop_states):

            delta, meta = self.model.bprop(
                delta,
                fprop_state=state,
                meta=dict(meta),
                return_state=True)

            meta['space_above'] = meta['space_below']
            del meta['space_below']

        return delta, meta


    def grads(self, delta, meta, fprop_state):
        meta = dict(meta)


        fprop_states = fprop_state['fprop_states']

        # this layer could have been applied zero times, but
        # we still need to return gradients
        if not fprop_states:
            return map(self._zeros_like, self.model.params())

        grads = []

        for state in reversed(fprop_states):

            new_grads = self.model.grads(
                delta,
                meta=dict(meta),
                fprop_state=state)

            delta, meta = self.model.bprop(
                delta,
                meta=dict(meta),
                fprop_state=state,
                return_state=True)

            meta['space_above'] = meta['space_below']

            if not grads:
                grads = new_grads
            else:
                for g, ng in zip(grads, new_grads):
                    g += ng

        return grads


    def params(self):
        return self.model.params()


    def move_to_cpu(self):
        from gpu.model.host_device_component_mapping import get_cpu_analog
        cpu_class = get_cpu_analog(self.__class__)

        return cpu_class(
            model=self.model.move_to_cpu(),
            stop_condition=self.stop_condition)


    def __repr__(self):
        return "{}(model={})".format(
            self.__class__.__name__,
            self.model)
