__author__ = 'mdenil'

import generic.model.layer
import cpu.model.layer

# These layers do nothing at all.  They exist so that when you move a model with mixed GPU/CPU layers to the CPU the
# structure will not change.  I'm not convinced this is a good idea, but it's what I'm going to go with for now.


class HostToDevice(generic.model.layer.NoOpLayer, cpu.model.layer.Layer):
    pass


class DeviceToHost(generic.model.layer.NoOpLayer, cpu.model.layer.Layer):
    pass

