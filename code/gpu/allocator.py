__author__ = 'mdenil'

import pycuda.autoinit

global_device_pool = pycuda.tools.DeviceMemoryPool()
global_device_allocator = global_device_pool.allocate