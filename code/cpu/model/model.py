__author__ = 'mdenil'

import numpy as np

import generic.model.model

import cpu.space
import cpu.model.layer


class CSM(generic.model.model.CSM, cpu.model.layer.Layer):
    Space = cpu.space.CPUSpace
    NDArray = np.ndarray