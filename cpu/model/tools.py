__author__ = 'mdenil'

import numpy as np

def permute_axes(X, current_axes, desired_axes):
    """
    Axis types:

    b: batch_size, index over data
    w: sentence_length, index over words
    f: n_feature_maps, index over feature maps
    d: n_input_dimensions, index over dimensions of the input in the non-sentence direction
    """
    if current_axes == desired_axes:
        return X

    assert set(current_axes) == set(desired_axes)

    X = np.transpose(X, [current_axes.index(d) for d in desired_axes])

    return X