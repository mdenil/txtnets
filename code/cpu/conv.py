__author__ = 'mdenil'

import numpy as np
import pyfftw

pyfftw.interfaces.cache.enable()

def fftconv1d(X, K, mode='full', n_threads=1):
    # always convolves along axis=1

    assert mode in ['full', 'valid']

    xw = X.shape[1]
    kw = K.shape[1]

    if mode == 'valid':
        assert xw >= kw

    # pad

    if xw >= kw:
        X = np.concatenate([X, np.zeros((X.shape[0], kw - 1))], axis=1)

        K = np.concatenate([
            K, np.zeros((K.shape[0], xw - 1)) # (xw - kw) + kw - 1
            ], axis=1)

    else:
        X = np.concatenate([
            X, np.zeros((X.shape[0], 2*kw - xw - 1)) # (kw - xw) + kw - 1
            ], axis=1)

        K = np.concatenate([K, np.zeros((K.shape[0], kw - 1))], axis=1)

    # compute

    X = pyfftw.interfaces.numpy_fft.fft(X, axis=1, threads=n_threads)
    K = pyfftw.interfaces.numpy_fft.fft(K, axis=1, threads=n_threads)
    X = pyfftw.interfaces.numpy_fft.ifft(X*K, axis=1, threads=n_threads).real

    # trim

    if kw > xw and mode == 'full':
        X = X[:, :xw + kw - 1]
    elif mode == 'valid':
        X = X[:, kw-1:-kw+1]

    return X
