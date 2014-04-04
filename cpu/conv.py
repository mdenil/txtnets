__author__ = 'mdenil'

import numpy as np
import pyfftw

pyfftw.interfaces.cache.enable()

def fftconv1d(X, K, mode='full', n_threads=1):
    assert mode in ['full', 'valid']

    xw = X.shape[1]
    kw = K.shape[1]
    # pad
    if xw >= kw:
        p = int((xw - kw) / 2)
        pairity = (xw - kw) % 2
        K = np.concatenate([
                               np.zeros((K.shape[0], p)), K, np.zeros((K.shape[0], p + pairity))
                           ], axis=1)
    else:
        p = int((kw - xw) / 2)
        pairity = (kw - xw) % 2
        X = np.concatenate([
                               np.zeros((X.shape[0], p)), X, np.zeros((X.shape[0], p + pairity))
                           ], axis=1)

    # compute

    X = np.concatenate([X, np.zeros_like(X)], axis=1)
    K = np.concatenate([K, np.zeros_like(K)], axis=1)

    X = pyfftw.interfaces.numpy_fft.fft(X, axis=1, threads=n_threads)
    K = pyfftw.interfaces.numpy_fft.fft(K, axis=1, threads=n_threads)
    X = pyfftw.interfaces.numpy_fft.ifft(X*K, axis=1, threads=n_threads).real

    # trim

    if mode == 'full':
        X = X[:, p:-1-p-pairity]
    else:
        X = X[:, p+kw-1:-p-pairity-kw]

    return X
