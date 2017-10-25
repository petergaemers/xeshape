"""Useful functions for waveform template modelling
(even if you don't like the blueice integration)
"""
import numpy as np

default_alignment_options = dict(method='max', area_fraction=0.1)

def centers_to_edges(ts):
    dt = ts[1] - ts[0]
    return np.concatenate([[ts[0] - dt/2], ts + dt/2])

def model_matrix(ts, y, method='max', area_fraction=0.1):
    """Return matrix of waveform models y shifted by various amounts

    Returns 2-tuple:
     - (n_samples, n_shifts) matrix of waveform models
     - index along second axis corresponding to no shift (under the indicated alignment method)
    """
    if method == 'max':
        i_noshift = np.where(ts == 0)[0][0]
    else:
        i_noshift = np.argmin(np.abs(np.cumsum(y) - area_fraction))

    return np.vstack([shift(y, i - i_noshift) for i in range(len(ts))]).T, i_noshift

def shift(x, n):
    """Shift the array x n samples to the right, adding zeros to the left."""
    if n > 0:
        return np.pad(x, (n, 0), mode='constant')[:len(x)]
    else:
        return np.pad(x, (0, -n), mode='constant')[-len(x):]
