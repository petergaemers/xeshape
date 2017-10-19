"""Useful functions for waveform template modelling
(even if you don't like the blueice integration)
"""

import numba
import numpy as np
from scipy import stats


##
# Simulation
##
def split_groups(x, n_x, distr=stats.randint(0, 100)):
    """Splits x into groups and return their histograms. Group size drawn from distr.
    Returns: integer array (n_x, n_groups)
    n_x: number of possible values in x. Assumed to be from 0 ... n_x - 1
    distr: distribution from scipy stats. Default stats.randint(0, 100)
    """
    # We want to exhaust the indices x as best as possible. Draw a generous amount of group sizes.
    n_groups_est = int(1.5 * len(x) / distr.mean())
    hits_per_group = distr.rvs(size=n_groups_est)

    result = np.zeros((n_x, n_groups_est), dtype=np.int)
    group_i = _split_groups(x, hits_per_group, result)
    return result[:,:group_i - 1]


@numba.jit(nopython=True)
def _split_groups(x, hits_per_group, result):
    # Inner loop of split_groups
    group_i = 0
    for i in x:
        if hits_per_group[group_i] == 0:
            group_i += 1
            continue
        result[i, group_i] += 1
        hits_per_group[group_i] -= 1
    return group_i


def shift(x, n):
    """Shift the array x n samples to the right, adding zeros to the left."""
    if n > 0:
        return np.pad(x, (n, 0), mode='constant')[:len(x)]
    else:
        return np.pad(x, (0, -n), mode='constant')[-len(x):]


##
# SPE simulation
##

def digitizer_response(pmt_pulse, offset, dt, samples_before, samples_after):
    """Get the output of  d/dt pmt_pulse(t) on a digitizer with sampling size dt.
    :param pmt_pulse: function that accepts a numpy array of times, return normalized integral of PMT pulse
    :param offset: Offset of the digitizer time grid (number in [0, dt])
    :param dt: sampling size (ns)
    :param samples_before: number of samples before the maximum to simulate
    :param samples_after: number of samples after the maximum to simulate
    """
    return np.diff(pmt_pulse(
        np.linspace(
            - offset - samples_before * dt,
            - offset + samples_after * dt,
            1 + samples_before + samples_after)
    )) / dt




##
# Data processing
##

def aligned_time_matrix(ts, wv_matrix, method='max', area_fraction=0.5):
    """Return time matrix that would align waveforms im wv_matrix
      ts: array of (n_samples), center times of digitizer bins
      wv_matrix: (n_waveforms, n_samples) amplitudes
    """
    n_s1 = wv_matrix.shape[1]

    if method == 'max':
        # Align on maximum sample
        inds = np.argmax(wv_matrix, axis=0)

    elif method == 'area_fraction':
        # Align on point when fixed fraction of area is reached
        # First compute cumulative normalized waveforms
        cum_norm_wvs = np.cumsum(wv_matrix, axis=1)
        cum_norm_wvs /= cum_norm_wvs[:, -1][:, np.newaxis]

        # Find index closest to point where target area fraction is reached
        inds = np.argmin(np.abs(cum_norm_wvs - area_fraction))

    else:
        raise ValueError('Unknown alignment method %s' % method)

    # Construct the time matrix
    time_matrix = np.repeat(ts, n_s1).reshape(wv_matrix.shape)
    t_shift = ts[inds]
    time_matrix -= t_shift[np.newaxis, :]
    return time_matrix, t_shift


def average_pulse(time_matrix, wv_matrix, ts):
    """Return average pulse, given time and waveform matrices
    Both are (n_waveforms, n_samples) matrices, see simulate_s1_pulse
    """
    # Compute edges of digitizer time bins
    # Not really necessary if an integer-sample granularity is used (then centers - epsilon suffices)
    # But maybe one day we'll have more general alignment methods.
    dt = np.diff(ts)[0]
    t_edges = np.concatenate([[ts[0] - dt/2], ts + dt/2])

    h, _ = np.histogram(time_matrix, bins=t_edges, weights=wv_matrix)
    h = h.astype(np.float)
    h /= h.sum()
    return h
