from copy import deepcopy
import numpy as np
import numba

from .utils import model_matrix, centers_to_edges, default_alignment_options


def simulate_peak_waveforms(photon_times, ts, spe_pulse, n_dist, n_offsets=4, alignment_options=None):
    """Simulate matrix (n_samples, n_peaks) of peak waveforms
    :param photon_times: 1d array of photon detection times (since e.g. the interaction origin) to draw from
    :param ts: centers of digitizer time grid
    :param spe_pulse: function, taking 1d array of times to
    :param n_dist: scipy.stats distribution of number of photons per peak.
    :param n_offsets: number of offsets in digitizer grid to consider as possible photon arrival times. Default 4.
    :param alignment_options: dictionary of alignment options:
        'method' ('max' or 'area_fraction') and 'area_fraction' (number between 0 and 1)
    :return: 1d numpy array of (n_samples, n_peaks) with waveforms. n_samples = len(ts). Not aligned. Not normalized.
    """
    if alignment_options is None:
        alignment_options = deepcopy(default_alignment_options)
    t_edges = centers_to_edges(ts)
    n_samples = len(ts)
    dt = ts[1] - ts[0]

    ##
    # Convert photon detection times to indices
    ##
    indices = np.searchsorted(t_edges, photon_times)
    # Remove times beyond t range
    indices = indices[True ^ ((indices == 0) | (indices == len(t_edges)))]
    # Convert to index in bin centers
    indices -= 1

    # TODO: gain variation simulation

    ##
    # Build instruction matrix, simulate waveforms
    ##
    model_ms = []
    for offset in np.linspace(0, dt, n_offsets + 1)[:-1]:
        # Make matrix (n_samples, n_shifts = n_samples) of single PE waveforms
        spe_y = digitizer_response(spe_pulse,
                                   offset=offset,
                                   dt=10,
                                   samples_before=len(ts)//2,
                                   samples_after=len(ts)//2 + 1)

        model_m, _ = model_matrix(ts, spe_y, **alignment_options)
        model_ms.append(model_m)

    # Stack to matrix of (n_samples, n_shifts * n_samples)
    model_m = np.hstack(model_ms)

    # Divide indices randomly over different SPE offsets
    indices += np.random.randint(0, n_offsets, size=len(indices)) * n_samples

    # Divide indices over peaks
    index_matrix = split_groups(indices, n_x=n_samples * n_offsets, n_dist=n_dist)

    # Create and return waveform matrix
    return np.dot(model_m, index_matrix)


def split_groups(x, n_x, n_dist):
    """Splits x into groups and return their histograms. Group size drawn from distr.
    Returns: integer array (n_x, n_groups)
    n_x: number of possible values in x. Assumed to be from 0 ... n_x - 1
    n_dist: distribution from scipy stats.
    """
    # We want to exhaust the indices x as best as possible. Draw a generous amount of group sizes.
    n_groups_est = int(1.5 * len(x) / n_dist.mean())
    hits_per_group = n_dist.rvs(size=n_groups_est)
    hits_per_group = hits_per_group.astype(np.int)

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
