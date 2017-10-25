from copy import deepcopy
import numba
import numpy as np

from .utils import centers_to_edges, model_matrix, default_alignment_options


class MatrixProcessor:
    """Processor for matrices of waveforms"""

    def __init__(self, ts, alignment_options=None):
        """Create processor for processing matrices of peak waveforms.
        :param ts: Centers of time grid. I may have assumed you have one center at 0.
        :param alignment_options: see [todo elsewhere]
        """
        self.ts = ts
        self.dt = ts[1] - ts[0]
        self.t_edges = centers_to_edges(self.ts)

        if alignment_options is None:
            self.alignment_options = deepcopy(default_alignment_options)
        else:
            self.alignment_options = alignment_options

    def process(self, wv_matrix, full=False, waveform_model=None):
        """Return dictionary of results from processing (n_samples, n_peaks) matrix of waveforms.

        Keys in returned dictionary:
            wv_matrix: the wv_matrix you passed in (normalized if you didn't already do it)
            areas: waveform integral (in the units you passed in)
            n_samples: wv_matrix.shape[0]
            n_peaks: wv_matrix.shape[1]
            time_shifts: time shifts that would align each waveform
            time_matrix: time matrix that would align each waveform
            average_waveform: average normalized aligned waveform

        if full=True, (default False) returns extra quantities you might like (obviously takes more time):
            width_std: array of waveform widths (measured by standard deviation)
            fraction_area_width: dictionary {area_fraction: widths}
            model_matrix: model matrix of template. See model_matrix documentation
            i_noshift: index in model matrix corresponding to no shift. See model_matrix documentation
            data_model_difs: matrix of same shape as wv_matrix containing data - model (aligned properly)
            data_model_ks: KS-distances between waveforms and model

        Waveform model (for data_model_difs etc.) can be specified by waveform_model argument. If not given,
        will use the data-extracted mean template.

        wv_matrix will be normalized (area = 1 for each waveform) once you pass it in.
        """
        # Normalize the waveforms
        areas = wv_matrix.sum(axis=0)
        wv_matrix /= areas[np.newaxis, :]
        n_samples, n_peaks = wv_matrix.shape

        ##
        # Find time matrix to align the waveforms
        ##
        if self.alignment_options['method'] == 'max':
            # Align on maximum sample
            inds = np.argmax(wv_matrix, axis=0)

        elif self.alignment_options['method'] == 'area_fraction':
            # Align on point when fixed fraction of area is reached
            # First compute cumulative normalized waveforms
            # (we're always assuming area already sums to one, so we'll do so here too)
            cum_norm_wvs = np.cumsum(wv_matrix, axis=0)

            # Find index closest to point where target area fraction is reached
            inds = np.argmin(np.abs(cum_norm_wvs - self.alignment_options['area_fraction']), axis=0)

        else:
            raise ValueError('Unknown alignment method %s' % self.alignment_options['method'])

        # Construct the time matrix
        time_matrix = np.repeat(self.ts, n_peaks).reshape(wv_matrix.shape)
        time_shifts = self.ts[inds]
        time_matrix -= time_shifts[np.newaxis, :]

        ##
        # Compute average normalized waveform model
        ##
        h, _ = np.histogram(time_matrix, bins=self.t_edges, weights=wv_matrix)
        h = h.astype(np.float)
        h /= h.sum()

        results = dict(wv_matrix=wv_matrix, n_samples=n_samples, n_peaks=n_peaks,
                       areas=areas,
                       time_matrix=time_matrix,
                       time_shifts=time_shifts,
                       average_waveform=h)
        if not full:
            return results

        ##
        # Compute width, other waveform properties later?
        ##
        results['cog'] = np.average(time_matrix, weights=wv_matrix, axis=0)
        results['width_std'] = np.average((time_matrix - results['cog'][np.newaxis, :])**2,
                                          weights=wv_matrix, axis=0)**0.5

        # Width at fraction of area (1-sample resolution only, interpolation hard to vectorize...)
        results['fraction_area_width'] = dict()
        cum_wvs = np.cumsum(wv_matrix, axis=0)
        for af in [0.5, 0.8]:
            res = []
            for f in [(1-af)/2, 1-(1-af)/2]:
                res.append(np.argmin(np.abs(cum_wvs - f), axis=0))
            results['fraction_area_width'][af] = (res[1] - res[0]) * self.dt

        ##
        # Compute aligned differences between waveforms and template
        ##
        if waveform_model is not None:
            # Use a pre-specified waveform model for computing data-model differences
            h = waveform_model

        results['model_matrix'], results['i_noshift'] = model_matrix(self.ts, h, **self.alignment_options)
        results['data_model_difs'] = difs = data_model_difs(wv_matrix,
                                                            dt=self.dt,
                                                            time_shifts=time_shifts,
                                                            mm=results['model_matrix'],
                                                            i_noshift=results['i_noshift'])
        results['data_model_ks'] = np.max(np.abs(np.cumsum(difs, axis=0)), axis=0)

        return results


@numba.jit(nopython=True)
def data_model_difs(wv_matrix, dt, time_shifts, mm, i_noshift):
    """Return data - model, for each waveform in wv_matrix, considering alignment properly
    Returns matrix of same shape as wv_matrix.
    """
    n_wvs = wv_matrix.shape[1]
    difs = np.zeros_like(wv_matrix)
    for wv_i in range(n_wvs):
        difs[:, wv_i] = wv_matrix[:, wv_i] - mm[:, i_noshift + time_shifts[wv_i]//dt]
    return difs


