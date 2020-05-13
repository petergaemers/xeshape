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
        areas_ = wv_matrix.sum(axis=1)
        areas = areas_[np.where(areas_>0)]
        wv_matrix = wv_matrix[np.where(areas_ > 0)]
        n_peaks, n_samples = wv_matrix.shape
        
        # Find time matrix to align the waveforms
        ##
        if self.alignment_options['method'] == 'max':
            # Align on maximum sample
            inds = np.argmax(wv_matrix, axis=1)

        elif self.alignment_options['method'] == 'area_fraction':
            # Align on point when fixed fraction of area is reached
            # First compute cumulative normalized waveforms
            # (we're always assuming area already sums to one, so we'll do so here too)
            cum_norm_wvs = np.cumsum(wv_matrix, axis=1)

            # Find index closest to point where target area fraction is reached
            inds = np.argmin(np.abs(cum_norm_wvs - self.alignment_options['area_fraction']), axis=1)

        else:
            raise ValueError('Unknown alignment method %s' % self.alignment_options['method'])

        # Construct the time matrix
        time_matrix = np.tile(self.ts,n_peaks).reshape(n_peaks,n_samples)
        time_shifts = self.ts[inds]
        time_matrix -= time_shifts[:,np.newaxis]

        ##
        # Compute average normalized waveform model
        ##
        h, _ = np.histogram(time_matrix, bins=self.t_edges, weights=wv_matrix)
        np.save('./tm.npy',time_matrix)
        np.save('./edges.npy',self.t_edges)
        np.save('./wvmatrix.npy',wv_matrix)
        np.save('./hist.npy',h)
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
        results['cog'] = np.average(time_matrix, weights=wv_matrix, axis=1)
        results['width_std'] = np.average((time_matrix - results['cog'][:,np.newaxis])**2,
                                          weights=wv_matrix, axis=1)**0.5

        # Width at fraction of area (1-sample resolution only, interpolation hard to vectorize...)
        results['fraction_area_width'] = compute_widths(wv_matrix)

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
        difs[wv_i, :] = wv_matrix[wv_i,:] - mm[i_noshift + time_shifts[wv_i]//dt,:]
    return difs

##
# Width computation
##

def compute_widths(wv_matrix, af_widths=(0.25, 0.5, 0.75, 0.9)):
    """Compute the width of the central area fraction for each waveform
    :param wv_matrix: (n_samples, n_peaks) waveform matrix (not necessarily aligned or normalized)
    :param af_widths: list/tuple/array of width fractions you're interested in. Default (0.25, 0.5, 0.75, 0.9)
    :returns: dictionary {area fraction: array (n_peaks) of widths)
    """
    af_widths = np.asarray(af_widths)
    n_peaks = wv_matrix.shape[0]

    # Which area fraction locations do we need?
    afs = np.sort(np.concatenate([(1-af_widths)/2, 1-(1-af_widths)/2]))

    # Compute them with numba
    results = np.zeros((n_peaks, len(afs)), dtype=np.float)
    _compute_afs(wv_matrix, afs, results)

    # Compute the widths from them
    afs = afs.tolist()
    width_results = dict()
    for i, afw in enumerate(af_widths):
        l_i = afs.index((1-afw)/2)
        r_i = afs.index(1-(1-afw)/2)
        width_results[afw] = results[:,r_i] - results[:,l_i]

    return width_results


@numba.jit(nopython=True)
def _compute_afs(wv_matrix, afs, results):
    # Compute area fraction locations for all waveforms in wv_matrix, put results in (n_afs, n_peaks) array results
    n_peaks = wv_matrix.shape[0]
    for wv_i in range(n_peaks):
        integrate_until_fraction(wv_matrix[wv_i,:], afs, results[wv_i,:])


# Below function copied from pax, with @numba.jit added back in (I hope numba's memory management is OK now)
@numba.jit(nopython=True)
def integrate_until_fraction(w, fractions_desired, results):
    """For array of fractions_desired, integrate w until fraction of area is reached, place sample index in results
    Will add last sample needed fractionally.
    eg. if you want 25% and a sample takes you from 20% to 30%, 0.5 will be added.
    Assumes fractions_desired is sorted and all in [0, 1]!
    """
    area_tot = w.sum()
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(w):
        # How much of the area is in this sample?
        fraction_this_sample = x/area_tot
        # Will this take us over the fraction we seek?
        # Must be while, not if, since we can pass several fractions_desired in one sample
        while fraction_seen + fraction_this_sample >= needed_fraction:
            # Yes, so we need to add the next sample fractionally
            area_needed = area_tot * (needed_fraction - fraction_seen)
            if x != 0:
                results[current_fraction_index] = i + area_needed/x
            else:
                results[current_fraction_index] = i
            # Advance to the next fraction
            current_fraction_index += 1
            if current_fraction_index > len(fractions_desired) - 1:
                return
            needed_fraction = fractions_desired[current_fraction_index]
        # Add this sample's area to the area seen, advance to the next sample
        fraction_seen += fraction_this_sample
    if needed_fraction == 1:
        results[current_fraction_index] = len(w)
    else:
        # Sorry, can't add the last fraction to the error message: numba doesn't allow it
        raise RuntimeError("Fraction not reached in waveform? What the ...?")


