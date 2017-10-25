"""Blueice extensions for waveform template shape matching

Always create a likelihood with only one source. Don't pass any arguments except a name to the source.
This is clunky. Maybe weave another layer of interface around it to make it nice...

"""
import numpy as np
import blueice as bi
from scipy import stats

from xeshape import utils


def _align_and_average(ts, config, wv_matrix):
    # "method" used in both WaveformSource and WaveformLikelihood to align and average based on config
    t_matrix, _ = utils.aligned_time_matrix(ts, wv_matrix,
                                            method=config['alignment_method'],
                                            area_fraction=config['alignment_area_fraction'])

    return utils.average_pulse(t_matrix, wv_matrix, ts)


class WaveformSource(bi.Source):
    waveform_model = None

    def __init__(self, config, *args, **kwargs):
        defaults = dict(alignment_method='max',
                        alignment_area_fraction=0.5,
                        events_per_day=1,
                        cache_attributes=['waveform_model'],
                        n_events_for_model=10)
        config = bi.utils.combine_dicts(defaults, config)

        self.ts = config['analysis_space'][0][1]
        self.dt = self.ts[1] - self.ts[0]

        bi.Source.__init__(self, config, *args, **kwargs)

    def get_pmf_grid(self):
        if not self.pdf_has_been_computed:
            raise bi.PDFNotComputedException("%s: Attempt to call a PDF that has not been computed" % self)
        return self.waveform_model, float('inf')

    def compute_pdf(self):
        # Simulate matrix (n_samples, n_waveforms) of amplitudes
        wv_matrix = self.simulate_wv_matrix(self.config['n_events_for_model'])

        # Align and average
        self.waveform_model = _align_and_average(self.ts, self.config, wv_matrix)

        bi.Source.compute_pdf(self)

    def simulate_wv_matrix(self, n_waveforms):
        raise NotImplementedError


class WaveformLikelihood(bi.BinnedLogLikelihood):
    """Likelihood (or 'Likelihood') for waveform template matching

    Error methods:
        'KS': Return -1 * the raw KS statistic (NOT a p-value) in place of a loglikelihood.
              You can still minimize this to get good fits, but forget about confidence intervals or using
              sampling-based minimizers like EMCEE.

        'manual': Use a fixed relative and/or absolute error.
                  Defaults to 0.001 absolute error (in fraction of total amplitude) per sample
    """

    def __init__(self, *args, **kwargs):
        bi.BinnedLogLikelihood.__init__(self, *args, **kwargs)

        self.ts = self.pdf_base_config['analysis_space'][0][1]
        if len(self.base_model.sources) > 1:
            raise ValueError("WaveformLikelihood only supports one source")

        self.config.setdefault('error_method', 'KS')
        self.config.setdefault('relative_error', 0)
        self.config.setdefault('absolute_error', 0.001)

    def set_data(self, waveform_matrix):
        # Align by the same method as simulated waveforms
        self.data_waveform = _align_and_average(self.ts,
                                                self.base_model.sources[0].config,
                                                waveform_matrix)

        # Arbitrary error estimate.
        if self.config['error_method'] == 'manual':
            self.errors = self.config['relative_error'] * self.data_waveform + self.config['absolute_error']

        self.is_data_set = True

    def _compute_likelihood(self, mus, pmfs):
        waveform_model = pmfs[0]

        if self.config['error_method'] == 'KS':
            # Minus sign ensures bad fits -> low values, just a like a loglikelihood
            return -np.max(np.abs(np.cumsum(waveform_model) - np.cumsum(self.data_waveform)))

        else:
            difs = (waveform_model - self.data_waveform)
            ndf = len(waveform_model)    # TODO: Subtract some for fitted parameters??
            return stats.chi2(ndf).logsf(np.sum(difs**2/self.errors))
