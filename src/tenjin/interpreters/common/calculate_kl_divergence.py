import numpy as np
from scipy import stats


def calculate_kl_div(seq1, seq2):
    kl_divergence = stats.entropy(seq1, seq2)
    return kl_divergence


def get_optimum_bin_size(feature_pd_series):
    '''
    Adopt Freedman-Diaconis rule
    '''
    q1 = feature_pd_series.quantile(0.25)
    q3 = feature_pd_series.quantile(0.75)
    inner_quantile = q3 - q1
    bin_width = (2 * inner_quantile) / (len(feature_pd_series) ** (1 / 3))
    bin_count = int(np.ceil((feature_pd_series.max() - feature_pd_series.min()) / bin_width))
    return bin_count
