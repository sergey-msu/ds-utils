import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint


def single_mean(data, m, s=None, alpha=0.05, alternative='two-sided', force_normal=False, verbose=False):

    if (check_binomial(data)):
        return test_binomial(data, m, alpha, alternative, verbose)

    if (check_normal(data, alpha, verbose=verbose) or force_normal):
        return test_normal(data, m, s, alpha, alternative, verbose)

    return test_nonparam(data, m, alpha, alternative, verbose)


def check_binomial(data):
    for item in data:
        if item != 0 and item != 1:
            return False

    return True


def test_binomial(data, p, alpha=0.05, alternative='two-sided', verbose=False):
    if verbose:
        print('Mean test for binomial sample:')

    m = sum(data)
    n = len(data)

    p_value = stats.binom_test(m, n, p=p, alternative=alternative)
    res = (p_value>alpha)

    if verbose:
        normal_interval = proportion_confint(m, n, alpha=alpha, method = 'normal')
        wilson_interval = proportion_confint(m, n, alpha=alpha, method = 'wilson')

        print('Proportion for binomial sample: {0} --> mean={1} p_value={2}, normal={3}, wilson={4}'
              .format(res, m/n, p_value, normal_interval, wilson_interval))

    return res


def check_normal(data, alpha=0.05, criteria=None, strategy=-1, verbose=False):
    if verbose:
        print('Normality test for data:')

    if criteria is None:
        criteria = ['shapiro', 'normaltest', 'anderson']
    if strategy <=0 :
        strategy = len(criteria)

    result = {}

    data = np.array(data)
    mean = data.mean()
    std  = data.std()
    if std==0:
        if verbose:
            print('data is constant')
        return False

    data = (data - mean)/std
    norms = []

    for cname in criteria:
        # Shapiro-Wilk test
        if cname == 'shapiro':
            r = stats.shapiro(data)
            norm = r[1]>alpha
            norms.append(norm)
            result['shapiro'] = { 'norm': norm, 'details': r }
            if verbose:
                print('Shap-Wilk:\t{2} --> T={0}, p_value={1}'.format(r[0], r[1], str(norm)))

        # D’Agostino’s K^2 test
        if cname == 'normaltest':
            r = stats.normaltest(data)
            norm = r[1]>alpha
            norms.append(norm)
            result['normaltest'] = { 'norm': norm, 'details': r }
            if verbose:
                print('DAgost K2:\t{2} --> T={0}, p_value={1}'.format(r[0], r[1], str(norm)))

        # Anderson-Darling test
        if cname == 'anderson':
            r = stats.anderson(data)
            norm = np.sum(r[0] >= r[1]) == 0
            norms.append(norm)
            result['anderson'] = { 'norm': norm, 'details': r }
            if verbose:
                print('And-Darl:\t{3} --> T={0}, {1}, {2}'.format(r[0], r[1], r[2], norm))

    return sum(norms) >= strategy


def test_normal(data, m, s=None, alpha=0.05, alternative='two-sided', verbose=False):
    if verbose:
        print('Mean test for normal sample:')

    # Fisher's z-test
    if s is not None and s>0:
        z = (data - m)/s
        p_value = 2*(1 - stats.norm.cdf(abs(z)))
        res = p_value>alpha
        if verbose:
            print('z-test:\t{2} --> T={0}, p_value={1}'.format(z, p_value, str(res)))
        return res

    # Student's t-test
    t, p_value = stats.ttest_1samp(data, m)
    res = p_value>alpha
    if verbose:
        print('t-test:\t{2} --> T={0}, p_value={1}'.format(t, p_value, str(res)))
    return res


def test_nonparam(data, m, alpha, alternative='two-sided', verbose=False):
    if verbose:
        print('Non-parametric mean test:')

    data = np.array(data)

    # Sign test
    M, p_value = sign_test(data, m)
    res = p_value>alpha
    if verbose:
        print('sign-test:\t{2} --> M={0}, p_value={1}'.format(M, p_value, str(res)))

    # Wilcoxon signe-rank test
    t, p_value = stats.wilcoxon(data - m)
    res = p_value>alpha
    if verbose:
        print('sign-rank-test:\t{2} --> T={0}, p_value={1}'.format(t, p_value, str(res)))

    # Permutation test
    t, p_value = permutation_test(data, m)
    res = p_value>alpha
    if verbose:
        print('perm-test:\t{2} --> T={0}, p_value={1}'.format(t, p_value, str(res)))

    return


def permutation_t_stat_1sample(sample, mean):
    t_stat = sum(map(lambda x: x - mean, sample))
    return t_stat


def permutation_zero_distr_1sample(sample, mean, max_permutations = None):
    centered_sample = list(map(lambda x: x - mean, sample))
    if max_permutations:
        signs_array = set([tuple(x) for x in 2 * np.random.randint(2, size = (max_permutations,
                                                                              len(sample))) - 1 ])
    else:
        signs_array =  itertools.product([-1, 1], repeat = len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]

    return distr


def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_1sample(sample, mean)

    zero_distr = permutation_zero_distr_1sample(sample, mean, max_permutations)

    if alternative == 'two-sided':
        return t_stat, sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return t_stat, sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return t_stat, sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)