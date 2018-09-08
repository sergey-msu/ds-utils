import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def single_mean(data, m, s=None, norm=None, binom=None, verbosity=None):



    return


def check_normal(data, criteria=None, verbose=False, alpha=0.05):
    if verbose:
        print('Normality test for data, length=', len(data))

    if criteria is None:
        criteria = ['qq', 'shapiro', 'normaltest', 'anderson']

    result = {}

    data = np.array(data)
    mean = data.mean()
    std  = data.std()
    if std==0:
        if verbose:
            print('data is constant')
        return None

    data = (data - mean)/std

    for cname in criteria:
        # QQ test
        if cname == 'qq':
            r = stats.probplot(data, dist='norm', plot=plt)
            result['qq'] = { 'norm': None, 'details': r }
            if verbose:
                print('QQ:\t\tslope={0}, intercept={1}, r={2}'.format(r[1][0], r[1][1], r[1][2]))

        # Shapiro-Wilk test
        if cname == 'shapiro':
            r = stats.shapiro(data)
            norm = r[1]>alpha
            result['shapiro'] = { 'norm': norm, 'details': r }
            if verbose:
                print('Shap-Wilk:\t{2} --> T={0}, p-value={1}'.format(r[0], r[1], str(norm)))

        # D’Agostino’s K^2 test
        if cname == 'normaltest':
            r = stats.normaltest(data)
            norm = r[1]>alpha
            result['normaltest'] = { 'norm': norm, 'details': r }
            if verbose:
                print('DAgost K2:\t{2} --> T={0}, p-value={1}'.format(r[0], r[1], str(norm)))

        # Anderson-Darling test
        if cname == 'anderson':
            r = stats.anderson(data)
            norm = np.sum(r[0] >= r[1]) == 0
            result['anderson'] = { 'norm': norm, 'details': r }
            if verbose:
                print('And-Darl:\t{3} --> T={0}, {1}, {2}'.format(r[0], r[1], r[2], norm))

    return