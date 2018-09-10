import numpy as np
import scipy
from stats.hypotheses import check_normal, single_mean


def main():

    #test_check_normal()
    #test_process_binomial()
    test_single_mean()

    return


def test_process_binomial():

    d = [1, 1, 0, 1, 1, 0, 1]
    single_mean(d, 0.5, verbose=True)

    np.random.seed(9)
    d = scipy.stats.bernoulli.rvs(0.4, size=1000)
    single_mean(d, 0.5, verbose=True)

    single_mean(d, 0.4, verbose=True)


    return


def test_check_normal():
    d2 = np.random.normal(1, 4, size=(1000,))
    res = check_normal(d2, verbose=True)
    print(res, '\n')

    d5 = np.random.normal(0, 1, size=(1000,))
    res = check_normal(d5, verbose=True)
    print(res, '\n')

    d6 = np.ones((1000,))
    res = check_normal(d6, verbose=True)
    print(res, '\n')

    d1 = np.linspace(0, 1, 1000)
    res = check_normal(d1, verbose=True)
    print(res, '\n')

    d3 = np.zeros((1000,))
    d3[0] = 1
    res = check_normal(d3, verbose=True)
    print(res, '\n')

    d4 = [i/100 for i in range(1000)]
    res = check_normal(d4, verbose=True)
    print(res, '\n')
    return


def test_single_mean():
    d2 = np.random.normal(1, 4, size=(1000,))
    res = single_mean(d2, 1, verbose=True)
    print(res, '\n')

    d5 = np.random.normal(0, 1, size=(1000,))
    res = single_mean(d5, 0.1, verbose=True)
    print(res, '\n')

    d6 = np.ones((1000,))
    res = single_mean(d6, 1, verbose=True)
    print(res, '\n')

    d1 = np.linspace(0, 1, 20)
    res = single_mean(d1, 5, verbose=True)
    print(res, '\n')

    d3 = np.zeros((100,))
    d3[0] = 1
    res = single_mean(d3, 0, verbose=True)
    print(res, '\n')

    d4 = [i/100 for i in range(20)]
    res = single_mean(d4, 0.5, verbose=True)
    print(res, '\n')
    return


main()