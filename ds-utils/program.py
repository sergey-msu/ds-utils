import numpy as np
from stats.hypotheses import check_normal


def main():
    d2 = np.random.normal(1, 4, size=(1000,))
    check_normal(d2, verbose=True)

    d5 = np.random.normal(0, 1, size=(1000,))
    check_normal(d5, verbose=True)

    d6 = np.ones((1000,))
    check_normal(d6, verbose=True)

    d1 = np.linspace(0, 1, 1000)
    check_normal(d1, verbose=True)

    d3 = np.zeros((1000,))
    d3[0] = 1
    check_normal(d3, verbose=True)

    d4 = [i/100 for i in range(1000)]
    check_normal(d4, verbose=True)


    return


main()