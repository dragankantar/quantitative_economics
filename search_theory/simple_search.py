"""
Generalization of a simple search theory model presented by
Mike Clark in https://www.youtube.com/watch?v=tpByCAN7sVc.
This implementation generalizes the model by allowing for
benefits (payoffs) to be distributed by any probability
distribution and for costs to be non-constant.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simple_search(costs,
                  benefit_bounds,
                  benefits_distribution,
                  granularity = 1000):
    """
    Performs a search for optimal stopping point

    :param costs: costs of search
    :type costs: np.ndarray
    :param benefit_bounds: smallest and largest benefit attainable
    :type benefit_bounds: list
    :param benefits_distribution: probability distribution of
                                  potential benefits
    :type benefits_distribution: scipy.stats.rv_continuous class
    :param granularity: how many values of costs and benefits to
                        consider when looking for the numerical
                        solution, defaults to 1000
    :type granularity: int, optional

    :return: solution, marginal costs, expected marginal benefits
    :rtype: float, np.ndarray, np.ndarray
    """

    w_space = np.linspace(benefit_bounds[0], benefit_bounds[1], num=granularity)
    p_higher = np.zeros(granularity)
    avg_benefit_w = np.zeros(granularity)

    for i in range(len(w_space)):
        p_higher[i] = 1 - benefits_distribution.cdf(w_space[i])
        avg_benefit_w[i] = benefits_distribution.expect(lb = w_space[i])

    e_benefit_w = p_higher*avg_benefit_w

    solution = w_space[np.argmin(np.abs(e_benefit_w-costs))]

    return solution, costs, e_benefit_w

def plot(costs, e_benefit):
    """
    Plots marginal costs and marginal benefits

    :param costs: marginal costs
    :type costs: np.ndarray
    :param e_gain: expected marginal benefits
    :type e_gain: np.ndarray
    """

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(costs, label = "Marginal Cost")
    ax.plot(e_benefit, label = "Marginal Benefit")
    ax.set_xlabel("Payoff")
    ax.set_ylabel("Maginal Cost/Benefit")
    ax.legend()
    plt.show()

def main():
    costs = np.lib.scimath.logn(50, np.linspace(1, 1000, 5000))
    bounds = [0, 6]

    solution, c, b = simple_search(costs = costs,
                            benefit_bounds = bounds,
                            benefits_distribution = stats.uniform(0, 6),
                            granularity = 5000)

    print(solution)

    plot(c, b)


if __name__=="__main__":
    main()
