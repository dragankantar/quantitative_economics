import numpy as np
import cvxpy as cvx
from get_data import GetData


def maximize_alpha_constrain_downside(alphas, returns, percentile, max_loss):
    """
    Solves for portfolio weights that maximize alpha constrained on historical downside

    :param alphas: Alphas, excess returns to benchmark
    :type alphas: ndarray
    :param returns: Time series of assets returns
    :type returns: pd.DataFrame
    :param percentile: Percent of daily assets returns that constitute average worst
                       day that our portfolio must exceed
    :type percentile: float
    :param max_loss: Maximum downside allowed for the bottom *:param percentile*
                     average worst days need to exceed
    :type max_loss: float
    :return: Portfolio weights
    :rtype: ndarray
    """

    weights = cvx.Variable(returns.shape[1])

    # Number of worst-case return periods to sample.
    nsamples = round(returns.shape[0] * percentile)

    portfolio_rets = returns * weights
    avg_worst_day = cvx.sum_smallest(portfolio_rets, nsamples) / nsamples

    objective = cvx.Maximize(weights.T * alphas)
    constraints = [avg_worst_day >= max_loss]

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return weights.value.round(3).ravel()


tickers = ["MSFT", "XOM"]
returns = GetData(tickers).get_dataframe()["Adj Close"]
alphas = np.linspace(-2, 2, 8)

result = maximize_alpha_constrain_downside(alphas, returns, percentile=0.05, max_loss=-0.05)
print("Portfolio:", result)

portfolio_rets = returns.dot(result)
worst_days = portfolio_rets[portfolio_rets <= np.percentile(portfolio_rets, 5)]
print("Average Bad Day:", worst_days.mean())
