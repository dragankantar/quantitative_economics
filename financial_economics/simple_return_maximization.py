import cvxpy as cvx
from get_data import GetData


def simple_return_maximization(returns):
    """
    Solves for portfolio weights that maximize expected returns, ignoring the variance

    :param returns: Time series of assets returns
    :type returns: pd.DataFrame
    :return: Portfolio weights
    :rtype: ndarray
    """

    expected_returns = returns.mean()

    weights = cvx.Variable(len(expected_returns))
    objective = cvx.Maximize(weights.T * expected_returns)
    constraints = [cvx.sum(weights) == 1,
                   weights >= 0]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return weights.value.round(3).ravel()


tickers = ["MSFT", "XOM"]
returns = GetData(tickers).get_dataframe()["Adj Close"]

simple_return_maximization(returns)
