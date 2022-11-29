import cvxpy as cvx
from get_data import GetData


def markowitz_optimization(returns, gamma):
    """
    Uses Markowitz's Mean-Variance optimization to solve for portfolio weights

    :param returns: Time series of assets returns
    :type returns: pd.DataFrame
    :param gamma: Risk aversion parameter
    :type gamma: float
    :return: Portfolio weights, portfolio expected return, portfolio expected volatility
    :rtype: ndarray, float, float
    """

    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    expected_portfolio_return = weights.T * expected_returns
    expected_volatility = cvx.quad_form(weights, cov_matrix)

    weights = cvx.Variable(len(expected_returns))
    gamma = cvx.Parameter(nonneg=True, value=gamma)

    objective = cvx.Maximize(expected_portfolio_return - gamma * expected_volatility)
    constraints = [cvx.sum(weights) == 1,
                   weights >= 0]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return weights.value.round(3).ravel(), expected_portfolio_return.value, expected_volatility.value


tickers = ["MSFT", "XOM"]
returns = GetData(tickers).get_dataframe()["Adj Close"]

markowitz_optimization(returns, 0.3)
