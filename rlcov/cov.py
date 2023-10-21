import warnings
from functools import wraps

import numpy as np
from sklearn.covariance import LedoitWolf


def lw_shrink(returns: np.ndarray, factor: float = 0.1) -> np.ndarray:
    cov_matrix = LedoitWolf().fit(returns).covariance_
    identity = np.eye(cov_matrix.shape[0])
    return (1 - factor) * cov_matrix + factor * identity / sum(np.diag(identity))


def shrink(mu: np.ndarray, cov_matrix: np.ndarray, factor: float = 0.0001) -> np.ndarray:
    identity = np.eye(cov_matrix.shape[0])
    return (1 - factor) * cov_matrix + factor * identity / sum(np.diag(identity))


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            response = f(*args, **kwargs)
        return response

    return inner


@ignore_warnings
def opt_weights(close, *, method_mu, method_cov, model, obj_func, rm, l=0, kelly=False, cov_callback: callable = None,
                **stats_kwargs):
    """
    close: pd.DataFrame
        Close prices, columns are assets
    rm: str
        Risk Measure
    model: str
        Classic (historical), BL (Black Litterman) or FM (Factor Model)
    obj_func: str
        MinRisk, MaxRet, Utility, ERC, or Sharpe
    l: float/int
        Risk aversion factor when obj_func="Utility"
    kelly: Union[str, bool]
        Method used to calculate mean return.
        - False: arithmetic mean return
        - “approx”: approximate mean logarithmic return using first and second moment
        - “exact”: mean logarithmic return
    stats_kwargs: kwargs
        Any additions params to pass to Portfolio.asset_stats() (i.e. d=0.94)
    """
    # Sort to ensure correct ordering later
    if list(close.columns) != list(sorted(close.columns)):
        raise Exception('Must sort df columns!')

    returns = close.dropna(axis=1).pct_change().dropna()

    port = rp.Portfolio(returns=returns)
    # Estimate expected returns and covariance matrix
    if isinstance(method_mu, str) and isinstance(method_cov, str):
        port.assets_stats(method_mu=method_mu,
                          method_cov=method_cov, **stats_kwargs)
    if callable(method_mu):
        port.mu = method_mu(returns)
    if callable(method_cov):
        port.cov = method_cov(returns)
    # else:
    #     raise Exception('method_mu and/or method_cov are invalid')
    if (port.mu is None) or (port.cov is None):
        port.assets_stats(method_mu='hist',
                          method_cov='hist', **stats_kwargs)
    if callable(cov_callback):
        port.cov = cov_callback(port.mu, port.cov)
    # ic(port.cov)
    # ic(port.mu)
    # Estimate optimal weights
    # port.solvers = ["MOSEK"]
    # port.solvers = ["CVXPY"]
    if obj_func == 'ERC':
        w = port.rp_optimization(
            model=model,
            rm=rm,
            rf=0,  # Risk free rate
            b=None,  # Constraints (None defaults to 1/n)
            hist=True,  # Use historical for risk measures
        )
    else:
        w = port.optimization(
            model=model,
            rm=rm,
            obj=obj_func,
            rf=0,  # Risk free rate
            l=l,
            hist=True,  # Use historical for risk measures
            kelly=kelly,
        )
    if w is None:
        print('\tNo solution found...')
        return np.repeat(0, close.shape[1])
    # Insert the assets that contained NA values back into the weights as NA
    w = w.T
    w = w[sorted(w.columns)]  # Sort to ensure correct ordering
    na_cols = w.columns.symmetric_difference(close.columns)
    w[na_cols] = [0] * len(na_cols)
    w = w[sorted(w.columns)]  # Sort to ensure correct ordering
    w = w.T
    weights = np.ravel(w.to_numpy())
    return weights
