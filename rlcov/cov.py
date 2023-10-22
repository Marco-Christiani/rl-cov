import warnings
from enum import Enum
from functools import wraps

import numpy as np
import riskfolio as rp


class ShrinkageTarget(Enum):
    Identity = 0
    MeanVar = 1


def shrink(
        cov_matrix: np.ndarray,
        shrinkage_target: ShrinkageTarget = ShrinkageTarget.Identity,
        factor: float = 0.0001
) -> np.ndarray:
    """Shrinks a covariance matrix towards a ShrinkageTarget."""
    n = cov_matrix.shape[0]
    if shrinkage_target == ShrinkageTarget.Identity:
        shrunk_cov = (1.0 - factor) * cov_matrix
        shrunk_cov.flat[:: n + 1] += factor / n
        return shrunk_cov
    elif shrinkage_target == ShrinkageTarget.MeanVar:
        shrunk_cov = (1.0 - factor) * cov_matrix
        shrunk_cov.flat[:: n + 1] += factor * np.trace(cov_matrix) / n
        return shrunk_cov
    else:
        raise ValueError('Invalid shrinkage target')


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            response = f(*args, **kwargs)
        return response

    return inner


def opt_weights(returns, *, mu: np.ndarray, cov, model, obj_func, rm, l: float = 2, kelly: bool = False, rf: float = 0.0):
    """
    returns: pd.DataFrame
        Returns, columns are assets
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
    rf: float
        Risk free rate
    """
    # Sort to ensure correct ordering later
    if list(returns.columns) != list(sorted(returns.columns)):
        raise Exception('Must sort df columns!')

    port = rp.Portfolio(returns=returns)
    port.mu = mu
    port.cov = cov

    # Estimate optimal weights
    # port.solvers = ["MOSEK"]
    # port.solvers = ["CVXPY"]

    if obj_func == 'ERC':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = port.rp_optimization(
                model=model,
                rm=rm,
                rf=rf,
                b=None,  # Constraints (None defaults to 1/n)
                hist=True,  # Use historical for risk measures
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = port.optimization(
                model=model,
                rm=rm,
                obj=obj_func,
                rf=rf,
                l=l,
                hist=True,  # Use historical for risk measures
                kelly=kelly,
            )
    if w is None:
        warnings.warn('No solution found. Returning zeros.')
        return np.repeat(0, returns.shape[1])
    if len(w) != len(returns.shape[1]):
        warnings.warn(
            f'There were NA values in the weights, replacing with zero.'
            f' len(w)={len(w)}, len(close.columns)={len(returns.columns)}')
        # Insert the assets that contained NA values back into the weights as 0
        w = w.T
        w = w[sorted(w.columns)]  # Sort to ensure correct ordering
        na_cols = w.columns.symmetric_difference(returns.columns)
        w[na_cols] = [0] * len(na_cols)
        w = w[sorted(w.columns)]  # Sort to ensure correct ordering
        w = w.T
        w = np.ravel(w.to_numpy())
    return w
