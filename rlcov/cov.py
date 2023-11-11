from __future__ import annotations

import warnings
from enum import Enum
from functools import wraps

import numpy as np
import pandas as pd
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
    shrunk_cov = (1.0 - factor) * cov_matrix
    if shrinkage_target == ShrinkageTarget.Identity:
        shrunk_cov.flat[:: n + 1] += factor / n
        return shrunk_cov
    elif shrinkage_target == ShrinkageTarget.MeanVar:
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


def opt_weights(returns: pd.DataFrame,
                *,
                mu: np.ndarray,
                cov: np.ndarray,
                model: str,
                obj_func: str,
                risk_metric,
                risk_aversion_factor: float = 2,
                kelly: bool = False,
                rf: float = 0.0,
                on_unsolvable: np.ndarray | str = 'zeros',
                ):
    """
    returns: pd.DataFrame
        Returns, columns are assets
    risk_metric: str
        Risk Measure
    model: str
        Classic (historical), BL (Black Litterman) or FM (Factor Model)
    obj_func: str
        MinRisk, MaxRet, Utility, ERC, or Sharpe
    risk_aversion_factor: float/int
        Risk aversion factor when obj_func="Utility"
    kelly: Union[str, bool]
        Method used to calculate mean return.
        - False: arithmetic mean return
        - “approx”: approximate mean logarithmic return using first and second moment
        - “exact”: mean logarithmic return
    rf: float
        Risk free rate
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError('Must be a dataframe')
    port = rp.Portfolio(returns=returns)
    port.mu = mu
    port.cov = cov

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if obj_func == 'ERC':
            w = port.rp_optimization(
                model=model,
                rm=risk_metric,
                rf=rf,
                b=None,  # Constraints (None defaults to 1/n)
                hist=True,  # Use historical for risk measures
            )
        else:
            w = port.optimization(
                model=model,
                rm=risk_metric,
                obj=obj_func,
                rf=rf,
                l=risk_aversion_factor,
                hist=True,  # Use historical for risk measures
                kelly=kelly,
            )
    if w is None:
        if on_unsolvable == 'zeros':
            warnings.warn('No solution found. Returning zeros according to on_unsolvable="zeros"')
            return np.repeat(0, returns.shape[1])
        # equal weights
        elif on_unsolvable == 'ew':
            warnings.warn('No solution found. Returning equal weights according to on_unsolvable="ew"')
            return np.repeat(1 / returns.shape[1], returns.shape[1])
        elif on_unsolvable == np.ndarray:
            warnings.warn('No solution found. Returning on_unsolvable ndarray')
            return on_unsolvable
    if len(w) != returns.shape[1]:
        warnings.warn('Solution does not match number of assets. Trying to fix by inserting zeros.')
        # will have to sort to ensure correct ordering when we insert zeros
        col_order = returns.columns
        # sort the columns
        returns = returns[sorted(returns.columns)]
        # Insert the assets that contained NA values back into the weights as 0
        w = w.T
        w = w[sorted(w.columns)]  # Sort to ensure correct ordering
        na_cols = w.columns.symmetric_difference(returns.columns)
        w[na_cols] = [0] * len(na_cols)
        # sort the columns back to the original order
        w = w[col_order]
        w = w.T
    w = np.ravel(w.to_numpy())
    return w


if __name__ == '__main__':
    # demonstrate the shrink function
    cov_matrix = np.array([
        [1, 0.5, 0.3],
        [0.5, .9, 0.1],
        [0.3, 0.1, 1]
    ])
    print(cov_matrix)
    print(shrink(cov_matrix, shrinkage_target=ShrinkageTarget.Identity, factor=0.1))
    print(shrink(cov_matrix, shrinkage_target=ShrinkageTarget.MeanVar, factor=0.1))
