"""Wraps allocation env, action space is shrinkage factor for covariance matrix."""
import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from statsmodels.stats.correlation_tools import cov_nearest

from rlcov import cov
from rlcov import utils
from rlcov.envs.rayenv import RayTradingEnv


class ShrinkEnv(RayTradingEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float64
        )
        self.cov_estimator = LedoitWolf()
        self.ewm_acc = utils.EWMAcc(n_cols=self.num_assets, halflife=.94)
        self.mu_method = config.get("mu_method", "ewm")
        self.shrinkage_target: cov.ShrinkageTarget = cov.ShrinkageTarget[config.get("shrinkage_target", "MeanVar")]
        if not self.tickers:
            raise ValueError("Must provide tickers")

    def step(self, shrinkage_factor: np.ndarray):
        shrinkage_factor = shrinkage_factor[0]
        # Expanding context window
        temp_prices = self.all_open_prices[:self._current_obs_window_end_idx + 1]
        returns = utils.pct_change_np(temp_prices)[1:]
        if self.mu_method == "ewm":
            # prices for current chunk for stateful ewm
            chunk_prices = self.all_open_prices[self._current_obs_window_start_idx - 1:self._current_obs_window_end_idx + 1]
            chunk_returns = utils.pct_change_np(chunk_prices)[1:]
            mu = self.ewm_acc.apply_chunk(chunk_returns)[-1]
        else:
            mu = returns.mean(axis=0)
        raw_cov_matrix = self.cov_estimator.fit(returns).covariance_
        cov_matrix = cov.shrink(
            cov_matrix=raw_cov_matrix,
            shrinkage_target=self.shrinkage_target,
            factor=shrinkage_factor
        )
        if not utils.is_psd(cov_matrix):
            cov_matrix = cov_nearest(cov_matrix, method='clipped')
        # TODO: need to fork this library or reimplement, it should not have to be a df for my use case.
        returns_df = pd.DataFrame(returns, columns=self.tickers)
        weights = cov.opt_weights(
            returns=returns_df,
            mu=mu,
            cov=cov_matrix,
            model='Classic',
            obj_func='Sharpe',
            risk_metric='MV',
            risk_aversion_factor=2,
            kelly=False,
            rf=0.0
        )
        obs, reward, done, truncated, info = super().step(weights)
        assert self.observation_space.contains(obs)

        info = {
            'raw_cov_matrix': raw_cov_matrix,
            'cov_matrix': cov_matrix,
            'weights': weights,
            'shrinkage_factor': shrinkage_factor,
            **info
        }
        return obs, reward, done, truncated, info

    def reset(self, *args, **kwargs):
        self.ewm_acc.reset()
        return super().reset(*args, **kwargs)
