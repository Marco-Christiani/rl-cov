"""Wraps allocation env, action space is shrinkage factor for covariance matrix."""
import gymnasium as gym
import numpy as np
from sklearn.covariance import LedoitWolf
from statsmodels.stats.correlation_tools import cov_nearest

from . import cov
from . import utils
from .rayenv import RayTradingEnv


class ShrinkEnv(RayTradingEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        self.cov_estimator = LedoitWolf()
        self.ewm_acc = utils.EWMAcc(n_cols=self.num_assets, halflife=.94)
        self.mu_method = config.get("mu_method", "ewm")

    def step(self, shrinkage_factor):
        # include an extra price at the beginning of the window to calculate returns
        temp_prices = self.all_open_prices[self._current_obs_window_start_idx - 1:self._current_obs_window_end_idx + 1]
        returns = utils.pct_change_np(temp_prices)[1:]
        if self.mu_method == "ewm":
            mu = self.ewm_acc.apply_chunk(returns)
        else:
            mu = returns.mean(axis=0)
        cov_matrix = self.cov_estimator.fit(returns).covariance_
        cov_matrix = cov.shrink(
            cov_matrix=cov_matrix,
            shrinkage_target=cov.ShrinkageTarget.MeanVar,
            factor=shrinkage_factor
        )
        if not utils.is_psd(cov_matrix):
            cov_matrix = cov_nearest(cov_matrix, method='clipped')
        weights = cov.opt_weights(
            returns=returns,
            mu=mu[-1],
            cov=cov_matrix,
            model='Classic',
            obj_func='MaxRet',
            rm='MV',
            l=2,
            kelly=False,
            rf=0.0
        )
        obs, reward, done, info = super().step(weights)
        return obs, reward, done, info

    def reset(self):
        self.ewm_acc.reset()
        return super().reset()
