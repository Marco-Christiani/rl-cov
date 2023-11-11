"""Wraps allocation env, action space is shrinkage factor for covariance matrix."""
import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
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
        self.cov_estimator = EmpiricalCovariance()
        # self.cov_estimator = LedoitWolf()
        self.ewm_acc = utils.EWMAcc(n_cols=self.num_assets, halflife=.94)
        self.mu_method = config.get("mu_method", "ewm")
        self.shrinkage_target: cov.ShrinkageTarget = cov.ShrinkageTarget[config.get("shrinkage_target", "MeanVar")]
        self.obs_type = config.get("obs_type", "prices")
        if not self.tickers:
            raise ValueError("Must provide tickers")

    def step(self, shrinkage_factor: np.ndarray):
        shrinkage_factor = shrinkage_factor[0]
        # Expanding context window
        temp_prices = self.all_open_prices[:self.end_idx() + 1]
        returns = utils.pct_change_np(temp_prices)[1:]
        if self.mu_method == "ewm":
            # prices for current chunk for stateful ewm
            chunk_prices = self.all_open_prices[self.start_idx() - 1:self.end_idx() + 1]
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
        if self.obs_type == "returns":
            # need to get an extra price for returns
            next_prices = self.all_open_prices[self.start_idx() - 1:self.end_idx()]
            next_returns = utils.pct_change_np(next_prices)[1:]
            obs = next_returns
            obs = obs.reshape(self.observation_space.shape)
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
        obs, info = super().reset(*args, **kwargs)
        if self.obs_type == "returns":
            # NOTE: a zero will be inserted for the first row (cant do -1 here since current step is 0)
            next_returns = utils.pct_change_np(obs)
            obs = next_returns
            obs = obs.reshape(self.observation_space.shape)
        return obs, info


def try_reset():
    import pandas as pd
    n = 20
    close_prices = np.arange(n * 2)+1
    open_prices = close_prices  # same for now
    # build multiindex
    # need to repeat dates for each symbol
    ts = np.array([pd.date_range('2023-10-10', periods=n, freq='d')] * 2)
    ts = ts.flatten()
    ts.sort()
    arrays = {
        'symbol': ['AAPL', 'GOOG'] * n,
        'timestamp': ts,
        'open': open_prices,
        'close': close_prices,
    }
    df = pd.DataFrame(arrays)
    df.set_index(['timestamp', 'symbol'], inplace=True)
    df = df.unstack(level='symbol')
    print(df)
    cfg = {
        'open_prices': df['close'].values,
        'close_prices': df['close'].values,
        'timestamps': df.index,
        'data_freq': 1,
        'freq_unit': 'd',
        'init_cash': 100_000,
        'txn_cost': 1e-3,
        'rebalance_freq': 2,
        'tickers': ['AAPL', 'GOOG'],
        'mu_method': 'hist',
        'shrinkage_target': 'Identity',
        'obs_type': 'returns',
    }
    sim = ShrinkEnv(config=cfg)
    print(sim.reset())
    print(sim.step(np.array([0.5, 0.5])))


if __name__ == '__main__':
    try_reset()
