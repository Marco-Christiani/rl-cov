import gymnasium as gym
import numpy as np

from rlcov.env import TradingEnv


class RayTradingEnv(TradingEnv, gym.Env):
    def __init__(self, config):
        """Theres an issue here with offset and rebalance freq."""
        self.offset = config["warmup"] - 1
        rebalance_open_prices = config["open_prices"][self.offset::config["rebalance_freq"]]
        rebalance_close_prices = config["close_prices"][self.offset::config["rebalance_freq"]]
        super().__init__(
            open_prices=rebalance_open_prices,
            close_prices=rebalance_close_prices,
            init_cash=config["init_cash"],
            txn_cost=config["txn_cost"]
        )
        self.all_open_prices = config["open_prices"]
        self.all_close_prices = config["close_prices"]
        self.rebalance_freq: int = config["rebalance_freq"]  # units of time between rebalances
        self.data_freq: int = config["data_freq"]  # units of time between data points
        self.freq_unit: str = config["freq_unit"]  # a pandas unit: i.e. T, h, d, etc
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets, self.rebalance_freq),
            dtype=np.float32
        )

    def step(self, actions):
        """Steps wrapped sim, adds intermediate prices to obs.

        The wrapped sim essentially steps the sim forward by one rebalance period, we fill in the gaps
        with intermediate prices.
        """
        # get the indices before we step
        start_price_idx = self.offset + (self.current_step * self.rebalance_freq) + 1
        end_price_idx = self.offset + (self.current_step + 1) * self.rebalance_freq
        # simulate one rebalance period
        next_open_prices, reward, done, info = super().step(actions)
        # obs are the open prices from beginning of rebalance period to end of rebalance period
        obs = self.all_open_prices[start_price_idx:end_price_idx + 1]
        # if our math is correct our last price should be the last price of the rebalance period
        assert np.equal(obs[-1], next_open_prices).all()
        assert len(obs) == self.rebalance_freq
        return obs, reward, done, info

    def reset(self, seed=None):
        """Reset the sim, return initial obs."""
        first_open, _ = super().reset()
        # HACK: ideally this takes advantage of the warmup period but would violate the gym interface
        obs = self.all_open_prices[self.offset:self.rebalance_freq+self.offset-1]  # oh also i kinda messed up the offset thing
        # obs = self.all_open_prices[:self.offset+1] # oh also i kinda messed up the offset thing
        # if our math is correct our last price of obs be the first open price of the first rebalance period
        print(first_open)
        print(obs)
        assert np.equal(obs[-1], first_open).all(), f'Got {obs[-1]} expected {first_open}.' \
                                                    f' offset={self.offset} rebalance_freq={self.rebalance_freq} ' \
                                                    f'obs={obs}'
        return obs, {}

    def _current_obs_window(self):
        return self.all_open_prices[self._current_obs_window_start_idx:self._current_obs_window_end_idx + 1]

    @property
    def _current_obs_window_start_idx(self):
        return self.offset + (self.current_step * self.rebalance_freq) + 1

    @property
    def _current_obs_window_end_idx(self):
        return self.offset + (self.current_step + 1) * self.rebalance_freq


if __name__ == '__main__':
    import pandas as pd

    df = pd.DataFrame({
        'timestamp': ['2023-10-10', '2023-10-10', '2023-10-11', '2023-10-11', '2023-10-12', '2023-10-12',
                      '2023-10-13', '2023-10-13', '2023-10-14', '2023-10-14'],
        'symbol': ['AAPL', 'GOOG', 'AAPL', 'GOOG', 'AAPL', 'GOOG', 'AAPL', 'GOOG', 'AAPL', 'GOOG'],
        'open': [150.0, 2800, 152, 2825, 153, 2830, 155, 2850, 154, 2840],
        'high': [155.0, 2850, 153, 2845, 157, 2860, 158, 2875, 159, 2880],
        'low': [148.0, 2790, 150, 2805, 151, 2815, 152, 2835, 153, 2825],
        'close': [150.0, 2850, 151, 2860, 151 * 2, 2870, 151, 2880, 155, 2890]
    })
    df.set_index(['timestamp', 'symbol'], inplace=True)

    df = df.unstack(level='symbol')
    print(df.columns)

    weight_list = np.array([
        [0.5, 0.5],
        [1, 0],
        [0, 0],
        [0, 0],
    ])
    print(df.open)
    sim = RayTradingEnv(config={
        'open_prices': df['open'].values,
        'close_prices': df['close'].values,
        'warmup': 4,
        'rebalance_freq': 2,
        'data_freq': 1,
        'freq_unit': 'd',
        'init_cash': 100_000,
        'txn_cost': 1e-3,
    })
    print(sim.reset())
