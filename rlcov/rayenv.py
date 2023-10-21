import gymnasium as gym
import numpy as np

from env import TradingEnv


class RayTradingEnv(TradingEnv, gym.Env):
    def __init__(self, config):
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
            shape=(self.num_assets,),
            dtype=np.float32
        )

    def step(self, actions):
        """Steps wrapped sim, adds intermediate prices to obs.

        The wrapped sim essentially steps the sim forward by one rebalance period, we fill in the gaps
        with intermediate prices.
        """
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

    def reset(self):
        """Reset the sim, return initial obs."""
        first_open = super().reset()
        obs = self.all_open_prices[:self.offset+1]
        # if our math is correct our last price of obs be the first open price of the first rebalance period
        assert np.equal(obs[-1], first_open).all()
