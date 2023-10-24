from pprint import pprint

import gymnasium as gym
import numpy as np

from rlcov.envs.env import TradingEnv


class RayTradingEnv(TradingEnv, gym.Env):
    def __init__(self, config):
        """Theres an issue here with offset and rebalance freq."""
        self.all_open_prices = np.array(config["open_prices"], dtype=np.float64)
        self.all_close_prices = np.array(config["close_prices"], dtype=np.float64)
        self.timestamps = config["timestamps"]

        self.rebalance_freq: int = config["rebalance_freq"]  # units of time between rebalances
        self.data_freq: int = config["data_freq"]  # units of time between data points
        self.freq_unit: str = config["freq_unit"]  # a pandas unit: i.e. T, h, d, etc
        # ensure we have a minimum of one rebalance period (the step size is inclusive of the first value)
        rebalance_open_prices = self.all_open_prices[self.rebalance_freq-1::config["rebalance_freq"]]
        rebalance_close_prices = self.all_close_prices[self.rebalance_freq-1::config["rebalance_freq"]]

        self.tickers = config.get("tickers", None)
        super().__init__(
            open_prices=rebalance_open_prices,
            close_prices=rebalance_close_prices,
            init_cash=config["init_cash"],
            txn_cost=config["txn_cost"]
        )
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.rebalance_freq, self.num_assets),
            dtype=np.float64
        )

    def step(self, actions):
        """Steps wrapped sim, adds intermediate prices to obs.

        The wrapped sim essentially steps the sim forward by one rebalance period, we fill in the gaps
        with intermediate prices.
        """
        # simulate one rebalance period
        next_open_prices, reward, done, truncated, info = super().step(actions)
        start_price_idx = self.start_idx()
        end_price_idx = self.end_idx()
        # obs are the open prices from beginning of rebalance period to end of rebalance period
        obs = self.all_open_prices[start_price_idx:end_price_idx]
        # if our math is correct our last price should be the last price of the rebalance period
        assert np.equal(obs[-1], next_open_prices).all()
        assert len(obs) == self.rebalance_freq
        truncated = False
        # check the dtypes
        assert obs.dtype == self.observation_space.dtype, f"Type mismatch. Expected {self.observation_space.dtype}, got {obs.dtype}"
        # check the shape
        assert np.shape(
            obs) == self.observation_space.shape, f"Shape mismatch. Expected {self.observation_space.shape}, got {np.shape(obs)}"
        assert self.observation_space.contains(obs)
        if done:
            pf = self.backtest_from_orders()
            info.update(pf.stats())
            reward += pf.sharpe_ratio
        return obs, reward, done, truncated, info

    def reset(self, *args, **kwargs):
        """Reset the sim, return initial obs."""
        # Calculate the observation window based on offset and rebalance frequency
        # obs = self.all_open_prices[self.offset - self.rebalance_freq + 1: self.offset + 1]
        obs = self._current_obs_window()
        first_open, _ = super().reset(*args, **kwargs)

        # if our math is correct our last price of obs be the first open price of the first rebalance period
        assert np.equal(obs[-1], first_open).all(), f'Got {obs[-1]} expected {first_open}.' \
                                                    f' rebalance_freq={self.rebalance_freq} ' \
                                                    f'obs={obs}'
        obs = obs.reshape(self.observation_space.shape)
        assert (self.observation_space.low <= obs).all() and (
                obs <= self.observation_space.high).all(), "Value out of bound."
        assert np.shape(
            obs) == self.observation_space.shape, f"Shape mismatch. Expected {self.observation_space.shape}, got {np.shape(obs)}"
        assert self.observation_space.contains(obs)
        return obs, {}

    def start_idx(self):
        return self.current_step * self.rebalance_freq

    def end_idx(self):
        return self.start_idx() + self.rebalance_freq

    def _current_obs_window(self):
        obs = self.all_open_prices[self.start_idx():self.end_idx()]
        return obs.reshape(self.observation_space.shape)
        # return self.all_open_prices[self._current_obs_window_start_idx:self._current_obs_window_end_idx + 1]

    def backtest_from_orders(self):
        import vectorbtpro as vbt
        import pandas as pd

        # use all prices for simulation
        opens = pd.DataFrame(self.all_open_prices, index=self.timestamps)
        closes = pd.DataFrame(self.all_close_prices, index=self.timestamps)
        # the simulation ends at the last rebalance date, need to slice until last rebalance date
        n = len(self.all_open_prices)
        opens = opens.iloc[:n - n % self.rebalance_freq]
        closes = closes.iloc[:n - n % self.rebalance_freq]

        reweight_dates = self.timestamps[self.rebalance_freq-1::self.rebalance_freq]
        weights = pd.DataFrame(self.weights_trace, index=reweight_dates,
                               columns=self.tickers)
        return vbt.Portfolio.from_orders(
            open=opens,
            close=closes,
            size=weights,
            init_cash=self.init_cash,
            size_type='targetpercent',
            call_seq='auto',  # first sell then buy
            group_by=True,  # one group
            cash_sharing=True,  # assets share the same cash
            fees=self.txn_cost,
            fixed_fees=0,
            slippage=0,  # costs
            # for no intermediate price movements, only rebalance dates
            # freq=f'{self.rebalance_freq}{self.freq_unit}'
            # for intermediate price movements (use all prices)
            freq=f'{self.data_freq}{self.freq_unit}',
        )


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
    print(df.close)

    weight_list = np.array([
        [0.5, 0.5],
        [1, 0],
        [0, 0],
        [0, 0],
    ])

    cfg = {
        'open_prices': df['close'].values,
        'close_prices': df['close'].values,
        'timestamps': df.index,
        'data_freq': 1,
        'freq_unit': 'd',
        'init_cash': 100_000,
        'txn_cost': 1e-3,
    }

    cfg['rebalance_freq'] = 2
    sim = RayTradingEnv(config=cfg)
    print(sim.reset())
    print(sim.step([0.5, 0.5]))
    print(sim.weights_trace)
    pf = sim.backtest_from_orders()
    print(pf.stats())
    print(sim.portfolio_value)
    pprint(sim.__dict__)
