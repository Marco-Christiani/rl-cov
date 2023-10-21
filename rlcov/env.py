import numpy as np
import vectorbtpro as vbt
import gymnasium as gym


class TradingEnv(gym.Env):
    def __init__(self, open_prices, close_prices, init_cash=100, txn_cost=1e-3):
        self.txn_cost = txn_cost
        self.open_prices = open_prices
        self.close_prices = close_prices
        self.init_cash = float(init_cash)
        self.num_assets = open_prices.shape[1]
        self.current_step = 0
        self.shared_cash = self.init_cash
        self.exec_states = [vbt.pf_enums.ExecState(
            cash=float(0.0),
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=np.nan,
            value=np.nan
        ) for _ in range(self.num_assets)]
        self.reset()

    def reset(self, **kwargs):
        self.exec_states = [vbt.pf_enums.ExecState(
            cash=float(0.0),
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=np.nan,
            value=np.nan
        ) for _ in range(self.num_assets)]
        self.shared_cash = self.init_cash
        self.current_step = 0
        return self.open_prices[self.current_step]

    def step(self, target_pcts):
        """Perform one rebalance step, time between steps is determined by the data provided.

        Parameters
        ----------
        target_pcts : np.ndarray
            Target percentages for each asset. Should sum to 1.
        """
        if self.current_step >= len(self.close_prices)-1:
            raise ValueError("Simulation has reached the end of the data.")
        starting_portfolio_value = self.portfolio_value.copy()
        target_values = [self.portfolio_value * pct for pct in target_pcts]

        # Execute sell orders first
        for i in range(self.num_assets):
            if not np.isnan(target_pcts[i]) and self.position_values[i] > target_values[i]:
                self._execute_order(i, target_pcts[i])

        # Execute buy orders
        for i in range(self.num_assets):
            if not np.isnan(target_pcts[i]) and self.position_values[i] < target_values[i]:
                self._execute_order(i, target_pcts[i])

        # stepping the env will change valuations, so freeze the current value to reflect the current state
        info = dict(
            portfolio_value=self.portfolio_value.copy(),
            position_values=self.position_values.copy(),
            position_pct=[self.position_values[i] / self.portfolio_value for i in range(self.num_assets)],
            cash=self.shared_cash,
        )
        one_step_return = (self.portfolio_value - starting_portfolio_value) / starting_portfolio_value
        self.current_step += 1
        done = self.current_step >= len(self.close_prices)-1
        return self.close_prices[self.current_step], one_step_return, done, info

    def _execute_order(self, asset_idx, target_pct):
        self.exec_states[asset_idx] = vbt.pf_enums.ExecState(
            cash=self.shared_cash,
            position=self.exec_states[asset_idx].position,
            debt=self.exec_states[asset_idx].debt,
            locked_cash=self.exec_states[asset_idx].locked_cash,
            free_cash=self.shared_cash,
            val_price=self.open_prices[self.current_step, asset_idx],
            value=self.portfolio_value
        )
        order = vbt.pf_nb.order_nb(
            size=target_pct,
            price=self.close_prices[self.current_step, asset_idx],
            size_type=vbt.pf_enums.SizeType.TargetPercent,
            fees=self.txn_cost,
        )
        _, self.exec_states[asset_idx] = vbt.pf_nb.execute_order_nb(
            exec_state=self.exec_states[asset_idx],
            order=order
        )
        self.shared_cash = self.exec_states[asset_idx].cash

    @property
    def position_values(self):
        # use the open price to calculate the current value
        return [self.exec_states[i].position * self.open_prices[self.current_step, i] for i in
                range(self.num_assets)]

    @property
    def portfolio_value(self):
        # use the open price to calculate the current value
        asset_values = [self.exec_states[i].position * self.open_prices[self.current_step, i] for i in
                        range(self.num_assets)]
        return self.shared_cash + sum(asset_values)

    def render(self):
        pass

if __name__ == '__main__':
    from icecream import ic
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
    print(df.close)
    simulator = TradingEnv(open_prices=df['open'].values, close_prices=df['close'].values)

    for weights in weight_list:
        obs, reward, done, info = simulator.step(weights)
        ic(obs, reward, done, info)
