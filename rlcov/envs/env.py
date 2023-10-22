import gymnasium as gym
import numpy as np
import vectorbtpro as vbt


class TradingEnv(gym.Env):
    def __init__(self, open_prices: np.ndarray, close_prices: np.ndarray, init_cash: float = 100.0,
                 txn_cost: float = 1e-3):
        self.txn_cost = txn_cost
        self.open_prices = open_prices
        self.close_prices = close_prices
        self.init_cash = np.float64(init_cash)
        self.num_assets = open_prices.shape[1]
        self.current_step = 0
        self.shared_cash = self.init_cash
        self.weights_trace = []
        self.exec_states = [vbt.pf_enums.ExecState(
            cash=np.float64(0.0),
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=np.nan,
            value=np.nan
        ) for _ in range(self.num_assets)]

    def reset(self, *args, **kwargs):
        self.exec_states = [vbt.pf_enums.ExecState(
            cash=np.float64(0.0),
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=np.nan,
            value=np.nan
        ) for _ in range(self.num_assets)]
        self.shared_cash = np.float64(self.init_cash)
        self.current_step = 0
        self.weights_trace = []
        return self.open_prices[self.current_step], {}

    def step(self, target_pcts):
        """Perform one rebalance step, time between steps is determined by the data provided.

        Parameters
        ----------
        target_pcts : np.ndarray
            Target percentages for each asset. Should sum to 1.

        Returns
        -------
        tuple
            (observation, reward, done, info)
            reward is the one step return
             i.e. the percentage change in portfolio value from beginning of the step to the end, using next open price
            Time between steps is determined by the data provided, hence limited by its frequency.
        """
        # make sure we have a non-zero sum, otherwise we can't normalize and we continue, exiting the market
        if not np.isclose(np.sum(target_pcts), 0.0):
            target_pcts = target_pcts / np.sum(target_pcts)  # enforce these are valid
        self.weights_trace.append(target_pcts)
        if self.current_step >= len(self.close_prices) - 1:
            raise ValueError("Simulation has reached the end of the data.")
        starting_portfolio_value = self.portfolio_value.copy()
        target_values = np.array([self.portfolio_value * pct for pct in target_pcts], dtype=np.float64)

        # Execute sell orders first
        for i in range(self.num_assets):
            if not np.isnan(target_pcts[i]) and self.position_values[i] > target_values[i]:
                self._execute_order(i, target_pcts[i])

        # Execute buy orders
        for i in range(self.num_assets):
            if not np.isnan(target_pcts[i]) and self.position_values[i] < target_values[i]:
                self._execute_order(i, target_pcts[i])

        end_portfolio_value = self.portfolio_value.copy()
        self.current_step += 1
        reward = (self.portfolio_value - starting_portfolio_value) / starting_portfolio_value
        info = dict(
            end_portfolio_value=end_portfolio_value,
            next_prices=self.open_prices[self.current_step],
            # one step return is the percentage change in portfolio value, using next open price
            one_step_return=(self.portfolio_value - starting_portfolio_value) / starting_portfolio_value,
            next_portfolio_value=self.portfolio_value,
            next_position_values=self.position_values,
            position_pct=np.array([self.position_values[i] / self.portfolio_value for i in range(self.num_assets)],
                                  dtype=np.float64),
            cash=self.shared_cash,
        )
        done = self.current_step >= len(self.close_prices) - 1
        truncated = False
        return self.open_prices[self.current_step], reward, done, truncated, info

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
            price=self.close_prices[self.current_step, asset_idx],  # execute at close price
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
    simulator = TradingEnv(open_prices=df['close'].values, close_prices=df['close'].values)

    for weights in weight_list:
        obs, reward, done, truncated, info = simulator.step(weights)
        ic(obs, reward, done, info)

    # Prices for backtesting (using closing prices)
    # prices = df['close'].unstack(level='symbol')
    ic(df['close'])
    portfolio = vbt.Portfolio.from_orders(
        open=df['open'][:-1],
        close=df['close'][:-1],
        size=pd.DataFrame({
            'AAPL': weight_list[:, 0],
            'GOOG': weight_list[:, 1],
        }),
        init_cash=100,
        size_type='targetpercent',
        call_seq='auto',  # first sell then buy
        group_by=True,  # one group
        cash_sharing=True,  # assets share the same cash
        fees=0.001,
        fixed_fees=0,
        slippage=0  # costs
    )

    # print(portfolio.asset_flow())
    # print(portfolio.cash_flow())
    ic(portfolio.get_asset_value(group_by=False))
    ic(portfolio.value)
    # print(portfolio.order_records)
    ic(portfolio.stats())
