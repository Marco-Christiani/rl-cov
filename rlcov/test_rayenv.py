import pandas as pd
from icecream import ic
import pytest

from . import rayenv
from . import utils


@pytest.fixture
def config():
    """Get a config for testing."""
    return {
        "tickers": [
            'BTCUSD',
            'ETHUSD',
        ],
        "start_date": '2021-01-01',
        "end_date": '2021-03-01',  # 2 months
        "warmup": 7 * 1,
        "rebalance_freq": 7,  # 1 week
        "data_freq": 1,
        "freq_unit": "d",
        "init_cash": 100,
        "txn_cost": 0.001,
    }


@pytest.fixture
def df_dict(config: dict):
    """Get dataframes of open/close prices testing. Dict keyed by ticker."""
    data = utils.load_local_data(list(config['tickers']), config['start_date'], config['end_date'],
                                 freq=f'{config["data_freq"]}{config["freq_unit"]}')
    return {
        'open': pd.DataFrame({sym: data[sym].open for sym in list(config["tickers"])}),
        'close': pd.DataFrame({sym: data[sym].close for sym in list(config["tickers"])})
    }


@pytest.fixture
def weights(config: dict):
    """Make a dataframe of weights for testing.

    Index is timestamps for rebalance dates.
    """
    offset = pd.Timedelta(f'{config["warmup"]}{config["freq_unit"]}')

    index = pd.date_range(start=pd.Timestamp(config['start_date']) + offset,
                          end=config['end_date'],
                          freq=f'{config["rebalance_freq"]}{config["freq_unit"]}')
    # create weights for each value in index
    # equally weighted, liquidate all assets on every third rebalance
    weight_df = pd.DataFrame([[0, 0] if i % 3 == 0 else [0.5, 0.5] for i in range(len(index))], index=index)
    weight_df.columns = config['tickers']
    return weight_df


def test_ray_trading_env(df_dict, config, weights):
    """Test that the RayTradingEnv works."""
    env = rayenv.RayTradingEnv({
        'open_prices': df_dict['open'].values,
        'close_prices': df_dict['close'].values,
        **config

    })
    assert env.action_space.shape == (len(config['tickers']),)
    ic(env.reset())

    for i, w in weights.iterrows():
        obs, reward, done, info = env.step(w.values)
        ic(obs, reward, done, info)
        if done:
            break

    # verify against vectorbt
    # vbt will account for timestamps
    import vectorbtpro as vbt
    # use a profiler to benchmark how long this takes

    portfolio = vbt.Portfolio.from_orders(
        open=df_dict['open'],
        close=df_dict['close'],
        size=weights,
        init_cash=100,
        size_type='targetpercent',
        call_seq='auto',  # first sell then buy
        group_by=True,  # one group
        cash_sharing=True,  # assets share the same cash
        fees=1e-3,
        fixed_fees=0,
        slippage=0  # costs
    )

    ic(portfolio.get_asset_value(group_by=False))
    ic(portfolio.get_value())
    # TODO: find out why these are different results

