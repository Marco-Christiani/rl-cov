import matplotlib
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from sklearn.covariance import LedoitWolf

from rlcov.utils import load_local_data


def get_allocations(returns, cov_matrix, frequency):
    # Calculate expected returns
    mu = expected_returns.mean_historical_return(returns, returns_data=True, frequency=frequency)

    # Use EfficientFrontier to get the optimal weights
    ef = EfficientFrontier(mu, cov_matrix)
    ef.max_sharpe(risk_free_rate=-1)  # Maximize the Sharpe ratio
    cleaned_weights = ef.clean_weights()  # Clean the raw weights, setting any weights whose absolute values are below the cutoff to zero, and rounding the rest
    return cleaned_weights


def shrink(cov_matrix: np.ndarray, factor: float = 0.5):
    identity = np.eye(cov_matrix.shape[0])
    return (1 - factor) * cov_matrix + factor * identity


def animate(config, stock_returns):
    matplotlib.use('Qt5Agg')
    fig, ax = plt.subplots()

    # Initial covariance and weights
    cov = LedoitWolf().fit(stock_returns[:config.warmup_period]).covariance_
    initial_weights = get_allocations(stock_returns[:config.warmup_period], cov, 252 * 24)

    # Setting up the initial bar chart
    bars = ax.bar(range(len(initial_weights)), list(initial_weights.values()), align='center')
    ax.set_xticks(range(len(initial_weights)))
    ax.set_xticklabels(list(initial_weights.keys()))
    ax.set_ylim(0, 1)  # Assuming weights are between 0 and 1
    ax.set_xlabel('Stocks')
    ax.set_ylabel('Weights')

    prev_weights = initial_weights

    def update(frame):
        # ignore FutureWarning
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        nonlocal prev_weights
        # Updating for each frame
        curr_returns = stock_returns[:frame * 24 * 7 + config.warmup_period]
        cov = LedoitWolf().fit(curr_returns).covariance_
        weights = get_allocations(curr_returns, cov, 252 * 24)
        temp = {k: weights[k] - prev_weights[k] for k in weights.keys()}
        prev_weights = weights
        weights = temp
        print(frame, curr_returns.shape, temp)
        # weights = get_allocations(curr_returns, shrink(cov, factor=0.001))
        # Set the height of each bar according to the new weights
        for bar, (_, weight) in zip(bars, weights.items()):
            bar.set_height(weight)

        return bars

    ani = animation.FuncAnimation(fig=fig, func=update, frames=100, repeat=False)
    plt.show()
    ani.save("weights_animation.gif", writer="pillow")


def main():
    config = OmegaConf.create({
        "tickers": [
            'ADAUSD',
            'BTCUSD',
            'CRVUSD',
            'ETHUSD',
            'FTTUSD',
            # 'LINKUSD',
            'LTCUSD',
            'MANAUSD',
            'MATICUSD',
            'XRPUSD',
        ],
        "start_date": '2019-01-01',
        "end_date": '2023-01-01',
        "warmup_period": 300,
    })
    freq = '1h'
    data = load_local_data(list(config.tickers), config.start_date, config.end_date, freq=freq)
    # stock_returns = fetch_stock_data(list(config.tickers), config.start_date, config.end_date, freq='1h')
    print(data['BTCUSD'].shape)
    stock_returns = pd.DataFrame({sym: data[sym].close.pct_change().dropna() for sym in list(config.tickers)})
    stock_returns.dropna(inplace=True)
    print(stock_returns.head())
    print(stock_returns.shape)
    # animation(config, stock_returns)
    # testing
    cov = LedoitWolf().fit(stock_returns.iloc[:100, :2]).covariance_
    cov = pd.DataFrame(np.array(cov, ndmin=2), columns=stock_returns.columns[:2], index=stock_returns.columns[:2])
    print(cov)
    from sklearn.covariance._shrunk_covariance import shrunk_covariance
    temp = pd.DataFrame(shrunk_covariance(cov, shrinkage=0.1), columns=stock_returns.columns[:2], index=stock_returns.columns[:2])
    print(temp)
    temp = pd.DataFrame(shrink(cov.values, factor=0.1), columns=stock_returns.columns[:2], index=stock_returns.columns[:2])
    print(temp)
    import riskfolio as rp
    rpp = rp.Portfolio(returns=stock_returns.iloc[:100, :2])
    rpp.assets_stats(method_mu='hist', method_cov='ledoit')
    print(rpp.cov)
    print(rpp.mu)


if __name__ == '__main__':
    main()
