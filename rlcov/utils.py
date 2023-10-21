import os

import pandas as pd
import yfinance as yf


def fetch_stock_data(tickers, start_date, end_date, freq='1d'):
    """
    Fetch historical stock data for given tickers from start_date to end_date.

    Parameters:
    - tickers: list of stock tickers
    - start_date: start date in format 'YYYY-MM-DD'
    - end_date: end date in format 'YYYY-MM-DD'

    Returns:
    - returns: a DataFrame containing daily returns of the stocks
    """

    stock_data = yf.download(tickers, start=start_date, end=end_date, interval=freq)['Adj Close']
    returns = stock_data.pct_change().dropna()  # Compute daily returns
    return returns


def load_local_data(tickers, start_date, end_date, freq='1d'):
    df_dict = {}
    base = f'{os.environ["HOME"]}/data/combined-minute-bars/parquets/'
    for sym in tickers:
        temp_df = pd.read_parquet(base + sym + '.parquet')[['open', 'high', 'low', 'close', 'time']]
        temp_df = temp_df.set_index(pd.to_datetime(temp_df.time, unit='s'))
        temp_df = temp_df.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df_dict[sym] = temp_df[start_date:end_date]
    return df_dict
