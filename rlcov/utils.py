import os
import re

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import linalg
from vectorbtpro.generic.enums import EWMMeanAIS
from vectorbtpro.generic.nb.rolling import ewm_mean_acc_nb


def pct_change_np(arr):
    diff = np.diff(arr, axis=0)
    return np.vstack((np.zeros(arr.shape[1]), diff / arr[:-1]))


def is_psd(A: np.ndarray, tol=1e-8):
    w: np.ndarray = linalg.eigh(A, lower=True, check_finite=True, eigvals_only=True)  # type: ignore
    return np.all(w >= tol)


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


def parse_duration_string(duration_str: str) -> np.timedelta64:
    """
    Parse a mixed unit string (with or without spaces) to a numpy.timedelta64 object.

    Parameters:
    - duration_str (str): A string representation of duration (e.g., "2h15m" or "2 h 15 m").

    Returns:
    - np.timedelta64: The corresponding timedelta object.
    """

    def convert_unit_for_numpy(freq_unit):
        """Convert pandas freq_unit to numpy's timedelta64 units."""
        conversion = {
            "min": "m",
            "T": "m",
            "d": "D",
            "w": "W"
        }
        return conversion.get(freq_unit, freq_unit)

    matches = re.findall(r'(\d+)?(\D+)', duration_str)

    total_duration = np.timedelta64(0)
    for value, unit in matches:
        # if unit only, without number, default to 1 of that unit
        numeric_value = int(value) if value else 1
        total_duration += np.timedelta64(numeric_value, convert_unit_for_numpy(unit.strip()))

    return total_duration


def calculate_interval_ratio(interval1: str, interval2_str: str) -> int:
    """
    Calculate the ratio of two time intervals based on the provided duration strings.

    I.e. how many interval2s fit into interval1.

    Parameters:
    - interval1 (str): A string representation of the first interval (e.g., "2h").
    - interval2_str (str): A string representation of the second interval (e.g., "1 h 15 m").

    Returns:
    - int: The ratio of the two intervals.
    """
    interval1_duration = parse_duration_string(interval1)
    interval2_duration = parse_duration_string(interval2_str)
    interval_ratio = interval1_duration // interval2_duration
    return int(interval_ratio)


class EWMAcc:
    __slots__ = ['alpha', 'minp', 'adjust', 'weighted_avg', 'nobs', 'old_wt', 'chunk_counter', 'n_cols']

    def __init__(self, n_cols: int, halflife: float = None, span: float = None, alpha: float = None, minp: int = None,
                 adjust: bool = False):
        params_provided = sum([halflife is not None, span is not None, alpha is not None])
        if params_provided != 1:
            raise ValueError("Must provide exactly one of halflife, span, or alpha.")

        if halflife is not None:
            self.alpha = 1 - np.exp(-np.log(2) / halflife)
        elif span is not None:
            self.alpha = 2.0 / (span + 1.0)
        else:
            self.alpha = alpha

        self.minp = minp if minp is not None else int(span) if span is not None else 1
        self.adjust = adjust
        self.n_cols = n_cols
        self.weighted_avg = np.zeros(self.n_cols, dtype=np.float64)
        self.nobs = np.zeros(self.n_cols, dtype=np.int64)
        self.old_wt = np.ones(self.n_cols, dtype=np.float64)
        self.chunk_counter = 0

    def reset(self):
        self.weighted_avg = np.zeros(self.n_cols, dtype=np.float64)
        self.nobs = np.zeros(self.n_cols, dtype=np.int64)
        self.old_wt = np.ones(self.n_cols, dtype=np.float64)
        self.chunk_counter = 0

    def apply_chunk(self, arr_2d: np.ndarray) -> np.ndarray:
        n_rows, cols = arr_2d.shape
        if cols != self.n_cols:
            raise ValueError("Mismatch in number of columns.")
        out = np.empty_like(arr_2d, dtype=np.float64)

        # Process each column
        for col in range(self.n_cols):
            for i in range(n_rows):
                # Adjust the index to account for chunks processed so far
                idx = i + self.chunk_counter * n_rows

                in_state = EWMMeanAIS(
                    i=idx,
                    value=arr_2d[i, col],
                    old_wt=self.old_wt[col],
                    weighted_avg=self.weighted_avg[col],
                    nobs=self.nobs[col],
                    alpha=self.alpha,
                    minp=self.minp,
                    adjust=self.adjust,
                )
                out_state = ewm_mean_acc_nb(in_state)

                # Update the state for the next iteration
                self.old_wt[col] = out_state.old_wt
                self.weighted_avg[col] = out_state.weighted_avg
                self.nobs[col] = out_state.nobs
                out[i, col] = out_state.value

        # Increment the chunk counter
        self.chunk_counter += 1

        return out

    def current_state(self):
        return self.weighted_avg, self.nobs, self.old_wt


if __name__ == '__main__':
    arr = np.array([[1, 2, 3],
                    [2, 3, 6],
                    [3, 5, 9]])
    print(pct_change_np(arr))
    # Sample usage:
    ewm = EWMAcc(halflife=.5, n_cols=2)
    arr = np.random.random((10, 2))
    print(arr)
    chunk1 = arr[:5, :]
    result1 = ewm.apply_chunk(chunk1)
    print(result1)

    chunk2 = arr[5:, :]
    result2 = ewm.apply_chunk(chunk2)
    print(result2)

    test_strings = ["2h15m", "1D24h", "2W14D", "75m1h15m", "2 h 15 m"]
    print([parse_duration_string(ts) for ts in test_strings])
    unit = "min"
    print([calculate_interval_ratio(ts, unit) for ts in test_strings])
