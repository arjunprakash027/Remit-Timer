import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def simple_moving_average(price_movement: pd.DataFrame, price_row: str, periods: int = 50) -> pd.DataFrame:
    """
    Performing simple moving average on raw prices data to detect
    regime changes and stuff
    """

    price_movement[f'SMA_last_{periods}'] = price_movement[price_row].rolling(window=periods).mean()

    return price_movement


def arima_train_and_pred(log_returns: pd.Series, pred_period: int = 5, summary: bool = False) -> np.float64:

    """
    Trains a ARIMA model and uses it for prediction on data for pred_period days
    """
    
    model = ARIMA(log_returns, order=(1,0,1))
    model_fit = model.fit()
    
    if summary:
        print(model_fit.summary())

    forecast = model_fit.forecast(steps=pred_period)
    
    return forecast.values[0]

def backtest_arima(log_returns: pd.Series) -> pd.DataFrame:
    
    """
    Backtesting ARIMA only
    """

    results = []
    actual = []
    for i in range(3, len(log_returns)):
        prev_rows = log_returns.iloc[:i]
        forecast = arima_train_and_pred(log_returns=prev_rows, pred_period=1)
        results.append(forecast)
        actual.append(log_returns.iloc[i])
    
    return pd.DataFrame (
        {
        "original":actual,
        "predicted":results
        }
    )







    