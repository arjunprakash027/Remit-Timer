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


def arima_train_and_pred(returns: np.ndarray, pred_period: int = 1, summary: bool = False) -> np.float64:

    """
    Trains a ARIMA model and uses it for prediction on data for pred_period days

    1 would mean the exhange value next day is less than what it was today and 0 is the opposite of that
    """
    
    model = ARIMA(returns, order=(2,1,2))
    model_fit = model.fit()
    
    if summary:
        print(model_fit.summary())

    forecast = model_fit.forecast(steps=pred_period)

    forecast_direction = 1 if forecast < returns[-1] else 0
    return forecast_direction

def backtest_arima(log_returns: pd.Series) -> pd.DataFrame:
    
    """
    Backtesting ARIMA only
    """

    #log_returns = log_returns.asfreq('D')
    #print("Log returns",len(log_returns))
    results = []
    actual = []
    for i in range(10, len(log_returns) - 1):
        prev_rows = log_returns.iloc[:i]
        forecast = arima_train_and_pred(log_returns=prev_rows, pred_period=1)
        results.append(forecast)
        actual.append(1 if log_returns.iloc[i+1] < 0 else 0)
    
    assert len(results) == len(actual), "Length of result and actual do not match"

    matches = (np.array(results) == np.array(actual))
    accuracy = matches.mean()

    print(f"Accuracy of direction changes: {accuracy}")

    return pd.DataFrame (
        {
        "original":actual,
        "predicted":results
        }
    )







    