import pandas as pd
import numpy as np

def simple_moving_average(price_movement: pd.DataFrame, price_row: str, periods: int = 50) -> pd.DataFrame:
    """
    Performing simple moving average on raw prices data to detect
    regime changes and stuff
    """

    price_movement[f'SMA_last_{periods}'] = price_movement[price_row].rolling(window=periods).mean()

    return price_movement

