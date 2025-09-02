"""
This file contains all the code to create a trainable dataframe and backtesting framework to test any algorithm on the data
"""

import pandas as pd
import numpy as np
from typing import Callable
from sklearn.metrics import accuracy_score, f1_score, classification_report


class CreateTrainableDf:

    def __init__(self, exchange_df: pd.DataFrame, config: dict) -> None:
        
        self.df = exchange_df
        self.price_col = config['price_col']
        self.lag = config['lag']
        self.df_created = False

    def create_t_lag(self) -> pd.DataFrame:

        for i in range(1, self.lag+1):
            self.df[f'{self.price_col}_lag_{i}'] = self.df[self.price_col].shift(i)

    def calculate_log_returns(self) -> pd.DataFrame:

        prices = self.df[self.price_col].astype(float)
        log_returns = np.log(prices / prices.shift(1))
        self.df['target'] = np.where(log_returns < 0, 1, 0)
        
        self.df.dropna(inplace=True)
    
    def create_trainable_df(self) -> pd.DataFrame:

        self.create_t_lag()
        self.calculate_log_returns()

        self.df_created = True

    def backtest(self, algorithm: Callable, last_n_days: int = 365) -> float:

        if self.df_created == False:
            print("Create a trainable df first by using create_trainable_df function")
            return
        
        backtest_df = self.df.copy().iloc[-last_n_days:]
        target_col = backtest_df['target'].to_list()
        backtest_df.drop(columns='target',inplace=True)
        outputs = []

        for day in range(0, last_n_days):
            return_row = np.array(backtest_df.iloc[day].to_list())
            output = algorithm(return_row)
            outputs.append(output)

        assert len(target_col) == len(outputs)

        acc_score = accuracy_score(target_col, outputs)

        print("Accuracy Score:", acc_score)







        

        


