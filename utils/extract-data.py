import yfinance as yf
import pandas as pd

def download_single_ticker_forex_data(ticker1: str, ticker2: str) -> pd.DataFrame:
    """
    Ticker1 is the current currency and ticker2 is the tagret currency, for ex if we wanna look
    how eur fares against usd, we can use ticker1 = EUR and ticker2 = USD
    """

    ticker1, ticker2 = ticker1.upper(), ticker2.upper()
    print(ticker1,ticker2)

    forex_data = yf.download(f"{ticker1}{ticker2}=X", start='2010-01-02', end='2025-10-01')
    
    print(forex_data)


if __name__ == "__main__":

    download_single_ticker_forex_data("EUR", "INR")