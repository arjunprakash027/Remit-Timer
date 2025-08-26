import yfinance as yf
import pandas as pd

def download_single_ticker_forex_data(ticker1: str, ticker2: str) -> pd.DataFrame:
    """
    Ticker1 is the current currency and ticker2 is the tagret currency, for ex if we wanna look
    how eur fares against usd, we can use ticker1 = EUR and ticker2 = USD
    """

    ticker1, ticker2 = ticker1.upper(), ticker2.upper()

    ticker_full = f"{ticker1}{ticker2}=X"
    forex_data = yf.download(ticker_full, start='2000-01-02', end='2025-10-01')
    
    forex_data.columns = ['_'.join(col) for col in forex_data.columns]
    
    return forex_data[[f'Close_{ticker_full}']]


if __name__ == "__main__":

    eur_to_inr = download_single_ticker_forex_data("EUR", "INR")
    print(eur_to_inr)