# src/data_prep/download_sp500_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_sp500_data(start_date='2000-01-01', end_date=None):
    """Download S&P 500 price data."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading S&P 500 data from {start_date} to {end_date}...")
    
    # Download data
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    
    # Calculate returns
    sp500['returns'] = sp500['Adj Close'].pct_change()
    
    # Prepare dataframe
    sp500_data = pd.DataFrame({
        'date': sp500.index,
        'close': sp500['Adj Close'].values,
        'sp500_returns': sp500['returns'].values
    })
    
    # Save
    sp500_data.to_csv('data/sp500_daily_returns.csv', index=False)
    print(f"Saved to data/sp500_daily_returns.csv")
    print(f"Total days: {len(sp500_data)}")
    
    return sp500_data

if __name__ == "__main__":
    download_sp500_data()