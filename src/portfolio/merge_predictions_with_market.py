# src/portfolio/merge_predictions_with_market.py
import pandas as pd

def merge_predictions_with_sp500():
    """Merge sentiment predictions with S&P 500 returns."""
    
    # Load predictions
    finllama = pd.read_csv('results/predictions/finllama_predictions.csv')
    finllama['date'] = pd.to_datetime(finllama['date'])
    
    # Load S&P 500 data
    sp500 = pd.read_csv('data/sp500_daily_returns.csv')
    sp500['date'] = pd.to_datetime(sp500['date'])
    
    # Merge on date
    merged = pd.merge(finllama, sp500, on='date', how='inner')
    
    print(f"FinLLaMA predictions: {len(finllama)} rows")
    print(f"S&P 500 data: {len(sp500)} rows")
    print(f"Merged data: {len(merged)} rows")
    
    # Save
    merged.to_csv('data/predictions_with_returns.csv', index=False)
    print("Saved to data/predictions_with_returns.csv")
    
    return merged

if __name__ == "__main__":
    merge_predictions_with_sp500()