import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# --- Configuration ---
PREDICTION_FILES = {
    "Base Llama (Untrained)": Path("../../results/predictions/Base_Llama_predictions.csv"),
    "FinLLaMA (LoRA Tuned)": Path("../../results/predictions/FinLLaMA_predictions.csv")
}
OUTPUT_DIR = Path("../../results/figures")
TICKER = "SPY"

class ImprovedBacktester:
    """
    Improved backtesting framework addressing:
    1. Long-only strategy (no shorting)
    2. Confidence-weighted positions
    3. Transaction costs
    4. Execution delays
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,      # 10 bps per trade
        execution_delay: int = 1,              # 1-day lag
        base_position: float = 1.0,            # Always invested
        max_overweight: float = 0.2,           # Max 20% overweight
        confidence_threshold: float = 0.5,     # Min confidence to act
        use_confidence_weighting: bool = True
    ):
        self.transaction_cost = transaction_cost
        self.execution_delay = execution_delay
        self.base_position = base_position
        self.max_overweight = max_overweight
        self.confidence_threshold = confidence_threshold
        self.use_confidence_weighting = use_confidence_weighting
    
    def sentiment_to_weight(
        self, 
        sentiment_z_score: float,
        confidence: float = None,
        is_uptrend: bool = True
    ) -> float:
        """
        Convert relative sentiment z-score to portfolio weight.
        
        CRITICAL CHANGE: Long-only strategy based on Relative Sentiment
        - Baseline: 100% invested (tracks market)
        - Z-Score > 1.0 → Overweight (up to 120%)
        - Z-Score < -1.0 → Underweight (down to 80%)
        - Neutral (-1.0 to 1.0) → Market weight (100%)
        - Momentum Filter: If SMA trend is down, disable overweights, and bad sentiment -> 0% Cash.
        """
        # Determine direction from z-score
        if sentiment_z_score > 1.0:
            direction = 1  # positive deviation
        elif sentiment_z_score < -1.0:
            direction = -1 # negative deviation
        else:
            direction = 0  # neutral
            
        # Apply Momentum Filter
        if not is_uptrend:
            if direction == -1:
                # Negative sentiment + Downtrend = Cash out
                return 0.0
            elif direction == 1:
                # Positive sentiment + Downtrend = Cancel overweight (treat as neutral)
                direction = 0
        
        # If we have confidence scores, use them
        if self.use_confidence_weighting and confidence is not None:
            # Low confidence → stay at market weight
            if confidence < self.confidence_threshold:
                return self.base_position
            
            # High confidence → adjust position based on direction
            if direction == 1:  # Positive
                adjustment = confidence * self.max_overweight
                return self.base_position + adjustment
            elif direction == -1:  # Negative
                adjustment = confidence * self.max_overweight
                return self.base_position - adjustment
            else:  # Neutral
                return self.base_position
        else:
            # No confidence scores - use z-score signals
            if direction == 1:  # Positive
                return self.base_position + self.max_overweight
            elif direction == -1:  # Negative
                return self.base_position - self.max_overweight
            else:  # Neutral
                return self.base_position
    
    def apply_transaction_costs(self, positions: pd.Series) -> pd.Series:
        """Calculate transaction costs when position changes."""
        position_changes = positions.diff().abs()
        costs = position_changes * self.transaction_cost
        return costs
    
    def calculate_metrics(self, returns: pd.Series, name: str) -> Dict:
        """Calculate performance metrics."""
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0 or returns_clean.std() == 0:
            return {
                'name': name,
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        cumulative = (1 + returns_clean).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics
        years = len(returns_clean) / 252
        annual_return = (cumulative.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
        annual_volatility = returns_clean.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (returns_clean.mean() / returns_clean.std()) * np.sqrt(252)
        
        # Maximum drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns_clean > 0).mean()
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }


def process_predictions_to_signals(file_path: Path, use_confidence: bool = True) -> pd.DataFrame:
    """
    Process prediction file into daily trading signals using Relative Sentiment (Z-Score).
    
    Returns DataFrame with columns: date, predicted_sentiment, confidence, sentiment_z_score
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.floor('D')
    
    # Check if we have confidence scores
    has_confidence = 'confidence' in df.columns or 'probability' in df.columns
    
    if has_confidence and use_confidence:
        if 'confidence' in df.columns:
            confidence_col = 'confidence'
        elif 'probability' in df.columns:
            confidence_col = 'probability'
        
        # Group by date, take mean of predictions and confidence
        daily_signals = df.groupby('date').agg({
            'predicted_sentiment': 'mean',
            confidence_col: 'mean'
        }).reset_index()
        daily_signals.rename(columns={confidence_col: 'confidence'}, inplace=True)
    else:
        # No confidence scores - continuous sentiment
        daily_signals = df.groupby('date')['predicted_sentiment'].mean().reset_index()
        daily_signals['confidence'] = 1.0  # Dummy confidence
    
    daily_signals.set_index('date', inplace=True)
    
    # Step 1: Calculate Relative Sentiment (Z-Score)
    rolling_mean = daily_signals['predicted_sentiment'].rolling(window=14, min_periods=1).mean()
    rolling_std = daily_signals['predicted_sentiment'].rolling(window=14, min_periods=1).std()
    
    # Handle division by zero for the first few periods by adding a small epsilon
    daily_signals['sentiment_z_score'] = (daily_signals['predicted_sentiment'] - rolling_mean) / (rolling_std + 1e-8)
    
    return daily_signals


def main():
    print("="*70)
    print("IMPROVED PORTFOLIO BACKTESTER: LONG-ONLY WITH CONFIDENCE WEIGHTING")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize backtester
    backtester = ImprovedBacktester(
        transaction_cost=0.001,          # 10 bps per trade
        execution_delay=1,                # 1-day delay
        base_position=1.0,                # Always 100% invested
        max_overweight=0.2,               # Max 20% deviation
        confidence_threshold=0.5,         # 50% min confidence
        use_confidence_weighting=True     # Use confidence if available
    )
    
    # Process prediction files
    portfolio_data = {}
    min_date = None
    max_date = None
    
    for model_name, file_path in PREDICTION_FILES.items():
        if not file_path.exists():
            print(f"⚠️  Skipping {model_name} - File not found: {file_path}")
            continue
        
        print(f"📊 Processing {model_name}...")
        signals = process_predictions_to_signals(file_path, use_confidence=True)
        
        current_min = signals.index.min()
        current_max = signals.index.max()
        if min_date is None or current_min < min_date:
            min_date = current_min
        if max_date is None or current_max > max_date:
            max_date = current_max
        
        portfolio_data[model_name] = signals
        print(f"   ✓ {len(signals)} daily signals from {current_min.date()} to {current_max.date()}")
    
    if not portfolio_data:
        print("❌ No prediction data available. Exiting.")
        return
    
    # Fetch market data
    fetch_end = max_date + pd.Timedelta(days=5)
    print(f"\n📈 Fetching {TICKER} data from {min_date.date()} to {fetch_end.date()}...")
    
    market_data = yf.download(TICKER, start=min_date, end=fetch_end, progress=False)
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
    
    if market_data.empty:
        print("❌ Failed to fetch market data. Exiting.")
        return
    
    market_data['Market_Return'] = market_data['Close'].pct_change()
    market_data['SMA_50'] = market_data['Close'].rolling(window=50).mean()
    market_data['Is_Uptrend'] = (market_data['Close'] > market_data['SMA_50']).fillna(True)
    market_data.index = market_data.index.tz_localize(None).floor('D')
    
    # Build backtest dataframe
    backtest_df = market_data[['Market_Return', 'Is_Uptrend']].copy()
    backtest_df['Buy_and_Hold'] = (1 + backtest_df['Market_Return']).cumprod()
    
    all_metrics = []
    
    # Run backtest for each model
    for model_name, signals in portfolio_data.items():
        print(f"\n🔄 Backtesting {model_name}...")
        
        # Merge signals with market data
        backtest_df = backtest_df.join(signals, how='left')
        backtest_df[f'{model_name}_z_score'] = backtest_df['sentiment_z_score'].ffill()
        backtest_df[f'{model_name}_confidence'] = backtest_df['confidence'].ffill()
        
        # Convert sentiment to weights
        weights = []
        for idx, row in backtest_df.iterrows():
            if pd.isna(row[f'{model_name}_z_score']):
                weights.append(backtester.base_position)
            else:
                weight = backtester.sentiment_to_weight(
                    row[f'{model_name}_z_score'],
                    row[f'{model_name}_confidence'],
                    row['Is_Uptrend']
                )
                weights.append(weight)
        
        backtest_df[f'{model_name}_target_weight'] = weights
        
        # Apply execution delay
        backtest_df[f'{model_name}_actual_weight'] = backtest_df[f'{model_name}_target_weight'].shift(
            backtester.execution_delay
        )
        backtest_df[f'{model_name}_actual_weight'].fillna(backtester.base_position, inplace=True)
        
        # Calculate returns
        backtest_df[f'{model_name}_gross_return'] = (
            backtest_df['Market_Return'] * backtest_df[f'{model_name}_actual_weight']
        )
        
        # Apply transaction costs
        backtest_df[f'{model_name}_txn_cost'] = backtester.apply_transaction_costs(
            backtest_df[f'{model_name}_actual_weight']
        )
        
        backtest_df[f'{model_name}_net_return'] = (
            backtest_df[f'{model_name}_gross_return'] - backtest_df[f'{model_name}_txn_cost']
        )
        
        # Cumulative returns
        backtest_df[f'{model_name}_cumulative'] = (
            1 + backtest_df[f'{model_name}_net_return']
        ).cumprod()
        
        # Calculate metrics
        metrics = backtester.calculate_metrics(
            backtest_df[f'{model_name}_net_return'],
            model_name
        )
        all_metrics.append(metrics)
        
        # Clean up temporary columns
        backtest_df.drop(['predicted_sentiment', 'confidence'], axis=1, errors='ignore', inplace=True)
    
    # Add baseline metrics
    baseline_metrics = backtester.calculate_metrics(
        backtest_df['Market_Return'],
        'S&P 500 (Buy & Hold)'
    )
    all_metrics.append(baseline_metrics)
    
    # Display results
    print("\n" + "="*90)
    print("FINAL PERFORMANCE METRICS")
    print("="*90)
    
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*90)
    
    # Save detailed results
    results_path = OUTPUT_DIR / "backtest_detailed_results.csv"
    backtest_df.to_csv(results_path)
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_results(backtest_df, portfolio_data.keys(), OUTPUT_DIR)
    
    return backtest_df, metrics_df


def plot_results(backtest_df: pd.DataFrame, model_names: list, output_dir: Path):
    """Generate comprehensive visualization."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Cumulative returns
    ax1 = axes[0]
    
    # Baseline
    ax1.plot(backtest_df.index, backtest_df['Buy_and_Hold'],
             label='S&P 500 (Buy & Hold)', color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Strategy colors
    colors = {
        'Base Llama (Untrained)': '#e74c3c',
        'FinLLaMA (LoRA Tuned)': '#2ecc71'
    }
    
    for model_name in model_names:
        ax1.plot(backtest_df.index, backtest_df[f'{model_name}_cumulative'],
                label=model_name, color=colors.get(model_name, '#3498db'), linewidth=2)
    
    ax1.set_ylabel('Cumulative Growth (1.0 = Initial Investment)', fontsize=12, fontweight='bold')
    ax1.set_title('Improved Long-Only Sentiment Strategy vs Buy & Hold',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')  # Log scale to see relative performance
    
    # Plot 2: Portfolio weights over time
    ax2 = axes[1]
    
    for model_name in model_names:
        if f'{model_name}_actual_weight' in backtest_df.columns:
            ax2.plot(backtest_df.index, backtest_df[f'{model_name}_actual_weight'],
                    label=f'{model_name} Weight', color=colors.get(model_name, '#3498db'),
                    linewidth=1.5, alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Market Weight (100%)')
    ax2.fill_between(backtest_df.index, 0.8, 1.2, alpha=0.1, color='gray',
                     label='±20% Band')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Dynamic Position Sizing Based on Sentiment + Confidence',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.7, 1.3])
    
    plt.tight_layout()
    
    plot_path = output_dir / "improved_backtest_chart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Chart saved to: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    backtest_df, metrics_df = main()