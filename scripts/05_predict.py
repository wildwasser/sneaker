#!/usr/bin/env python3
"""
Step 5: Make Predictions on Live Data

Downloads recent data, adds features, and generates trading signals.

Usage:
    export BINANCE_API='your_key'
    export BINANCE_SECRET='your_secret'
    python scripts/05_predict.py --pair BTCUSDT --threshold 4.0
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sneaker.data import download_live_data
from sneaker.features import add_all_features
from sneaker.model import load_model, generate_signals
from sneaker.logging import setup_logger


def plot_predictions(df: pd.DataFrame, signals: list, threshold: float, pair: str, output_dir: str = 'visualizations'):
    """Create visualization of predictions and signals."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Plot 1: Price with signals
    ax1.plot(df['datetime'], df['close'], 'k-', linewidth=1, label='Price')

    # Mark buy/sell signals
    buy_mask = signals == 1
    sell_mask = signals == -1

    if buy_mask.any():
        ax1.scatter(df.loc[buy_mask, 'datetime'], df.loc[buy_mask, 'close'],
                   color='green', marker='^', s=100, label=f'BUY (>{threshold}σ)', zorder=5)

    if sell_mask.any():
        ax1.scatter(df.loc[sell_mask, 'datetime'], df.loc[sell_mask, 'close'],
                   color='red', marker='v', s=100, label=f'SELL (<-{threshold}σ)', zorder=5)

    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.set_title(f'{pair} - Predictions with {threshold}σ Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted magnitudes
    predictions = df['prediction'].values

    colors = ['green' if p > threshold else 'red' if p < -threshold else 'gray'
              for p in predictions]

    ax2.bar(df['datetime'], predictions, color=colors, width=0.03, alpha=0.7)
    ax2.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'Buy Threshold (+{threshold}σ)')
    ax2.axhline(-threshold, color='red', linestyle='--', linewidth=2, label=f'Sell Threshold (-{threshold}σ)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)

    ax2.set_ylabel('Predicted Reversal (σ)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{pair}_predictions_{threshold}sigma.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Generate trading signals')
    parser.add_argument('--pair', type=str, default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--hours', type=int, default=180, help='Hours of data to fetch (default: 180)')
    parser.add_argument('--threshold', type=float, default=4.0, help='Signal threshold in sigma (default: 4.0)')
    parser.add_argument('--model', type=str, default='models/production.txt', help='Model path')

    args = parser.parse_args()

    logger = setup_logger('predict')

    logger.info("="*80)
    logger.info("Step 5: Prediction on Live Data")
    logger.info("="*80)

    logger.info(f"\nParameters:")
    logger.info(f"  Pair: {args.pair}")
    logger.info(f"  Hours: {args.hours}")
    logger.info(f"  Threshold: {args.threshold}σ")
    logger.info(f"  Model: {args.model}")

    try:
        # 1. Download live data
        logger.info(f"\nDownloading {args.hours}h of live data for {args.pair}...")
        df = download_live_data(args.pair, hours=args.hours)

        if df.empty:
            logger.error("No data downloaded")
            return 1

        logger.info(f"  ✓ Downloaded {len(df)} candles")
        logger.info(f"  Time range: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} to "
                   f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
        logger.info(f"  Current price: ${df['close'].iloc[-1]:.4f}")

        # 2. Add features
        logger.info(f"\nAdding 83 Enhanced V3 features...")
        df = add_all_features(df)
        logger.info(f"  ✓ Features added")

        # 3. Load model
        logger.info(f"\nLoading model from {args.model}...")
        model = load_model(args.model)
        logger.info(f"  ✓ Model loaded")

        # 4. Prepare feature matrix
        feature_cols = [
            # Original 20
            'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel',
            'bb_position', 'bb_position_vel',
            'macd_hist', 'macd_hist_vel',
            'stoch', 'stoch_vel',
            'di_diff', 'di_diff_vel',
            'adx',
            'adr', 'adr_up_bars', 'adr_down_bars',
            'is_up_bar',
            'vol_ratio', 'vol_ratio_vel',
            'vwap_20',
            # Batch 1: 24 momentum
            'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20',
            'price_accel_5', 'price_accel_10',
            'rsi_accel', 'rsi_7_accel', 'bb_position_accel',
            'macd_hist_accel', 'stoch_accel', 'di_diff_accel',
            'vol_regime_vel', 'vol_ratio_accel', 'atr_14', 'atr_vel',
            'rsi_2x', 'bb_pos_2x', 'macd_hist_2x', 'price_change_2x',
            'is_up_streak', 'dist_from_high_20', 'dist_from_low_20', 'dist_from_vwap',
            # Batch 2: 35 advanced
            'rsi_4x', 'bb_pos_4x', 'macd_hist_4x', 'price_change_4x', 'vol_ratio_4x',
            'rsi_bb_interaction', 'macd_vol_interaction', 'rsi_stoch_interaction',
            'bb_vol_interaction', 'adx_di_interaction', 'price_rsi_momentum_align',
            'vol_percentile', 'vol_regime_low', 'vol_regime_med', 'vol_regime_high', 'vol_zscore',
            'is_new_high_20', 'is_new_low_20', 'price_range_position',
            'consecutive_higher_highs', 'consecutive_lower_lows', 'vwap_distance_pct',
            'price_20_high',
            'price_rsi_divergence', 'price_macd_divergence', 'price_stoch_divergence',
            'rsi_divergence_strength', 'macd_divergence_strength',
            'vol_momentum_5', 'is_high_volume', 'price_20_low',
            'adx_vel', 'adx_accel', 'is_strong_trend', 'is_weak_trend',
            # Batch 3: 4 statistical
            'hurst_exponent',
            'permutation_entropy',
            'cusum_signal',
            'squeeze_duration'
        ]

        X = df[feature_cols].values

        # 5. Generate signals
        logger.info(f"\nGenerating predictions...")
        signals, summary = generate_signals(model, X, threshold=args.threshold)

        # Store predictions in df
        df['prediction'] = model.predict(X)
        df['signal'] = signals

        logger.info(f"\n" + "="*80)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*80)
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Mean: {summary['pred_mean']:.4f}σ")
        logger.info(f"  Std:  {summary['pred_std']:.4f}σ")
        logger.info(f"  Min:  {summary['pred_min']:.4f}σ")
        logger.info(f"  Max:  {summary['pred_max']:.4f}σ")

        logger.info(f"\nSignals at {args.threshold}σ threshold:")
        logger.info(f"  BUY:  {summary['buy_count']} ({summary['buy_pct']:.1f}%)")
        logger.info(f"  SELL: {summary['sell_count']} ({summary['sell_pct']:.1f}%)")
        logger.info(f"  HOLD: {summary['hold_count']} ({summary['hold_pct']:.1f}%)")
        logger.info(f"  Total signals: {summary['buy_count'] + summary['sell_count']} ({summary['signal_pct']:.1f}%)")

        # 6. Current market assessment
        current_pred = df['prediction'].iloc[-1]
        current_signal = df['signal'].iloc[-1]

        logger.info(f"\n" + "="*80)
        logger.info("CURRENT MARKET ASSESSMENT")
        logger.info("="*80)
        logger.info(f"\nPair: {args.pair}")
        logger.info(f"Current Price: ${df['close'].iloc[-1]:.4f}")
        logger.info(f"Prediction: {current_pred:+.4f}σ")

        if current_signal == 1:
            logger.info(f"Signal: 🟢 BUY (prediction > +{args.threshold}σ)")
        elif current_signal == -1:
            logger.info(f"Signal: 🔴 SELL (prediction < -{args.threshold}σ)")
        else:
            logger.info(f"Signal: ⚪ HOLD (|prediction| ≤ {args.threshold}σ)")

        logger.info("="*80)

        # 7. Create visualization
        logger.info(f"\nCreating visualization...")
        output_path = plot_predictions(df, signals, args.threshold, args.pair)
        logger.info(f"  ✓ Saved to: {output_path}")

        logger.info("\n" + "="*80)
        logger.info("✅ PREDICTION COMPLETE")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Prediction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
