#!/usr/bin/env python3
"""
Backtest Script - LINKUSDT 256-Hour Validation

Simple backtest to validate model predictions on recent data.
Tests on LINKUSDT with financial data and futures data available.

MANDATORY: Run this before merging any model changes to main.

Usage:
    .venv/bin/python scripts/backtest.py [--pair PAIR] [--hours HOURS] [--threshold SIGMA]

Pass Criteria:
    - Sharpe ratio ≥ 1.0
    - Win rate ≥ 50%
    - Max drawdown ≤ 25%
    - Reasonable trade count (2-15 in 256h)

Trading Logic:
    - BUY when prediction > +threshold σ
    - SELL when prediction < -threshold σ
    - Exit when prediction crosses zero
    - No leverage, simple spot trading simulation
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import (
    setup_logger,
    download_live_data,
    add_all_features,
    load_model,
    generate_signals
)

# Feature list (83 Enhanced V3 features)
ENHANCED_V3_FEATURES = [
    # Core indicators (20)
    'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel', 'bb_position', 'bb_position_vel',
    'macd_hist', 'macd_hist_vel', 'stoch', 'stoch_vel', 'di_diff', 'di_diff_vel',
    'adx', 'adr', 'adr_up_bars', 'adr_down_bars', 'is_up_bar', 'vol_ratio',
    'vol_ratio_vel', 'vwap_20',
    # Momentum features (24)
    'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20', 'price_accel_5',
    'price_accel_10', 'rsi_accel', 'rsi_7_accel', 'bb_pos_accel', 'macd_hist_accel',
    'stoch_accel', 'di_diff_accel', 'vol_regime_vel', 'vol_ratio_accel', 'atr',
    'atr_vel', 'rsi_2x', 'bb_pos_2x', 'macd_hist_2x', 'price_chg_2x', 'streak',
    'dist_from_high', 'dist_from_low', 'vwap_dist',
    # Advanced features (35)
    'rsi_4x', 'bb_pos_4x', 'macd_hist_4x', 'price_chg_4x', 'vol_ratio_4x',
    'rsi_bb_cross', 'macd_stoch_align', 'rsi_di_align', 'bb_macd_cross',
    'stoch_di_align', 'bb_stoch_cross', 'vol_regime_stable', 'vol_regime_rising',
    'vol_regime_falling', 'vol_regime_extreme', 'vol_spike', 'vol_drought',
    'new_high_20', 'new_low_20', 'range_position', 'higher_high', 'lower_low',
    'higher_low', 'lower_high', 'above_vwap_20', 'trend_consistent', 'adx_rising',
    'adx_falling', 'adx_stable', 'rsi_price_div', 'macd_price_div', 'stoch_price_div',
    'bb_price_div', 'di_price_div', 'vol_spike_extreme', 'vol_drought_extreme',
    # Statistical features (4)
    'hurst_exponent', 'permutation_entropy', 'cusum_signal', 'squeeze_duration'
]

# Pass criteria
PASS_CRITERIA = {
    'sharpe_min': 1.0,
    'win_rate_min': 0.50,
    'max_drawdown_max': 0.25,
    'min_trades': 2,
    'max_trades': 15
}


class SimpleBacktest:
    """Simple backtest simulator for spot trading."""

    def __init__(self, df, predictions, threshold=4.0, logger=None):
        self.df = df.copy()
        self.df['prediction'] = predictions
        self.df['signal'] = 0  # 0=hold, 1=long, -1=short
        self.threshold = threshold
        self.logger = logger or setup_logger('backtest')

        # Generate signals
        self.df.loc[self.df['prediction'] > threshold, 'signal'] = 1   # Buy signal
        self.df.loc[self.df['prediction'] < -threshold, 'signal'] = -1  # Sell signal

        # Trading state
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [1.0]  # Start with $1
        self.equity = 1.0

    def run(self):
        """Execute backtest."""
        self.logger.info(f"Running backtest with threshold={self.threshold}σ")
        self.logger.info(f"Data period: {len(self.df)} candles")

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            self._process_candle(i, row)

        # Close any open position
        if self.position != 0:
            self._close_position(len(self.df)-1, self.df.iloc[-1], reason='end')

        self.logger.info(f"Backtest complete. Total trades: {len(self.trades)}")

        return self._calculate_metrics()

    def _process_candle(self, i, row):
        """Process single candle."""
        signal = row['signal']
        pred = row['prediction']
        price = row['close']

        # Check for exit (prediction crosses zero)
        if self.position != 0:
            if (self.position == 1 and pred < 0) or (self.position == -1 and pred > 0):
                self._close_position(i, row, reason='zero_cross')

        # Check for entry
        if self.position == 0:
            if signal == 1:  # Buy signal
                self._open_position(i, row, direction=1)
            elif signal == -1:  # Sell signal
                self._open_position(i, row, direction=-1)

        # Update equity curve (mark-to-market)
        if self.position != 0:
            pnl_pct = (price / self.entry_price - 1) * self.position
            self.equity = 1.0 + pnl_pct
        else:
            self.equity = 1.0

        self.equity_curve.append(self.equity)

    def _open_position(self, i, row, direction):
        """Open new position."""
        self.position = direction
        self.entry_price = row['close']
        self.logger.debug(f"[{i}] OPEN {direction:+d} @ {self.entry_price:.2f} (pred={row['prediction']:.2f}σ)")

    def _close_position(self, i, row, reason='signal'):
        """Close existing position."""
        exit_price = row['close']
        pnl_pct = (exit_price / self.entry_price - 1) * self.position
        pnl_usd = pnl_pct * 1.0  # $1 position size

        trade = {
            'entry_idx': i - 1,  # Approximate
            'exit_idx': i,
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'reason': reason
        }
        self.trades.append(trade)

        self.logger.debug(f"[{i}] CLOSE {self.position:+d} @ {exit_price:.2f} | PnL: {pnl_pct:+.4f} ({reason})")

        self.position = 0
        self.entry_price = 0

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            self.logger.warning("No trades executed - cannot calculate metrics")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Win/loss stats
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]

        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0

        # Total return
        total_return = trades_df['pnl_pct'].sum()

        # Sharpe ratio (simplified - assumes 1H candles)
        returns = trades_df['pnl_pct'].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns))
        else:
            sharpe_ratio = 0

        # Max drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Profit factor
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'longs': len(trades_df[trades_df['direction'] == 1]),
            'shorts': len(trades_df[trades_df['direction'] == -1])
        }


def run_backtest(pair='LINKUSDT', hours=256, threshold=4.0, model_path=None):
    """Run complete backtest pipeline."""
    logger = setup_logger('backtest')

    # Check API credentials
    if not os.getenv('BINANCE_API') or not os.getenv('BINANCE_SECRET'):
        logger.error("BINANCE_API and BINANCE_SECRET environment variables required")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"BACKTEST - {pair} {hours}H @ {threshold}σ threshold")
    logger.info("=" * 80)

    # Load model
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'production.txt'

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run: .venv/bin/python scripts/04_train_model.py")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}")
    model = load_model(str(model_path))

    # Download data
    logger.info(f"\nDownloading {hours}H of {pair} data...")
    df = download_live_data(pair, hours=hours)
    logger.info(f"Downloaded {len(df)} candles")

    # Add features
    logger.info("\nAdding 83 Enhanced V3 features...")
    df = add_all_features(df)
    logger.info(f"Features added. Shape: {df.shape}")

    # Check for required features
    missing = [f for f in ENHANCED_V3_FEATURES if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        sys.exit(1)

    # Generate predictions
    logger.info("\nGenerating predictions...")
    X = df[ENHANCED_V3_FEATURES].values
    predictions = model.predict(X)
    df['prediction'] = predictions

    logger.info(f"Predictions generated. Range: [{predictions.min():.2f}, {predictions.max():.2f}]σ")

    # Run backtest
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 80)

    bt = SimpleBacktest(df, predictions, threshold=threshold, logger=logger)
    metrics = bt.run()

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nTotal Trades:    {metrics['total_trades']}")
    logger.info(f"  Longs:         {metrics['longs']}")
    logger.info(f"  Shorts:        {metrics['shorts']}")
    logger.info(f"\nWin Rate:        {metrics['win_rate']:.2%} ({'PASS' if metrics['win_rate'] >= PASS_CRITERIA['win_rate_min'] else 'FAIL'})")
    logger.info(f"Avg Win:         {metrics['avg_win']:+.4f}")
    logger.info(f"Avg Loss:        {metrics['avg_loss']:+.4f}")
    logger.info(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    logger.info(f"\nTotal Return:    {metrics['total_return']:+.4f}")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f} ({'PASS' if metrics['sharpe_ratio'] >= PASS_CRITERIA['sharpe_min'] else 'FAIL'})")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.2%} ({'PASS' if metrics['max_drawdown'] <= PASS_CRITERIA['max_drawdown_max'] else 'FAIL'})")

    # Pass/fail checks
    logger.info("\n" + "=" * 80)
    logger.info("PASS/FAIL CRITERIA")
    logger.info("=" * 80)

    checks = {
        'Sharpe Ratio ≥ 1.0': metrics['sharpe_ratio'] >= PASS_CRITERIA['sharpe_min'],
        'Win Rate ≥ 50%': metrics['win_rate'] >= PASS_CRITERIA['win_rate_min'],
        'Max Drawdown ≤ 25%': metrics['max_drawdown'] <= PASS_CRITERIA['max_drawdown_max'],
        'Trade Count Reasonable (2-15)': PASS_CRITERIA['min_trades'] <= metrics['total_trades'] <= PASS_CRITERIA['max_trades']
    }

    all_passed = True
    for check, passed in checks.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        logger.info(f"{status} - {check}")
        if not passed:
            all_passed = False

    # Red flags
    logger.info("\n" + "=" * 80)
    logger.info("RED FLAG CHECKS")
    logger.info("=" * 80)

    red_flags = []

    if metrics['total_trades'] == 0:
        red_flags.append("🚩 NO TRADES: Model generated no signals")

    if metrics['win_rate'] == 1.0 and metrics['total_trades'] > 0:
        red_flags.append("🚩 PERFECT WIN RATE: 100% wins is suspicious")

    if metrics['sharpe_ratio'] > 5.0:
        red_flags.append("🚩 SUSPICIOUSLY HIGH SHARPE: >5.0 is unrealistic")

    if metrics['max_drawdown'] > 0.50:
        red_flags.append("🚩 EXCESSIVE DRAWDOWN: >50% is unacceptable")

    if red_flags:
        for flag in red_flags:
            logger.error(flag)
        all_passed = False
    else:
        logger.info("✅ No red flags detected")

    # Final verdict
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ BACKTEST PASSED - Model performance acceptable")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("❌ BACKTEST FAILED - Performance issues detected")
        logger.error("=" * 80)
        logger.error("\nDO NOT MERGE TO MAIN. Investigate issues before proceeding.")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Backtest model on recent data')
    parser.add_argument('--pair', type=str, default='LINKUSDT', help='Trading pair (default: LINKUSDT)')
    parser.add_argument('--hours', type=int, default=256, help='Hours of data (default: 256)')
    parser.add_argument('--threshold', type=float, default=4.0, help='Signal threshold in σ (default: 4.0)')
    parser.add_argument('--model', type=str, help='Model path (default: models/production.txt)')

    args = parser.parse_args()

    sys.exit(run_backtest(
        pair=args.pair,
        hours=args.hours,
        threshold=args.threshold,
        model_path=args.model
    ))


if __name__ == '__main__':
    main()
