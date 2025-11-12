# Issue #1.1: Download Training Binance Data

**Parent:** Issue #1 (Pipeline Restructuring Epic)
**Phase:** 1 - Data Collection
**Dependencies:** None
**Estimated Effort:** 2-3 hours

## 🎯 Objective

Create a script to download long-term 1H candle data for 20 trading pairs from Binance, saving raw data to a reusable JSON file.

## 📋 Requirements

### Functional Requirements

1. Download 1H OHLCV candles for 20 pairs
2. Date range: 2021-01-01 to present (or configurable)
3. ~50,000 candles per pair expected
4. Save as raw JSON (no processing)
5. Handle API rate limits gracefully
6. Resume capability if download fails
7. Verify data completeness

### Non-Functional Requirements

- Robust error handling
- Progress logging
- Data validation
- Configurable pair list
- Configurable date range

## 📁 Files to Create/Modify

###Create: `scripts/01_download_training_binance.py`

```python
#!/usr/bin/env python3
"""
Download Training Binance Data - 20 Pairs, Long-Term

Downloads 1H OHLCV candles for 20 trading pairs from Binance.
Saves raw data to data/raw/training/binance_20pairs_1H.json.

Usage:
    .venv/bin/python scripts/01_download_training_binance.py [--start START] [--end END]

Arguments:
    --start: Start date (YYYY-MM-DD), default: 2021-01-01
    --end: End date (YYYY-MM-DD), default: today
    --pairs: Comma-separated list, default: 20 baseline pairs
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

from sneaker import setup_logger, download_historical_data

# 20 baseline pairs
BASELINE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "SUIUSDT", "LINKUSDT",
    "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "TRXUSDT",
    "ALGOUSDT", "APTUSDT", "AAVEUSDT", "XLMUSDT", "XMRUSDT"
]

def main():
    parser = argparse.ArgumentParser(description='Download training Binance data')
    parser.add_argument('--start', type=str, default='2021-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--pairs', type=str, default=None, help='Comma-separated pairs')

    args = parser.parse_args()

    logger = setup_logger('download_training_binance')

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()

    # Parse pairs
    pairs = args.pairs.split(',') if args.pairs else BASELINE_PAIRS

    logger.info(f"Downloading {len(pairs)} pairs from {start_date} to {end_date}")

    # Download all pairs
    all_data = []
    for i, pair in enumerate(pairs, 1):
        logger.info(f"[{i}/{len(pairs)}] Downloading {pair}...")

        # Download with retry logic
        df = download_historical_data(pair, start_date, end_date, interval='1h')

        if df is None or df.empty:
            logger.error(f"Failed to download {pair}")
            continue

        # Convert to records (list of dicts)
        df['pair'] = pair
        records = df.to_dict('records')
        all_data.extend(records)

        logger.info(f"  Downloaded {len(records)} candles for {pair}")

        # Rate limit protection
        time.sleep(0.5)

    # Save to file
    output_path = Path('data/raw/training/binance_20pairs_1H.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

    logger.info(f"Saved {len(all_data)} total candles to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Summary
    logger.info("\nSummary:")
    logger.info(f"  Pairs: {len(pairs)}")
    logger.info(f"  Total candles: {len(all_data)}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Output: {output_path}")

if __name__ == '__main__':
    main()
```

### Modify: `sneaker/data.py`

Add function:
```python
def download_historical_data(pair, start_date, end_date, interval='1h'):
    """
    Download historical OHLCV data from Binance.

    Args:
        pair: Trading pair (e.g., 'BTCUSDT')
        start_date: datetime object
        end_date: datetime object
        interval: Candle interval ('1h', '4h', '1d', etc.)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Implementation using python-binance
    # Handle pagination (Binance 1000 candle limit per request)
    # Handle rate limits
    # Retry logic on failures
    pass
```

## 🧪 Testing & Validation

### Unit Tests
- [ ] Test date parsing
- [ ] Test API call with single pair
- [ ] Test rate limit handling
- [ ] Test file writing

### Integration Tests
- [ ] Download 1 pair successfully
- [ ] Download all 20 pairs successfully
- [ ] Verify data completeness (no gaps)
- [ ] Verify file format (valid JSON)

### Manual Validation
```bash
# 1. Run script
.venv/bin/python scripts/01_download_training_binance.py

# 2. Verify output file exists
ls -lh data/raw/training/binance_20pairs_1H.json

# 3. Verify JSON structure
python -c "import json; data = json.load(open('data/raw/training/binance_20pairs_1H.json')); print(f'Records: {len(data)}'); print(f'Keys: {data[0].keys()}')"

# 4. Verify date range
python -c "import json; data = json.load(open('data/raw/training/binance_20pairs_1H.json')); print(f'First: {data[0][\"timestamp\"]}'); print(f'Last: {data[-1][\"timestamp\"]}')"

# 5. Verify all pairs present
python -c "import json; data = json.load(open('data/raw/training/binance_20pairs_1H.json')); pairs = set(d['pair'] for d in data); print(f'Pairs: {sorted(pairs)}')"
```

## ✅ Success Criteria

- [ ] Script downloads all 20 pairs without errors
- [ ] File `data/raw/training/binance_20pairs_1H.json` created
- [ ] File size ~500-1000 MB (depending on date range)
- [ ] Valid JSON format
- [ ] Expected columns present: timestamp, open, high, low, close, volume, pair
- [ ] No missing dates (or gaps documented)
- [ ] Date range matches parameters
- [ ] All 20 pairs present in data

## 📊 Expected Output

**File:** `data/raw/training/binance_20pairs_1H.json`

**Format:**
```json
[
  {
    "timestamp": 1609459200000,
    "open": 29000.0,
    "high": 29100.0,
    "low": 28900.0,
    "close": 29050.0,
    "volume": 1234.56,
    "pair": "BTCUSDT"
  },
  ...
]
```

**Size:** ~500-1000 MB
**Records:** ~1,000,000 candles (20 pairs × ~50,000 candles)

## 🚨 Edge Cases to Handle

1. **API rate limits** - Binance limits requests, need backoff
2. **Network failures** - Implement retry logic
3. **Incomplete data** - Some pairs may have gaps
4. **Large file size** - Ensure enough disk space
5. **Date boundaries** - Handle timezone correctly (Binance uses UTC)

## 🔗 Next Steps

After completion:
- [ ] Commit script to issue-1 branch
- [ ] Document any gaps or issues in data
- [ ] Move to Issue #1.2 (macro data)

## 📚 References

- Current `scripts/01_collect_data.py` - Existing download logic
- `sneaker/data.py` - Current Binance API integration
- [python-binance docs](https://python-binance.readthedocs.io/)
- Binance API rate limits: 1200 requests/minute

## 💬 Notes

- **DO NOT** include this file in .gitignore (raw data should be tracked)
- First download will take ~10-15 minutes
- Can be re-run to update data
- Consider compressed format if file too large
