#!/bin/bash
# Run complete Issue #23 pipeline: scripts 05-10
# Tests reduced feature set (89 vs 93)

set -e  # Exit on error

ISSUE="issue-23"

echo "================================================================================"
echo "Issue #23 Pipeline: Testing 89 Feature Set"
echo "================================================================================"
echo ""
echo "Changes:"
echo "  - Removed 4 macro close features (zero importance)"
echo "  - 93 → 89 shared features"
echo "  - 1,116 → 1,068 windowed features (12-candle)"
echo ""
echo "================================================================================"

# Wait for script 06 if still running
echo ""
echo "[1/6] Waiting for script 06 to complete..."
while pgrep -f "06_add_training_features.py" > /dev/null; do
    sleep 5
    echo "  Still running..."
done
echo "✓ Script 06 completed"

# Remove old windowed data
echo ""
echo "[2/6] Removing old windowed data..."
rm -f data/features/windowed_training_data.json
echo "✓ Old windowed data removed"

# Run script 07 - Create windows
echo ""
echo "[3/6] Running script 07: Create windows (12-candle, 4 workers)..."
.venv/bin/python scripts/07_create_windows.py --workers 4
echo "✓ Script 07 completed"

# Run script 08 - Train model
echo ""
echo "[4/6] Running script 08: Train model (issue-23)..."
.venv/bin/python scripts/08_train_model.py --issue $ISSUE
echo "✓ Script 08 completed"

# Run script 09 - Create prediction windows
echo ""
echo "[5/6] Running script 09: Create prediction windows..."
.venv/bin/python scripts/09_create_prediction_windows.py
echo "✓ Script 09 completed"

# Run script 10 - Generate predictions
echo ""
echo "[6/6] Running script 10: Generate predictions..."
.venv/bin/python scripts/10_generate_predictions.py
echo "✓ Script 10 completed"

echo ""
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - proof/$ISSUE/"
echo "  - models/issue-1/model.txt (updated)"
echo "  - data/predictions/latest.json"
echo ""
echo "Review:"
echo "  - cat proof/$ISSUE/training_report_*.txt"
echo "  - open proof/$ISSUE/*.png"
echo ""
