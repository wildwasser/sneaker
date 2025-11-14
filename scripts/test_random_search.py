#!/usr/bin/env python3
"""
Random Search Hyperparameter Test

Tests 8 random combinations of LightGBM hyperparameters to evaluate impact
on model performance.

Part of Issue #26

Usage:
    .venv/bin/python scripts/test_random_search.py

Output:
    - Trains 8 models with different hyperparameters
    - Saves proof artifacts to proof/issue-26a/ through proof/issue-26h/
    - Generates random_search_results.txt with comparison table
"""

import subprocess
import sys
import random
import time
from pathlib import Path
from datetime import datetime

# Setup random seed for reproducibility
random.seed(42)

# Parameter ranges for random search
PARAM_RANGES = {
    'num_leaves': [31, 63, 127, 255, 511],
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500, 2000, 2500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# Baseline for comparison (Issue #23 results)
BASELINE = {
    'issue': 'issue-23',
    'test_signal_r2': 0.1714,
    'train_signal_r2': 0.6214,
    'test_dir_acc': 3.57,
    'params': {
        'num_leaves': 255,
        'max_depth': 8,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}


def generate_random_params():
    """Generate a random parameter combination."""
    return {
        param: random.choice(values)
        for param, values in PARAM_RANGES.items()
    }


def generate_n_combinations(n):
    """Generate N unique random parameter combinations."""
    combinations = []
    seen = set()

    while len(combinations) < n:
        params = generate_random_params()
        # Create hashable representation
        param_tuple = tuple(sorted(params.items()))

        if param_tuple not in seen:
            seen.add(param_tuple)
            combinations.append(params)

    return combinations


def run_training(issue_name, params):
    """
    Run training with specified parameters.

    Args:
        issue_name: Issue folder name (e.g., 'issue-26a')
        params: Dictionary of hyperparameters

    Returns:
        (success, duration) tuple
    """
    print("")
    print("=" * 80)
    print(f"RUNNING: {issue_name}")
    print("=" * 80)
    print("Parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    print("")

    # Build command
    cmd = [
        '.venv/bin/python',
        'scripts/08_train_model.py',
        '--issue', issue_name,
        '--num-leaves', str(params['num_leaves']),
        '--max-depth', str(params['max_depth']),
        '--learning-rate', str(params['learning_rate']),
        '--n-estimators', str(params['n_estimators']),
        '--subsample', str(params['subsample']),
        '--colsample-bytree', str(params['colsample_bytree'])
    ]

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Training complete in {duration:.1f}s ({duration/60:.1f} minutes)")
            return True, duration
        else:
            print(f"✗ Training failed!")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"✗ Training timed out after {duration:.1f}s")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ Training error: {e}")
        return False, duration


def parse_training_report(issue_name):
    """
    Parse training report to extract key metrics.

    Args:
        issue_name: Issue folder name (e.g., 'issue-26a')

    Returns:
        Dictionary of metrics or None if parsing fails
    """
    proof_folder = Path('proof') / issue_name

    # Find most recent training report
    reports = list(proof_folder.glob('training_report_*.txt'))
    if not reports:
        print(f"  ⚠️  No training report found in {proof_folder}")
        return None

    report_path = max(reports, key=lambda p: p.stat().st_mtime)

    # Parse report
    try:
        with open(report_path, 'r') as f:
            content = f.read()

        metrics = {}

        # Extract metrics using string parsing
        for line in content.split('\n'):
            if 'Train Signal R²:' in line:
                metrics['train_signal_r2'] = float(line.split(':')[1].strip())
            elif 'Test Signal R²:' in line:
                metrics['test_signal_r2'] = float(line.split(':')[1].strip())
            elif 'Train Signal MAE:' in line:
                metrics['train_signal_mae'] = float(line.split(':')[1].strip().rstrip('σ'))
            elif 'Test Signal MAE:' in line:
                metrics['test_signal_mae'] = float(line.split(':')[1].strip().rstrip('σ'))
            elif line.strip().startswith('Test:') and 'Direction Accuracy' in content[:content.index(line) + 100]:
                # This is the direction accuracy test line
                metrics['test_dir_acc'] = float(line.split(':')[1].strip().rstrip('%'))

        if 'test_signal_r2' in metrics:
            print(f"  ✓ Parsed metrics: Test Signal R² = {metrics['test_signal_r2']:.4f}")
            return metrics
        else:
            print(f"  ⚠️  Failed to parse Test Signal R² from report")
            return None

    except Exception as e:
        print(f"  ⚠️  Error parsing report: {e}")
        return None


def generate_results_table(results):
    """
    Generate comparison table of all results.

    Args:
        results: List of (issue_name, params, metrics, duration) tuples

    Returns:
        String containing formatted table
    """
    output = []
    output.append("=" * 120)
    output.append("RANDOM SEARCH RESULTS - ISSUE #26")
    output.append("=" * 120)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    output.append("BASELINE (Issue #23):")
    output.append(f"  Test Signal R²:    {BASELINE['test_signal_r2']:.4f}")
    output.append(f"  Train Signal R²:   {BASELINE['train_signal_r2']:.4f}")
    output.append(f"  Test Direction:    {BASELINE['test_dir_acc']:.2f}%")
    output.append(f"  Parameters: {BASELINE['params']}")
    output.append("")
    output.append("=" * 120)
    output.append("RESULTS TABLE")
    output.append("=" * 120)
    output.append("")

    # Header
    header = f"{'Issue':<12} {'Test R²':>10} {'Train R²':>10} {'Dir %':>8} {'Time':>8} {'num_leaves':>11} {'max_depth':>10} {'lr':>8} {'n_est':>6} {'subsample':>10} {'colsample':>10}"
    output.append(header)
    output.append("-" * 120)

    # Sort by Test Signal R² (descending)
    sorted_results = sorted(results, key=lambda x: x[2]['test_signal_r2'] if x[2] else -1, reverse=True)

    # Rows
    for issue_name, params, metrics, duration in sorted_results:
        if metrics:
            test_r2 = metrics['test_signal_r2']
            train_r2 = metrics.get('train_signal_r2', 0)
            test_dir = metrics.get('test_dir_acc', 0)

            # Highlight if better than baseline
            marker = "★" if test_r2 > BASELINE['test_signal_r2'] else " "

            row = (
                f"{marker}{issue_name:<11} "
                f"{test_r2:>10.4f} "
                f"{train_r2:>10.4f} "
                f"{test_dir:>7.2f}% "
                f"{duration/60:>7.1f}m "
                f"{params['num_leaves']:>11} "
                f"{params['max_depth']:>10} "
                f"{params['learning_rate']:>8.3f} "
                f"{params['n_estimators']:>6} "
                f"{params['subsample']:>10.2f} "
                f"{params['colsample_bytree']:>10.2f}"
            )
            output.append(row)
        else:
            output.append(f"✗{issue_name:<11} FAILED")

    output.append("")
    output.append("=" * 120)
    output.append("SUMMARY")
    output.append("=" * 120)

    # Find best result
    successful_results = [(issue, params, metrics, dur) for issue, params, metrics, dur in sorted_results if metrics]

    if successful_results:
        best_issue, best_params, best_metrics, best_dur = successful_results[0]
        output.append(f"Best Test Signal R²: {best_metrics['test_signal_r2']:.4f} ({best_issue})")
        output.append(f"Improvement over baseline: {(best_metrics['test_signal_r2'] - BASELINE['test_signal_r2']):.4f} ({(best_metrics['test_signal_r2'] / BASELINE['test_signal_r2'] - 1) * 100:+.1f}%)")
        output.append("")
        output.append(f"Best parameters:")
        for param, value in best_params.items():
            output.append(f"  {param}: {value}")
        output.append("")

        # Count improvements
        improvements = sum(1 for _, _, m, _ in successful_results if m and m['test_signal_r2'] > BASELINE['test_signal_r2'])
        output.append(f"Combinations better than baseline: {improvements}/{len(successful_results)}")
    else:
        output.append("No successful runs!")

    output.append("")
    output.append("=" * 120)

    return "\n".join(output)


def main():
    """Main execution."""
    print("=" * 80)
    print("RANDOM SEARCH HYPERPARAMETER TEST - ISSUE #26")
    print("=" * 80)
    print("")
    print("Testing 8 random parameter combinations to evaluate impact on performance")
    print("")

    # Check that windowed data exists
    windowed_data = Path('data/features/windowed_training_data.json')
    if not windowed_data.exists():
        print(f"✗ Error: {windowed_data} not found!")
        print("  Run scripts/07_create_windows.py first")
        return 1

    print(f"✓ Windowed data found: {windowed_data}")
    print("")

    # Generate 8 random combinations
    print("Generating 8 random parameter combinations...")
    combinations = generate_n_combinations(8)
    print(f"✓ Generated {len(combinations)} unique combinations")
    print("")

    # Run training for each combination
    results = []
    issue_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    start_time_total = time.time()

    for i, params in enumerate(combinations):
        issue_name = f"issue-26{issue_letters[i]}"

        # Run training
        success, duration = run_training(issue_name, params)

        # Parse results
        if success:
            metrics = parse_training_report(issue_name)
        else:
            metrics = None

        results.append((issue_name, params, metrics, duration))

        # Small delay between runs to avoid resource contention
        if i < len(combinations) - 1:
            print("Waiting 5s before next run...")
            time.sleep(5)

    total_duration = time.time() - start_time_total

    print("")
    print("=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print("")

    # Generate results table
    table = generate_results_table(results)

    # Save to file
    output_path = Path('random_search_results.txt')
    with open(output_path, 'w') as f:
        f.write(table)

    print(f"✓ Results saved to: {output_path}")
    print("")

    # Print table to console
    print(table)

    return 0


if __name__ == '__main__':
    sys.exit(main())
