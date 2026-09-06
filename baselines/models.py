"""
Baseline models and evaluation logic for ARC-like puzzles.

1. Copy-Input Baseline:
   Predicts test output as an exact unchanged copy of the test input.
   Useful for quantifying how much cell-level accuracy is attributable to
   trivial identity structure (e.g. background cells or static regions).

2. Majority-Color Baseline:
   Predicts every cell in the 5x5 grid as the most frequent color across
   all demonstration outputs.
   Useful for establishing the frequency-based prior baseline.
"""

import time
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import torch

GRID_SIZE = 5
GRID_CELLS = GRID_SIZE * GRID_SIZE  # 25
NUM_COLORS = 3


def predict_copy_input(puzzle: Dict[str, Any], num_demos: Optional[int] = None) -> List[List[int]]:
    """Predict test output as an exact copy of test input.

    Parameters
    ----------
    puzzle : dict
        Puzzle dictionary containing 'test_pair' with 'input'.
    num_demos : int, optional
        Unused for copy-input, but accepted for uniform interface.

    Returns
    -------
    List[List[int]]
        5x5 grid copied from test input.
    """
    test_in = puzzle['test_pair']['input']
    return [row[:] for row in test_in]


def predict_majority_color(puzzle: Dict[str, Any], num_demos: Optional[int] = None) -> List[List[int]]:
    """Predict every cell as the most frequent color across demo outputs.

    Parameters
    ----------
    puzzle : dict
        Puzzle dictionary containing 'demo_pairs' and 'test_pair'.
    num_demos : int, optional
        Number of demo pairs to consider. If None, uses all available demos.

    Returns
    -------
    List[List[int]]
        5x5 grid filled entirely with the majority color.
    """
    demos = puzzle.get('demo_pairs', [])
    if num_demos is not None:
        demos = demos[:num_demos]

    counts = Counter()
    for d in demos:
        out_grid = d.get('output', [])
        for row in out_grid:
            for val in row:
                counts[val] += 1

    if counts:
        # Most frequent color across demo outputs (tie-break by lowest index)
        majority_color = max(counts.keys(), key=lambda c: (counts[c], -c))
    else:
        majority_color = 0

    return [[majority_color for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


class CopyInputBaseline:
    """Copy-input baseline model class."""

    def __init__(self):
        self.name = 'copy_input'

    def predict(self, puzzle: Dict[str, Any], num_demos: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prediction = predict_copy_input(puzzle, num_demos)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        res = {
            'model': self.name,
            'prediction': prediction,
            'latency_ms': round(latency_ms, 3),
            'confidence': 1.0,
        }

        if 'output' in puzzle.get('test_pair', {}):
            gt = puzzle['test_pair']['output']
            res['ground_truth'] = gt
            res['correct'] = (prediction == gt)

        return res


class MajorityColorBaseline:
    """Majority-color baseline model class."""

    def __init__(self):
        self.name = 'majority_color'

    def predict(self, puzzle: Dict[str, Any], num_demos: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prediction = predict_majority_color(puzzle, num_demos)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        res = {
            'model': self.name,
            'prediction': prediction,
            'latency_ms': round(latency_ms, 3),
            'confidence': 1.0,
        }

        if 'output' in puzzle.get('test_pair', {}):
            gt = puzzle['test_pair']['output']
            res['ground_truth'] = gt
            res['correct'] = (prediction == gt)

        return res


def eval_baseline_on_puzzles(
    baseline_type: str,
    puzzles: List[Dict[str, Any]],
    num_demos: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a baseline model on a list of puzzles.

    Parameters
    ----------
    baseline_type : str
        'copy_input' or 'majority_color'
    puzzles : list of dict
        Puzzle instances.
    num_demos : int, optional
        Number of demos to use.

    Returns
    -------
    dict with exact_match, cell_accuracy, avg_latency_ms, n_samples.
    """
    if baseline_type == 'copy_input':
        predict_fn = predict_copy_input
    elif baseline_type == 'majority_color':
        predict_fn = predict_majority_color
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    total = len(puzzles)
    if total == 0:
        return {
            'exact_match': 0.0,
            'cell_accuracy': 0.0,
            'avg_latency_ms': 0.0,
            'n_samples': 0,
        }

    correct_exact = 0
    cell_correct = 0
    cell_total = 0
    latencies = []

    for p in puzzles:
        t0 = time.perf_counter()
        pred = predict_fn(p, num_demos)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        gt = p['test_pair']['output']
        if pred == gt:
            correct_exact += 1

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if pred[r][c] == gt[r][c]:
                    cell_correct += 1
        cell_total += GRID_CELLS

    return {
        'exact_match': correct_exact / total,
        'cell_accuracy': cell_correct / cell_total if cell_total > 0 else 0.0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0.0,
        'n_samples': total,
    }
