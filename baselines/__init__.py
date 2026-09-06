"""
Baseline evaluators for ARC puzzle transformations.

Provides:
- CopyInputBaseline: Predicts test output as an unchanged copy of test input.
- MajorityColorBaseline: Predicts every cell as the most frequent color across demonstration outputs.
"""

from baselines.models import (
    CopyInputBaseline,
    MajorityColorBaseline,
    predict_copy_input,
    predict_majority_color,
    eval_baseline_on_puzzles,
)

__all__ = [
    'CopyInputBaseline',
    'MajorityColorBaseline',
    'predict_copy_input',
    'predict_majority_color',
    'eval_baseline_on_puzzles',
]
