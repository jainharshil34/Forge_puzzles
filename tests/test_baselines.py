import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.models import (
    CopyInputBaseline,
    MajorityColorBaseline,
    predict_copy_input,
    predict_majority_color,
    eval_baseline_on_puzzles,
)


class TestBaselines(unittest.TestCase):

    def setUp(self):
        # Synthetic sample puzzle
        self.sample_puzzle = {
            'id': 'test_puzzle_001',
            'rule_type': 'translate',
            'rule_params': {'direction': 'right', 'shift': 1},
            'demo_pairs': [
                {
                    'input': [
                        [0, 1, 2, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, 0, 1, 2],
                        [1, 2, 0, 0, 0],
                        [2, 0, 0, 0, 1],
                    ],
                    'output': [
                        [0, 0, 1, 2, 0],
                        [0, 0, 0, 1, 2],
                        [0, 0, 0, 0, 1],
                        [0, 1, 2, 0, 0],
                        [0, 2, 0, 0, 0],
                    ],
                },
                {
                    'input': [
                        [1, 1, 1, 0, 0],
                        [0, 2, 2, 2, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    'output': [
                        [0, 1, 1, 1, 0],
                        [0, 0, 2, 2, 2],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                },
            ],
            'test_pair': {
                'input': [
                    [2, 2, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                'output': [
                    [0, 2, 2, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            }
        }

    def test_copy_input_predict(self):
        pred = predict_copy_input(self.sample_puzzle)
        self.assertEqual(pred, self.sample_puzzle['test_pair']['input'])
        self.assertNotEqual(pred, self.sample_puzzle['test_pair']['output'])

    def test_majority_color_predict(self):
        # Demo outputs have color 0 as dominant majority
        pred = predict_majority_color(self.sample_puzzle)
        expected = [[0 for _ in range(5)] for _ in range(5)]
        self.assertEqual(pred, expected)

    def test_eval_baseline_on_puzzles(self):
        puzzles = [self.sample_puzzle]
        metrics_copy = eval_baseline_on_puzzles('copy_input', puzzles)
        self.assertEqual(metrics_copy['exact_match'], 0.0)
        # Test input and output share 21 of 25 matching background/overlapping cells
        self.assertGreater(metrics_copy['cell_accuracy'], 0.5)

        metrics_maj = eval_baseline_on_puzzles('majority_color', puzzles)
        self.assertEqual(metrics_maj['exact_match'], 0.0)
        # Output has 21 zero cells out of 25 (0.84)
        self.assertEqual(metrics_maj['cell_accuracy'], 21 / 25)

    def test_baseline_classes(self):
        cb = CopyInputBaseline()
        res_cb = cb.predict(self.sample_puzzle)
        self.assertIn('prediction', res_cb)
        self.assertIn('correct', res_cb)
        self.assertFalse(res_cb['correct'])

        mb = MajorityColorBaseline()
        res_mb = mb.predict(self.sample_puzzle)
        self.assertIn('prediction', res_mb)
        self.assertIn('correct', res_mb)
        self.assertFalse(res_mb['correct'])


if __name__ == '__main__':
    unittest.main()
