"""
Baseline model prediction script.

Usage:
    python -m baselines.predict --baseline copy_input --puzzle puzzle.json
    python -m baselines.predict --baseline majority_color --puzzle_jsonl data/test.jsonl --limit 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.models import (
    CopyInputBaseline,
    MajorityColorBaseline,
    predict_copy_input,
    predict_majority_color,
)


def main():
    parser = argparse.ArgumentParser(description='Baseline model prediction')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['copy_input', 'majority_color'],
                        help='Which baseline model to run')
    parser.add_argument('--puzzle', type=str, default=None,
                        help='Path to single puzzle JSON file')
    parser.add_argument('--puzzle_jsonl', type=str, default=None,
                        help='Path to JSONL file with puzzles')
    parser.add_argument('--limit', type=int, default=5,
                        help='Max puzzles to process from JSONL')
    parser.add_argument('--num_demos', type=int, default=5,
                        help='Number of demos to use (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: stdout)')
    args = parser.parse_args()

    model = CopyInputBaseline() if args.baseline == 'copy_input' else MajorityColorBaseline()
    results = []

    if args.puzzle:
        with open(args.puzzle, 'r', encoding='utf-8') as f:
            puzzle = json.load(f)
        result = model.predict(puzzle, num_demos=args.num_demos)
        results.append(result)

    elif args.puzzle_jsonl:
        with open(args.puzzle_jsonl, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                puzzle = json.loads(line.strip())
                result = model.predict(puzzle, num_demos=args.num_demos)
                result['puzzle_id'] = puzzle.get('id', f'puzzle_{i}')
                results.append(result)

    output_str = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_str)

    if results:
        n_correct = sum(1 for r in results if r.get('correct', False))
        print(f"\nSummary [{args.baseline}]: {n_correct}/{len(results)} correct "
              f"({n_correct/len(results)*100:.1f}%)", file=sys.stderr)


if __name__ == '__main__':
    main()
