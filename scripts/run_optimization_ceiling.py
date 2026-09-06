"""
Ceiling sweep for optimization-based inference adaptation.

Evaluates whether scaling demonstration counts {10, 25, 50, 100} and
gradient steps K {10, 25, 50} enables the Optimization Model to break
the 0% exact-match barrier on ARC puzzle rules (translate, mirror, recolor).

Saves results to results/optimization_ceiling.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    puzzle_to_tensors, decode_grid, GRID_CELLS, NUM_COLORS
)
from optimization_model.model import OptimizationModel
from puzzle_generator.generator import generate_puzzle
from puzzle_generator.rules import get_all_params
from scripts.generate_dataset import _get_train_novelty_params


def load_opt_model(checkpoint_path: str, device: torch.device):
    """Load trained optimization model without modifying it."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']
    model = OptimizationModel(
        hidden_dim=args['hidden_dim'],
        num_layers=args.get('num_layers', 3),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, args


def generate_ceiling_test_puzzles(
    rule_types: List[str],
    puzzles_per_rule: int = 50,
    max_demos: int = 100,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate fixed test puzzle instances with max_demos demonstration pairs."""
    splits = _get_train_novelty_params()
    puzzles_by_rule = {}

    for rule_type in rule_types:
        params_list = splits[rule_type]['train']
        puzzles = []
        for i in range(puzzles_per_rule):
            rng = np.random.default_rng(seed + i * 100 + hash(rule_type) % 10000)
            param_idx = i % len(params_list)
            rule_params = params_list[param_idx]
            puzzle = generate_puzzle(
                rule_type=rule_type,
                rule_params=rule_params,
                num_demos=max_demos,
                rng=rng,
                index=i,
                max_attempts=500,
            )
            puzzles.append(puzzle)
        puzzles_by_rule[rule_type] = puzzles

    return puzzles_by_rule


def eval_opt_ceiling_configuration(
    model: OptimizationModel,
    puzzles: List[Dict[str, Any]],
    num_demos: int,
    K: int,
    inner_lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate optimization model on a puzzle set with given num_demos and K."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    cell_correct = 0
    cell_total = 0
    total = len(puzzles)
    latencies = []
    param_changes = []
    init_losses = []
    final_losses = []

    for p in puzzles:
        demos, test_in, target = puzzle_to_tensors(p, num_demos)
        demos = demos.to(device)
        test_in = test_in.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        result = model.adapt_and_predict(
            demos, test_in, K=K, inner_lr=inner_lr, criterion=criterion
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = result['logits'].squeeze(0)  # (25, 3)
        pred = logits.argmax(dim=-1)
        target_dev = target.to(device)
        if (pred == target_dev).all():
            correct += 1
        cell_correct += (pred == target_dev).sum().item()
        cell_total += GRID_CELLS

        param_changes.append(result['param_change_magnitude'])
        if result['loss_curve']:
            init_losses.append(result['loss_curve'][0])
            final_losses.append(result['loss_curve'][-1])

    return {
        'exact_match': round(correct / total, 4) if total > 0 else 0.0,
        'cell_accuracy': round(cell_correct / cell_total, 4) if cell_total > 0 else 0.0,
        'avg_latency_ms': round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        'avg_param_change': round(sum(param_changes) / len(param_changes), 4) if param_changes else 0.0,
        'avg_initial_loss': round(sum(init_losses) / len(init_losses), 4) if init_losses else 0.0,
        'avg_final_loss': round(sum(final_losses) / len(final_losses), 4) if final_losses else 0.0,
        'n_samples': total,
    }


def run_ceiling_sweep(
    checkpoint_path: str = 'optimization_model/checkpoint.pt',
    output_path: str = 'results/optimization_ceiling.json',
    puzzles_per_rule: int = 50,
    inner_lr: float = 0.01,
):
    """Run full demo count {10, 25, 50, 100} x K {10, 25, 50} ceiling sweep."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Optimization Ceiling Sweep on {device}...")
    model, args = load_opt_model(checkpoint_path, device)

    demo_counts = [10, 25, 50, 100]
    K_values = [10, 25, 50]
    rule_types = ['translate', 'mirror', 'recolor']

    print(f"Generating test puzzles (50 per rule, up to 100 demos)...")
    puzzles_by_rule = generate_ceiling_test_puzzles(
        rule_types=rule_types,
        puzzles_per_rule=puzzles_per_rule,
        max_demos=max(demo_counts),
        seed=42,
    )

    all_results = []
    first_nonzero_exact_match = None

    print("\n--- Starting Evaluation Grid ---")
    for rule in rule_types:
        puzzles = puzzles_by_rule[rule]
        for demos in demo_counts:
            for K in K_values:
                metrics = eval_opt_ceiling_configuration(
                    model=model,
                    puzzles=puzzles,
                    num_demos=demos,
                    K=K,
                    inner_lr=inner_lr,
                    device=device,
                )

                entry = {
                    'model': 'optimization',
                    'rule_type': rule,
                    'demo_count': demos,
                    'K_steps': K,
                    'inner_lr': inner_lr,
                    **metrics,
                }
                all_results.append(entry)

                if metrics['exact_match'] > 0 and first_nonzero_exact_match is None:
                    first_nonzero_exact_match = entry

                print(
                    f"Rule: {rule:<10} | Demos: {demos:3d} | K: {K:2d} | "
                    f"Exact: {metrics['exact_match']*100:5.1f}% | "
                    f"Cell Acc: {metrics['cell_accuracy']*100:5.2f}% | "
                    f"Final Loss: {metrics['avg_final_loss']:.4f} | "
                    f"Latency: {metrics['avg_latency_ms']:6.2f} ms"
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_output = {
        'experiment': 'optimization_ceiling_sweep',
        'tested_demo_counts': demo_counts,
        'tested_K_values': K_values,
        'tested_rules': rule_types,
        'puzzles_per_rule': puzzles_per_rule,
        'inner_lr': inner_lr,
        'first_nonzero_exact_match_threshold': first_nonzero_exact_match,
        'results': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(summary_output, f, indent=2)

    print(f"\nSaved ceiling results to {output_path}")

    # Print conclusion
    if first_nonzero_exact_match:
        print(f"\n[THRESHOLD FOUND]: Exact match first exceeded 0% at:")
        print(f"  Rule: {first_nonzero_exact_match['rule_type']}")
        print(f"  Demos: {first_nonzero_exact_match['demo_count']}")
        print(f"  K: {first_nonzero_exact_match['K_steps']}")
        print(f"  Exact Match: {first_nonzero_exact_match['exact_match']*100:.1f}%")
        print(f"  Cell Accuracy: {first_nonzero_exact_match['cell_accuracy']*100:.2f}%")
    else:
        print("\n[NO THRESHOLD FOUND]: Exact match remained strictly 0.0% across all tested regimes in range ({10, 25, 50, 100} demos x K {10, 25, 50}).")

    return summary_output


if __name__ == '__main__':
    run_ceiling_sweep()
