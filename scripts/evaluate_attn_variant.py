"""
Evaluation script for Attention-based Context Model vs GRU Context Model.

Compares:
1. Sweep performance across demo counts (1-5) and rule families (translate, mirror, recolor).
2. Unseen rule generalization (held-out recolor).

Saves to:
  results/sweep_attn_variant.json
  results/generalization_attn_variant.json
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    PuzzleDataset, collate_puzzles, load_puzzles, puzzle_to_tensors,
    GRID_CELLS, NUM_COLORS
)
from context_model.model import ContextModel
from context_model.model_attn import ContextModelAttn
from scripts.run_experiments import (
    load_context_model, filter_by_rule
)


def load_attn_context_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']
    model = ContextModelAttn(
        state_dim=args['state_dim'],
        hidden_dim=args['hidden_dim'],
        num_heads=args.get('num_heads', 4),
        num_layers=args.get('num_layers', 1),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def eval_attn_model(model: ContextModelAttn, puzzles: List[Dict[str, Any]], num_demos: int, device: torch.device):
    model.eval()
    correct = 0
    cell_correct = 0
    cell_total = 0
    total = len(puzzles)
    latencies = []

    for p in puzzles:
        demos, test_in, target = puzzle_to_tensors(p, num_demos)
        demos = demos.to(device)
        test_in = test_in.to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            logits, _ = model.forward_with_state_trace(demos, test_in)
            latencies.append((time.perf_counter() - t0) * 1000)

        pred = logits.argmax(dim=-1)
        if (pred == target.to(device)).all():
            correct += 1
        cell_correct += (pred == target.to(device)).sum().item()
        cell_total += GRID_CELLS

    return {
        'exact_match': correct / total if total > 0 else 0.0,
        'cell_accuracy': cell_correct / cell_total if cell_total > 0 else 0.0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0.0,
        'n_samples': total,
    }


def run_sweep_attn(
    attn_model: ContextModelAttn,
    data_dir: str = 'data',
    device: torch.device = torch.device('cpu'),
    max_puzzles_per_group: int = 50,
) -> List[Dict[str, Any]]:
    splits = {}
    for name in ['test', 'novelty']:
        path = os.path.join(data_dir, f'{name}.jsonl')
        if os.path.exists(path):
            splits[name] = load_puzzles(path)

    demo_counts = [1, 2, 3, 4, 5]
    rule_types = ['translate', 'mirror', 'recolor']
    novelty_labels = {'test': 'seen', 'novelty': 'unseen'}

    results = []
    for split_name, puzzles in splits.items():
        novelty_label = novelty_labels[split_name]
        for rule_type in rule_types:
            rule_puzzles = filter_by_rule(puzzles, rule_type)
            if not rule_puzzles:
                continue
            subset = rule_puzzles[:max_puzzles_per_group]

            for num_demos in demo_counts:
                metrics = eval_attn_model(attn_model, subset, num_demos, device)
                results.append({
                    'model': 'context_attention',
                    'rule_type': rule_type,
                    'novelty': novelty_label,
                    'demo_count': num_demos,
                    'state_size': attn_model.state_dim,
                    'num_heads': attn_model.num_heads,
                    'exact_match': round(metrics['exact_match'], 4),
                    'cell_accuracy': round(metrics['cell_accuracy'], 4),
                    'latency_ms': round(metrics['avg_latency_ms'], 2),
                })

    return results


def run_generalization_attn(
    data_dir: str = 'data',
    device: torch.device = torch.device('cpu'),
    state_dim: int = 128,
    hidden_dim: int = 256,
    num_heads: int = 4,
    epochs: int = 15,
    batch_size: int = 64,
    num_demos: int = 5,
    seed: int = 42,
    max_eval: int = 200,
) -> Dict[str, Any]:
    print("\n--- Training Attention Model on Translate+Mirror (Held-out Recolor) ---")
    import tempfile
    from context_model.train_attn import train_epoch
    from context_model.train import evaluate

    all_train = load_puzzles(os.path.join(data_dir, 'train.jsonl'))
    all_val = load_puzzles(os.path.join(data_dir, 'validation.jsonl'))

    train_tm = [p for p in all_train if p['rule_type'] in ('translate', 'mirror')]
    val_tm = [p for p in all_val if p['rule_type'] in ('translate', 'mirror')]

    tmp_dir = tempfile.mkdtemp()
    train_path = os.path.join(tmp_dir, 'train_tm.jsonl')
    val_path = os.path.join(tmp_dir, 'val_tm.jsonl')

    with open(train_path, 'w') as f:
        for p in train_tm:
            f.write(json.dumps(p) + '\n')
    with open(val_path, 'w') as f:
        for p in val_tm:
            f.write(json.dumps(p) + '\n')

    train_ds = PuzzleDataset(train_path, num_demos=num_demos)
    val_ds = PuzzleDataset(val_path, num_demos=num_demos)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_puzzles)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_puzzles)

    torch.manual_seed(seed)
    model = ContextModelAttn(state_dim=state_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_exact = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if val_metrics['exact_match'] > best_val_exact:
            best_val_exact = val_metrics['exact_match']
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_test = load_puzzles(os.path.join(data_dir, 'test.jsonl'))
    all_novelty = load_puzzles(os.path.join(data_dir, 'novelty.jsonl'))

    results = {
        'training_rules': ['translate', 'mirror'],
        'held_out_rule': 'recolor',
        'evaluations': [],
    }

    for rule_type in ['translate', 'mirror', 'recolor']:
        for split_name, puzzles_all in [('test', all_test), ('novelty', all_novelty)]:
            rule_puzzles = filter_by_rule(puzzles_all, rule_type)[:max_eval]
            if not rule_puzzles:
                continue
            metrics = eval_attn_model(model, rule_puzzles, num_demos, device)
            results['evaluations'].append({
                'model': 'context_attention',
                'rule_type': rule_type,
                'split': split_name,
                'seen_during_training': rule_type in ('translate', 'mirror'),
                'exact_match': round(metrics['exact_match'], 4),
                'cell_accuracy': round(metrics['cell_accuracy'], 4),
                'n_samples': metrics['n_samples'],
            })

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate attention context model variant')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--attn_checkpoint', type=str, default='context_model/checkpoint_attn.pt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--max_puzzles', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.attn_checkpoint):
        print(f"Error: {args.attn_checkpoint} not found. Please train first.")
        sys.exit(1)

    print(f"Loading attention model from {args.attn_checkpoint}...")
    attn_model, _ = load_attn_context_model(args.attn_checkpoint, device)

    # 1. Sweep
    print("\n=== Running Attention Model Demo Sweep ===")
    sweep_results = run_sweep_attn(attn_model, data_dir=args.data_dir, device=device, max_puzzles_per_group=args.max_puzzles)
    sweep_file = os.path.join(args.results_dir, 'sweep_attn_variant.json')
    with open(sweep_file, 'w', encoding='utf-8') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved sweep to {sweep_file}")

    # 2. Generalization
    print("\n=== Running Attention Model Generalization Experiment ===")
    gen_results = run_generalization_attn(data_dir=args.data_dir, device=device)
    gen_file = os.path.join(args.results_dir, 'generalization_attn_variant.json')
    with open(gen_file, 'w', encoding='utf-8') as f:
        json.dump(gen_results, f, indent=2)
    print(f"Saved generalization to {gen_file}")


if __name__ == '__main__':
    main()
