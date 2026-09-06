"""
Training script for the Attention-based Context Model variant.

Mirrors train.py under identical data splits, loss, and training rigor.

Usage:
    python -m context_model.train_attn [--state_dim 128] [--epochs 35] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import PuzzleDataset, collate_puzzles
from context_model.model_attn import ContextModelAttn
from context_model.train import compute_metrics, evaluate


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_exact = 0.0
    total_cell = 0.0
    n_batches = 0

    for demo_pairs, test_inputs, targets in loader:
        demo_pairs = [d.to(device) for d in demo_pairs]
        test_inputs = test_inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(demo_pairs, test_inputs)

        loss = criterion(logits.view(-1, 3), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics = compute_metrics(logits.detach(), targets)
        total_loss += loss.item()
        total_exact += metrics['exact_match']
        total_cell += metrics['cell_accuracy']
        n_batches += 1

    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'exact_match': total_exact / n_batches if n_batches > 0 else 0.0,
        'cell_accuracy': total_cell / n_batches if n_batches > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Train attention-based context model')
    parser.add_argument('--state_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='context_model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Architecture: ContextModelAttn (Self-Attention Pooling)")

    # Load datasets
    print("Loading datasets...")
    train_ds = PuzzleDataset(
        os.path.join(args.data_dir, 'train.jsonl'),
        num_demos=args.num_demos
    )
    val_ds = PuzzleDataset(
        os.path.join(args.data_dir, 'validation.jsonl'),
        num_demos=args.num_demos
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_puzzles, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_puzzles, num_workers=0
    )

    print(f"Train: {len(train_ds)} instances, Val: {len(val_ds)} instances")

    # Create model
    model = ContextModelAttn(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    best_val_exact = 0.0
    patience_counter = 0
    history = []

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint_attn.pt')

    print(f"\nTraining Attention Context Model for up to {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'TrLoss':>7} | {'TrExact':>7} | {'TrCell':>7} | "
          f"{'VaLoss':>7} | {'VaExact':>7} | {'VaCell':>7} | {'LR':>8} | {'Time':>5}")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:5d} | {train_metrics['loss']:7.4f} | "
              f"{train_metrics['exact_match']:7.4f} | "
              f"{train_metrics['cell_accuracy']:7.4f} | "
              f"{val_metrics['loss']:7.4f} | "
              f"{val_metrics['exact_match']:7.4f} | "
              f"{val_metrics['cell_accuracy']:7.4f} | "
              f"{current_lr:.1e} | {elapsed:5.1f}s")

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'elapsed': elapsed,
        })

        if val_metrics['exact_match'] > best_val_exact:
            best_val_exact = val_metrics['exact_match']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
                'param_count': param_count,
            }, checkpoint_path)
            print(f"  -> Saved best checkpoint (val_exact={best_val_exact:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience reached).")
                break

    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history_attn.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete. Best validation exact-match: {best_val_exact:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
