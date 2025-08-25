#!/usr/bin/env python3
"""
Entry point to train encoders/decoders via CipherTrainer.

Run:
    python3 -m alshicrypt.generator --epochs 200 --outdir artifacts/run1
"""

from __future__ import annotations
import argparse

from .trainer import CipherTrainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Generate encoder/decoder by training on a random substitution cipher.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=None, help="Training seed (None=non-deterministic)")
    p.add_argument("--perm-seed", type=int, default=None, help="Cipher Permutation seed (None=random key)")
    p.add_argument("--target-acc", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        seed=args.seed,
        perm_seed=args.perm_seed,
        target_acc=args.target_acc,
        outdir=args.outdir,
        use_cpu=args.cpu,
    )
    # Training runs during initialization
    CipherTrainer(cfg=cfg, autostart=True)


if __name__ == "__main__":
    main()
