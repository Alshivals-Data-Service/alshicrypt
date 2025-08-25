#!/usr/bin/env python3
from __future__ import annotations
import argparse
from .trainer import CipherTrainer, TrainerConfig

def main():
    p = argparse.ArgumentParser("alshicrypt generator")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--perm-seed", type=int, default=None)
    p.add_argument("--target-acc", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    cfg = TrainerConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        emb_dim=args.emb_dim, seed=args.seed, perm_seed=args.perm_seed,
                        target_acc=args.target_acc, outdir=args.outdir, use_cpu=args.cpu)
    CipherTrainer(cfg=cfg, autostart=True)

if __name__ == "__main__":
    main()
