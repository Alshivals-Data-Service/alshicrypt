from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.utils.data as data

from .cipher import Cipher
from .datasets import CipherDataset
from .model import Architecture


@dataclass
class TrainerConfig:
    epochs: int = 500
    batch_size: int = 32
    lr: float = 2e-3
    emb_dim: int = 64
    seed: int | None = None          # training RNG (None = non-deterministic)
    perm_seed: int | None = None     # permutation RNG (None = new random key)
    target_acc: float = 1.0
    outdir: str | None = None        # default artifacts/YYYYMMDD-HHMMSS
    use_cpu: bool = False            # force CPU
    num_workers: int = 0             # keep 0 for simplicity/repro; bump if you like
    pin_memory: bool = False         # set True if training on CUDA for small speedup


class CipherTrainer:
    """
    Trains encoder/decoder that learn a random substitution cipher.

    Instantiating this class runs training if autostart=True.

    Artifacts:
      - encoder.pth, decoder.pth
      - encoder_best.pth, decoder_best.pth
      - vocab.json
      - mapping.json   (includes perm_seed)
      - history.json   (per-epoch metrics)
      - README.txt
    """
    def __init__(self, cfg: TrainerConfig = TrainerConfig(), autostart: bool = True):
        self.cfg = cfg
        self.device = torch.device("cpu" if cfg.use_cpu or not torch.cuda.is_available() else "cuda")

        # Training reproducibility (optional). Production default is None -> leave random.
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(cfg.seed)

        # Data / permutation (perm_seed controls the KEY reproducibility)
        self.cipher = Cipher(seed=cfg.perm_seed)
        V = self.cipher.char.num_characters

        # Datasets / loaders
        self.enc_ds = CipherDataset(self.cipher.training_data.encoder)
        self.dec_ds = CipherDataset(self.cipher.training_data.decoder)
        self.enc_loader = data.DataLoader(
            self.enc_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )
        self.dec_loader = data.DataLoader(
            self.dec_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )

        # Models
        self.encoder = Architecture(V, emb_dim=cfg.emb_dim).to(self.device)
        self.decoder = Architecture(V, emb_dim=cfg.emb_dim).to(self.device)

        # Optimizers / loss
        self.criterion = nn.CrossEntropyLoss()
        self.enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr)

        # Outdir
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.outdir = Path(cfg.outdir or f"artifacts/{ts}")
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.best_enc, self.best_dec = 0.0, 0.0
        self.history: List[Dict[str, Any]] = []

        if autostart:
            self.run()

    # -------- public API --------
    def run(self):
        for epoch in range(1, self.cfg.epochs + 1):
            enc_loss_tr = self._train_one(self.encoder, self.enc_loader, self.enc_opt)
            dec_loss_tr = self._train_one(self.decoder, self.dec_loader, self.dec_opt)

            enc_loss_ev, enc_acc = self._evaluate(self.encoder, self.cipher.training_data.encoder)
            dec_loss_ev, dec_acc = self._evaluate(self.decoder, self.cipher.training_data.decoder)

            row = {
                "epoch": epoch,
                "enc_train_loss": enc_loss_tr,
                "enc_eval_loss": enc_loss_ev,
                "enc_acc": enc_acc,
                "dec_train_loss": dec_loss_tr,
                "dec_eval_loss": dec_loss_ev,
                "dec_acc": dec_acc,
            }
            self.history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"enc train {enc_loss_tr:.4f} eval {enc_loss_ev:.4f} acc {enc_acc:.3f} | "
                f"dec train {dec_loss_tr:.4f} eval {dec_loss_ev:.4f} acc {dec_acc:.3f}"
            )

            improved = False
            if enc_acc > self.best_enc:
                torch.save(self.encoder.state_dict(), self.outdir / "encoder_best.pth")
                self.best_enc = enc_acc
                improved = True
            if dec_acc > self.best_dec:
                torch.save(self.decoder.state_dict(), self.outdir / "decoder_best.pth")
                self.best_dec = dec_acc
                improved = True
            if improved:
                self._save_vocab_and_mapping()

            if enc_acc >= self.cfg.target_acc and dec_acc >= self.cfg.target_acc:
                self._finalize()
                print("✅ Both encoder and decoder reached target accuracy. Done.")
                return

        # If no early stop, still save finals
        self._finalize()

    # -------- internals --------
    def _train_one(self, model: nn.Module, loader: data.DataLoader, opt: torch.optim.Optimizer) -> float:
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=self.cfg.pin_memory)
            y = y.to(self.device, non_blocking=self.cfg.pin_memory)
            opt.zero_grad(set_to_none=True)
            logits = model(x)  # [B, V]
            loss = self.criterion(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, pairs_tensor: torch.Tensor) -> Tuple[float, float]:
        model.eval()
        x = pairs_tensor[:, 0].to(self.device)
        y = pairs_tensor[:, 1].to(self.device)
        logits = model(x)
        loss = self.criterion(logits, y).item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return loss, acc

    def _save_vocab_and_mapping(self):
        (self.outdir / "vocab.json").write_text(
            json.dumps({"characters": self.cipher.char.characters}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.outdir / "mapping.json").write_text(
            json.dumps(
                {
                    "original_indices": self.cipher.original_indices,
                    "shuffled_indices": self.cipher.shuffled_indices,
                    "perm_seed": self.cfg.perm_seed,   # record key-generation seed (None if random)
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _finalize(self):
        torch.save(self.encoder.state_dict(), self.outdir / "encoder.pth")
        torch.save(self.decoder.state_dict(), self.outdir / "decoder.pth")

        # Save history + config for auditing
        (self.outdir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        (self.outdir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")

        (self.outdir / "README.txt").write_text(
            "Artifacts for trained encoder/decoder on a random substitution cipher.\n"
            f"- device: {self.device}\n"
            f"- see config.json for full TrainerConfig\n"
            f"- see mapping.json for permutation details\n",
            encoding="utf-8",
        )
        self._save_vocab_and_mapping()
