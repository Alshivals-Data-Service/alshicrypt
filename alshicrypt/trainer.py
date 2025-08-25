from __future__ import annotations
import json, random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch, torch.nn as nn, torch.utils.data as data
from .cipher import Cipher
from .datasets import CipherDataset
from .model import Architecture

@dataclass
class TrainerConfig:
    epochs: int = 500
    batch_size: int = 32
    lr: float = 2e-3
    emb_dim: int = 64
    seed: int | None = None          # training RNG
    perm_seed: int | None = None     # permutation RNG (key)
    target_acc: float = 1.0
    outdir: str | None = None
    use_cpu: bool = False
    num_workers: int = 0
    pin_memory: bool = False

class CipherTrainer:
    def __init__(self, cfg: TrainerConfig = TrainerConfig(), autostart: bool = True):
        self.cfg = cfg
        self.device = torch.device("cpu" if cfg.use_cpu or not torch.cuda.is_available() else "cuda")
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed); random.seed(cfg.seed)
            if self.device.type == "cuda": torch.cuda.manual_seed_all(cfg.seed)
        self.cipher = Cipher(seed=cfg.perm_seed)
        V = self.cipher.char.num_characters
        self.enc_ds = CipherDataset(self.cipher.training_data.encoder)
        self.dec_ds = CipherDataset(self.cipher.training_data.decoder)
        self.enc_loader = data.DataLoader(self.enc_ds, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        self.dec_loader = data.DataLoader(self.dec_ds, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        self.encoder = Architecture(V, emb_dim=cfg.emb_dim).to(self.device)
        self.decoder = Architecture(V, emb_dim=cfg.emb_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.outdir = Path(cfg.outdir or f"artifacts/{ts}"); self.outdir.mkdir(parents=True, exist_ok=True)
        self.best_enc = self.best_dec = 0.0
        self.history: List[Dict[str, Any]] = []
        if autostart: self.run()

    def run(self):
        for epoch in range(1, self.cfg.epochs + 1):
            enc_tr = self._train_one(self.encoder, self.enc_loader, self.enc_opt)
            dec_tr = self._train_one(self.decoder, self.dec_loader, self.dec_opt)
            enc_ev, enc_acc = self._evaluate(self.encoder, self.cipher.training_data.encoder)
            dec_ev, dec_acc = self._evaluate(self.decoder, self.cipher.training_data.decoder)
            self.history.append({"epoch": epoch, "enc_train_loss": enc_tr, "enc_eval_loss": enc_ev,
                                 "enc_acc": enc_acc, "dec_train_loss": dec_tr, "dec_eval_loss": dec_ev,
                                 "dec_acc": dec_acc})
            improved = False
            if enc_acc > self.best_enc: torch.save(self.encoder.state_dict(), self.outdir / "encoder_best.pth"); self.best_enc = enc_acc; improved = True
            if dec_acc > self.best_dec: torch.save(self.decoder.state_dict(), self.outdir / "decoder_best.pth"); self.best_dec = dec_acc; improved = True
            if improved: self._save_vocab_and_mapping()
            if enc_acc >= self.cfg.target_acc and dec_acc >= self.cfg.target_acc:
                self._finalize(); return
        self._finalize()

    def _train_one(self, model: nn.Module, loader: data.DataLoader, opt: torch.optim.Optimizer) -> float:
        model.train(); total = count = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=self.cfg.pin_memory)
            y = y.to(self.device, non_blocking=self.cfg.pin_memory)
            opt.zero_grad(set_to_none=True); logits = model(x); loss = self.criterion(logits, y)
            loss.backward(); opt.step(); total += loss.item() * y.size(0); count += y.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, pairs_tensor: torch.Tensor) -> Tuple[float, float]:
        model.eval(); x = pairs_tensor[:, 0].to(self.device); y = pairs_tensor[:, 1].to(self.device)
        logits = model(x); loss = self.criterion(logits, y).item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return loss, acc

    def _save_vocab_and_mapping(self):
        (self.outdir / "vocab.json").write_text(
            json.dumps({"characters": self.cipher.char.characters}, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.outdir / "mapping.json").write_text(
            json.dumps({"original_indices": self.cipher.original_indices,
                        "shuffled_indices": self.cipher.shuffled_indices,
                        "perm_seed": self.cfg.perm_seed}, indent=2), encoding="utf-8")

    def _finalize(self):
        torch.save(self.encoder.state_dict(), self.outdir / "encoder.pth")
        torch.save(self.decoder.state_dict(), self.outdir / "decoder.pth")
        (self.outdir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        (self.outdir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        (self.outdir / "README.txt").write_text(
            "Artifacts for trained encoder/decoder on a random substitution cipher.\n"
            "- see config.json for TrainerConfig\n- see mapping.json for permutation\n", encoding="utf-8")
        self._save_vocab_and_mapping()
