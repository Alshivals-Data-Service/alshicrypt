from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import json, torch
from .characters import Characters
from .model import Architecture

@dataclass
class Crypt:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    chars: Characters
    device: torch.device
    outdir: Optional[Path] = None

    def _apply(self, model: torch.nn.Module, text: str) -> str:
        model.eval()
        idxs, mask = [], []
        for ch in text:
            if ch in self.chars.characters: idxs.append(self.chars.characters.index(ch)); mask.append(True)
            else:                            idxs.append(None);                                mask.append(False)
        if any(mask):
            x = torch.tensor([i for i, m in zip(idxs, mask) if m], dtype=torch.long, device=self.device)
            with torch.no_grad(): pred = model(x).argmax(dim=-1).tolist()
        else:
            pred = []
        out, j = [], 0
        for ch, m in zip(text, mask):
            out.append(self.chars.characters[pred[j]] if m else ch)
            j += int(m)
        return ''.join(out)

    def encode(self, text: str) -> str: return self._apply(self.encoder, text)
    def decode(self, text: str) -> str: return self._apply(self.decoder, text)

    @classmethod
    def from_artifacts(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "Crypt":
        path = Path(path); device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        chars = Characters(); chars.characters = vocab["characters"]; chars.num_characters = len(chars.characters)
        emb_dim = 64
        cfg = path / "config.json"
        if cfg.exists():
            try: emb_dim = int(json.loads(cfg.read_text())["emb_dim"])
            except Exception: pass
        V = chars.num_characters
        enc = Architecture(V, emb_dim=emb_dim).to(device)
        dec = Architecture(V, emb_dim=emb_dim).to(device)
        enc.load_state_dict(torch.load(path / "encoder.pth", map_location=device))
        dec.load_state_dict(torch.load(path / "decoder.pth", map_location=device))
        return cls(encoder=enc, decoder=dec, chars=chars, device=device, outdir=path)
