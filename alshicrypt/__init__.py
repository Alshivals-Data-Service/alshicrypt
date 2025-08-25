from .characters import Characters
from .cipher import Cipher
from .datasets import CipherDataset
from .model import Architecture
from .trainer import CipherTrainer, TrainerConfig
from .runtime import Crypt

__all__ = [
    "Characters", "Cipher", "CipherDataset", "Architecture",
    "CipherTrainer", "TrainerConfig", "Crypt",
    "generate", "load",
]
__version__ = "0.1.0"

def generate(*,
             epochs: int = 500,
             batch_size: int = 32,
             lr: float = 2e-3,
             emb_dim: int = 64,
             seed: int | None = None,          # training RNG (None = random)
             perm_seed: int | None = None,     # key RNG (None = random key)
             target_acc: float = 1.0,
             outdir: str | None = None,
             use_cpu: bool = False) -> Crypt:
    """Train a fresh encoder/decoder pair and return a Crypt runtime."""
    cfg = TrainerConfig(
        epochs=epochs, batch_size=batch_size, lr=lr, emb_dim=emb_dim,
        seed=seed, perm_seed=perm_seed, target_acc=target_acc,
        outdir=outdir, use_cpu=use_cpu
    )
    trainer = CipherTrainer(cfg=cfg, autostart=True)
    return Crypt.from_artifacts(trainer.outdir)

def load(path: str) -> Crypt:
    """Load a previously generated crypt from an artifacts folder."""
    return Crypt.from_artifacts(path)
