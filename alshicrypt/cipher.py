import random
import torch
from types import SimpleNamespace as SN
from .characters import Characters

class Cipher:
    def __init__(self, seed: int | None = None):
        self.char = Characters()
        n = self.char.num_characters
        self.original_indices = list(range(n))
        self.shuffled_indices = self.original_indices.copy()
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(self.shuffled_indices)
        self.training_data = SN()
        self.training_data.encoder = torch.tensor(
            [self.original_indices, self.shuffled_indices], dtype=torch.long
        ).T  # original -> shuffled
        self.training_data.decoder = torch.tensor(
            [self.shuffled_indices, self.original_indices], dtype=torch.long
        ).T  # shuffled -> original
