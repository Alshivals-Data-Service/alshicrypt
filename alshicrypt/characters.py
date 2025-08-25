import string
import torch

class Characters:
    def __init__(self):
        LETTERS = string.ascii_letters
        DIGITS = string.digits
        PUNCTUATION = string.punctuation
        WHITESPACE = ' \t\n\r\x0b\x0c'
        self.characters = LETTERS + DIGITS + PUNCTUATION + WHITESPACE
        self.num_characters = len(self.characters)

    def read(self, indices: torch.Tensor) -> str:
        return ''.join(self.characters[int(i)] for i in indices)

    def index(self, text: str) -> torch.Tensor:
        return torch.tensor([self.characters.index(c) for c in text], dtype=torch.long)
