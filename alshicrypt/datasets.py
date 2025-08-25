import torch.utils.data as data

class CipherDataset(data.Dataset):
    def __init__(self, pairs_tensor):
        self.data = pairs_tensor
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.long(), y.long()
