import torch
from numpy import ndarray
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, X: ndarray, y: ndarray, cnn_mode=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        if cnn_mode:
            self.X = self.X.unsqueeze(1)  # CNN models require 4 dimensions
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
