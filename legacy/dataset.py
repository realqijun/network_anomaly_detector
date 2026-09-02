import torch
from torch.utils.data import Dataset


class TemporalDataset(Dataset):
    def __init__(self, data, window_size=10, stride=5):
        self.data = torch.tensor(data, dtype=torch.float32).clone().detach()
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size

        window = self.data[start:end]
        window = window.transpose(0, 1)

        return window, window
