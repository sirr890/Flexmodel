import torch
from torch.utils.data import Dataset
import random


class DummyClass(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Agregar ruido aleatorio controlado
        noisy_value = self.data[index] + random.random()
        original_value = self.data[index]

        # Convertir a tensores
        return torch.tensor(noisy_value, dtype=torch.float32), torch.tensor(
            original_value, dtype=torch.float32
        )
