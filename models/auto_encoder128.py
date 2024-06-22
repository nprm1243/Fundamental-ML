import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from sklearn.model_selection import train_test_split

class EmotionDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]).float(), self.labels[idx]
    
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ELU(), nn.BatchNorm2d(64), #64x48x48
            nn.Conv2d(64, 64, 3), nn.ELU(), nn.BatchNorm2d(64),#64x46x46
            nn.Conv2d(64, 4, 3), nn.ELU(), nn.BatchNorm2d(4),  #4x44x44
        )
        self.flattern = nn.Flatten()
        self.linear1 = nn.Linear(4*44*44, 256)
        self.linear2 = nn.Linear(256, 4*44*44)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, 3), nn.ELU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3), nn.ELU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 1), nn.ELU(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flattern(x)
        x = self.linear1(x)
        return x

    def decode(self, x):
        x = self.linear2(x)
        x = x.reshape(-1, 4, 44, 44)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x