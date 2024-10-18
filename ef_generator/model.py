import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST


class CNNAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class EMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # Download the dataset
        EMNIST(self.data_dir, split='letters', train=True, download=True)
        EMNIST(self.data_dir, split='letters', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            emnist_full = EMNIST(self.data_dir, split='letters', train=True, transform=self.transform)
            self.emnist_train, self.emnist_val = random_split(emnist_full, [0.9, 0.1])

        if stage == 'test' or stage is None:
            self.emnist_test = EMNIST(self.data_dir, split='letters', train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.emnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.emnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.emnist_test, batch_size=self.batch_size)