import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=26):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = 0.2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, mu, log_var = self(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, mu, log_var = self(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
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