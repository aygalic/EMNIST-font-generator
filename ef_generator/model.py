import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = 0

        self.feature_multiplier = 4

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16*self.feature_multiplier, 3, stride=2, padding=1),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(16*self.feature_multiplier, 32*self.feature_multiplier, 3, stride=2, padding=1),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(32*self.feature_multiplier, 64, 7),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, self.latent_dim)
        self.fc_var = nn.Linear(64, self.latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(self.latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 32*self.feature_multiplier, 7),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(32*self.feature_multiplier, 16*self.feature_multiplier, 3, stride=2, padding=1, output_padding=1),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(16*self.feature_multiplier, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Dropout(self.dropout),
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
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    
        # Scale KL by ratio of dimensions
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) * (self.latent_dim / 784)
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

