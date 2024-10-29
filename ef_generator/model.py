from typing import Literal
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainedVAE(pl.LightningModule):
    def __init__(self, latent_dim=4, n_annealing_steps=1000, sigma = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = 0

        self.feature_multiplier = 4

        self.training_phase : Literal["supervised", "vae"] = "supervised"
        # Annealing parameters
        self.n_annealing_steps = n_annealing_steps
        self.sigma = sigma

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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 26),  # 26 letters, logits
        )

    
    def set_training_phase(self, training_phase : Literal["supervised", "vae"]):
        assert training_phase in ["supervised", "vae"]
        self.training_phase = training_phase
    
    def get_kl_weight(self):
        # Linear annealing from 0 to 1 over n_annealing_steps
        if self.trainer.global_step > self.n_annealing_steps:
            return 1.0 * self.sigma
        return (self.trainer.global_step / self.n_annealing_steps) * self.sigma

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
        encoded = self.encoder(x)
        if self.training_phase == 'supervised':
            return self.classifier(encoded)
        mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.training_phase == 'supervised':
            # Classification training
            logits = self(x)
            loss = F.cross_entropy(logits, y-1)  # Subtract 1 because EMNIST labels start at 1
            acc = (logits.argmax(dim=1) == (y-1)).float().mean()
            self.log('train_loss', loss)
            self.log('train_acc', acc)
            return loss
        else:
            # VAE training
            x_hat, mu, log_var = self(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) * (self.latent_dim / 784)
            kl_weight = self.get_kl_weight()
            loss = recon_loss + kl_weight * kl_div
            self.log('train_loss', loss)
            self.log('train_recon_loss', recon_loss)
            self.log('train_kl_div', kl_div)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.training_phase == 'supervised':
            logits = self(x)
            loss = F.cross_entropy(logits, y-1)
            acc = (logits.argmax(dim=1) == (y-1)).float().mean()
            self.log('val_loss', loss)
            self.log('val_acc', acc)
        else:
            x_hat, mu, log_var = self(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) * (self.latent_dim / 784)
            kl_weight = self.get_kl_weight()
            loss = recon_loss + kl_weight * kl_div
            self.log('val_loss', loss)
            self.log('val_recon_loss', recon_loss)
            self.log('val_kl_div', kl_div)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

