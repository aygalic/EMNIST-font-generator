import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim=2,
        subloss_weights=[1, 1, 1],
        lr_encoder=1e-4,
        lr_decoder=1e-3,
        lr_classifier=1e-3,
        reconstruction_weight=1.0,
        classification_weight=0.1,
        beta=4.0,
        decoder_depth=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = 0.1
        self.feature_multiplier = 8
        self.poly_power = 2
        self.beta = beta
        self.subloss_weights = subloss_weights

        # Register weight tensor as a buffer
        self.register_buffer("_weight_tensor", torch.zeros(3))

        # Encoder (unchanged)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16 * self.feature_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(16 * self.feature_multiplier),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                16 * self.feature_multiplier,
                32 * self.feature_multiplier,
                3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32 * self.feature_multiplier),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32 * self.feature_multiplier, 64, 7),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        # Latent space mappings (unchanged)
        self.fc_mu = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, self.latent_dim)
        )

        self.fc_var = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, self.latent_dim)
        )

        # Decoder input with proper dimensionality
        decoder_input_layers = []
        current_dim = self.latent_dim
        target_dim = 64  # Final dimension needed before spatial operations

        # Calculate dimension steps
        dim_step = int((target_dim - current_dim) / decoder_depth)

        for i in range(decoder_depth):
            next_dim = (
                min(current_dim + dim_step, target_dim)
                if i < decoder_depth - 1
                else target_dim
            )
            decoder_input_layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(self.dropout),
                ]
            )
            current_dim = next_dim

        # Reshape to starting spatial dimensions (1x1)
        decoder_input_layers.append(nn.Unflatten(1, (64, 1, 1)))
        self.decoder_input = nn.Sequential(*decoder_input_layers)

        # Decoder with correct upscaling to 28x28
        self.decoder = nn.ModuleList()

        # First deconv block: 1x1 -> 7x7
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32 * self.feature_multiplier, 7),
                nn.BatchNorm2d(32 * self.feature_multiplier),
                nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2),
            )
        )

        # Second deconv block: 7x7 -> 14x14
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    32 * self.feature_multiplier,
                    16 * self.feature_multiplier,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(16 * self.feature_multiplier),
                nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2),
            )
        )

        # Final deconv block: 14x14 -> 28x28
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    16 * self.feature_multiplier,
                    1,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Sigmoid(),
            )
        )

        # Classifier (unchanged)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(128, 26),
        )

        # Initialize other parameters
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_classifier = lr_classifier
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.validation_step_outputs = []

    def get_training_weights(self):
        """Update cached weights and return them as tuple for logging"""
        progress = min(
            self.trainer.global_step / self.trainer.estimated_stepping_batches, 1.0
        )

        # Classification weight starts high and gradually decreases
        classification_weight = (
            1.0 - 0.5 * (progress**self.poly_power) * self.subloss_weights[0]
        )
        # Reconstruction weight increases gradually
        reconstruction_weight = (
            progress ** (self.poly_power / 2) * self.subloss_weights[1]
        )
        # KL weight increases more slowly
        kl_weight = (progress**self.poly_power) * self.subloss_weights[2]

        # Update weights in-place
        with torch.no_grad():
            # Classification weight
            self._weight_tensor[0] = classification_weight
            # Reconstruction weight
            self._weight_tensor[1] = reconstruction_weight
            # KL weight
            self._weight_tensor[2] = kl_weight

        # Return as tuple for logging purposes
        return classification_weight, reconstruction_weight, kl_weight

    def freeze_encoder(self):
        """Freeze encoder weights except for latent mappings"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Keep latent mappings trainable
        for param in self.fc_mu.parameters():
            param.requires_grad = True
        for param in self.fc_var.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze all encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True

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
        x = self.decoder_input(z)
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)

        mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Get encoded features
        encoded = self.encoder(x)
        mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)

        # Enhanced loss computation
        logits = self.classifier(encoded)
        classification_loss = F.cross_entropy(logits, y - 1)

        # Use both MSE and L1 loss for reconstruction
        mse_loss = F.mse_loss(x_hat, x, reduction="mean")
        l1_loss = F.l1_loss(x_hat, x, reduction="mean")
        reconstruction_loss = 0.8 * mse_loss + 0.2 * l1_loss

        # Modified KL divergence with beta-VAE formulation
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        weighted_kl_div = self.beta * kl_div * (self.latent_dim / 784)

        # Stack losses for vectorized computation
        losses = torch.stack(
            [classification_loss, reconstruction_loss, weighted_kl_div]
        )

        # Update cached weights
        class_weight, recon_weight, kl_weight = self.get_training_weights()

        # Compute total loss using cached weights
        total_loss = torch.sum(losses * self._weight_tensor)

        # Compute accuracy
        acc = (logits.argmax(dim=1) == (y - 1)).float().mean()

        # Enhanced logging
        self.log_dict(
            {
                "loss": total_loss,
                "train_class_loss": classification_loss,
                "train_recon_loss": reconstruction_loss,
                "train_mse_loss": mse_loss,
                "train_l1_loss": l1_loss,
                "train_kl_div": kl_div,
                "train_weighted_kl_div": weighted_kl_div,
                "train_acc": acc,
                "class_weight": class_weight,
                "recon_weight": recon_weight,
                "kl_weight": kl_weight,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Get encoded features
        encoded = self.encoder(x)
        mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)

        # Compute losses
        logits = self.classifier(encoded)
        classification_loss = F.cross_entropy(logits, y - 1)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_div = (
            -0.5
            * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            * (self.latent_dim / 784)
        )

        # Stack losses for vectorized computation
        losses = torch.stack([classification_loss, reconstruction_loss, kl_div])

        # Compute total loss using cached weights
        total_loss = torch.sum(losses * self._weight_tensor)

        # Compute accuracy
        acc = (logits.argmax(dim=1) == (y - 1)).float().mean()

        # Logging
        self.log_dict(
            {
                "val_total_loss": total_loss,
                "val_class_loss": classification_loss,
                "val_recon_loss": reconstruction_loss,
                "val_kl_div": kl_div,
                "val_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"val_loss": total_loss, "val_acc": acc}

    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        encoder_params = list(self.encoder.parameters())
        latent_params = list(self.fc_mu.parameters()) + list(self.fc_var.parameters())
        decoder_params = list(self.decoder.parameters()) + list(
            self.decoder_input.parameters()
        )
        classifier_params = list(self.classifier.parameters())

        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_encoder},
                {"params": latent_params, "lr": self.lr_encoder},
                {"params": decoder_params, "lr": self.lr_decoder},
                {"params": classifier_params, "lr": self.lr_classifier},
            ]
        )

        # Create scheduler with calculated total steps
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=self.trainer.estimated_stepping_batches,
            power=self.poly_power / 2,
        )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        """Initialize weights when training starts"""
        super().on_fit_start()
        # Initial weight update
        _ = self.get_training_weights()
