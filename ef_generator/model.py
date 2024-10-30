from typing import Literal
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim=2,
        n_annealing_steps=1000,
        sigma=1,
        lr_encoder=1e-4,  # Lower learning rate for pretrained encoder
        lr_decoder=1e-3,  # Higher learning rate for decoder
        lr_classifier=1e-3,  # Standard learning rate for classifier
        reconstruction_weight=1.0,
        classification_weight=0.1,  # Weight for optional classification during VAE phase
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = 0

        self.feature_multiplier = 8
        self.poly_power = 2


        self.training_phase: Literal["supervised", "vae"] = "supervised"
        # Annealing parameters
        self.n_annealing_steps = n_annealing_steps
        self.sigma = sigma
        self.vae_step_counter = 0  # Add counter specifically for VAE training
        
        self._total_steps = None

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16 * self.feature_multiplier, 3, stride=2, padding=1),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(
                16 * self.feature_multiplier,
                32 * self.feature_multiplier,
                3,
                stride=2,
                padding=1,
            ),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(32 * self.feature_multiplier, 64, 7),
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
            nn.ConvTranspose2d(64, 32 * self.feature_multiplier, 7),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32 * self.feature_multiplier,
                16 * self.feature_multiplier,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16 * self.feature_multiplier,
                1,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Dropout(self.dropout),
            nn.Sigmoid(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 26),  # 26 letters, logits
        )

        # Learning rates
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_classifier = lr_classifier
        
        # Training weights
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        
        # Track validation metrics
        self.validation_step_outputs = []

    def get_total_steps(self) -> int:
        """Calculate total training steps for the entire training process."""
        if self._total_steps is None:
            # Get dataloader and trainer
            if self.trainer is None:
                return 1000  # Default value if trainer is not yet available
                
            # Calculate total steps
            dataset_size = len(self.trainer.datamodule.train_dataloader().dataset)
            batch_size = self.trainer.datamodule.train_dataloader().batch_size
            max_epochs = self.trainer.max_epochs
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
            
            steps_per_epoch = dataset_size // (batch_size * accumulate_grad_batches)
            self._total_steps = steps_per_epoch * max_epochs
            
        return self._total_steps

    def get_training_weights(self):
        """Calculate weights for different objectives based on training progress"""
        total_steps = self.get_total_steps()
        
        # Get current progress (0 to 1)
        progress = min(self.trainer.global_step / total_steps, 1.0)
        
        # Classification weight starts high and gradually decreases
        classification_weight = 1.0 - 0.5 * (progress ** self.poly_power)
        
        # Reconstruction weight increases gradually
        reconstruction_weight = progress ** (self.poly_power / 2)
        
        # KL weight increases more slowly
        kl_weight = (progress ** self.poly_power) * self.sigma
        
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

    def set_training_phase(self, training_phase: Literal["supervised", "vae"]):
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
        if self.training_phase == "supervised":
            return self.classifier(encoded)
        mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch
        total_loss = 0
        logs = {}

        # Get encoded features
        encoded = self.encoder(x)
        
        if self.training_phase == 'supervised':
            # Classification training
            logits = self.classifier(encoded)
            loss = F.cross_entropy(logits, y-1)
            acc = (logits.argmax(dim=1) == (y-1)).float().mean()
            
            # Explicit logging for supervised phase
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
            
            return {'loss': loss}
        
        else:  # VAE training
            # VAE components
            mu, log_var = self.fc_mu(encoded), self.fc_var(encoded)
            z = self.reparameterize(mu, log_var)
            x_hat = self.decode(z)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
            
            # KL divergence
            kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) * (self.latent_dim / 784)
            kl_weight = self.get_kl_weight()
            
            # VAE loss
            vae_loss = self.reconstruction_weight * recon_loss + kl_weight * kl_div
            total_loss += vae_loss
            
            # Optional: Add classification loss during VAE training
            if self.classification_weight > 0:
                class_logits = self.classifier(encoded.detach())
                class_loss = F.cross_entropy(class_logits, y-1)
                total_loss += self.classification_weight * class_loss
                acc = (class_logits.argmax(dim=1) == (y-1)).float().mean()
                
                self.log('train_class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('train_class_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
            
            # Explicit logging for VAE phase
            self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_kl_div', kl_div, on_step=True, on_epoch=True, prog_bar=True)
            self.log('kl_weight', kl_weight, on_step=True, on_epoch=True, prog_bar=True)
            
            return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Get encoded features
        encoded = self.encoder(x)
        
        # Always compute classification metrics
        logits = self.classifier(encoded)
        class_loss = F.cross_entropy(logits, y-1)
        acc = (logits.argmax(dim=1) == (y-1)).float().mean()
        
        # Explicit logging for classification metrics
        self.log('val_class_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.training_phase == 'vae':
            # VAE validation metrics
            x_hat, mu, log_var = self(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) * (self.latent_dim / 784)
            
            # Explicit logging for VAE metrics
            self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_kl_div', kl_div, on_step=False, on_epoch=True, prog_bar=True)


    
    """
    def on_validation_epoch_end(self):
        # Aggregate validation metrics
        metrics = {}
        for key in self.validation_step_outputs[0].keys():
            metrics[key] = torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        
        self.log_dict(metrics)
        self.validation_step_outputs.clear()    
    """


    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        encoder_params = list(self.encoder.parameters())
        latent_params = list(self.fc_mu.parameters()) + list(self.fc_var.parameters())
        decoder_params = list(self.decoder.parameters()) + list(self.decoder_input.parameters())
        classifier_params = list(self.classifier.parameters())
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.lr_encoder},
            {'params': latent_params, 'lr': self.lr_encoder},
            {'params': decoder_params, 'lr': self.lr_decoder},
            {'params': classifier_params, 'lr': self.lr_classifier}
        ])
        
        # Get total steps for scheduler
        total_steps = self.get_total_steps()
        
        # Create scheduler with calculated total steps
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=self.poly_power/2
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def on_fit_start(self):
        """Called when fit begins. Calculate total steps here when we're sure trainer is set."""
        super().on_fit_start()
        # Force recalculation of total steps
        self._total_steps = None
        _ = self.get_total_steps()
        
        # Log the total steps for verification
        #self.log("total_training_steps", self._total_steps)