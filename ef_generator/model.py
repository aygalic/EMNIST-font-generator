import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torchvision



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
        x = batch['image']
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class EMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the dataset
        load_dataset("tanganke/emnist_letters")

    def setup(self, stage=None):
        # Load the dataset
        dataset = load_dataset("tanganke/emnist_letters")
        
        # Preprocess function
        def preprocess(examples):
            tensor_img = [torchvision.transforms.functional.pil_to_tensor(img) for img in examples['image']]
            examples['image'] = [torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0 for img in tensor_img]
            return examples

        # Apply preprocessing
        dataset = dataset.map(preprocess, batched=True)

        # Split the training set into train and validation
        train_val_data = dataset['train']
        train_indices, val_indices = train_test_split(range(len(train_val_data)), test_size=0.1, random_state=42)

        self.train_data = train_val_data.select(train_indices)
        self.val_data = train_val_data.select(val_indices)
        self.test_data = dataset['test']

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

