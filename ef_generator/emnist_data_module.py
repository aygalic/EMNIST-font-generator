from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST
import pytorch_lightning as pl

class EMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation([270,270]),
            transforms.RandomHorizontalFlip(p = 1),
            
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