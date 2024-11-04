import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import EMNIST

class ProcessedEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, force_process=False):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Define processed data path
        mode = 'train' if train else 'test'
        self.processed_path = os.path.join(root, f'processed_{mode}.pt')
        
        # Process data if needed
        if not os.path.exists(self.processed_path) or force_process:
            self._process_and_save()
            
        # Load processed data
        self.data_dict = torch.load(self.processed_path)
        self.data = self.data_dict['data']
        self.targets = self.data_dict['targets']
        
    def _process_and_save(self):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
            transforms.Lambda(lambda x: transforms.functional.hflip(x))
        ])
        
        # Load and process dataset
        dataset = EMNIST(self.root, split='letters', train=self.train, 
                        download=True, transform=transform)
        
        # Convert to tensors and save
        data_list = []
        target_list = []
        for data, target in dataset:
            data_list.append(data)
            target_list.append(target)
            
        processed_data = {
            'data': torch.stack(data_list),
            'targets': torch.tensor(target_list)
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        torch.save(processed_data, self.processed_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class EMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # This will process and cache the data if needed
        ProcessedEMNIST(self.data_dir, train=True)
        ProcessedEMNIST(self.data_dir, train=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            emnist_full = ProcessedEMNIST(self.data_dir, train=True)
            self.emnist_train, self.emnist_val = torch.utils.data.random_split(
                emnist_full, [0.9, 0.1]
            )

        if stage == 'test' or stage is None:
            self.emnist_test = ProcessedEMNIST(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.emnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,  # Adjust based on your CPU cores
            #multiprocessing_context='fork'  # This might help on MacOS
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(self.emnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.emnist_test, batch_size=self.batch_size)