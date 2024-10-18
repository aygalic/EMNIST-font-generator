from ef_generator.model import CNNAutoencoder, EMNISTDataModule
import pytorch_lightning as pl


model = CNNAutoencoder()
data_module = EMNISTDataModule()

trainer = pl.Trainer(max_epochs=10,precision="16-mixed")
trainer.fit(model, data_module)