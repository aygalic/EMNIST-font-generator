import pytorch_lightning as pl

from ef_generator.model import VAE, EMNISTDataModule
from ef_generator.visualization import (
    visualize_generated_samples,
    visualize_latent_manifold,
    visualize_latent_space,
    visualize_reconstructions,
    visualize_samples,
)

model = VAE()
data_module = EMNISTDataModule()

# visualize_samples(data_module, num_samples=25, cols=5)

trainer = pl.Trainer(max_epochs=3, precision="16-mixed")
trainer.fit(model, data_module)

visualize_reconstructions(model, data_module, num_samples=5)


# visualize_latent_space(model, data_module, n_samples=1000, perplexity=30)

# visualize_generated_samples(model)

visualize_latent_manifold(model)
