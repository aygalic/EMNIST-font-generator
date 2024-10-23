
import pytorch_lightning as pl

from ef_generator.model import VAE, EMNISTDataModule

from ef_generator.visualization import (
    visualize_latent_manifold,
    visualize_samples,
    visualize_reconstructions,
    visualize_latent_space,
    visualize_generated_samples,
    visualize_latent_manifold
)



# Usage example:
# vae = VAE(latent_dim=26)
# visualize_latent_manifold(vae)


model = VAE()
data_module = EMNISTDataModule()

# visualize_samples(data_module, num_samples=25, cols=5)

trainer = pl.Trainer(max_epochs=3, precision="16-mixed")
trainer.fit(model, data_module)

visualize_reconstructions(model, data_module, num_samples=5)

# Load the trained model
# visualize_latent_space(model, data_module, n_samples=1000, perplexity=30)

# visualize_generated_samples(model)

visualize_latent_manifold(model)
