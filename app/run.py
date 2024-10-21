from ef_generator.model import VAE, EMNISTDataModule
import pytorch_lightning as pl


import matplotlib.pyplot as plt

from torchvision.utils import make_grid


import torch

from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA


import torch
import matplotlib.pyplot as plt
import numpy as np



def visualize_samples(data_module, num_samples=25, cols=5):
    # Ensure the data is prepared and set up
    data_module.prepare_data()
    data_module.setup()

    # Get a batch of data
    dataloader = data_module.train_dataloader()
    batch = next(iter(dataloader))
    images, labels = batch

    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Create a grid of images
    grid = make_grid(images, nrow=cols, normalize=True, padding=2)

    # Convert to numpy for displaying
    img_grid = grid.permute(1, 2, 0).cpu().numpy()

    # Plot the grid
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_grid, cmap='gray')
    ax.axis('off')

    # Add labels
    for i, label in enumerate(labels):
        row = i // cols
        col = i % cols
        ax.text(col * (images.shape[2] + 2) + images.shape[2]/2, 
                row * (images.shape[3] + 2) + images.shape[3] + 2, 
                chr(label.item() + 96),  # Convert label to corresponding letter
                ha='center', va='center')

    plt.tight_layout()
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_latent_space(model, data_module, n_samples=1000, perplexity=30):
    # Set the model to evaluation mode
    model.eval()

    # Prepare the data
    data_module.setup()
    dataloader = data_module.train_dataloader()

    # Collect samples and their latent representations
    samples = []
    latent_representations = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, batch_labels = batch
            # Get the latent representation (mean)
            mu, _ = model.encode(inputs)
            
            samples.extend(inputs.cpu().numpy())
            latent_representations.extend(mu.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

            if len(samples) >= n_samples:
                break

    # Convert to numpy arrays
    latent_representations = np.array(latent_representations[:n_samples])
    labels = np.array(labels[:n_samples])

    # Apply PCA
    pca = PCA(n_components=2)
    latent_compressed = pca.fit_transform(latent_representations)

    # Plot the results
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_compressed[:, 0], latent_compressed[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('PCA visualization of the latent space')
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    plt.show()

    # Optionally, you can also use t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_tsne = tsne.fit_transform(latent_representations)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of the latent space')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

def visualize_reconstructions(model, data_module, num_samples=5):
    # Set the model to evaluation mode
    model.eval()

    # Prepare the data
    data_module.setup()
    dataloader = data_module.train_dataloader()

    # Get a batch of data
    batch = next(iter(dataloader))
    images, labels = batch

    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Get reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(images)

    # Plot original images and reconstructions
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Original: {chr(labels[i].item() + 96)}")
        axes[0, i].axis('off')

        # Reconstructed image
        axes[1, i].imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Compute and print reconstruction error
    mse = torch.nn.functional.mse_loss(images, reconstructions)
    print(f"Average reconstruction error (MSE): {mse.item():.4f}")

def visualize_generated_samples(model, num_samples=5):
    model.eval()
    
    with torch.no_grad():
        # Sample from the latent space
        z = torch.randn(num_samples, model.latent_dim)
        
        # Generate images
        generated_images = model.decode(z)

    # Plot generated images
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    for i in range(num_samples):
        axes[i].imshow(generated_images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title("Generated")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

model = VAE()
data_module = EMNISTDataModule()

visualize_samples(data_module, num_samples=25, cols=5)

trainer = pl.Trainer(max_epochs=1, precision="16-mixed")
trainer.fit(model, data_module)

visualize_reconstructions(model, data_module, num_samples=5)

# Load the trained model
visualize_latent_space(model, data_module, n_samples=1000, perplexity=30)


visualize_generated_samples(model)