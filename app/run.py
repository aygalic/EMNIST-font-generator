from ef_generator.model import CNNAutoencoder, EMNISTDataModule
import pytorch_lightning as pl


import matplotlib.pyplot as plt

from torchvision.utils import make_grid


import torch

from sklearn.manifold import TSNE
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
            # Get the latent representation
            latent = model.encoder(inputs)
            latent = latent.squeeze()

            samples.extend(inputs.cpu().numpy())
            latent_representations.extend(latent.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
            if len(samples) >= n_samples:
                break
    
    # Convert to numpy arrays
    latent_representations = np.array(latent_representations[:n_samples])
    labels = np.array(labels[:n_samples])
    breakpoint()
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_tsne = tsne.fit_transform(latent_representations)
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of the latent space')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()


model = CNNAutoencoder()
data_module = EMNISTDataModule()

visualize_samples(data_module, num_samples=25, cols=5)

trainer = pl.Trainer(max_epochs=1,precision="16-mixed")
trainer.fit(model, data_module)



# Load the trained model
visualize_latent_space(model, data_module, n_samples=1000, perplexity=30)

