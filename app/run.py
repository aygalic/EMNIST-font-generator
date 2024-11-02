import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ef_generator.model import PretrainedVAE
from ef_generator.emnist_data_module import EMNISTDataModule
from ef_generator.visualization import (
    visualize_generated_samples,
    visualize_latent_manifold,
    visualize_latent_space,
    visualize_reconstructions,
    visualize_samples,
)
from ef_generator.callbacks import ValidationLossCallback, ClassificationMetricsCallback
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler


# Initialize logger
logger = TensorBoardLogger("lightning_logs", name="my_model")
model = PretrainedVAE(
    subloss_weights=[1,10,1]
    )
data_module = EMNISTDataModule()


# visualize_samples(data_module, num_samples=25, cols=5)

trainer = pl.Trainer(
    accelerator="mps",
    max_epochs=2,
    callbacks=[ClassificationMetricsCallback(num_classes=26)],
    profiler="simple",
    devices="auto",
    )
trainer.fit(model, data_module)


visualize_reconstructions(model, data_module, num_samples=5)


visualize_latent_space(model, data_module, n_samples=10000, perplexity=30)

# visualize_generated_samples(model)

visualize_latent_manifold(model)
