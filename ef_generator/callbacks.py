from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall


class ValidationLossCallback(pl.Callback):
    """
    A PyTorch Lightning callback that prints validation loss after training
    with optional formatting and epoch tracking.
    """

    def __init__(self, print_epoch: bool = True, decimal_places: int = 4):
        super().__init__()
        self.print_epoch = print_epoch
        self.decimal_places = decimal_places
        self.best_loss = float("inf")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the validation epoch ends."""
        current_loss = trainer.callback_metrics.get("val_loss")

        if current_loss is not None:
            # Format the loss value
            loss_str = f"{current_loss:.{self.decimal_places}f}"

            # Track best loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                is_best = " (Best)"
            else:
                is_best = ""

            # Create the output string
            if self.print_epoch:
                epoch_str = f"Epoch {trainer.current_epoch}: "
            else:
                epoch_str = ""

            print(f"{epoch_str}Validation Loss: {loss_str}{is_best}")


class ClassificationMetricsCallback(pl.Callback):
    """
    A PyTorch Lightning callback that tracks and prints multiple classification metrics
    including balanced accuracy, F1-score, precision, and recall.
    """

    def __init__(
        self, num_classes=26, print_epoch: bool = True, decimal_places: int = 4
    ):
        super().__init__()
        self.print_epoch = print_epoch
        self.decimal_places = decimal_places
        self.num_classes = num_classes

        # Best metrics tracking
        self.best_metrics = {
            "val_loss": float("inf"),
            "balanced_acc": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

        # Initialize metrics (device will be set in setup)
        self.metrics = None

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = None
    ) -> None:
        """Initialize metrics on the correct device"""
        device = pl_module.device
        self.metrics = {
            "accuracy": Accuracy(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ).to(device),
            "f1_macro": F1Score(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ).to(device),
            "f1_weighted": F1Score(
                task="multiclass", num_classes=self.num_classes, average="weighted"
            ).to(device),
            "precision": Precision(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ).to(device),
            "recall": Recall(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ).to(device),
            "confusion_matrix": ConfusionMatrix(
                task="multiclass", num_classes=self.num_classes
            ).to(device),
        }

    def _compute_balanced_accuracy(self, y_true, y_pred):
        # Move tensors to CPU for sklearn metric
        return balanced_accuracy_score(y_true.cpu(), y_pred.cpu())

    def _format_metric(self, name: str, value: float, is_best: bool = False) -> str:
        """Format a metric with consistent decimal places and optional 'Best' marker"""
        metric_str = f"{value:.{self.decimal_places}f}"
        return f"{name}: {metric_str}" + (" (Best)" if is_best else "")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the validation epoch ends."""
        # Get predictions and targets from validation step
        val_preds = []
        val_targets = []

        # Collect predictions and targets from validation dataloader
        val_dataloader = trainer.val_dataloaders
        device = pl_module.device

        pl_module.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                # Get encoded features
                encoded = pl_module.encoder(x)
                # Always compute classification metrics
                logits = pl_module.classifier(encoded)
                preds = torch.argmax(logits, dim=1)
                # Adjust for EMNIST labels starting at 1
                val_preds.append(preds)
                val_targets.append(y - 1)

        # Concatenate all batches
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        # Compute metrics
        metrics = {
            "val_loss": trainer.callback_metrics.get("val_loss", float("inf")),
            "balanced_acc": self._compute_balanced_accuracy(val_targets, val_preds),
            "macro_f1": self.metrics["f1_macro"](val_preds, val_targets),
            "weighted_f1": self.metrics["f1_weighted"](val_preds, val_targets),
            "precision": self.metrics["precision"](val_preds, val_targets),
            "recall": self.metrics["recall"](val_preds, val_targets),
        }

        # Update best metrics
        is_best = {key: False for key in metrics.keys()}
        for metric_name, value in metrics.items():
            if metric_name == "val_loss" and value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                is_best[metric_name] = True
            elif metric_name != "val_loss" and value > self.best_metrics.get(
                metric_name, 0.0
            ):
                self.best_metrics[metric_name] = value
                is_best[metric_name] = True

        # Create output string
        epoch_str = f"Epoch {trainer.current_epoch}: " if self.print_epoch else ""
        metrics_str = [
            self._format_metric("Loss", metrics["val_loss"], is_best["val_loss"]),
            self._format_metric(
                "Balanced Acc", metrics["balanced_acc"], is_best["balanced_acc"]
            ),
            self._format_metric("Macro F1", metrics["macro_f1"], is_best["macro_f1"]),
            self._format_metric(
                "Weighted F1", metrics["weighted_f1"], is_best["weighted_f1"]
            ),
            self._format_metric("Precision", metrics["precision"], False),
            self._format_metric("Recall", metrics["recall"], False),
        ]

        print(f"{epoch_str}{' | '.join(metrics_str)}")

        # Log all metrics to trainer
        for name, value in metrics.items():
            trainer.logger.log_metrics({f"val_{name}": value}, step=trainer.global_step)

        # print confusion matrix stats

        conf_matrix = self.metrics["confusion_matrix"](val_preds, val_targets)
        per_class_acc = conf_matrix.diag() / conf_matrix.sum(1)
        worst_classes = torch.argsort(per_class_acc)[:3]
        print("\nWorst performing classes:")
        for cls in worst_classes:
            print(f"Class {chr(cls.item() + 65)}: {per_class_acc[cls]:.3f}")


class LatentSpaceVisualizer(pl.Callback):
    def __init__(
        self, every_n_batches=100, output_dir="latent_vis", n=20, digit_size=28
    ):
        """
        Args:
            every_n_batches: Generate visualization every N batches
            output_dir: Directory to save the visualization images
            n: Grid size for visualization
            digit_size: Size of each digit image
        """
        super().__init__()
        self.every_n_batches = every_n_batches
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n = n
        self.digit_size = digit_size
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count % self.every_n_batches == 0:
            self.visualize_latent_space(pl_module)

    def on_train_end(self, trainer, pl_module):
        # Create animation from saved images
        self.create_animation()

    def visualize_latent_space(self, model):
        model.eval()
        grid_x = np.linspace(-3, 3, self.n)
        grid_y = np.linspace(-3, 3, self.n)[::-1]
        figure = np.zeros((self.digit_size * self.n, self.digit_size * self.n))

        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = torch.tensor(
                        [[xi, yi] + [0] * (model.latent_dim - 2)],
                        dtype=torch.float32,
                        device=model.device,
                    )
                    x_decoded = model.decode(z_sample)
                    digit = x_decoded[0].reshape(self.digit_size, self.digit_size)
                    figure[
                        i * self.digit_size : (i + 1) * self.digit_size,
                        j * self.digit_size : (j + 1) * self.digit_size,
                    ] = digit.cpu().numpy()

        plt.figure(figsize=(10, 10))
        start_range = self.digit_size // 2
        end_range = self.n * self.digit_size + start_range
        pixel_range = np.arange(start_range, end_range, self.digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.title(f"Latent Space Manifold (Batch {self.batch_count})")

        # Save the figure
        plt.savefig(self.output_dir / f"latent_space_{self.batch_count:06d}.png")
        plt.close()

    def create_animation(self, fps=10):
        """Create a GIF animation from saved images"""
        images = []
        image_files = sorted(self.output_dir.glob("latent_space_*.png"))

        for filename in image_files:
            images.append(imageio.imread(filename))

        # Save as GIF
        imageio.mimsave(self.output_dir / "latent_space_evolution.gif", images, fps=fps)
