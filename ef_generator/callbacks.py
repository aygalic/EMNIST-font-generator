import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

from sklearn.metrics import balanced_accuracy_score

class ValidationLossCallback(pl.Callback):
    """
    A PyTorch Lightning callback that prints validation loss after training
    with optional formatting and epoch tracking.
    """
    def __init__(self, print_epoch: bool = True, decimal_places: int = 4):
        super().__init__()
        self.print_epoch = print_epoch
        self.decimal_places = decimal_places
        self.best_loss = float('inf')
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the validation epoch ends."""
        current_loss = trainer.callback_metrics.get('val_loss')
        
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
    def __init__(self, num_classes=26, print_epoch: bool = True, decimal_places: int = 4):
        super().__init__()
        self.print_epoch = print_epoch
        self.decimal_places = decimal_places
        self.num_classes = num_classes
        
        # Best metrics tracking
        self.best_metrics = {
            'val_loss': float('inf'),
            'balanced_acc': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0
        }
        
        # Initialize metrics (device will be set in setup)
        self.metrics = None
        
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = None) -> None:
        """Initialize metrics on the correct device"""
        device = pl_module.device
        self.metrics = {
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes, average='macro').to(device),
            'f1_macro': F1Score(task="multiclass", num_classes=self.num_classes, average='macro').to(device),
            'f1_weighted': F1Score(task="multiclass", num_classes=self.num_classes, average='weighted').to(device),
            'precision': Precision(task="multiclass", num_classes=self.num_classes, average='macro').to(device),
            'recall': Recall(task="multiclass", num_classes=self.num_classes, average='macro').to(device),
            'confusion_matrix': ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(device)
        }
    
    def _compute_balanced_accuracy(self, y_true, y_pred):
        # Move tensors to CPU for sklearn metric
        return balanced_accuracy_score(y_true.cpu(), y_pred.cpu())
    
    def _format_metric(self, name: str, value: float, is_best: bool = False) -> str:
        """Format a metric with consistent decimal places and optional 'Best' marker"""
        metric_str = f"{value:.{self.decimal_places}f}"
        return f"{name}: {metric_str}" + (" (Best)" if is_best else "")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
            'val_loss': trainer.callback_metrics.get('val_loss', float('inf')),
            'balanced_acc': self._compute_balanced_accuracy(val_targets, val_preds),
            'macro_f1': self.metrics['f1_macro'](val_preds, val_targets),
            'weighted_f1': self.metrics['f1_weighted'](val_preds, val_targets),
            'precision': self.metrics['precision'](val_preds, val_targets),
            'recall': self.metrics['recall'](val_preds, val_targets)
        }
        
        # Update best metrics
        is_best = {key: False for key in metrics.keys()}
        for metric_name, value in metrics.items():
            if metric_name == 'val_loss' and value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                is_best[metric_name] = True
            elif metric_name != 'val_loss' and value > self.best_metrics.get(metric_name, 0.0):
                self.best_metrics[metric_name] = value
                is_best[metric_name] = True
        
        # Create output string
        epoch_str = f"Epoch {trainer.current_epoch}: " if self.print_epoch else ""
        metrics_str = [
            self._format_metric("Loss", metrics['val_loss'], is_best['val_loss']),
            self._format_metric("Balanced Acc", metrics['balanced_acc'], is_best['balanced_acc']),
            self._format_metric("Macro F1", metrics['macro_f1'], is_best['macro_f1']),
            self._format_metric("Weighted F1", metrics['weighted_f1'], is_best['weighted_f1']),
            self._format_metric("Precision", metrics['precision'], False),
            self._format_metric("Recall", metrics['recall'], False)
        ]
        
        print(f"{epoch_str}{' | '.join(metrics_str)}")
        
        # Log all metrics to trainer
        for name, value in metrics.items():
            trainer.logger.log_metrics({f"val_{name}": value}, step=trainer.global_step)
        
        # print confusion matrix stats

        conf_matrix = self.metrics['confusion_matrix'](val_preds, val_targets)
        per_class_acc = conf_matrix.diag() / conf_matrix.sum(1)
        worst_classes = torch.argsort(per_class_acc)[:3]
        print("\nWorst performing classes:")
        for cls in worst_classes:
            print(f"Class {chr(cls.item() + 65)}: {per_class_acc[cls]:.3f}")