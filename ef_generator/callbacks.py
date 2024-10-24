import pytorch_lightning as pl

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
