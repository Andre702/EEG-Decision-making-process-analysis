"""Trening: wrapper Lightning, walidacja Cross-Subject (5-fold) z auto-resume, wybór checkpointu."""
import pytorch_lightning as pl
import torch

# (...) importy (ModelCheckpoint, EarlyStopping, loggery, KFold/StratifiedKFold, DataLoader,
# os, glob, copy, gc, shutil) dopasujemy po wklejeniu ciał funkcji


class EEGLightningWrapper(pl.LightningModule):
    """
    PyTorch Lightning Wrapper handling training logic, loss calculation, and optimizers.
    """

    def __init__(self, model_class, input_size, d_model, nhead, lr=1e-3, noise_std=0.01):
        ...  # (...)

    # (...) forward i ewentualne pozostałe metody pominięte w wklejce

    def _shared_step(self, batch, batch_idx, phase):
        x, y = batch
        outputs = self(x)

        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Log metrics to TensorBoard and progress bar
        self.log(f"{phase}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


def train_cross_individual_torch(
    subject_data: dict,
    model_class,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    d_model=128,
    nhead=8,
    dataset_name="EEG_Data",
    training_patience=7,
    sequence_first=True,
    noise_std=0.01,
    feature_method="raw",
    model_input_size=None,
    checkpoints_dir: str | None = None,  # NOWE: katalog checkpointów z cfg.checkpoints_dir
):
    """
    Performs 5-Fold Cross-Subject validation on the remaining training data.
    Includes Auto-Resume function - it will search for last checkpoint to start with!

    Args:
        ...
        sequence_first (bool): Crucial for Architecture mapping!
            - True: Temporal Transformers. Permutes input to (Batch, Time, Channels).
            - False: CNNs & Spatial Transformers. Keeps input as (Batch, Channels, Time).
    """
    ...  # (...)


def get_best_model_checkpoint(checkpoints_base_dir: str) -> str:
    """
    Scanns folders with checkpoints.
    Returns path to the file containing best model weights (best by val_acc).
    """
    ...  # (...)
