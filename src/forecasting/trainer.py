"""
Training Loop for Stock LSTM Model.

Handles training, validation, early stopping,
learning rate scheduling, and checkpoint management.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .lstm_model import StockLSTM, CombinedLoss
from .dataset import StockSequenceDataset

logger = logging.getLogger(__name__)


class StockTrainer:
    """Train and evaluate the Stock LSTM model."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_config = self.config.get("model", {})
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_layers = model_config.get("num_layers", 2)
        self.dropout = model_config.get("dropout", 0.3)
        self.bidirectional = model_config.get("bidirectional", True)
        self.lr = model_config.get("learning_rate", 0.001)
        self.batch_size = model_config.get("batch_size", 64)
        self.epochs = model_config.get("epochs", 100)
        self.patience = model_config.get("patience", 10)
        self.checkpoint_dir = model_config.get("checkpoint_dir", "models")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def build_model(self, num_features: int) -> StockLSTM:
        """Initialize the LSTM model, optimizer, scheduler, and loss."""
        self.model = StockLSTM(
            num_features=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

        self.criterion = CombinedLoss(alpha=0.7)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model built: {total_params:,} params ({trainable:,} trainable) "
            f"on {self.device}"
        )

        return self.model

    def train(
        self,
        train_dataset: StockSequenceDataset,
        val_dataset: StockSequenceDataset,
    ) -> dict:
        """
        Full training loop with early stopping.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.

        Returns:
            Training history dict.
        """
        if self.model is None:
            self.build_model(train_dataset.num_features)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"batch_size={self.batch_size}, lr={self.lr}"
        )

        for epoch in range(self.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.patience})"
                    )
                    break

        # Load best model
        self._load_checkpoint("best_model.pth")
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return self.history

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for sequences, targets in dataloader:
            sequences = sequences.to(self.device)
            direction_target = targets["direction"].to(self.device)
            return_target = targets["return"].to(self.device)

            self.optimizer.zero_grad()

            pred_direction, pred_return = self.model(sequences)

            loss, _ = self.criterion(
                pred_direction, pred_return,
                direction_target, return_target,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run validation and return loss + directional accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for sequences, targets in dataloader:
            sequences = sequences.to(self.device)
            direction_target = targets["direction"].to(self.device)
            return_target = targets["return"].to(self.device)

            pred_direction, pred_return = self.model(sequences)

            loss, _ = self.criterion(
                pred_direction, pred_return,
                direction_target, return_target,
            )

            total_loss += loss.item()

            predicted = (pred_direction.squeeze() > 0.5).float()
            correct += (predicted == direction_target).sum().item()
            total += direction_target.size(0)

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    @torch.no_grad()
    def predict(self, dataset: StockSequenceDataset) -> dict:
        """
        Run predictions on a dataset.

        Returns:
            Dict with 'directions' (probabilities), 'returns' (values),
            and 'predicted_labels' (0/1).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a checkpoint first.")

        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_directions = []
        all_returns = []

        for sequences, _ in dataloader:
            sequences = sequences.to(self.device)
            pred_dir, pred_ret = self.model(sequences)
            all_directions.append(pred_dir.cpu().numpy())
            all_returns.append(pred_ret.cpu().numpy())

        directions = np.concatenate(all_directions).squeeze()
        returns = np.concatenate(all_returns).squeeze()

        return {
            "directions": directions,
            "returns": returns,
            "predicted_labels": (directions > 0.5).astype(int),
        }

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "num_features": self.model.num_features,
        }, path)

        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.info(f"Checkpoint loaded: {path}")

    def load_for_inference(self, filename: str = "best_model.pth", num_features: int = None):
        """Load a saved model for inference only.

        If num_features is not provided, it is read from the checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        if num_features is None and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            num_features = checkpoint.get("num_features", 14)

        self.build_model(num_features or 14)
        self._load_checkpoint(filename)
        self.model.eval()
