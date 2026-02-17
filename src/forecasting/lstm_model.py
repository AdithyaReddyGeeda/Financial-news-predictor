"""
Bidirectional LSTM Model for Stock Price Prediction.

Dual-head architecture:
  1. Classification head: predicts next-day direction (up/down)
  2. Regression head: predicts next-day return magnitude
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Bidirectional LSTM with dual prediction heads.

    Architecture:
        Input (batch, seq_len, num_features)
        -> 2-layer BiLSTM (hidden=128)
        -> Dropout (0.3)
        -> FC layers
        -> Direction head (sigmoid) + Return head (linear)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_size = hidden_size * self.num_directions

        self.dropout = nn.Dropout(dropout)

        # Shared feature extractor
        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head: next-day direction (up/down)
        self.fc_direction = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Regression head: next-day return magnitude
        self.fc_return = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).

        Returns:
            direction: (batch, 1) probabilities for up direction.
            returns: (batch, 1) predicted return values.
        """
        # LSTM output: (batch, seq_len, hidden * num_directions)
        lstm_out, _ = self.lstm(x)

        # Take the last timestep output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)

        # Shared features
        shared = self.fc_shared(last_output)

        # Dual heads
        direction = self.fc_direction(shared)
        returns = self.fc_return(shared)

        return direction, returns


class CombinedLoss(nn.Module):
    """
    Combined loss for dual-head prediction.

    Loss = alpha * BCE(direction) + (1 - alpha) * MSE(return)
    """

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for direction loss (0-1).
                   Higher = more focus on direction accuracy.
        """
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_direction: torch.Tensor,
        pred_return: torch.Tensor,
        target_direction: torch.Tensor,
        target_return: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Returns:
            total_loss: Weighted sum of BCE and MSE.
            loss_dict: Individual loss values for logging.
        """
        dir_loss = self.bce(
            pred_direction.squeeze(), target_direction.float()
        )
        ret_loss = self.mse(
            pred_return.squeeze(), target_return.float()
        )

        total = self.alpha * dir_loss + (1 - self.alpha) * ret_loss

        loss_dict = {
            "total": total.item(),
            "direction": dir_loss.item(),
            "return": ret_loss.item(),
        }

        return total, loss_dict
