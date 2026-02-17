"""
Evaluation Metrics and Backtesting for Stock Prediction.

Provides directional accuracy, classification metrics,
Sharpe ratio, max drawdown, and a full backtesting simulation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class ModelEvaluator:
    """Evaluate stock prediction model performance."""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
        """
        self.risk_free_rate = risk_free_rate

    def directional_accuracy(
        self,
        actual_directions: np.ndarray,
        predicted_directions: np.ndarray,
    ) -> float:
        """
        Percentage of days where predicted direction matches actual.

        Args:
            actual_directions: Ground truth (0=down, 1=up).
            predicted_directions: Model predictions (0=down, 1=up).

        Returns:
            Accuracy as a float (0-1).
        """
        return float(accuracy_score(actual_directions, predicted_directions))

    def classification_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """
        Full classification metrics for buy/sell signals.

        Returns:
            Dict with accuracy, precision, recall, f1,
            confusion_matrix, and per-class report.
        """
        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
            "report": classification_report(
                actual, predicted,
                target_names=["Down", "Up"],
                zero_division=0,
            ),
        }

    def sharpe_ratio(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Calculate Sharpe ratio of a return series.

        Args:
            returns: Array of daily returns.
            annualize: If True, annualizes the ratio.

        Returns:
            Sharpe ratio (higher is better, >1.0 is good, >2.0 is excellent).
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / TRADING_DAYS_PER_YEAR)
        ratio = np.mean(excess_returns) / np.std(excess_returns)

        if annualize:
            ratio *= np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(ratio)

    def max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from peak.

        Args:
            cumulative_returns: Cumulative return series (starting from 1.0).

        Returns:
            Max drawdown as a negative percentage (e.g., -0.15 = -15%).
        """
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return float(np.min(drawdown))

    def backtest(
        self,
        actual_returns: np.ndarray,
        predicted_directions: np.ndarray,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
    ) -> dict:
        """
        Simulate a trading strategy based on model predictions.

        Strategy:
            - Predicted UP (1): go long (buy)
            - Predicted DOWN (0): go short (sell) or stay flat

        Args:
            actual_returns: Daily actual returns of the stock.
            predicted_directions: Model's predicted directions (0/1).
            initial_capital: Starting capital.
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%).

        Returns:
            Dict with strategy performance metrics and daily data.
        """
        n = min(len(actual_returns), len(predicted_directions))
        actual_returns = actual_returns[:n]
        predicted_directions = predicted_directions[:n]

        # Strategy returns: long when predicted up, short when predicted down
        positions = np.where(predicted_directions == 1, 1.0, -1.0)

        # Detect position changes for transaction costs
        position_changes = np.abs(np.diff(positions, prepend=positions[0]))
        costs = position_changes * transaction_cost

        strategy_returns = positions * actual_returns - costs

        # Buy-and-hold baseline
        buyhold_returns = actual_returns

        # Cumulative returns
        strategy_cumulative = np.cumprod(1 + strategy_returns)
        buyhold_cumulative = np.cumprod(1 + buyhold_returns)

        # Scale to initial capital
        strategy_value = initial_capital * strategy_cumulative
        buyhold_value = initial_capital * buyhold_cumulative

        # Metrics
        strategy_sharpe = self.sharpe_ratio(strategy_returns)
        buyhold_sharpe = self.sharpe_ratio(buyhold_returns)
        strategy_mdd = self.max_drawdown(strategy_cumulative)
        buyhold_mdd = self.max_drawdown(buyhold_cumulative)

        total_strategy_return = float(strategy_cumulative[-1] - 1) if len(strategy_cumulative) > 0 else 0
        total_buyhold_return = float(buyhold_cumulative[-1] - 1) if len(buyhold_cumulative) > 0 else 0

        excess_return = total_strategy_return - total_buyhold_return
        n_trades = int(np.sum(position_changes > 0))

        results = {
            "strategy": {
                "total_return": total_strategy_return,
                "sharpe_ratio": strategy_sharpe,
                "max_drawdown": strategy_mdd,
                "final_value": float(strategy_value[-1]) if len(strategy_value) > 0 else initial_capital,
            },
            "buy_and_hold": {
                "total_return": total_buyhold_return,
                "sharpe_ratio": buyhold_sharpe,
                "max_drawdown": buyhold_mdd,
                "final_value": float(buyhold_value[-1]) if len(buyhold_value) > 0 else initial_capital,
            },
            "comparison": {
                "excess_return": excess_return,
                "improvement_pct": (excess_return / abs(total_buyhold_return) * 100)
                    if total_buyhold_return != 0 else 0.0,
                "n_trades": n_trades,
                "total_costs": float(np.sum(costs)),
            },
            "daily_data": pd.DataFrame({
                "actual_return": actual_returns,
                "predicted_direction": predicted_directions,
                "position": positions,
                "strategy_return": strategy_returns,
                "buyhold_return": buyhold_returns,
                "strategy_cumulative": strategy_cumulative,
                "buyhold_cumulative": buyhold_cumulative,
                "strategy_value": strategy_value,
                "buyhold_value": buyhold_value,
            }),
        }

        logger.info(
            f"Backtest results — "
            f"Strategy: {total_strategy_return:.2%} (Sharpe: {strategy_sharpe:.2f}), "
            f"Buy&Hold: {total_buyhold_return:.2%} (Sharpe: {buyhold_sharpe:.2f}), "
            f"Excess: {excess_return:.2%}"
        )

        return results

    def full_evaluation(
        self,
        actual_directions: np.ndarray,
        predicted_directions: np.ndarray,
        actual_returns: np.ndarray,
        predicted_returns: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run all evaluation metrics at once.

        Args:
            actual_directions: Ground truth directions (0/1).
            predicted_directions: Predicted directions (0/1).
            actual_returns: Actual daily returns.
            predicted_returns: Predicted return magnitudes (optional).

        Returns:
            Comprehensive evaluation dict.
        """
        dir_accuracy = self.directional_accuracy(actual_directions, predicted_directions)
        clf_metrics = self.classification_metrics(actual_directions, predicted_directions)
        backtest_results = self.backtest(actual_returns, predicted_directions)

        evaluation = {
            "directional_accuracy": dir_accuracy,
            "classification": clf_metrics,
            "backtest": {
                k: v for k, v in backtest_results.items() if k != "daily_data"
            },
            "backtest_daily_data": backtest_results["daily_data"],
        }

        if predicted_returns is not None:
            mse = float(np.mean((actual_returns - predicted_returns) ** 2))
            mae = float(np.mean(np.abs(actual_returns - predicted_returns)))
            evaluation["return_prediction"] = {"mse": mse, "mae": mae}

        logger.info(
            f"Full evaluation — "
            f"Directional Accuracy: {dir_accuracy:.2%}, "
            f"F1: {clf_metrics['f1']:.2%}, "
            f"Strategy Sharpe: {backtest_results['strategy']['sharpe_ratio']:.2f}"
        )

        return evaluation
