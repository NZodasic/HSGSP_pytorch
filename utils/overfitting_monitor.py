import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch.nn as nn

class OverfittingMonitor:
    """Monitor and detect overfitting during training"""
    
    def __init__(self, patience: int = 5, threshold: float = 0.05):
        self.patience = patience
        self.threshold = threshold
        self.train_history = []
        self.val_history = []
        
    def update(self, train_metric: float, val_metric: float) -> bool:
        """
        Update monitor with new metrics
        
        Args:
            train_metric: Training metric (e.g., accuracy)
            val_metric: Validation metric
            
        Returns:
            True if overfitting is detected
        """
        self.train_history.append(train_metric)
        self.val_history.append(val_metric)
        
        if len(self.train_history) < self.patience:
            return False
        
        # Check if gap between train and val is increasing
        recent_gaps = []
        for i in range(-self.patience, 0):
            gap = self.train_history[i] - self.val_history[i]
            recent_gaps.append(gap)
        
        # Overfitting if gap is consistently increasing
        increasing_count = sum(1 for i in range(1, len(recent_gaps))
                              if recent_gaps[i] > recent_gaps[i-1])
        
        avg_gap = np.mean(recent_gaps)
        
        return increasing_count >= self.patience - 1 or avg_gap > self.threshold
    
    def get_overfitting_score(self) -> float:
        """
        Calculate overfitting score (0-1, higher means more overfitting)
        
        Returns:
            Overfitting score
        """
        if len(self.train_history) < 2:
            return 0.0
        
        train_improvement = self.train_history[-1] - self.train_history[0]
        val_improvement = self.val_history[-1] - self.val_history[0]
        
        if train_improvement == 0:
            return 0.0
        
        # Score based on relative improvement difference
        score = max(0, min(1, (train_improvement - val_improvement) / abs(train_improvement)))
        
        # Factor in the gap between train and val
        current_gap = self.train_history[-1] - self.val_history[-1]
        gap_score = max(0, min(1, current_gap))
        
        # Combined score
        return 0.5 * score + 0.5 * gap_score
    
    def plot_overfitting_analysis(self, save_path: str = None):
        """
        Plot overfitting analysis
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.train_history) + 1)
        
        # Plot 1: Train vs Val metrics
        axes[0, 0].plot(epochs, self.train_history, 'b-', label='Training')
        axes[0, 0].plot(epochs, self.val_history, 'r-', label='Validation')
        axes[0, 0].fill_between(epochs, self.train_history, self.val_history,
                               alpha=0.3, color='gray')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Metric')
        axes[0, 0].set_title('Training vs Validation Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Overfitting gap
        gaps = [t - v for t, v in zip(self.train_history, self.val_history)]
        axes[0, 1].plot(epochs, gaps, 'g-', linewidth=2)
        axes[0, 1].axhline(y=self.threshold, color='r', linestyle='--',
                          label=f'Threshold ({self.threshold})')
        axes[0, 1].fill_between(epochs, 0, gaps, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Train - Val Gap')
        axes[0, 1].set_title('Overfitting Gap Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Rate of change
        if len(self.train_history) > 1:
            train_changes = np.diff(self.train_history)
            val_changes = np.diff(self.val_history)
            axes[1, 0].plot(epochs[1:], train_changes, 'b-', label='Training Δ')
            axes[1, 0].plot(epochs[1:], val_changes, 'r-', label='Validation Δ')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Change in Metric')
            axes[1, 0].set_title('Rate of Improvement')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Overfitting score
        scores = []
        for i in range(len(self.train_history)):
            if i > 0:
                train_imp = self.train_history[i] - self.train_history[0]
                val_imp = self.val_history[i] - self.val_history[0]
                score = max(0, min(1, (train_imp - val_imp) / (abs(train_imp) + 1e-8)))
                scores.append(score)
        
        if scores:
            axes[1, 1].plot(epochs[1:], scores, 'purple', linewidth=2)
            axes[1, 1].fill_between(epochs[1:], 0, scores, alpha=0.3, color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Overfitting Score')
            axes[1, 1].set_title('Overfitting Score (0-1)')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class AdaptiveRegularization:
    """Dynamically adjust regularization based on overfitting detection"""
    
    def __init__(self, initial_dropout: float = 0.3,
                 max_dropout: float = 0.7,
                 adjustment_rate: float = 0.01):
        self.current_dropout = initial_dropout
        self.max_dropout = max_dropout
        self.min_dropout = initial_dropout
        self.adjustment_rate = adjustment_rate
        self.history = []
        
    def adjust(self, overfitting_score: float) -> float:
        """
        Adjust dropout rate based on overfitting score
        
        Args:
            overfitting_score: Current overfitting score (0-1)
            
        Returns:
            New dropout rate
        """
        if overfitting_score > 0.7:
            # Strong overfitting - increase dropout
            self.current_dropout = min(
                self.max_dropout,
                self.current_dropout + self.adjustment_rate * 2
            )
        elif overfitting_score > 0.4:
            # Moderate overfitting - slightly increase dropout
            self.current_dropout = min(
                self.max_dropout,
                self.current_dropout + self.adjustment_rate
            )
        elif overfitting_score < 0.2:
            # No overfitting - can reduce dropout
            self.current_dropout = max(
                self.min_dropout,
                self.current_dropout - self.adjustment_rate
            )
        
        self.history.append(self.current_dropout)
        return self.current_dropout
    
    def update_model_dropout(self, model: nn.Module, new_rate: float):
        """
        Update dropout rates in the model
        
        Args:
            model: PyTorch model
            new_rate: New dropout rate
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_rate
            elif isinstance(module, nn.Dropout2d):
                module.p = new_rate * 0.5  # Use half rate for spatial dropout