"""
Trainer for PatchTransformer with MAE (L1) loss.
"""

import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .model_patch import PatchTransformer


class PatchTrainer:
    """Trainer for time-series forecasting with MAE loss."""

    def __init__(
        self,
        model: PatchTransformer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # AdamW optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )

        # MAE (L1) loss - encourages median prediction
        self.criterion = nn.L1Loss()

        self.train_losses = []
        self.test_losses = []
        self.test_maes = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.2f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)
            mae = torch.abs(pred - y).mean()

            total_loss += loss.item()
            total_mae += mae.item() * y.size(0)
            n_samples += y.size(0)

        return total_loss / len(self.test_loader), total_mae / n_samples

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "test_maes": self.test_maes,
            "config": self.config,
        }
        path = self.config.checkpoint_dir / "model.pt"
        torch.save(checkpoint, path)

    def train(self) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        best_mae = float('inf')

        for epoch in range(self.config.epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            test_loss, test_mae = self.evaluate()

            # Step scheduler
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.test_maes.append(test_mae)

            elapsed = time.time() - start_time

            # Track best MAE
            if test_mae < best_mae:
                best_mae = test_mae
                self.save_checkpoint(epoch)

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train: {train_loss:.2f} | "
                f"Test: {test_loss:.2f} | "
                f"MAE: ${test_mae:.2f} | "
                f"Best: ${best_mae:.2f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        # Save final checkpoint
        self.save_checkpoint(self.config.epochs)

        return {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "test_maes": self.test_maes,
            "best_mae": best_mae,
        }
