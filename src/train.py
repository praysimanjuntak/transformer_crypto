import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .model import CandleTransformer


class Trainer:
    """Simple trainer matching paper's approach."""

    def __init__(
        self,
        model: CandleTransformer,
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

        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.test_losses = []

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
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)
            mae = torch.abs(pred - y).mean()

            total_loss += loss.item()
            total_mae += mae.item()

        return total_loss / len(self.test_loader), total_mae / len(self.test_loader)

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "config": self.config,
        }
        path = self.config.checkpoint_dir / "model.pt"
        torch.save(checkpoint, path)

    def train(self) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            test_loss, test_mae = self.evaluate()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | "
                f"Test MAE: {test_mae:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        self.save_checkpoint(self.config.epochs)

        return {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
        }
