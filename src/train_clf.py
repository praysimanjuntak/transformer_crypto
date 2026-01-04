"""
Trainer for trend classification with BCE loss.
"""

import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .model_clf import TrendClassifier


class ClassifierTrainer:
    """Trainer for binary trend classification."""

    def __init__(
        self,
        model: TrendClassifier,
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
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_losses = []
        self.test_losses = []
        self.test_accs = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            # Compute accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()

        accuracy = correct / total
        return total_loss / len(self.test_loader), accuracy

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "test_accs": self.test_accs,
            "config": self.config,
        }
        path = self.config.checkpoint_dir / "model.pt"
        torch.save(checkpoint, path)

    def train(self) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        best_acc = 0.0

        for epoch in range(self.config.epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            elapsed = time.time() - start_time

            # Track best accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_checkpoint(epoch)

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc*100:.2f}% | "
                f"Best: {best_acc*100:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

        # Save final checkpoint
        self.save_checkpoint(self.config.epochs)

        return {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "test_accs": self.test_accs,
            "best_acc": best_acc,
        }
