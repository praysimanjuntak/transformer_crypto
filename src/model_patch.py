"""
Modern Time-Series Transformer with:
- RevIN (Reversible Instance Normalization)
- Patching (group candles into patches)
- Predicts multiple future steps directly
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Normalizes input based on the current window's statistics,
    then denormalizes output using the same statistics.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mode: str = "norm"):
        """
        Args:
            x: (batch, seq_len, features) or (batch, pred_len, features)
            mode: "norm" for normalization, "denorm" for denormalization
        """
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        elif mode == "denorm":
            return x * self.std + self.mean
        else:
            raise ValueError(f"Unknown mode: {mode}")


class PatchEmbedding(nn.Module):
    """Convert sequence into patches and embed."""

    def __init__(self, patch_size: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_size = patch_size
        # Each patch contains patch_size * n_features values
        self.proj = nn.Linear(patch_size * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, num_patches, d_model)
        """
        B, L, F = x.shape
        # Reshape to patches: (batch, num_patches, patch_size * features)
        num_patches = L // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]  # Trim to fit patches
        x = x.reshape(B, num_patches, self.patch_size * F)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderBlock(nn.Module):
    """Single decoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class PatchTransformer(nn.Module):
    """
    Modern Time-Series Transformer with RevIN and Patching.

    Input: (batch, seq_len, n_features) - raw OHLC values (not returns!)
    Output: (batch, pred_len) - predicted close prices
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_size = getattr(config, 'patch_size', 16)
        self.pred_len = config.pred_horizon

        # RevIN for normalization
        self.revin = RevIN(config.n_features)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            self.patch_size, config.d_model, config.n_features
        )

        # Calculate number of patches
        self.num_patches = config.max_seq_len // self.patch_size

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, self.num_patches + 10, config.dropout
        )

        # Decoder blocks
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output projection: flatten patches and project to predictions
        # We use the last patch's representation to predict all future steps
        self.output_proj = nn.Linear(config.d_model, self.pred_len)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) - raw OHLC prices

        Returns:
            (batch, pred_len) - predicted close prices (denormalized)
        """
        # 1. RevIN: Normalize based on input window
        x_norm = self.revin(x, mode="norm")

        # 2. Patch embedding
        x_embed = self.patch_embed(x_norm)  # (batch, num_patches, d_model)

        # 3. Add positional encoding
        x_embed = self.pos_encoding(x_embed)

        # 4. Pass through decoder blocks
        for block in self.blocks:
            x_embed = block(x_embed)

        # 5. Final layer norm
        x_embed = self.ln_final(x_embed)

        # 6. Get last patch and project to predictions
        x_last = x_embed[:, -1, :]  # (batch, d_model)
        pred_norm = self.output_proj(x_last)  # (batch, pred_len)

        # 7. Denormalize: use the close price statistics
        # We need to expand pred_norm to match RevIN's expected shape
        # pred_norm is (batch, pred_len), we need (batch, pred_len, 1) for denorm
        pred_norm = pred_norm.unsqueeze(-1)  # (batch, pred_len, 1)

        # Use only close price stats (feature index 3 in OHLC)
        # But RevIN stored stats for all features, so we use the overall mean/std
        pred = pred_norm * self.revin.std[:, :, 3:4] + self.revin.mean[:, :, 3:4]

        return pred.squeeze(-1)  # (batch, pred_len)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
