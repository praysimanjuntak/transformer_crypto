from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Model hyperparameters (matching paper's decoder-only)
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    n_features: int = 4  # OHLC features
    pred_horizon: int = 4  # predict next N steps

    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100

    # Data (paper uses 70/30 train/test)
    data_dir: Path = Path("/home/pray/Documents/btc_transformer/btcusdt/1h")
    train_ratio: float = 0.7
    test_ratio: float = 0.3

    # Checkpoints
    checkpoint_dir: Path = Path("/home/pray/Documents/btc_transformer/checkpoints")

    # Device
    device: str = "cuda"

    # Patch size for PatchTransformer
    patch_size: int = 16

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PatchConfig(Config):
    """Extended config for PatchTransformer."""
    pass  # patch_size is now in base Config
