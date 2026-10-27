"""
PyTorch Dataset for the multimodal stock-chart forecasting samples.

Reads from manifest_split.csv (produced by make_splits.py) and the .npz
sample files (produced by build_dataset.py). Converts uint8 [0,255]
storage back to float32 [-1, 1] (the standard input range for diffusion
models - see diffusion.py).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

import config


def _to_model_range(x_u8: np.ndarray) -> torch.Tensor:
    """uint8 [0,255] -> float32 [-1,1] (standard diffusion model input range)."""
    x = torch.from_numpy(x_u8.astype(np.float32) / 255.0)
    return x * 2.0 - 1.0


class StockChartDataset(Dataset):
    """
    Yields (X, Y, horizon_days) where:
      X: (3, H, W) float32 in [-1, 1] - the conditioning input (fundamentals,
         news, price channels stacked - see render_channels.stack_channels)
      Y: (1, H, W) float32 in [-1, 1] - the target chart at t+horizon_days
      horizon_days: scalar int - which of the configured horizons this
         sample targets, so the model can condition on "how far ahead"
    """

    def __init__(self, split: str, manifest_path=None):
        manifest_path = manifest_path or (config.PROCESSED_DIR / "manifest_split.csv")
        manifest = pd.read_csv(manifest_path)
        if "split" not in manifest.columns:
            raise ValueError(f"{manifest_path} has no 'split' column - run make_splits.py first.")
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError(f"No samples found for split='{split}'. Check make_splits.py's output/fractions.")

        # Horizon days -> a small integer class index, used for the model's
        # horizon-conditioning embedding (see model.py). Sorted so the
        # mapping is stable and human-readable (0=shortest horizon, etc.)
        self.horizon_to_idx = {h: i for i, h in enumerate(sorted(config.HORIZON_DAYS_OPTIONS))}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        data = np.load(config.SAMPLES_DIR / row["file"])
        X = _to_model_range(data["X"]).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
        Y = _to_model_range(data["Y"]).unsqueeze(0)       # (H,W)   -> (1,H,W)
        horizon_idx = self.horizon_to_idx[int(row["horizon_days"])]
        return X, Y, horizon_idx
