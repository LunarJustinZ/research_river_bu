"""Minimal example: a PyTorch Dataset over one split (image, log(1 + discharge)).

Usage: python examples/load_split.py --split val
Needs the images downloaded first (python download.py --split val) and: pip install pandas pillow torch
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent


class StreamDischarge(Dataset):
    def __init__(self, split, images=ROOT / "data" / "images"):
        self.rows = pd.read_csv(ROOT / "data" / "discharge" / f"{split}.csv")
        self.images = pathlib.Path(images)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(self.images / r.site / r.date[:4] / r.filename).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # uint8, 3 x 960 x 1296
        y = torch.tensor(np.log1p(r.discharge_lps), dtype=torch.float32)
        return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ds = StreamDischarge(ap.parse_args().split)
    x, y = ds[0]
    r = ds.rows.iloc[0]
    print(f"{len(ds)} images; first: {r.date} {r.time_local}, image {tuple(x.shape)}, "
          f"discharge {r.discharge_lps:.1f} L/s (target log1p = {y.item():.3f})")
