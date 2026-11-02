"""Step 1 of the baseline: frozen DINOv3 features for every image of the given splits (GPU recommended).

Usage: python baseline/extract_features.py [--splits train val test test_oksr] [--model facebook/dinov3-vitb16-pretrain-lvd1689m]
Needs the images (python download.py) and: pip install torch "transformers>=4.56" pillow pandas numpy
DINOv3 is a gated model on Hugging Face: accept its license on the model page and run `huggingface-cli login` once.

Each image: the camera's timestamp banner (top 84 px) is cut off, the rest resized to 448 x 304 (multiples of the 16-px
patch) and ImageNet-normalised. Feature = [CLS token, mean of the patch tokens] -> 2 x 768 numbers for ViT-B.
Writes baseline/features/<split>.npz with image_id and feats.
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANNER_PX, SIZE = 84, (448, 304)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Images(Dataset):
    def __init__(self, rows, images):
        self.rows, self.images = rows, images

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        im = Image.open(self.images / r.site / r.date[:4] / r.filename).convert("RGB")
        im = im.crop((0, BANNER_PX, im.width, im.height)).resize(SIZE, Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "test_oksr"])
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 where the GPU supports it fast (A100+); fp32 otherwise (fp16 overflows in DINOv3)
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    n_skip = 1 + model.config.num_register_tokens  # CLS + register tokens come before the patch tokens
    out = ROOT / "baseline" / "features"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model} on {dev} ({dtype})", flush=True)

    for split in args.splits:
        rows = pd.read_csv(ROOT / "data" / "discharge" / f"{split}.csv")
        loader = DataLoader(Images(rows, pathlib.Path(args.images)), batch_size=args.batch, num_workers=8)
        feats, t0 = [], time.time()
        with torch.no_grad():
            for x in loader:
                h = model(pixel_values=x.to(dev, dtype)).last_hidden_state.float()
                feats.append(torch.cat([h[:, 0], h[:, n_skip:].mean(1)], 1).cpu())
        feats = torch.cat(feats).numpy()
        np.savez(out / f"{split}.npz", image_id=rows.image_id.to_numpy(), feats=feats)
        print(f"{split}: {feats.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
