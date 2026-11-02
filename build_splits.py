"""Maintainer script: write data/discharge/{train,val,test,test_oksr}.csv and stats.json from the working files.

Usage: python build_splits.py WORK_ROOT
  WORK_ROOT holds dataset_discharge.csv (splits; from the project's build_dataset.py), labels/perimage_discharge.csv
  (per-image labels; from build_perimage_labels.py) and the images themselves (to compute sha256).
Users of the dataset never need this script; they only run download.py.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent / "data" / "discharge"
ARCHIVE = "https://phenocam.nau.edu/data/archive"
SITES = {"HOPB": "NEON.D01.HOPB.DP1.20002", "OKSR": "NEON.D18.OKSR.DP1.20002"}
COLUMNS = ["image_id", "site", "date", "time_local", "slot", "split", "fewshot", "url", "filename", "discharge_lps",
           "n_records", "sha256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(work):
    work = pathlib.Path(work)
    d = pd.read_csv(work / "dataset_discharge.csv", parse_dates=["date"])
    lab = pd.read_csv(work / "labels" / "perimage_discharge.csv")
    d = d.merge(lab, on="path", how="left")
    n_all = len(d)
    d = d.dropna(subset=["q_img"]).copy()
    print(f"{n_all} images, {n_all - len(d)} without a per-image label dropped")
    d["filename"] = d.path.str.rsplit("/", n=1).str[-1]

    stamp = pd.to_datetime(d.filename.str.extract(r"_(\d{4}_\d{2}_\d{2}_\d{6})\.jpg")[0], format="%Y_%m_%d_%H%M%S")
    d["time_local"] = stamp.dt.strftime("%H:%M:%S")
    d["url"] = [f"{ARCHIVE}/{SITES[s]}/{t:%Y/%m}/{f}" for s, t, f in zip(d.site, stamp, d.filename)]
    d["image_id"] = d.filename.str.replace(".jpg", "", regex=False)
    d["discharge_lps"] = d.q_img.round(4)
    d["n_records"] = d.n_img.astype(int)
    d["fewshot"] = d.oksr_fewshot.fillna("")
    d["date"] = d.date.dt.strftime("%Y-%m-%d")
    print("hashing images ...", flush=True)
    d["sha256"] = [sha256(work / p) for p in d.path]

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ["train", "val", "test", "test_oksr"]:
        s = d[d.split == split].sort_values(["date", "slot"])[COLUMNS]
        s.to_csv(OUT / f"{split}.csv", index=False)
        q = s.discharge_lps
        stats[split] = {
            "site": s.site.iloc[0], "images": int(len(s)), "days": int(s.date.nunique()),
            "first_date": s.date.min(), "last_date": s.date.max(),
            "images_per_day": round(len(s) / s.date.nunique(), 2),
            "discharge_lps": {k: round(float(v), 2) for k, v in
                              zip(["min", "p05", "median", "p95", "max"], np.quantile(q, [0, .05, .5, .95, 1]))},
            "zero_flow_images": int((q <= 0.5).sum()),
        }
        print(split, stats[split]["days"], "days", stats[split]["images"], "images")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
