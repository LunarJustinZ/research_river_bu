"""Step 3 (optional): one-year collage of photos above measured vs predicted discharge.

Usage: python baseline/plot_collage.py [--split test]       (after train_linear.py; needs the images)
Blue frames: one 12:00 photo per month (the available day nearest the 15th). Orange frames: the year's three largest
sudden rises in flow (jump above the minimum of the three days before, events >= 20 days apart). Each thumbnail
shows measured and predicted discharge and whether the prediction is within x2. Below: daily measured (black) and
predicted (orange) discharge with the x2 band, on the same time axis.
Writes baseline/results/collage_<split>.png
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#8a8984", "#2a78d6", "#eb6834"
BANNER_PX = 84


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--year", type=int, default=None, help="for train: which year (default: the last)")
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    args = ap.parse_args()

    p = pd.read_csv(ROOT / "baseline" / "results" / f"predictions_{args.split}.csv", parse_dates=["date"])
    year = args.year or int(p.date.dt.year.max())
    p = p[p.date.dt.year == year]
    day = p.groupby("date").agg(q=("discharge_lps", "mean"), pred=("pred_discharge_lps", lambda v: np.expm1(np.log1p(v).mean())))
    noon = p[p.slot == "12:00"].set_index("date")

    days = noon.index
    monthly = [days[days.month == m][np.argmin(np.abs((days[days.month == m] - pd.Timestamp(year, m, 15)).days))]
               for m in range(1, 13) if (days.month == m).any()]
    ly = np.log1p(day.q)
    rise = (ly - ly.rolling(4, min_periods=1).min().shift(1)).dropna()
    events = []
    for d, r in rise.sort_values(ascending=False).items():
        if len(events) == 3 or r < 0.7:
            break
        if d in days and all(abs((d - e).days) > 20 for e, _ in events):
            events.append((d, r))

    fig = plt.figure(figsize=(22, 8.4))
    ax = fig.add_axes([0.06, 0.08, 0.92, 0.32])
    x0, x1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    ax.fill_between(day.index, day.pred / 2, day.pred * 2, color=ORANGE, alpha=0.15, lw=0, label="prediction ×2 band")
    ax.plot(day.index, day.q, color=INK, lw=1.6, label="measured (NEON)")
    ax.plot(day.index, day.pred, color=ORANGE, lw=1.3, label="predicted (DINOv3 + linear head)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 10, 100, 1000])
    ax.set_yticklabels(["0", "10", "100", "1,000"])
    ax.set_ylim(-1, 6000)
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("noon discharge (L/s, log)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=3, fontsize=9, frameon=False)
    hit = (np.abs(np.log1p(day.pred) - np.log1p(day.q)) <= np.log(2)).mean()
    ax.set_title(f"Lower Hop Brook {year} ({args.split}): photos above measured vs predicted discharge; "
                 f"{hit:.0%} of days within ×2 (blue = monthly photo, orange = sudden rise)", loc="left", fontsize=11)

    def place(d, row, colour, head):
        w = 0.075
        fx = min(max(0.06 + 0.92 * (d - x0).days / (x1 - x0).days, 0.06 + w / 2), 0.98 - w / 2)
        bottom = 0.70 if row == 0 else 0.47
        a = fig.add_axes([fx - w / 2, bottom, w, w * 22 / 8.4 / 1.5])
        im = Image.open(pathlib.Path(args.images) / noon.site[d] / str(year) / noon.filename[d]).convert("RGB")
        a.imshow(im.crop((0, BANNER_PX, im.width, im.height)).resize((300, 200)))
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(colour)
            s.set_linewidth(2.5)
        q, pr = day.q[d], day.pred[d]
        ok = "✓" if abs(np.log1p(pr) - np.log1p(q)) <= np.log(2) else "✗"
        a.set_title(f"{head}\nmeas {q:,.0f} | pred {pr:,.0f} {ok}", fontsize=7.5, pad=2)
        ax.plot(d, q, "o", color=colour, ms=6, mec="white", mew=1.2, zorder=5)
        fig.add_artist(ConnectionPatch(xyA=(0.5, 0), coordsA=a.transAxes, xyB=(mdates.date2num(d), q),
                                       coordsB=ax.transData, color=colour, lw=0.8, alpha=0.6))

    for d in monthly:
        place(d, 0, BLUE, f"{d:%d %b}")
    for d, r in events:
        place(d, 1, ORANGE, f"{d:%d %b}: rise ×{np.exp(r):.0f}")
    out = ROOT / "baseline" / "results" / f"collage_{args.split}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.15)
    print("saved", out)


if __name__ == "__main__":
    main()
