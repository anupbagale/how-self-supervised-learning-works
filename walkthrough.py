"""
Step 3 · Draw every figure in the README from the trained MAE and the saved results.

    python walkthrough.py        # about 1 minute
"""
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

from selfsup.mae import MAE, PATCH, GRID, N_PATCHES, random_masking, masking_from_hidden, patchify
from selfsup.anomaly import anomaly_map, anomaly_score, pixel_anomaly_map, auroc
from selfsup.phantom import make_slice, make_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
EXAMPLE_SEED = 6          # the same slice used in the CNN and ViT repos

# ---- one visual language (matches the CNN and ViT repos) ---------------------------
SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
HIDE = "#b9cbe3"                                           # colour of a hidden patch (light blue-gray)
HEAT = LinearSegmentedColormap.from_list("heat", ["#f5b48f", "#eb6834", "#a8401a"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
})


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRIDC)


def title(ax, t, sub=None):
    ax.set_title(t, fontsize=10.5, fontweight="bold", pad=4)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved assets/{name}")


def gray(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    clean(ax)


def masked_view(img, hidden):
    """RGB image with hidden patches painted dark gray."""
    rgb = np.repeat(np.clip(img, 0, 1)[..., None], 3, axis=2)
    col = np.array(matplotlib.colors.to_rgb(HIDE))
    for i in np.where(hidden)[0]:
        r, c = divmod(i, GRID)
        rgb[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = col
    return rgb


def outline(ax, idx, color=ORANGE, lw=2.0, ls="-"):
    r, c = divmod(idx, GRID)
    ax.add_patch(Rectangle((c * PATCH - .5, r * PATCH - .5), PATCH, PATCH, fill=False, ec=color, lw=lw, ls=ls))


def heat_overlay(ax, img, amap, vmax):
    """amap: 96 x 96 pixel-level map (or 8 x 8 patch map, which is upsampled)."""
    gray(ax, img)
    if amap.shape[0] == GRID:
        amap = np.kron(amap, np.ones((PATCH, PATCH)))
    a = np.clip(amap / vmax, 0, 1)
    rgba = HEAT(a)
    rgba[..., 3] = np.where(a < 0.2, 0, 0.25 + 0.7 * a)
    ax.imshow(rgba, interpolation="nearest")


def lesion_patches(info):
    """Indices of patches the lesion touches."""
    out = set()
    for dy in (-info["radius"], 0, info["radius"]):
        for dx in (-info["radius"], 0, info["radius"]):
            r, c = int((info["row"] + dy) // PATCH), int((info["col"] + dx) // PATCH)
            out.add(r * GRID + c)
    return sorted(out)


# ==================================================================================
print("loading the pretrained MAE ...")
mae = MAE()
mae.load_state_dict(torch.load(ROOT / "weights" / "mae.pt"))
mae.eval()
T = lambda a: torch.tensor(np.asarray(a)[None] if np.asarray(a).ndim == 2 else np.asarray(a), dtype=torch.float32)

img, info = make_slice(96, True, np.random.default_rng(EXAMPLE_SEED))
twin, _ = make_slice(96, False, np.random.default_rng(EXAMPLE_SEED))
L = int(info["row"] // PATCH) * GRID + int(info["col"] // PATCH)
LP = lesion_patches(info)

# a fixed 75% mask for the main example (seed matches pretrain.py's snapshot mask)
demo_mask = random_masking(1, 0.75, torch.Generator().manual_seed(3))
hidden = demo_mask[1][0].numpy().astype(bool)
rec, _ = mae.reconstruct(T(img), demo_mask)
rec = rec[0].numpy()
paste = np.where(np.kron(hidden.reshape(GRID, GRID), np.ones((PATCH, PATCH))).astype(bool), rec, img)

# zero-label anomaly map for the example and its lesion-free twin
pmap = pixel_anomaly_map(mae, T(np.stack([img, twin]))).numpy()     # for display
AV = pmap[0].max()

# results from evaluate.py
rows = list(csv.DictReader(open(ROOT / "results" / "label_efficiency.csv")))
METHODS = [("scratch", "trained from scratch", GRAY),
           ("mae_linear", "MAE, frozen + linear layer", AQUA),
           ("mae_finetune", "MAE, fine-tuned", BLUE),
           ("mae_anomaly", "MAE anomaly score + 1 threshold", ORANGE)]
budgets = sorted({int(r["n_labels"]) for r in rows})
stats = {}
for m, *_ in METHODS:
    acc = np.array([[float(r["test_accuracy"]) for r in rows if r["method"] == m and int(r["n_labels"]) == n]
                    for n in budgets])
    stats[m] = (acc.mean(1), acc.min(1), acc.max(1))
scores_test = np.loadtxt(ROOT / "results" / "anomaly_scores_test.csv", delimiter=",", skiprows=1)
zero_auroc = auroc(scores_test[:, 0], scores_test[:, 1])
print(f"  zero-label AUROC {zero_auroc:.3f}")


def efficiency_plot(ax, small=False):
    for m, name, col in METHODS:
        mu, lo, hi = stats[m]
        ax.fill_between(budgets, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(budgets, mu, color=col, lw=2.4 if m != "scratch" else 2, marker="o", ms=5 if not small else 4,
                label=name, zorder=3)
    ax.axhline(0.5, color=GRIDC, lw=1, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0.45, 1.01)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["50%", "60", "70", "80", "90", "100%"])
    ax.grid(axis="y", color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if not small:
        ax.set_xlabel("number of labelled scans used for training")
        ax.set_ylabel("accuracy on 400 unseen test scans")
        ax.text(budgets[-1], 0.492, "50% = guessing", fontsize=8.5, color=MUTED, va="top", ha="right")


# ---- HERO ----------------------------------------------------------------------------
print("drawing figures ...")
fig = plt.figure(figsize=(17, 5.2))
H, y0 = 0.56, 0.19
sq = H * 5.2 / 17
stages = [("Unlabelled scans", "no diagnosis needed", "20,000 slices, labels thrown away", sq),
          ("Hide 75%", "48 of 64 patches", "the network only sees 16 pieces", sq),
          ("Fill in the gaps", "self-supervised training", "the image is its own answer key", sq),
          ("Zero labels", f"AUROC {zero_auroc:.2f}", "leftover error = lesion", sq),
          ("Few labels", "accuracy vs labels", "pretrained beats scratch from 64 labels", sq * 1.25)]
gap = (1 - 0.03 - sum(s[3] for s in stages)) / (len(stages) - 1)
x, boxes = 0.015, []
for i, (t, sub, expl, w) in enumerate(stages):
    boxes.append((x, w))
    fig.text(x + w / 2, y0 + H + 0.14, f"{i + 1}  {t}", ha="center", fontsize=12.5, fontweight="bold")
    fig.text(x + w / 2, y0 + H + 0.075, sub, ha="center", fontsize=10, color=BLUE, family="DejaVu Sans Mono")
    fig.text(x + w / 2, y0 - 0.08, expl, ha="center", fontsize=9.5, color=INK2)
    x += w + gap
# 1: a small stack of slices
xb, wb = boxes[0]
stack = make_dataset(4, seed=77)[0]
for k, s_img in enumerate(stack[::-1]):
    off = (3 - k) * 0.012
    a = fig.add_axes([xb + off, y0 + off * 17 / 5.2, wb * 0.86, H * 0.86 - 0.012 * 3 * 17 / 5.2 + 0.03]); gray(a, s_img)
a = fig.add_axes([boxes[1][0], y0, boxes[1][1], H]); a.imshow(masked_view(img, hidden), interpolation="nearest"); clean(a)
a = fig.add_axes([boxes[2][0], y0, boxes[2][1], H]); gray(a, paste)
for i in np.where(hidden)[0]:
    r, c = divmod(i, GRID)
    a.add_patch(Rectangle((c * PATCH - .5, r * PATCH - .5), PATCH, PATCH, fill=False, ec=BLUE, lw=0.5, alpha=0.5))
a = fig.add_axes([boxes[3][0], y0, boxes[3][1], H]); heat_overlay(a, img, pmap[0], AV)
a = fig.add_axes([boxes[4][0] + 0.03, y0 + 0.03, boxes[4][1] - 0.03, H - 0.03]); efficiency_plot(a, small=True)
a.set_title("orange: anomaly · blue: MAE fine-tuned · gray: scratch", fontsize=7.5, color=INK2, pad=3)
for (xa, wa), (xc, _) in zip(boxes[:-1], boxes[1:]):
    fig.patches.append(FancyArrowPatch((xa + wa + 0.006, y0 + H / 2), (xc - 0.006, y0 + H / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                       color=MUTED, lw=1.4))
save(fig, "hero.png")

# ---- 01 · masking -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.62, 1]})
gray(axs[0], img); title(axs[0], "1 · original slice", "64 patches of 12 × 12 pixels")
axs[1].imshow(masked_view(img, hidden), interpolation="nearest"); clean(axs[1])
title(axs[1], "2 · hide 75% at random", "48 patches hidden, 16 visible")
vis = np.where(~hidden)[0]
axs[2].axis("off"); title(axs[2], "3 · all the encoder sees", "16 patches + their positions")
for k, i in enumerate(vis):
    r, c = divmod(k, 4)
    a = axs[2].inset_axes([c / 4 + 0.01, 1 - (r + 1) / 4 + 0.01, 1 / 4 - 0.02, 1 / 4 - 0.02])
    rr, cc = divmod(i, GRID)
    a.imshow(img[rr * PATCH:(rr + 1) * PATCH, cc * PATCH:(cc + 1) * PATCH], cmap="gray", vmin=0, vmax=1)
    clean(a)
gray(axs[3], paste)
for i in np.where(hidden)[0]:
    r, c = divmod(i, GRID)
    axs[3].add_patch(Rectangle((c * PATCH - .5, r * PATCH - .5), PATCH, PATCH, fill=False, ec=BLUE, lw=0.6, alpha=0.6))
title(axs[3], "4 · the decoder fills in the rest", "blue outlines = painted by the model")
fig.text(0.5, -0.05, "Training signal: how far the painted patches are from the real ones (mean squared error). "
         "No human ever labelled anything.", ha="center", fontsize=9.5, color=INK2)
save(fig, "01_masking.png")

# ---- 02 · learning to see (snapshots + loss) ---------------------------------------------------
snap = np.load(ROOT / "results" / "recon_snapshots.npz")
ep_keys = sorted([k for k in snap.files if k.startswith("epoch_")], key=lambda k: int(k.split("_")[1]))
loss = np.loadtxt(ROOT / "results" / "pretrain_loss.csv", delimiter=",", skiprows=1)
s_hidden = snap["hidden"].astype(bool)
s_mask = np.kron(s_hidden.reshape(GRID, GRID), np.ones((PATCH, PATCH))).astype(bool)
fig = plt.figure(figsize=(15.5, 5.8))
gs = fig.add_gridspec(2, len(ep_keys) + 1, height_ratios=[1, 0.9], hspace=0.28, wspace=0.08)
a = fig.add_subplot(gs[0, 0]); a.imshow(masked_view(snap["image"], s_hidden), interpolation="nearest"); clean(a)
title(a, "input (75% hidden)")
for j, k in enumerate(ep_keys):
    a = fig.add_subplot(gs[0, j + 1]); gray(a, np.where(s_mask, snap[k], snap["image"]))
    title(a, f"after epoch {k.split('_')[1]}")
a = fig.add_subplot(gs[1, 1:-1])
ep = np.arange(1, len(loss) + 1)
a.plot(ep, loss, color=BLUE, lw=2.2)
a.set_yscale("log"); a.set_xlabel("epoch (one pass over 20,000 unlabelled slices)")
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_yticks([0.03, 0.05, 0.1, 0.2, 0.4])
a.set_ylabel("reconstruction error\n(hidden patches, log scale)")
a.grid(color=GRIDC, lw=0.8); a.set_axisbelow(True); a.tick_params(length=0)
for sp in ["top", "right"]:
    a.spines[sp].set_visible(False)
a.annotate(f"{loss[-1]:.3f}", (ep[-1], loss[-1]), xytext=(6, 0), textcoords="offset points",
           va="center", fontsize=9, color=INK)
save(fig, "02_learning_to_see.png")

frames = []
fig, axs = plt.subplots(1, 3, figsize=(9, 3.5))
for k in ep_keys:
    for a in axs:
        a.clear()
    axs[0].imshow(masked_view(snap["image"], s_hidden), interpolation="nearest"); clean(axs[0])
    axs[0].set_title("what the model sees", fontsize=11, fontweight="bold")
    gray(axs[1], np.where(s_mask, snap[k], snap["image"]))
    axs[1].set_title(f"its guess · epoch {k.split('_')[1]}", fontsize=11, fontweight="bold")
    gray(axs[2], snap["image"]); axs[2].set_title("the real slice", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
plt.close(fig)
small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS).quantize(128) for f in frames]
small[0].save(OUT / "02_learning_to_see.gif", save_all=True, append_images=small[1:],
              duration=[900] * (len(small) - 1) + [2500], loop=0, optimize=True)
print("  saved assets/02_learning_to_see.gif")

# ---- 03 · more examples ------------------------------------------------------------------------
ex = make_dataset(6, seed=99)[0]
masks = random_masking(len(ex), 0.75, torch.Generator().manual_seed(11))
recs, _ = mae.reconstruct(T(ex), masks)
fig, axs = plt.subplots(3, len(ex), figsize=(14, 7.2))
for j in range(len(ex)):
    h = masks[1][j].numpy().astype(bool)
    hm = np.kron(h.reshape(GRID, GRID), np.ones((PATCH, PATCH))).astype(bool)
    axs[0, j].imshow(masked_view(ex[j], h), interpolation="nearest"); clean(axs[0, j])
    gray(axs[1, j], np.where(hm, recs[j].numpy(), ex[j])); gray(axs[2, j], ex[j])
for r, lab in enumerate(["model sees", "model's guess", "real slice"]):
    axs[r, 0].set_ylabel(lab, fontsize=10.5, fontweight="bold", color=INK, labelpad=8)
fig.text(0.5, 0.0, "New slices it never saw in training. From 16 pieces it recovers the skull, ventricles and tissue.\n"
         "But a hidden lesion can't be guessed from its surroundings (column 4: gone in the guess). "
         "Keep that in mind for the next step.", ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.6)
fig.subplots_adjust(wspace=0.05, hspace=0.06, bottom=0.05)
save(fig, "03_examples.png")

# ---- 04 · the model "heals" the lesion -----------------------------------------------------------
g = np.random.default_rng(5)
others = [i for i in range(N_PATCHES) if i not in LP]
h4 = np.zeros(N_PATCHES, bool); h4[LP] = True
h4[g.choice(others, 48 - len(LP), replace=False)] = True
rec4, _ = mae.reconstruct(T(np.stack([img, twin])), masking_from_hidden(np.stack([h4, h4])))
rec4 = rec4.numpy()
hm4 = np.kron(h4.reshape(GRID, GRID), np.ones((PATCH, PATCH))).astype(bool)
fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
for r, (im, lab) in enumerate([(img, "with lesion"), (twin, "same slice, no lesion")]):
    axs[r, 0].imshow(masked_view(im, h4), interpolation="nearest"); clean(axs[r, 0])
    title(axs[r, 0], f"{lab} · lesion area hidden")
    gray(axs[r, 1], np.where(hm4, rec4[r], im)); title(axs[r, 1], "model fills in")
    gray(axs[r, 2], im); title(axs[r, 2], "real slice")
    heat_overlay(axs[r, 3], im, pmap[r], AV); title(axs[r, 3], "real − predicted (brighter than expected)")
    for i in LP:
        for c in range(4):
            outline(axs[r, c], i, color=ORANGE, lw=1.2, ls=(0, (3, 2)))
fig.text(0.5, 0.02, "The model learned what slices usually look like. Asked to fill in the lesion area, it paints "
         "normal brain tissue: it 'heals' the scan.\nSubtract its guess from reality and the lesion is left as a "
         "round blob, found with no label. The skull edge lights up a little in BOTH rows:\nthe model can't predict "
         "its exact position. Real pipelines remove the skull first for this reason.",
         ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.6)
fig.subplots_adjust(wspace=0.06, hspace=0.18, bottom=0.12)
save(fig, "04_healing.png")

# ---- 05 · anomaly detection on the test set ---------------------------------------------------------
XT, yT = make_dataset(400, seed=2)
pick = list(np.where(yT == 1)[0][:4]) + list(np.where(yT == 0)[0][:2])
am_patch = anomaly_map(mae, T(XT[pick]))[0].numpy()              # the score uses patch maps
am = pixel_anomaly_map(mae, T(XT[pick])).numpy()                    # the picture uses pixel maps
fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.15, 2.2], hspace=0.22, wspace=0.06)
for k, i in enumerate(pick):
    a = fig.add_subplot(gs[k // 3, k % 3]); heat_overlay(a, XT[i], am[k], AV)
    sc = am_patch[k].max()
    a.set_title(f"{'lesion' if yT[i] else 'normal'} · score {sc:.2f}", fontsize=10, fontweight="bold", pad=3,
                color=INK)
a = fig.add_subplot(gs[:, 4])
sc, lab = scores_test[:, 0], scores_test[:, 1].astype(bool)
bins = np.linspace(0, np.percentile(sc, 99.5), 40)
a.hist(sc[~lab], bins=bins, color=BLUE, alpha=0.85, label="normal scans", edgecolor=SURFACE, lw=0.6)
a.hist(sc[lab], bins=bins, color=ORANGE, alpha=0.85, label="scans with a lesion", edgecolor=SURFACE, lw=0.6)
a.set_xlabel("anomaly score (most surprising patch)"); a.set_ylabel("number of test scans")
a.legend(frameon=False, loc="upper right")
a.set_title(f"400 test scans · AUROC {zero_auroc:.3f} with zero labels", fontsize=10.5, fontweight="bold")
for sp in ["top", "right"]:
    a.spines[sp].set_visible(False)
a.tick_params(length=0); a.grid(axis="y", color=GRIDC, lw=0.8); a.set_axisbelow(True)
fig.text(0.5, -0.02, "AUROC = chance that a random lesion scan gets a higher score than a random normal scan "
         "(1.0 = perfect, 0.5 = guessing).", ha="center", fontsize=9.5, color=INK2)
save(fig, "05_anomaly_detection.png")

# ---- 06 · label efficiency --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
efficiency_plot(ax)
for m, name, col in METHODS:
    nudge = {"mae_anomaly": 5, "mae_finetune": -5, "mae_linear": 4, "scratch": -4}[m]
    ax.annotate(name, (budgets[-1], stats[m][0][-1]), xytext=(10, nudge), textcoords="offset points",
                va="center", fontsize=9.5, color=INK)
    ax.plot([budgets[-1] * 1.04], [stats[m][0][-1]], marker="o", color=col, ms=0)
ax.set_title("How many labelled scans does each approach need?", fontsize=12, fontweight="bold", loc="left", pad=24)
ax.text(0, 1.0, "lines = mean of 3 random draws of labelled scans, shaded = min to max",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
save(fig, "06_label_efficiency.png")

print("done")
