"""
Step 1 · Self-supervised pretraining. NO LABELS are used anywhere in this file.

    python pretrain.py            # about 15 minutes on a laptop CPU

Generates 20,000 synthetic slices (some with a lesion, most people would never
know which), throws the labels away, and trains a masked autoencoder to fill in
75% hidden patches. Saves weights/mae.pt and results/pretrain_loss.csv.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from selfsup.mae import MAE, random_masking
from selfsup.phantom import make_slice, make_dataset

ROOT = Path(__file__).parent


def main(n_images=20000, epochs=40, batch=64, lr=1.5e-3, seed=0):
    torch.manual_seed(seed)
    print(f"generating {n_images:,} unlabeled slices ...")
    X, _labels_thrown_away = make_dataset(n_images, seed=21)
    X = torch.tensor(X, dtype=torch.float32)

    model = MAE()
    print(f"MAE parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    steps = epochs * math.ceil(n_images / batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    # one fixed slice + one fixed mask, to watch the reconstruction improve during training
    demo = torch.tensor(make_slice(96, True, np.random.default_rng(6))[0][None], dtype=torch.float32)
    demo_mask = random_masking(1, 0.75, torch.Generator().manual_seed(3))
    snap_epochs = {1, 2, 3, 5, 10, 20, epochs}
    snaps = {}

    log = []
    for ep in range(epochs):
        t, total = time.time(), 0.0
        perm = torch.randperm(n_images)
        model.train()
        for i in range(0, n_images, batch):
            x = X[perm[i:i + batch]]
            if torch.rand(1) < 0.5:
                x = x.flip(-1)                                   # left-right flip
            pred, mask = model(x)
            loss = model.loss(x, pred, mask)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(x)
        log.append(total / n_images)
        if ep + 1 in snap_epochs:
            model.eval()
            rec, _ = model.reconstruct(demo, demo_mask)
            snaps[f"epoch_{ep + 1}"] = rec[0].numpy()
        print(f"epoch {ep + 1:2d}  reconstruction loss {log[-1]:.4f}  ({time.time() - t:.0f}s)", flush=True)

    (ROOT / "weights").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "weights" / "mae.pt")
    np.savez_compressed(ROOT / "results" / "recon_snapshots.npz", image=demo[0].numpy(),
                        hidden=demo_mask[1][0].numpy(), **snaps)
    np.savetxt(ROOT / "results" / "pretrain_loss.csv", np.array(log), delimiter=",",
               header="reconstruction_loss", comments="")
    print("saved weights/mae.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--images", type=int, default=20000)
    a = ap.parse_args()
    main(n_images=a.images, epochs=a.epochs)
