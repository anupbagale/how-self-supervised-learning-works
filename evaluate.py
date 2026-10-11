"""
Step 2 · How much do the self-supervised features help when labels are scarce?

    python evaluate.py            # about 20 minutes on a laptop CPU

For each budget of labelled scans (16 ... 1024), compares:
  scratch      ViT encoder trained from random weights on the labelled scans only
  mae_finetune same encoder, but starting from the MAE weights, then fine-tuned
  mae_linear   MAE encoder frozen; only a linear layer is trained (a "linear probe")
  mae_anomaly  MAE reconstruction error; the labels are used only to pick ONE threshold

Fairness: every method picks its learning rate on a separate validation set, never on
the test set. Each setting is repeated with 3 random draws of the labelled scans.
Also reports the zero-label anomaly-detection AUROC. Writes results/*.csv.
"""
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from selfsup.mae import MAE, Classifier
from selfsup.anomaly import anomaly_map, anomaly_score, auroc
from selfsup.phantom import make_dataset

ROOT = Path(__file__).parent
BUDGETS = [16, 32, 64, 128, 256, 1024]
SEEDS = [0, 1, 2]
LRS = [3e-4, 1e-3]
STEPS = 500


def pick(labels, n, seed):
    """n labelled scans, half with a lesion and half without."""
    g = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    return np.concatenate([g.choice(pos, n // 2, replace=False), g.choice(neg, n // 2, replace=False)])


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train_classifier(encoder, X, y, lr, seed):
    torch.manual_seed(seed)
    clf = Classifier(copy.deepcopy(encoder) if encoder is not None else None)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    bs = min(32, len(X))
    for _ in range(STEPS):
        clf.train()
        b = torch.randint(0, len(X), (bs,))
        xb = X[b].flip(-1) if torch.rand(1) < 0.5 else X[b]
        loss = F.cross_entropy(clf(xb), y[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return clf


def linear_probe(encoder, Xtr, ytr, Xev_list):
    """Frozen features -> standardise -> L2-regularised logistic regression."""
    with torch.no_grad():
        Ftr = encoder(Xtr).mean(1)
        Fev = [encoder(X).mean(1) for X in Xev_list]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd
    W = torch.zeros(Ftr.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Ftr @ W + b, ytr) + 1e-2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return [((((F_ - mu) / sd) @ W + b).argmax(1)) for F_ in Fev]


def best_threshold(scores, labels):
    """The single cut-off that classifies the given labelled scans best."""
    s = np.sort(scores)
    cands = np.concatenate([[s[0] - 1], (s[1:] + s[:-1]) / 2, [s[-1] + 1]])
    accs = [((scores > c) == labels).mean() for c in cands]
    return cands[int(np.argmax(accs))]


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print("generating labelled pool, validation and test sets ...")
    XL, yL = make_dataset(2000, seed=31)           # pool we draw labelled scans from
    XV, yV = make_dataset(400, seed=41)            # validation: only for choosing learning rates
    XT, yT = make_dataset(400, seed=2)             # test: same as the CNN and ViT repos
    XL, XV, XT = (torch.tensor(a, dtype=torch.float32) for a in (XL, XV, XT))
    tL, tV, tT = (torch.tensor(a) for a in (yL, yV, yT))

    mae = MAE()
    mae.load_state_dict(torch.load(ROOT / "weights" / "mae.pt"))
    mae.eval()
    enc = mae.encoder

    print("zero-label anomaly scores ...")
    sL = anomaly_score(anomaly_map(mae, XL)[0]).numpy()
    sT = anomaly_score(anomaly_map(mae, XT)[0]).numpy()
    zero_label_auroc = auroc(sT, yT)
    print(f"  AUROC with zero labels: {zero_label_auroc:.3f}")
    np.savetxt(ROOT / "results" / "anomaly_scores_test.csv", np.stack([sT, yT], 1), delimiter=",",
               header="anomaly_score,true_label", comments="")

    rows = []
    for n in BUDGETS:
        t = time.time()
        for method in ["scratch", "mae_finetune"]:
            init = None if method == "scratch" else enc
            per_lr = {}
            for lr in LRS:
                per_lr[lr] = [train_classifier(init, XL[pick(yL, n, s)], tL[pick(yL, n, s)], lr, s) for s in SEEDS]
            best = max(LRS, key=lambda lr: np.mean([accuracy(c, XV, tV) for c in per_lr[lr]]))
            for s, clf in zip(SEEDS, per_lr[best]):
                rows.append((n, method, s, accuracy(clf, XT, tT), best))
        for s in SEEDS:
            idx = pick(yL, n, s)
            pred = linear_probe(enc, XL[idx], tL[idx], [XT])[0]
            rows.append((n, "mae_linear", s, (pred == tT).float().mean().item(), 0))
            thr = best_threshold(sL[idx], yL[idx])
            rows.append((n, "mae_anomaly", s, float(((sT > thr) == yT).mean()), 0))
        summary = {m: np.mean([r[3] for r in rows if r[0] == n and r[1] == m])
                   for m in ["scratch", "mae_finetune", "mae_linear", "mae_anomaly"]}
        print(f"  {n:5d} labels  " + "  ".join(f"{k} {v:.1%}" for k, v in summary.items())
              + f"  ({time.time() - t:.0f}s)", flush=True)

    with open(ROOT / "results" / "label_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_labels", "method", "seed", "test_accuracy", "chosen_lr"])
        w.writerows(rows)
    with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["zero_label_anomaly_auroc", f"{zero_label_auroc:.4f}"])
    print("saved results/label_efficiency.csv")


if __name__ == "__main__":
    main()
