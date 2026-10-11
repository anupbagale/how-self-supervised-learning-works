"""
Finding lesions with ZERO labels, using only the pretrained masked autoencoder.

The MAE has only ever learned what slices usually look like. Hide a patch that
contains a lesion and it paints in what *should* be there: ordinary brain
tissue. Compare the painting with the real image, and the lesion is what's
left over. This is the idea behind reconstruction-based unsupervised anomaly
detection in brain MRI.
"""
import numpy as np
import torch

from .mae import N_PATCHES, PIXEL_MEAN, PIXEL_STD, GRID, patchify, unpatchify, masking_from_hidden


def balanced_masks(B, rounds=2, seed=0):
    """
    Masks where every patch is hidden equally often. Each round shuffles the 64
    patches into 4 groups of 16; each mask shows one group and hides the rest
    (75% hidden, same as training). So each patch is hidden 3 times per round.
    """
    g = np.random.default_rng(seed)
    masks = []
    for _ in range(rounds):
        perm = g.permutation(N_PATCHES)
        for j in range(4):
            hidden = np.ones(N_PATCHES, bool)
            hidden[perm[16 * j:16 * (j + 1)]] = False
            masks.append(np.tile(hidden, (B, 1)))
    return masks


@torch.no_grad()
def reconstruct_every_patch(model, img, rounds=2, seed=0):
    """Predict every patch from its surroundings (averaged over masks where it was hidden)."""
    B = img.shape[0]
    acc = torch.zeros(B, N_PATCHES, 144)
    cnt = torch.zeros(B, N_PATCHES)
    for hidden in balanced_masks(B, rounds, seed):
        pred, mask = model(img, masking=masking_from_hidden(hidden))
        acc += pred * mask[..., None]
        cnt += mask
    pred = acc / cnt[..., None]
    return unpatchify(pred * PIXEL_STD + PIXEL_MEAN)          # (B, 96, 96), pixel units


@torch.no_grad()
def anomaly_map(model, img, rounds=2):
    """
    Per-patch anomaly: how much brighter the real patch is than the model expected.
    Only 'brighter than expected' counts, because these lesions are bright.
    Returns (map (B, 8, 8), reconstruction (B, 96, 96)).
    """
    rec = reconstruct_every_patch(model, img, rounds)
    excess = (img - rec).clamp(min=0) / PIXEL_STD
    amap = patchify(excess ** 2).mean(-1).reshape(-1, GRID, GRID)
    return amap, rec


def anomaly_score(amap):
    """One number per image: the most surprising patch."""
    return amap.flatten(1).max(1).values


def auroc(scores, labels):
    """Area under the ROC curve: chance that a random lesion scan scores higher than a random normal one."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


@torch.no_grad()
def pixel_anomaly_map(model, img, rounds=4, sigma=1.5):
    """
    Same idea at pixel level, for display: squared 'brighter than expected' error,
    lightly smoothed so a lesion shows as a blob. Returns (B, 96, 96).
    """
    rec = reconstruct_every_patch(model, img, rounds)
    excess = ((img - rec).clamp(min=0) / PIXEL_STD) ** 2
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = torch.outer(k, k)
    k = (k / k.sum())[None, None]
    return torch.nn.functional.conv2d(excess[:, None], k, padding=r)[:, 0]
