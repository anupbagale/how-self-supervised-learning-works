"""
A tiny Masked Autoencoder (MAE, He et al. 2021) for 96 x 96 MRI-like slices.

The idea in one line: hide most of the image and train the network to paint it
back in. No labels are needed, because the image is its own answer key.

  image (96 x 96) -> 64 patches of 12 x 12
  -> randomly hide 75% (48 patches), keep 16
  -> ENCODER (ViT) sees only the 16 visible patches
  -> DECODER gets the encoded patches + 48 "mask tokens" and predicts every pixel
  -> loss = mean squared error on the hidden patches only

The same encoder is later reused, with a small classification head, to
detect lesions from very few labelled scans.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH, GRID = 12, 8
N_PATCHES = GRID * GRID
DIM, HEADS, ENC_DEPTH = 64, 4, 2
DEC_DIM, DEC_DEPTH = 64, 1
MASK_RATIO = 0.75
PIXEL_MEAN, PIXEL_STD = 0.25, 0.26


# ---- patches ----------------------------------------------------------------------
def patchify(img):
    """(B, 96, 96) -> (B, 64, 144), patches in reading order."""
    B = img.shape[0]
    return img.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, N_PATCHES, PATCH * PATCH)


def unpatchify(p):
    """(B, 64, 144) -> (B, 96, 96)."""
    B = p.shape[0]
    return p.reshape(B, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(B, GRID * PATCH, GRID * PATCH)


def sincos_pos(dim, grid=GRID):
    """Fixed 2-D sine/cosine position codes (not learned), shape (grid*grid, dim)."""
    def one_axis(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2) / (d / 2))
        out = pos.reshape(-1)[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    gy, gx = np.meshgrid(np.arange(grid, dtype=float), np.arange(grid, dtype=float), indexing="ij")
    emb = np.concatenate([one_axis(dim // 2, gy), one_axis(dim // 2, gx)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)


def random_masking(B, ratio, generator=None):
    """
    For each image, pick which patches stay visible.
    Returns ids_keep (B, n_keep), mask (B, 64) with 1 = hidden, ids_restore (B, 64).
    """
    n_keep = int(round(N_PATCHES * (1 - ratio)))
    noise = torch.rand(B, N_PATCHES, generator=generator)
    ids_shuffle = noise.argsort(dim=1)            # lowest noise = kept
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    mask = torch.ones(B, N_PATCHES)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)     # back to reading order
    return ids_keep, mask, ids_restore


# ---- transformer block (same as the ViT repo) -------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, 2 * dim), nn.Linear(2 * dim, dim)
        self.attn = None

    def forward(self, x):
        B, T, D = x.shape
        d = D // self.heads
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (m.view(B, T, self.heads, d).transpose(1, 2) for m in (q, k, v))
        attn = (q @ k.transpose(-2, -1) / math.sqrt(d)).softmax(-1)
        self.attn = attn.detach()
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


# ---- encoder: a plain ViT that can skip hidden patches ------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)
        self.register_buffer("pos", sincos_pos(DIM))
        self.blocks = nn.ModuleList([Block(DIM, HEADS) for _ in range(ENC_DEPTH)])
        self.norm = nn.LayerNorm(DIM)

    def forward(self, img, ids_keep=None):
        x = patchify((img - PIXEL_MEAN) / PIXEL_STD)
        x = self.embed(x) + self.pos                               # (B, 64, D)
        if ids_keep is not None:                                    # drop the hidden patches
            x = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, DIM))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---- MAE = encoder + light decoder -----------------------------------------------------
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.dec_embed = nn.Linear(DIM, DEC_DIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_DIM))
        self.register_buffer("dec_pos", sincos_pos(DEC_DIM))
        self.dec_blocks = nn.ModuleList([Block(DEC_DIM, HEADS) for _ in range(DEC_DEPTH)])
        self.dec_norm = nn.LayerNorm(DEC_DIM)
        self.dec_pred = nn.Linear(DEC_DIM, PATCH * PATCH)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, img, mask_ratio=MASK_RATIO, generator=None, masking=None):
        B = img.shape[0]
        ids_keep, mask, ids_restore = masking if masking is not None else random_masking(B, mask_ratio, generator)
        latent = self.encoder(img, ids_keep)                       # (B, n_keep, D)
        x = self.dec_embed(latent)
        n_hidden = N_PATCHES - x.shape[1]
        x = torch.cat([x, self.mask_token.expand(B, n_hidden, -1)], dim=1)
        x = torch.gather(x, 1, ids_restore[..., None].expand(-1, -1, DEC_DIM))   # back to reading order
        x = x + self.dec_pos
        for blk in self.dec_blocks:
            x = blk(x)
        pred = self.dec_pred(self.dec_norm(x))                     # (B, 64, 144), normalised pixels
        return pred, mask

    def loss(self, img, pred, mask):
        target = patchify((img - PIXEL_MEAN) / PIXEL_STD)
        per_patch = ((pred - target) ** 2).mean(-1)                # (B, 64)
        return (per_patch * mask).sum() / mask.sum()               # hidden patches only

    @torch.no_grad()
    def reconstruct(self, img, masking):
        """Return the reconstructed image in pixel units (B, 96, 96)."""
        pred, mask = self.forward(img, masking=masking)
        return unpatchify(pred * PIXEL_STD + PIXEL_MEAN), mask


def masking_from_hidden(hidden):
    """Build (ids_keep, mask, ids_restore) from a boolean (B, 64) array of hidden patches."""
    hidden = torch.as_tensor(hidden, dtype=torch.bool)
    B = hidden.shape[0]
    order = torch.argsort(hidden.float() + torch.arange(N_PATCHES) * 1e-4, dim=1)   # visible first
    n_keep = int((~hidden[0]).sum())
    return order[:, :n_keep], hidden.float(), order.argsort(dim=1)


# ---- classifier on top of the encoder ------------------------------------------------------
class Classifier(nn.Module):
    """Encoder + average of patch tokens + linear layer. Used both from scratch and from MAE."""
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.head = nn.Linear(DIM, 2)

    def features(self, img):
        return self.encoder(img).mean(1)

    def forward(self, img):
        return self.head(self.features(img))
