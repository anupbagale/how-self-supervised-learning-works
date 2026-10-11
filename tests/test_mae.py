"""Checks for the masked autoencoder's moving parts."""
import torch

from selfsup.mae import (MAE, Classifier, patchify, unpatchify, random_masking, masking_from_hidden,
                         PIXEL_MEAN, PIXEL_STD, N_PATCHES)

torch.manual_seed(0)


def test_patchify_round_trip():
    x = torch.rand(3, 96, 96)
    p = patchify(x)
    assert p.shape == (3, 64, 144)
    assert torch.equal(p[0, 9].reshape(12, 12), x[0, 12:24, 12:24])     # patch 9 = row 1, col 1
    assert torch.equal(unpatchify(p), x)


def test_random_masking_hides_75_percent():
    ids_keep, mask, ids_restore = random_masking(5, 0.75)
    assert ids_keep.shape == (5, 16)
    assert torch.all(mask.sum(1) == 48)
    for b in range(5):                         # the kept ids are exactly the visible patches
        assert set(ids_keep[b].tolist()) == set(torch.where(mask[b] == 0)[0].tolist())


def test_loss_ignores_visible_patches():
    m = MAE()
    x = torch.rand(2, 96, 96)
    _, mask, _ = random_masking(2, 0.75)
    target = patchify((x - PIXEL_MEAN) / PIXEL_STD)
    pred = target.clone()
    pred[mask == 0] += 100.0                   # garbage on visible patches only
    assert m.loss(x, pred, mask).item() < 1e-10


def test_masking_from_hidden_and_shapes():
    hidden = torch.zeros(2, N_PATCHES, dtype=torch.bool)
    hidden[:, ::4] = True
    ids_keep, mask, ids_restore = masking_from_hidden(hidden)
    assert ids_keep.shape == (2, 48) and torch.equal(mask.bool(), hidden)
    pred, m = MAE()(torch.rand(2, 96, 96), masking=(ids_keep, mask, ids_restore))
    assert pred.shape == (2, 64, 144)
    assert Classifier()(torch.rand(2, 96, 96)).shape == (2, 2)
