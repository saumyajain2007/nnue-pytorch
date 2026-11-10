"""Tests for the fitted per-piece-count lambda (model/phase_lambda.py)."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import LambdaConfig
from model.phase_lambda import (
    NUM_PIECE_COUNTS,
    PhaseStats,
    fit_lambda_table,
    load_lambda_table,
    lookup_lambda,
)


def _synthetic(n, sp, st, sq, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(n, generator=g)
    p = v + sp * torch.randn(n, generator=g)
    t = v + st * torch.randn(n, generator=g)
    q = v + sq * torch.randn(n, generator=g)
    return p, t, q


def test_triple_collocation_recovers_error_variances():
    sp, st, sq = 0.05, 0.30, 0.08
    stats = PhaseStats()
    p, t, q = _synthetic(400_000, sp, st, sq)
    stats.update(p, t, q, torch.full((p.numel(),), 20))
    out = fit_lambda_table(stats, window=0, min_count=1000)
    assert out["var_search"][20] == pytest.approx(sp**2, rel=0.05)
    assert out["var_outcome"][20] == pytest.approx(st**2, rel=0.02)
    assert out["var_net"][20] == pytest.approx(sq**2, rel=0.05)
    expected = st**2 / (sp**2 + st**2)
    assert out["lambda_by_piece_count"][20] == pytest.approx(expected, abs=0.005)


def test_lambda_follows_relative_noise_across_phases():
    """Noisier outcomes in the opening -> higher lambda there."""
    stats = PhaseStats()
    for pc, st in ((30, 0.40), (6, 0.10)):
        p, t, q = _synthetic(200_000, 0.08, st, 0.06, seed=pc)
        stats.update(p, t, q, torch.full((p.numel(),), pc))
    out = fit_lambda_table(stats, window=0, min_count=1000)
    lam = out["lambda_by_piece_count"]
    assert lam[30] > lam[6]
    assert len(lam) == NUM_PIECE_COUNTS
    # Unpopulated piece counts are filled from the nearest populated one.
    assert lam[32] == lam[30] and lam[2] == lam[6]
    assert out["well_populated"][30] and not out["well_populated"][31]


def test_fit_requires_some_data():
    with pytest.raises(ValueError):
        fit_lambda_table(PhaseStats(), min_count=1)


def test_load_and_lookup(tmp_path):
    values = [i / 32 for i in range(NUM_PIECE_COUNTS)]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"lambda_by_piece_count": values}))
    table = load_lambda_table(str(path))
    like = torch.zeros(4, 1)
    got = lookup_lambda(table, torch.tensor([32, 16, 2, 40]), like)
    assert got.shape == (4, 1)
    torch.testing.assert_close(got.flatten(), torch.tensor([1.0, 0.5, 2 / 32, 1.0]))


@pytest.mark.parametrize("bad", [[0.5] * 10, [1.5] * NUM_PIECE_COUNTS])
def test_load_rejects_bad_tables(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_lambda_table(str(path))


def test_table_and_delta_are_exclusive():
    with pytest.raises(ValueError):
        LambdaConfig(lambda_phase_table="x.json", lambda_phase_delta=-0.2)
