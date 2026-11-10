"""Tests for the experimental loss knobs documented in docs/loss_experiments.md."""

import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import LambdaConfig, LossParams
from model.nnue import (
    apply_phase_lambda,
    calculate_reference_metrics,
    calculate_sf_loss,
    expected_score,
)


def _inputs(batch=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    scorenet = (torch.randn(batch, 1, generator=g) * 300).clamp(-3000, 3000)
    score = (torch.randn(batch, 1, generator=g) * 200).clamp(-2000, 2000)
    outcome = torch.randint(0, 3, (batch, 1), generator=g).float() / 2
    return scorenet, score, outcome


def _master_loss(scorenet, score, outcome, p, lam):
    """Verbatim copy of the loss at b261450, as the reference implementation."""
    from model.nnue import remap_tablebase_score

    score = remap_tablebase_score(
        score, p.tb_remap_base, p.tb_remap_scale, p.tb_remap_decay
    )
    q = (scorenet - p.in_offset) / p.in_scaling
    qm = (-scorenet - p.in_offset) / p.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())
    s = (score - p.out_offset) / p.out_scaling
    sm = (-score - p.out_offset) / p.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())
    pt = pf * lam + outcome * (1.0 - lam)
    loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
    if p.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
    w = 1 + (2.0**p.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), p.w2)
    return (loss * w).sum() / w.sum()


@pytest.mark.parametrize("lam", [1.0, 0.7])
def test_defaults_reproduce_master_bit_exact(lam):
    p = LossParams(qp_asymmetry=0.2, w1=1.5)
    x = _inputs()
    lam_t = x[0].new_tensor(lam)
    assert torch.equal(calculate_sf_loss(*x, p, lam_t), _master_loss(*x, p, lam_t))


def test_outcome_aux_matches_original_commit():
    """weight=0.1 reproduces the hard-coded term from b882e6c."""
    p = LossParams(outcome_aux_weight=0.1)
    scorenet, score, outcome = _inputs()
    qf = expected_score(scorenet, p.in_offset, p.in_scaling)
    pf = expected_score(score, p.out_offset, p.out_scaling)
    search = (pf - qf).abs().pow(p.pow_exp)
    w = 1 + (2.0**p.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), p.w2)
    expected = ((search + 0.1 * (qf - outcome).square()) * w).sum() / w.sum()
    got = calculate_sf_loss(scorenet, score, outcome, p, scorenet.new_tensor(1.0))
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("w", [0.1, 0.5])
def test_outcome_aux_is_a_lambda_reparameterisation_at_pow2(w):
    """With pow_exp=2 and no asymmetry, adding w*(q-t)^2 at lambda=1 has the
    same gradient as (1+w) * the plain loss at lambda = 1/(1+w).

    (q-p)^2 + w(q-t)^2 = (1+w)(q - (p+wt)/(1+w))^2 + const(p, t).
    """
    scorenet, score, outcome = _inputs()
    aux = LossParams(pow_exp=2.0, outcome_aux_weight=w)
    plain = LossParams(pow_exp=2.0)

    a = scorenet.clone().requires_grad_()
    calculate_sf_loss(a, score, outcome, aux, a.new_tensor(1.0)).backward()

    b = scorenet.clone().requires_grad_()
    (
        (1 + w) * calculate_sf_loss(b, score, outcome, plain, b.new_tensor(1 / (1 + w)))
    ).backward()

    torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-10)


def test_phase_lambda_endpoints_and_clamp():
    like = torch.zeros(4, 1)
    pc = torch.tensor([32, 17, 2, 2])
    out = apply_phase_lambda(like.new_tensor(0.9), pc, -0.5, like)
    torch.testing.assert_close(
        out.flatten(), torch.tensor([0.9, 0.65, 0.4, 0.4]), rtol=0, atol=1e-6
    )
    # Clamped to [0, 1].
    out = apply_phase_lambda(like.new_tensor(0.2), pc, -0.5, like)
    assert out.min().item() == 0.0
    out = apply_phase_lambda(like.new_tensor(0.9), pc, 0.5, like)
    assert out.max().item() == 1.0


def test_phase_lambda_keeps_batch_shape():
    """piece_count is (B,); the result must stay (B, 1), never (B, B)."""
    like = torch.zeros(8, 1)
    out = apply_phase_lambda(like.new_tensor(1.0), torch.full((8,), 10), -0.3, like)
    assert out.shape == (8, 1)


def test_phase_delta_validation():
    with pytest.raises(ValueError):
        LambdaConfig(lambda_phase_delta=1.5)
    with pytest.raises(ValueError):
        LossParams(outcome_aux_weight=-0.1)


def test_reference_metrics_ignore_objective_knobs():
    x = _inputs()
    base = LossParams()
    changed = dataclasses.replace(
        base,
        outcome_aux_weight=0.3,
        lambda_config=LambdaConfig(lambda_=0.5, lambda_phase_delta=-0.4),
    )
    a = calculate_reference_metrics(*x, base)
    b = calculate_reference_metrics(*x, changed)
    for k in ("ref_loss", "outcome_mse"):
        assert torch.equal(a[k], b[k]), k
    # ref_loss is exactly the master loss at lambda = 1.
    assert torch.equal(a["ref_loss"], _master_loss(*x, base, x[0].new_tensor(1.0)))


def test_compiled_loss_with_aux_matches_eager():
    p = LossParams(outcome_aux_weight=0.1, qp_asymmetry=0.2)
    x = _inputs(1024)
    lam = torch.rand(1024, 1)
    eager = calculate_sf_loss(*x, p, lam)
    compiled = torch.compile(calculate_sf_loss)(*x, p, lam)
    torch.testing.assert_close(eager, compiled, rtol=1e-6, atol=1e-8)
