"""Per-phase lambda: fit and apply a lambda value for every piece count.

The training target is pt = lambda * pf + (1 - lambda) * t, where pf is the
search score and t the game result (both as expected scores in [0, 1]).
Viewing both as noisy estimates of the position's true value V,

    pf = V + e_p,    t = V + e_t,

and assuming the two errors are uncorrelated, the expected squared error of
the target is lambda^2 var_p + (1 - lambda)^2 var_t. It is minimised by

    lambda* = var_t / (var_p + var_t)          (inverse-variance weighting).

V is unknown, so var_p and var_t can't be measured directly. Triple
collocation recovers them from a *third* estimate q (a trained net's
prediction) whose error is assumed uncorrelated with the other two:

    var_p = E[(p - t)(p - q)]
    var_t = E[(t - p)(t - q)]
    var_q = E[(q - p)(q - t)]

Caveat (documented in docs/loss_experiments.md): a net trained at lambda = 1
has errors positively correlated with the search's errors. That makes var_p
look smaller than it is, so the fitted lambda leans toward search, i.e. it
errs on the conservative side relative to the lambda = 1 baseline.

This module is pure tensor math, shared by scripts/fit_phase_lambda.py, the
trainer (--lambda-phase-table) and the tests.
"""

import json

import torch
from torch import Tensor

NUM_PIECE_COUNTS = 33  # index = number of pieces on the board, 0..32


class PhaseStats:
    """Running per-piece-count sums for triple collocation."""

    def __init__(self):
        z = lambda: torch.zeros(NUM_PIECE_COUNTS, dtype=torch.float64)
        self.n = z()
        self.s_p = z()  # sum of (p - t)(p - q)
        self.s_t = z()  # sum of (t - p)(t - q)
        self.s_q = z()  # sum of (q - p)(q - t)
        self.s_pt2 = z()  # sum of (p - t)^2, diagnostic

    @torch.no_grad()
    def update(self, p: Tensor, t: Tensor, q: Tensor, piece_count: Tensor) -> None:
        p, t, q = (x.reshape(-1).double().cpu() for x in (p, t, q))
        pc = piece_count.reshape(-1).long().cpu().clamp(0, NUM_PIECE_COUNTS - 1)
        self.n += torch.bincount(pc, minlength=NUM_PIECE_COUNTS).double()
        for acc, vals in (
            (self.s_p, (p - t) * (p - q)),
            (self.s_t, (t - p) * (t - q)),
            (self.s_q, (q - p) * (q - t)),
            (self.s_pt2, (p - t) ** 2),
        ):
            acc += torch.bincount(pc, weights=vals, minlength=NUM_PIECE_COUNTS)


def fit_lambda_table(
    stats: PhaseStats,
    window: int = 2,
    min_count: int = 20000,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
) -> dict:
    """Turn accumulated sums into a 33-entry lambda table.

    Variances are pooled over a +-``window`` piece-count neighbourhood
    (sample-weighted) before the ratio is taken, which smooths the curve and
    stabilises sparse piece counts. Piece counts whose pooled sample count is
    below ``min_count`` take the value of the nearest well-populated one.
    """
    n = stats.n
    k = torch.ones(2 * window + 1, dtype=torch.float64)

    def pooled(x):
        return torch.nn.functional.conv1d(
            x.view(1, 1, -1), k.view(1, 1, -1), padding=window
        ).view(-1)

    pn = pooled(n)
    var_p = pooled(stats.s_p) / pn.clamp(min=1)
    var_t = pooled(stats.s_t) / pn.clamp(min=1)
    var_q = pooled(stats.s_q) / pn.clamp(min=1)

    # Sampling noise can make a variance estimate slightly negative.
    vp = var_p.clamp(min=1e-9)
    vt = var_t.clamp(min=1e-9)
    lam = (vt / (vp + vt)).clamp(lambda_min, lambda_max)

    ok = pn >= min_count
    if not ok.any():
        raise ValueError("no piece count has enough samples; lower min_count")
    good = ok.nonzero().view(-1)
    for i in range(NUM_PIECE_COUNTS):
        if not ok[i]:
            nearest = good[(good - i).abs().argmin()]
            lam[i] = lam[nearest]

    return {
        "lambda_by_piece_count": [round(float(v), 4) for v in lam],
        "samples": [int(v) for v in n],
        "pooled_samples": [int(v) for v in pn],
        "var_search": [float(v) for v in var_p],
        "var_outcome": [float(v) for v in var_t],
        "var_net": [float(v) for v in var_q],
        "well_populated": [bool(v) for v in ok],
        "params": {
            "window": window,
            "min_count": min_count,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
        },
    }


def load_lambda_table(path: str) -> Tensor:
    """Load ``lambda_by_piece_count`` from a JSON file as a (33,) tensor."""
    with open(path) as f:
        data = json.load(f)
    values = data["lambda_by_piece_count"] if isinstance(data, dict) else data
    if len(values) != NUM_PIECE_COUNTS:
        raise ValueError(
            f"{path}: expected {NUM_PIECE_COUNTS} lambda values (piece counts 0..32), got {len(values)}"
        )
    table = torch.tensor(values, dtype=torch.float32)
    if not bool(((table >= 0) & (table <= 1)).all()):
        raise ValueError(f"{path}: lambda values must be in [0, 1]")
    return table


def lookup_lambda(table: Tensor, piece_count: Tensor, like: Tensor) -> Tensor:
    """Per-sample lambda from the table, shaped like ``like`` (B, 1)."""
    idx = piece_count.reshape(-1).long().clamp(0, NUM_PIECE_COUNTS - 1)
    return table.to(like.device)[idx].to(like.dtype).reshape(like.shape)
