# Loss experiments

This branch adds three switches to the training objective and its evaluation.
Every switch is **off by default**: with default flags, the loss is
bit-for-bit identical to the base commit (`aa2fff2`, the SF19-era trainer), and so is the compiled loss
graph (tested in `tests/test_loss_experiments.py`).

| # | What | Flag | Default | Kind |
|---|------|------|---------|------|
| 1 | Auxiliary outcome term (from `b882e6c`, now configurable) | `--outcome-aux-weight W` | `0.0` | objective |
| 2 | Phase-dependent lambda | `--lambda-phase-delta D` | `0.0` | objective |
| 3 | Validation metrics that don't depend on the objective | (always on) | — | measurement |

Changes 1 and 2 change what the net is trained to predict. Change 3 adds no
training behaviour; it makes runs with different objectives comparable.

None of these has been validated with games yet. Treat them as hypotheses to
test, not as improvements.

---

## Background: the existing loss

For each position, the trainer turns three numbers into expected game results
in [0, 1] using the two-sigmoid win-rate model (`expected_score`):

* `qf`: the net's prediction (scaled by `in_offset` / `in_scaling`)
* `pf`: the search score from the data (scaled by `out_offset` / `out_scaling`)
* `t`: the actual game result (1 / 0.5 / 0)

The target is the blend `pt = lambda * pf + (1 - lambda) * t`. The loss is
`|pt - qf| ^ pow_exp` (2.5 by default), optionally made asymmetric with
`qp_asymmetry`, then averaged with confidence weights that depend on `pf`.

At the default `--lambda 1.0`, game results are ignored entirely.

---

## 1. Auxiliary outcome term: `--outcome-aux-weight`

### What it does

```
loss_i = |pt - qf| ^ pow_exp  [* asymmetry]  +  W * (qf - t)^2
```

This is added to each sample before the confidence weighting. The original
commit `b882e6c` hard-coded `W = 0.1`, and `--outcome-aux-weight 0.1`
reproduces that commit exactly (tested).

### Review of the original change

**Kept**, with two changes:

* **The weight is now a CLI flag** (`LossParams.outcome_aux_weight`). A
  hard-coded constant can't be swept or turned off for an A/B run, and every
  value change was a code edit.
* **The default is 0.0, not 0.1.** The term is experimental, and master
  behaviour should be what you get when you don't ask for anything. The
  disabled path is a Python-level `if`, so `torch.compile` traces exactly the
  old graph.

### How it relates to lambda

This matters most for interpreting results. **At `pow_exp = 2` the term is
exactly a reparameterisation of lambda**:

```
(qf - pf)^2 + W (qf - t)^2  =  (1 + W) * (qf - m)^2 + const,
    where  m = (pf + W t) / (1 + W)
```

`m` is the lambda blend with `lambda = 1 / (1 + W)`. The constant depends
only on the data, so the gradients are identical up to a factor of `(1 + W)`,
which the optimizer mostly absorbs. `W = 0.1` at `--lambda 1.0` then behaves
like `--lambda 0.909` (`test_outcome_aux_is_a_lambda_reparameterisation_at_pow2`).

At the default `pow_exp = 2.5` the two terms have different exponents, and
the result behaves like a **per-sample lambda that depends on how much search
and result disagree** (`d = |pf - t|`). Balancing the two gradients
(`2.5 x^1.5 = 2W (d - x)`, with `x` the distance `qf` moves from `pf` toward
`t`) gives, for `W = 0.1`:

| search/result disagreement `d` | effective lambda |
|---|---|
| 1.0 (search says win, game lost) | ≈ 0.84 |
| 0.1 | ≈ 0.69 |
| 0.01 | ≈ 0.48 |

So it pulls hardest toward the result, in relative terms, where the result
and the search already nearly agree. In absolute terms those moves are tiny.
Where they disagree strongly, it acts like a mild `lambda ≈ 0.84`. Whether
this shape beats a plain constant lambda is exactly what the benchmark should
answer.

**Recommended control:** include a plain `--lambda 0.9` run alongside
`--outcome-aux-weight 0.1`. If the aux run doesn't beat the `lambda 0.9` run,
the term adds nothing that lambda doesn't already give, and can be dropped.

Interactions:

* `qp_asymmetry` applies to the search term only, not the aux term, as in the
  original commit.
* With `lambda < 1`, outcomes enter twice: once through `pt` and once through
  the aux term. That is allowed, but keep it in mind when you choose values.

---

## 2. Phase-dependent lambda: `--lambda-phase-delta`

### What it does

After the lambda scheduler (including any jitter), each sample's lambda is
shifted according to how much material is left:

```
phase  = clamp((32 - piece_count) / 30, 0, 1)   # 0 = all 32 pieces, 1 = bare kings
lambda = clamp(lambda + D * phase, 0, 1)
```

With `--lambda 1.0 --lambda-phase-delta -0.3`:

| pieces on board | lambda |
|---|---|
| 32 | 1.00 |
| 24 | 0.92 |
| 16 | 0.84 |
| 8 | 0.76 |
| 2 | 0.70 |

### Why

A game result is a noisy label for a position, because many moves are still
to be played. That noise is largest in the opening, where the result depends
on dozens of future decisions. It is smallest in simplified endgames, where
the result is close to determined by the position. Search scores have the
opposite problem: they are reliable in the middlegame but systematically
miscalibrated in some endgames (fortresses, drawish material imbalances,
long conversions).

A single global lambda can't reflect both effects. Making lambda depend on
game phase uses outcomes where they carry real information and search scores
where they do. Unlike change 1, this is not expressible with any existing
flag.

### Details

* The shift applies in both training and validation, because it is part of
  the objective. (Lambda jitter, by contrast, is a regulariser.)
* `piece_count` comes from the data loader as a flat `(B,)` tensor and is
  reshaped to the `(B, 1)` score shape explicitly, so it can never broadcast
  into a `(B, B)` matrix (tested).
* `D` must be in `[-1, 1]`. Negative values mean "more outcome in endgames",
  which is the intended use.
* **Porting note:** upstream `master` (#570, SFNNv20) replaced `piece_count`
  and `queen_bucket` in the batch with precomputed `layer_stack_indices`.
  Porting this change there needs either `piece_count` back in the batch or
  a coarse phase derived from the layer-stack index.

---

## 3. Validation metrics that don't depend on the objective

### Problem

`val_loss_epoch` is computed with the run's own objective. A run with
`--outcome-aux-weight 0.1` or `--lambda 0.8` reports a loss on a different
scale, measuring a different thing, so comparing its `val_loss_epoch` with
the baseline's means nothing.

### What's added

Every validation epoch now also logs:

* **`val_ref_loss_epoch`**: the master SF loss at `lambda = 1`, with no aux
  term and no phase shift. It measures how well the net fits the search
  scores. It uses the run's `pow_exp`, `qp_asymmetry`, `w1`, `w2` and scaling
  constants, so compare it only between runs that share those.
* **`val_outcome_mse_epoch`**: the unweighted mean of `(qf - t)^2`, the Brier
  score of the net's prediction against the actual game results. It depends
  only on `in_offset` / `in_scaling`.

Both are computed identically regardless of `--lambda*` or
`--outcome-aux-weight` (tested). A change that helps should usually not make
`val_ref_loss` much worse, and outcome-based changes should improve
`val_outcome_mse`. **Neither replaces games:** the SPRT match is the deciding
test.

Implementation: `NNUE.val_step` returns the extra `val_*` keys. The training
engine now averages every `val_*` output over the validation epoch, reduces
it across DDP ranks, and logs it as `<key>_epoch` next to `val_loss_epoch`.

---

## Suggested benchmark

Use identical data, seed, epochs, features and network size for every run,
and change only the objective. Use a held-out validation file
(`--validation-datasets`), so the validation metrics aren't measured on
training data.

```bash
COMMON="train.binpack --validation-datasets val.binpack --seed 42 \
  --max-epochs 400 --batch-size 16384 --gpus 0 \
  --features Full_Threats+PP_3Wide+HalfKAv2_hm^"

python train.py $COMMON --default-root-dir runs/A_baseline
python train.py $COMMON --default-root-dir runs/B_aux0.1    --outcome-aux-weight 0.1
python train.py $COMMON --default-root-dir runs/C_lambda0.9 --lambda 0.9
python train.py $COMMON --default-root-dir runs/D_phase-0.3 --lambda-phase-delta -0.3
```

1. Compare `val_ref_loss_epoch` and `val_outcome_mse_epoch` across runs.
2. Export each net with `serialize.py`.
3. Run SPRT matches of each candidate against A: same engine binary, fixed
   nodes, `Threads=1`, balanced openings, e.g. `elo0=0 elo1=5 alpha=0.05
   beta=0.05`.

B vs C answers whether change 1 is anything more than a lambda shift.

## Tests

`tests/test_loss_experiments.py` checks:

* Default flags reproduce the master loss bit-for-bit, at lambda 1 and 0.7.
* `--outcome-aux-weight 0.1` reproduces commit `b882e6c` exactly.
* At `pow_exp = 2`, the aux term gives the same gradient as a lambda shift.
* Phase-lambda endpoints, clamping, and the `(B, 1)` shape.
* Flag validation for both new flags.
* The reference metrics ignore the objective flags, and `ref_loss` equals
  the master loss at lambda 1.
* The compiled loss with the aux term matches the eager loss.
