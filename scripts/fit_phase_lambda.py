"""Fit a per-piece-count lambda table (see model/phase_lambda.py) from data.

Streams positions through the trainer's own data loader, with the same
skipping/filtering as training, and evaluates them with a reference net. For
every piece count it then estimates how noisy the search score and the game
result are (triple collocation) and picks the inverse-variance lambda.

Example:
  python scripts/fit_phase_lambda.py --net A_baseline.nnue --data train.binpack \
      --positions 20000000 --out phase_lambda.json
Then train with:  --lambda-phase-table phase_lambda.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader
import model as M
from model.config import LossParams
from model.nnue import expected_score, remap_tablebase_score
from model.phase_lambda import PhaseStats, fit_lambda_table

DEFAULT_FEATURES = "Full_Threats+PP_3Wide+HalfKAv2_hm^"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="reference .nnue (e.g. the lambda=1 baseline)")
    ap.add_argument("--data", required=True, nargs="+", help=".binpack file(s)")
    ap.add_argument("--out", required=True, help="output JSON table")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--positions", type=int, default=20_000_000)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=20000)
    ap.add_argument("--lambda-min", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = M.NNUELightningConfig(features=args.features)
    with open(args.net, "rb") as f:
        net = M.NNUEReader(f, cfg.features, cfg.model_config).model
    net.to(args.device).eval()
    lp = LossParams()  # default scaling constants, same as the trainer

    # The C++ loader takes the input-feature name (no factorizer suffix).
    provider = data_loader.SparseBatchProvider(
        net.input.INPUT_FEATURE_NAME,
        args.data,
        args.batch_size,
        cyclic=True,
        num_workers=args.workers,
        config=data_loader.DataloaderSkipConfig(),
        device=args.device,
    )
    stats = PhaseStats()
    seen = 0
    with torch.no_grad():
        for us, them, wi, bi, outcome, score, piece_count in provider:
            scorenet = net(us, them, wi, bi, piece_count, False, False) * net.quantization.nnue2score
            score = remap_tablebase_score(score, lp.tb_remap_base, lp.tb_remap_scale, lp.tb_remap_decay)
            q = expected_score(scorenet, lp.in_offset, lp.in_scaling)
            p = expected_score(score, lp.out_offset, lp.out_scaling)
            stats.update(p, outcome, q, piece_count)
            seen += us.shape[0]
            if seen % (args.batch_size * 200) < args.batch_size:
                print(f"{seen} positions", flush=True)
            if seen >= args.positions:
                break
    del provider

    result = fit_lambda_table(stats, window=args.window, min_count=args.min_count, lambda_min=args.lambda_min)
    result["meta"] = {
        "net": os.path.abspath(args.net),
        "data": [os.path.abspath(d) for d in args.data],
        "positions": seen,
        "features": args.features,
        "method": "triple collocation (search, outcome, net), inverse-variance lambda",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"wrote {args.out} from {seen} positions")
    print("pieces  samples    var_search  var_outcome  var_net    lambda")
    for i in range(2, 33):
        print(
            f"{i:6d}  {result['samples'][i]:9d}  {result['var_search'][i]:.6f}    "
            f"{result['var_outcome'][i]:.6f}     {result['var_net'][i]:+.6f}  "
            f"{result['lambda_by_piece_count'][i]:.3f}{'' if result['well_populated'][i] else '  (filled)'}"
        )


if __name__ == "__main__":
    main()
