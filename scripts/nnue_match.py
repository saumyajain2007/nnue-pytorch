"""Minimal paired-opening SPRT match between two NNUE nets on one Stockfish binary.

Written because no match tool (cutechess-cli / fastchess / c-chess-cli) is
installed in the container. Uses only python-chess's UCI driver.

* Each opening is played twice with colours swapped (a game pair), so results
  are scored pentanomially (pair score 0, 0.25, 0.5, 0.75, 1 for net A).
* SPRT: GSPRT with the logistic Elo model, normal approximation on the
  pentanomial pair scores (the same approximation cutechess / fastchess use
  for their "logistic" model). LLR = N (s1-s0)(2m - s0 - s1) / (2 var).
* Adjudication: win if |eval| >= RESIGN_CP for 8 consecutive plies with the
  same sign; draw after ply 68 if |eval| <= DRAW_CP for 8 consecutive plies;
  draw at MAX_PLIES. Normal rules (mate, stalemate, 3-fold, 50-move,
  insufficient material) are applied by python-chess.

Usage:
  nnue_match.py --engine SF --net-a A.nnue --net-b B.nnue --book BOOK.epd \
      --nodes 20000 --concurrency 16 --max-pairs 20000 --elo0 0 --elo1 5 \
      --log results.jsonl
"""

import argparse
import contextlib
import json
import math
import queue
import sys
import threading
import time

import chess
import chess.engine

RESIGN_CP = 1000
RESIGN_PLIES = 8
DRAW_CP = 10
DRAW_PLIES = 8
DRAW_MIN_PLY = 68
MAX_PLIES = 500
MATE_CP = 100000


def logistic_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(s):
    s = min(max(s, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / s - 1.0)


def sprt_stats(penta, elo0, elo1):
    """penta[i] = number of pairs where net A scored i/4 of the pair points."""
    n = sum(penta)
    if n == 0:
        return 0.0, float("nan"), float("nan")
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    mean = sum(c * s for c, s in zip(penta, scores)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(penta, scores)) / n
    if var <= 0:
        return 0.0, score_to_elo(mean), float("nan")
    s0, s1 = logistic_score(elo0), logistic_score(elo1)
    llr = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)
    half = 1.96 * math.sqrt(var / n)
    elo = score_to_elo(mean)
    err = (score_to_elo(min(mean + half, 1 - 1e-9)) - score_to_elo(max(mean - half, 1e-9))) / 2
    return llr, elo, err


def cp_from_info(info, color):
    sc = info.get("score")
    if sc is None:
        return None
    return sc.pov(color).score(mate_score=MATE_CP)


def play_game(engines, fen, white_key, nodes, game_id):
    """engines: {'A': engine, 'B': engine}. Returns (result str, plies, reason)."""
    board = chess.Board(fen)
    keys = {chess.WHITE: white_key, chess.BLACK: "B" if white_key == "A" else "A"}
    white_pov = []  # eval from White's point of view after each ply
    limit = chess.engine.Limit(nodes=nodes)
    plies = 0
    while True:
        if board.is_game_over(claim_draw=True):
            return board.result(claim_draw=True), plies, "rules"
        if plies >= MAX_PLIES:
            return "1/2-1/2", plies, "max_plies"
        mover = board.turn
        res = engines[keys[mover]].play(board, limit, game=game_id, info=chess.engine.INFO_SCORE)
        if res.move is None:
            raise RuntimeError(f"engine returned no move in {board.fen()}")
        cp = cp_from_info(res.info, chess.WHITE)
        board.push(res.move)
        plies += 1
        white_pov.append(cp)
        last = white_pov[-RESIGN_PLIES:]
        if len(last) == RESIGN_PLIES and all(v is not None for v in last):
            if all(v >= RESIGN_CP for v in last):
                return "1-0", plies, "adj_win"
            if all(v <= -RESIGN_CP for v in last):
                return "0-1", plies, "adj_win"
        last = white_pov[-DRAW_PLIES:]
        if (
            plies >= DRAW_MIN_PLY
            and len(last) == DRAW_PLIES
            and all(v is not None and abs(v) <= DRAW_CP for v in last)
        ):
            return "1/2-1/2", plies, "adj_draw"


def points_for_a(result, white_key):
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if (white_won == (white_key == "A")) else 0.0


ENGINE_ERRORS = (chess.engine.EngineError, chess.engine.EngineTerminatedError,
                 OSError, RuntimeError, TimeoutError)


def close_engines(engines):
    for eng in engines.values():
        with contextlib.suppress(*ENGINE_ERRORS):
            eng.quit()


def open_engine(path, net, hash_mb):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"EvalFile": net, "Threads": 1, "Hash": hash_mb})
    return eng


def worker(args, jobs, results, stop):
    engines = {"A": open_engine(args.engine, args.net_a, args.hash),
               "B": open_engine(args.engine, args.net_b, args.hash)}
    try:
        while not stop.is_set():
            try:
                idx, fen = jobs.get_nowait()
            except queue.Empty:
                return
            pair = []
            try:
                for white_key in ("A", "B"):
                    gid = object()  # new key -> python-chess sends ucinewgame
                    r, plies, why = play_game(engines, fen, white_key, args.nodes, gid)
                    pair.append({"opening": idx, "white": white_key, "result": r,
                                 "plies": plies, "end": why,
                                 "a_points": points_for_a(r, white_key)})
            except ENGINE_ERRORS as e:  # engine crash etc.: drop the pair, restart engines
                results.put(("error", idx, repr(e)))
                close_engines(engines)
                engines = {"A": open_engine(args.engine, args.net_a, args.hash),
                           "B": open_engine(args.engine, args.net_b, args.hash)}
                continue
            results.put(("pair", idx, pair))
    finally:
        close_engines(engines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net-a", required=True, help="candidate (Elo is reported for A vs B)")
    ap.add_argument("--net-b", required=True, help="baseline")
    ap.add_argument("--book", required=True)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--log", required=True, help="JSON-lines file with every game")
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    with open(args.book) as f:
        fens = [line.strip() for line in f if line.strip()]
    jobs = queue.Queue()
    for i in range(min(args.max_pairs, len(fens))):
        jobs.put((i, fens[i]))

    results = queue.Queue()
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(args, jobs, results, stop), daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()

    penta = [0] * 5
    wdl = {"W": 0, "D": 0, "L": 0}
    ends = {}
    errors = 0
    verdict = "max_pairs reached (inconclusive)"
    t0 = time.time()
    llr = elo = err = float("nan")
    with open(args.log, "a") as logf:
        while any(t.is_alive() for t in threads) or not results.empty():
            try:
                kind, idx, payload = results.get(timeout=1.0)
            except queue.Empty:
                continue
            if kind == "error":
                errors += 1
                print(f"[error] opening {idx}: {payload}", file=sys.stderr, flush=True)
                continue
            pts = 0.0
            for g in payload:
                logf.write(json.dumps(g) + "\n")
                pts += g["a_points"]
                wdl["W" if g["a_points"] == 1 else "D" if g["a_points"] == 0.5 else "L"] += 1
                ends[g["end"]] = ends.get(g["end"], 0) + 1
            penta[round(pts * 2)] += 1
            n = sum(penta)
            llr, elo, err = sprt_stats(penta, args.elo0, args.elo1)
            if n % args.report_every == 0:
                rate = 2 * n / (time.time() - t0)
                print(f"pairs={n} games={2*n} W/D/L={wdl['W']}/{wdl['D']}/{wdl['L']} "
                      f"penta={penta} elo={elo:+.2f}±{err:.2f} LLR={llr:.2f} "
                      f"[{lower:.2f},{upper:.2f}] {rate:.1f} games/s", flush=True)
            if llr >= upper:
                verdict = "H1 accepted (A stronger by >= elo1 is favoured)"
                stop.set()
                break
            if llr <= lower:
                verdict = "H0 accepted (A not stronger by elo1)"
                stop.set()
                break
    stop.set()
    for t in threads:
        t.join(timeout=60)

    n = sum(penta)
    summary = {
        "verdict": verdict, "pairs": n, "games": 2 * n, "wdl_for_A": wdl, "pentanomial": penta,
        "elo_A_minus_B": round(elo, 2), "elo_95ci_halfwidth": round(err, 2),
        "llr": round(llr, 3), "bounds": [round(lower, 3), round(upper, 3)],
        "sprt": {"elo0": args.elo0, "elo1": args.elo1, "alpha": args.alpha, "beta": args.beta,
                 "model": "logistic, pentanomial normal approximation"},
        "nodes": args.nodes, "hash": args.hash, "threads": 1, "errors_dropped_pairs": errors,
        "endings": ends, "seconds": round(time.time() - t0, 1),
    }
    print("FINAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
