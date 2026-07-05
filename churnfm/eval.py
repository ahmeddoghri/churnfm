"""Does drift-triggered retraining actually pay for itself?

We generate a churn stream with a concept drift at the midpoint (a pricing
change makes price-sensitivity the dominant churn driver — the model trained
on the old relationship silently degrades). Two policies are compared:

  * static    — train once on the first window, never retrain
  * adaptive  — ChurnFM: monitor PSI on prediction distributions, retrain on
                accumulated labeled history whenever drift crosses threshold

We report precision on batches *before* and *after* the drift to show the
static model rotting while the adaptive one recovers.

    python -m churnfm.eval
"""
from __future__ import annotations

import argparse

from .data import generate
from .model import ChurnModel
from .monitor import ChurnMonitor, _precision_recall


def _batches(rows: list, batch_size: int) -> list[list]:
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


def run(n: int = 3000, batch_size: int = 200, drift_at: float = 0.5, seed: int = 0):
    rows = generate(n=n, seed=seed, drift_at=drift_at)
    ref_window = rows[: batch_size * 2]        # first two batches = reference/training
    stream = rows[batch_size * 2:]
    batches = _batches(stream, batch_size)
    drift_batch_index = int(len(batches) * drift_at) - 2  # roughly where drift kicks in

    # --- static: train once, never retrain ---
    static_model = ChurnModel().fit(ref_window)
    static_results = []
    for b in batches:
        probs = static_model.predict_proba_batch(b)
        prec, rec = _precision_recall(b, probs)
        static_results.append((prec, rec))

    # --- adaptive: ChurnFM monitor with drift-triggered retraining ---
    # Retrain on a *recent* sliding window, not all accumulated history — mixing
    # in stale pre-drift examples would keep re-triggering drift indefinitely
    # and the model would never stabilize after the underlying relationship
    # actually changes.
    window_size = batch_size * 2
    monitor = ChurnMonitor.fit(ref_window)
    recent = list(ref_window)
    adaptive_results = []
    retrain_batches = []
    for i, b in enumerate(batches):
        res = monitor.process_batch(i, b, recent[-window_size:])
        adaptive_results.append((res.precision, res.recall))
        if res.retrained:
            retrain_batches.append(i)
        recent.extend(b)

    def avg(results, lo, hi):
        vals = [p for p, _ in results[lo:hi]]
        return sum(vals) / len(vals) if vals else 0.0

    mid = len(batches) // 2
    return {
        "n_batches": len(batches),
        "static_pre_drift": avg(static_results, 0, mid),
        "static_post_drift": avg(static_results, mid, None),
        "adaptive_pre_drift": avg(adaptive_results, 0, mid),
        "adaptive_post_drift": avg(adaptive_results, mid, None),
        "retrain_batches": retrain_batches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=200)
    args = p.parse_args()
    res = run(n=args.n, batch_size=args.batch_size)

    print(f"batches: {res['n_batches']}  (drift injected at stream midpoint)\n")
    print(f"{'policy':<12}{'pre-drift precision':>22}{'post-drift precision':>24}")
    print(f"{'static':<12}{res['static_pre_drift']:>21.0%}{res['static_post_drift']:>23.0%}")
    print(f"{'adaptive':<12}{res['adaptive_pre_drift']:>21.0%}{res['adaptive_post_drift']:>23.0%}")
    print(f"\nadaptive retrained at batches: {res['retrain_batches']}")


if __name__ == "__main__":
    main()
