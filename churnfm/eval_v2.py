"""Does PSI-only monitoring actually catch drift, or just the drift that
happens to be easy?

The original benchmark reports precision recovering from a static model's 80%
to an adaptive one's 89% post-drift. Both numbers are real, and both are
measured on a scenario where the injected "drift" also multiplies the churn
base rate by 20x (3.2% to 68.4%), which makes the post-drift problem
*inherently* easier and is most of why even the static model scores 80%.

This module runs the same monitor-and-retrain mechanism against
:func:`churnfm.scenarios.pure_concept_drift`, where the base rate and feature
distributions are held constant and only the feature-outcome relationship
inverts. That isolates what PSI is actually supposed to detect.

    python -m churnfm.eval_v2
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

from .drift import assess
from .model import ChurnModel
from .monitor import ChurnMonitor, _precision_recall
from .monitor_v2 import ChurnMonitorV2
from .outcome_drift import assess_outcomes
from .scenarios import covariate_shift_confound, describe, pure_concept_drift


def _batches(rows: list, batch_size: int) -> List[list]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def run_static_vs_psi_only(rows, batch_size: int, ref_size: int) -> Dict:
    ref_window = rows[:ref_size]
    stream = rows[ref_size:]
    batches = _batches(stream, batch_size)

    static_model = ChurnModel().fit(ref_window)
    static_precisions = []
    for batch in batches:
        probs = static_model.predict_proba_batch(batch)
        precision, _ = _precision_recall(batch, probs)
        static_precisions.append(precision)

    window_size = batch_size * 2
    monitor = ChurnMonitor.fit(ref_window)
    recent = list(ref_window)
    psi_precisions = []
    retrain_batches = []
    for i, batch in enumerate(batches):
        result = monitor.process_batch(i, batch, recent[-window_size:])
        psi_precisions.append(result.precision)
        if result.retrained:
            retrain_batches.append(i)
        recent.extend(batch)

    mid = len(batches) // 2
    return {
        "n_batches": len(batches),
        "static_pre": _avg(static_precisions[:mid]),
        "static_post": _avg(static_precisions[mid:]),
        "psi_only_pre": _avg(psi_precisions[:mid]),
        "psi_only_post": _avg(psi_precisions[mid:]),
        "psi_retrains": retrain_batches,
    }


def run_static_vs_dual(rows, batch_size: int, ref_size: int) -> Dict:
    ref_window = rows[:ref_size]
    stream = rows[ref_size:]
    batches = _batches(stream, batch_size)

    static_model = ChurnModel().fit(ref_window)
    static_precisions = []
    for batch in batches:
        probs = static_model.predict_proba_batch(batch)
        precision, _ = _precision_recall(batch, probs)
        static_precisions.append(precision)

    # Retrain on just the batch that triggered the alarm, not a wider sliding
    # window. Under real concept drift the wider window mixes labeled examples
    # from the old and new regimes, which teach opposite relationships; a
    # model fit on that mixture came back at 9-12% precision, no better than
    # not retraining at all. Fit on the single freshest, unambiguously
    # post-drift batch instead, and precision recovered to 38-51%.
    monitor = ChurnMonitorV2.fit(ref_window)
    dual_precisions = []
    triggers = []
    for i, batch in enumerate(batches):
        result = monitor.process_batch(i, batch, batch)
        dual_precisions.append(result.precision)
        if result.retrained:
            triggers.append((i, result.trigger))

    mid = len(batches) // 2
    return {
        "n_batches": len(batches),
        "static_pre": _avg(static_precisions[:mid]),
        "static_post": _avg(static_precisions[mid:]),
        "dual_pre": _avg(dual_precisions[:mid]),
        "dual_post": _avg(dual_precisions[mid:]),
        "retrains": triggers,
        "outcome_only_retrains": monitor.outcome_triggered_count,
    }


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_report(n: int = 3000, batch_size: int = 200) -> Dict:
    ref_size = batch_size * 2
    confound_rows = covariate_shift_confound(n=n)
    pure_rows = pure_concept_drift(n=n)

    return {
        "scenario_check": {
            "covariate_shift_confound": describe(confound_rows, name="covariate_shift_confound").__dict__,
            "pure_concept_drift": describe(pure_rows, name="pure_concept_drift").__dict__,
        },
        "confounded_scenario": run_static_vs_psi_only(confound_rows, batch_size, ref_size),
        "pure_concept_drift": {
            "psi_only": run_static_vs_psi_only(pure_rows, batch_size, ref_size),
            "dual_signal": run_static_vs_dual(pure_rows, batch_size, ref_size),
        },
    }


def format_report(report: Dict) -> str:
    lines = ["Scenario check: what actually moved across the drift point", "=" * 70]
    for name, check in report["scenario_check"].items():
        lines.append(
            f"  {name:<26} base rate {check['pre_base_rate']:.1%} -> "
            f"{check['post_base_rate']:.1%}   feature shift {check['feature_shift']:.1%}"
        )
    lines.append("")

    lines += [
        "Confounded scenario (coefficient + input distribution + base rate all",
        "change together, matching the original benchmark)",
        "=" * 70,
    ]
    c = report["confounded_scenario"]
    lines.append(f"{'policy':<12}{'pre-drift':>12}{'post-drift':>13}")
    lines.append(f"{'static':<12}{c['static_pre']:>11.0%}{c['static_post']:>13.0%}")
    lines.append(f"{'psi_only':<12}{c['psi_only_pre']:>11.0%}{c['psi_only_post']:>13.0%}")
    lines.append("")

    lines += [
        "Pure concept drift (base rate and inputs held constant; only the",
        "feature-outcome relationship inverts)",
        "=" * 70,
    ]
    pcd = report["pure_concept_drift"]
    psi = pcd["psi_only"]
    dual = pcd["dual_signal"]
    lines.append(f"{'policy':<12}{'pre-drift':>12}{'post-drift':>13}{'retrains':>12}")
    lines.append(f"{'static':<12}{psi['static_pre']:>11.0%}{psi['static_post']:>13.0%}{'':>12}")
    lines.append(
        f"{'psi_only':<12}{psi['psi_only_pre']:>11.0%}{psi['psi_only_post']:>13.0%}"
        f"{len(psi['psi_retrains']):>12}"
    )
    lines.append(
        f"{'dual_signal':<12}{dual['dual_pre']:>11.0%}{dual['dual_post']:>13.0%}"
        f"{len(dual['retrains']):>12}"
    )
    lines.append("")
    lines.append(
        f"Of {len(dual['retrains'])} dual_signal retrains, "
        f"{dual['outcome_only_retrains']} were triggered by outcome drift alone -- "
        "retrains PSI would have missed entirely."
    )
    lines.append(f"PSI-only retrained {len(psi['psi_retrains'])} time(s) on this scenario.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report(n=args.n, batch_size=args.batch_size)
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
