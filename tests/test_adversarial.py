"""Tests for the confound in the original drift scenario, and the fix for it."""

from __future__ import annotations

import pytest

from churnfm.data import generate
from churnfm.drift import assess
from churnfm.eval_v2 import (
    build_report,
    run_static_vs_dual,
    run_static_vs_psi_only,
)
from churnfm.model import ChurnModel
from churnfm.monitor import ChurnMonitor, _precision_recall
from churnfm.monitor_v2 import ChurnMonitorV2
from churnfm.outcome_drift import assess_outcomes, log_loss
from churnfm.scenarios import (
    covariate_shift_confound,
    describe,
    holdout_concept_drift,
    pure_concept_drift,
)


# --- the confound in the original scenario ----------------------------------

def test_original_scenario_confounds_base_rate_with_drift():
    """The 3.2% -> 68.4% swing this project's own headline number rests on."""
    rows = covariate_shift_confound()
    check = describe(rows, name="confound")
    assert check.post_base_rate > check.pre_base_rate * 5


def test_pure_concept_drift_holds_the_base_rate_constant():
    rows = pure_concept_drift()
    check = describe(rows, name="pure")
    assert abs(check.post_base_rate - check.pre_base_rate) < 0.10


def test_pure_concept_drift_does_not_shift_the_inputs():
    rows = pure_concept_drift()
    check = describe(rows, name="pure")
    assert check.feature_shift < 0.05


def test_holdout_scenario_also_isolates_the_relationship():
    rows = holdout_concept_drift()
    check = describe(rows, name="holdout")
    assert abs(check.post_base_rate - check.pre_base_rate) < 0.10
    assert check.feature_shift < 0.05


# --- the finding: PSI is blind to a relationship inversion ------------------

def test_psi_stays_near_zero_across_a_pure_concept_drift():
    """The relationship inverts completely; PSI barely notices."""
    rows = pure_concept_drift(n=3000)
    ref = rows[:1500]
    model = ChurnModel().fit(ref)
    ref_scores = model.predict_proba_batch(ref)

    post_batch = rows[2700:3000]
    report = assess(ref_scores, model.predict_proba_batch(post_batch), threshold=0.25)
    assert report.psi < 0.10
    assert not report.drifted


def test_precision_collapses_while_psi_stays_silent():
    """The gap this whole module exists to close."""
    rows = pure_concept_drift(n=3000)
    ref = rows[:1500]
    model = ChurnModel().fit(ref)
    ref_scores = model.predict_proba_batch(ref)

    pre_batch = rows[500:800]
    post_batch = rows[2700:3000]
    pre_precision, _ = _precision_recall(pre_batch, model.predict_proba_batch(pre_batch))
    post_precision, _ = _precision_recall(post_batch, model.predict_proba_batch(post_batch))
    post_psi = assess(ref_scores, model.predict_proba_batch(post_batch), threshold=0.25)

    assert post_precision < pre_precision / 2
    assert not post_psi.drifted


def test_psi_only_monitor_never_retrains_on_pure_concept_drift():
    result = run_static_vs_psi_only(pure_concept_drift(n=3000), batch_size=200, ref_size=400)
    assert result["psi_retrains"] == []
    # Without a retrain the "adaptive" monitor is just the static model.
    assert result["psi_only_post"] == pytest.approx(result["static_post"], abs=0.02)


# --- outcome drift catches it -----------------------------------------------

def test_log_loss_of_a_perfect_prediction_is_near_zero():
    assert log_loss([0.99, 0.01], [1, 0]) < 0.02


def test_log_loss_penalizes_confident_wrong_predictions():
    confident_wrong = log_loss([0.01, 0.99], [1, 0])
    uncertain = log_loss([0.5, 0.5], [1, 0])
    assert confident_wrong > uncertain


def test_outcome_drift_fires_after_a_relationship_inversion():
    rows = pure_concept_drift(n=3000)
    ref = rows[:1500]
    model = ChurnModel().fit(ref)
    ref_scores = model.predict_proba_batch(ref)
    ref_labels = [r.churned for r in ref]

    post_batch = rows[2700:3000]
    report = assess_outcomes(
        ref_scores, ref_labels,
        model.predict_proba_batch(post_batch), [r.churned for r in post_batch],
    )
    assert report.drifted


def test_outcome_drift_does_not_fire_pre_drift():
    rows = pure_concept_drift(n=3000)
    ref = rows[:1500]
    model = ChurnModel().fit(ref)
    ref_scores = model.predict_proba_batch(ref)
    ref_labels = [r.churned for r in ref]

    later_pre_batch = rows[1200:1500]
    report = assess_outcomes(
        ref_scores, ref_labels,
        model.predict_proba_batch(later_pre_batch), [r.churned for r in later_pre_batch],
    )
    assert not report.drifted


def test_outcome_drift_declines_to_judge_a_tiny_batch():
    """A ratio computed from five labels is noise, not a signal."""
    report = assess_outcomes([0.5] * 100, [0] * 100, [0.9, 0.9], [0, 0], min_batch_size=30)
    assert not report.drifted
    assert report.batch_size == 2


def test_outcome_drift_handles_a_perfect_reference():
    """Zero reference loss must not raise a division error."""
    report = assess_outcomes([0.999] * 40, [1] * 40, [0.5] * 40, [1] * 40, min_batch_size=30)
    assert isinstance(report.degradation_ratio, float)


# --- the dual-signal monitor -------------------------------------------------

def test_dual_signal_monitor_retrains_where_psi_only_does_not():
    rows = pure_concept_drift(n=3000)
    result = run_static_vs_dual(rows, batch_size=200, ref_size=400)
    assert result["retrains"]
    assert result["outcome_only_retrains"] >= 1


def test_dual_signal_monitor_recovers_precision():
    rows = pure_concept_drift(n=3000)
    result = run_static_vs_dual(rows, batch_size=200, ref_size=400)
    assert result["dual_post"] > result["static_post"] + 0.15


def test_batch_result_v2_reports_which_signal_triggered():
    rows = pure_concept_drift(n=3000)
    ref_window = rows[:400]
    monitor = ChurnMonitorV2.fit(ref_window)
    stream = rows[400:]
    batches = [stream[i : i + 200] for i in range(0, len(stream), 200)]
    triggers = []
    for i, batch in enumerate(batches):
        result = monitor.process_batch(i, batch, batch)
        if result.retrained:
            triggers.append(result.trigger)
    assert triggers
    assert all(t in {"psi", "outcome", "psi+outcome"} for t in triggers)


def test_dual_signal_monitor_still_works_on_the_confounded_scenario():
    """The old scenario is not broken by adding the new signal."""
    result = run_static_vs_dual(covariate_shift_confound(n=2000), batch_size=200, ref_size=400)
    assert result["dual_post"] >= result["static_post"]


# --- the retrain-window lesson ----------------------------------------------

def test_retraining_on_a_mixed_regime_window_does_not_help():
    """Half old-regime, half new-regime labels teach opposite relationships."""
    rows = pure_concept_drift(n=3000)
    ref_window = rows[:400]
    stream = rows[400:]
    batch_size = 200
    batches = [stream[i : i + batch_size] for i in range(0, len(stream), batch_size)]

    monitor = ChurnMonitorV2.fit(ref_window)
    recent = list(ref_window)
    window_size = batch_size * 2
    mixed_precisions = []
    for i, batch in enumerate(batches):
        result = monitor.process_batch(i, batch, recent[-window_size:])
        mixed_precisions.append(result.precision)
        recent.extend(batch)

    late = sum(mixed_precisions[-4:]) / 4
    assert late < 0.30  # stays broken; mixed-regime retraining does not fix it


def test_retraining_on_only_the_triggering_batch_recovers():
    rows = pure_concept_drift(n=3000)
    ref_window = rows[:400]
    stream = rows[400:]
    batch_size = 200
    batches = [stream[i : i + batch_size] for i in range(0, len(stream), batch_size)]

    monitor = ChurnMonitorV2.fit(ref_window)
    fresh_precisions = []
    for i, batch in enumerate(batches):
        result = monitor.process_batch(i, batch, batch)
        fresh_precisions.append(result.precision)

    late = sum(fresh_precisions[-4:]) / 4
    assert late > 0.30


# --- held out, evaluated once ------------------------------------------------

def test_holdout_scenario_psi_only_fails_to_recover():
    rows = holdout_concept_drift()
    result = run_static_vs_psi_only(rows, batch_size=200, ref_size=400)
    assert result["psi_retrains"] == []


def test_holdout_scenario_dual_signal_recovers():
    rows = holdout_concept_drift()
    result = run_static_vs_dual(rows, batch_size=200, ref_size=400)
    assert result["dual_post"] > result["static_post"] + 0.15
    assert result["retrains"]


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report(n=1500, batch_size=150) == build_report(n=1500, batch_size=150)


def test_report_shows_psi_missing_and_dual_catching():
    report = build_report(n=3000, batch_size=200)
    pcd = report["pure_concept_drift"]
    assert len(pcd["psi_only"]["psi_retrains"]) == 0
    assert len(pcd["dual_signal"]["retrains"]) >= 1


def test_original_scenario_is_unaffected_by_the_new_modules():
    """The original benchmark's own claims still reproduce."""
    rows = generate(n=3000, seed=0)
    ref_window = rows[:400]
    static_model = ChurnModel().fit(ref_window)
    monitor = ChurnMonitor.fit(ref_window)
    stream = rows[400:]
    batches = [stream[i : i + 200] for i in range(0, len(stream), 200)]
    recent = list(ref_window)
    static_post, adaptive_post = [], []
    mid = len(batches) // 2
    for i, batch in enumerate(batches):
        static_precision, _ = _precision_recall(batch, static_model.predict_proba_batch(batch))
        result = monitor.process_batch(i, batch, recent[-400:])
        recent.extend(batch)
        if i >= mid:
            static_post.append(static_precision)
            adaptive_post.append(result.precision)
    assert sum(adaptive_post) / len(adaptive_post) >= sum(static_post) / len(static_post)
