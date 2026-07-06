from churnfm import ChurnModel, ChurnMonitor, assess, generate
from churnfm.eval import run


def test_model_predicts_probability_in_range():
    rows = generate(n=500, seed=1)
    model = ChurnModel().fit(rows)
    for r in rows[:20]:
        p = model.predict_proba(r)
        assert 0.0 <= p <= 1.0


def test_model_learns_signal_better_than_chance():
    rows = generate(n=1500, seed=2, drift_at=2.0)  # no drift within this sample
    train, test = rows[:1000], rows[1000:]
    model = ChurnModel().fit(train)
    probs = model.predict_proba_batch(test)
    # simple separation check: mean predicted prob should be higher for churners
    churn_probs = [p for p, r in zip(probs, test) if r.churned == 1]
    stay_probs = [p for p, r in zip(probs, test) if r.churned == 0]
    assert sum(churn_probs) / len(churn_probs) > sum(stay_probs) / len(stay_probs)


def test_psi_detects_shifted_distribution():
    ref = [0.11, 0.13, 0.15, 0.12, 0.14] * 20
    same = [0.12, 0.14, 0.13, 0.11, 0.15] * 20   # same bin, reshuffled
    shifted = [0.8, 0.85, 0.9, 0.75, 0.82] * 20
    assert assess(ref, same).psi < 0.1
    assert assess(ref, shifted).drifted


def test_monitor_retrains_on_drift():
    rows = generate(n=2000, seed=0, drift_at=0.5)
    ref = rows[:400]
    monitor = ChurnMonitor.fit(ref)
    history = list(ref)
    stream = rows[400:]
    batches = [stream[i:i + 200] for i in range(0, len(stream), 200)]
    retrained_any = False
    for i, b in enumerate(batches):
        res = monitor.process_batch(i, b, history)
        history.extend(b)
        if res.retrained:
            retrained_any = True
    assert retrained_any


def test_adaptive_beats_static_after_drift():
    res = run(n=3000, batch_size=200, seed=0)
    assert res["adaptive_post_drift"] > res["static_post_drift"]
    assert res["retrain_batches"]  # at least one retrain happened


def test_adaptive_matches_static_before_drift():
    res = run(n=3000, batch_size=200, seed=0)
    # before drift, adaptive shouldn't be meaningfully worse than static
    assert res["adaptive_pre_drift"] >= res["static_pre_drift"] - 0.05
