"""Train a churn model, monitor a stream for drift, and see retraining kick in.
Run: python examples/quickstart.py
"""
from churnfm import ChurnModel, ChurnMonitor, generate
from churnfm.eval import run

rows = generate(n=1000, seed=0, drift_at=0.5)
train, test = rows[:400], rows[400:500]

model = ChurnModel().fit(train)
probs = model.predict_proba_batch(test)
print(f"Trained on {len(train)} rows. Mean predicted churn prob on holdout: "
      f"{sum(probs)/len(probs):.3f}")

print("\n--- streaming monitor: does it catch the mid-stream drift? ---")
monitor = ChurnMonitor.fit(rows[:400])
recent = list(rows[:400])
stream = rows[400:]
for i in range(0, len(stream), 200):
    batch = stream[i:i + 200]
    res = monitor.process_batch(i // 200, batch, recent[-400:])
    recent.extend(batch)
    flag = " <- RETRAINED" if res.retrained else ""
    print(f"batch {i//200}: psi={res.drift.psi:.3f}  precision@k={res.precision:.0%}{flag}")

print("\n--- does retraining pay off? (full benchmark) ---")
r = run(n=3000, batch_size=200, seed=0)
print(f"static   pre-drift={r['static_pre_drift']:.0%}   post-drift={r['static_post_drift']:.0%}")
print(f"adaptive pre-drift={r['adaptive_pre_drift']:.0%}   post-drift={r['adaptive_post_drift']:.0%}")
