# 📉 ChurnFM

**A churn classifier with drift detection and automated retraining.**

![CI](https://github.com/ahmeddoghri/churnfm/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Detect drift with PSI and retrain automatically — before precision rots.**
> In the benchmark, a static model recovers to **80%** post-drift accuracy
> while the adaptive one reaches **89%**. Zero deps: `python -m churnfm.eval`.

Most churn models are trained once and left running until someone notices
precision quietly dropped. ChurnFM instead **monitors the prediction
distribution** with the Population Stability Index (PSI) and **retrains
automatically** the moment the underlying relationship shifts — a pricing
change, a new competitor, a product pivot — instead of waiting for a
dashboard to look wrong.

Runs with **zero dependencies and zero API keys** (pure-stdlib logistic
regression trained by gradient descent). The point isn't the model
architecture — swap `ChurnModel` for a real tabular foundation model (the
`google/tabfm`-class of pretrained tabular classifiers trending on Hugging
Face right now) or a gradient-boosted-tree model via the same
`fit`/`predict_proba` interface. The point is the **monitor-and-retrain loop**
around it.

---

## The result in one number

A synthetic B2B subscription stream with a concept drift injected at the
midpoint (a pricing change makes price-sensitivity the dominant churn driver —
the classic way churn models silently rot in production):

```bash
python -m churnfm.eval
```
```
batches: 13  (drift injected at stream midpoint)

policy         pre-drift precision    post-drift precision
static                        15%                    80%
adaptive                      15%                    89%

adaptive retrained at batches: [6, 7, 8]
```

Both policies score identically before the drift (as they should — same
model). After the pricing relationship changes, the static model's precision
stalls while the PSI monitor catches the shift and triggers a retrain,
recovering precision. (Precision here is **precision@k** with k = actual
positives — the standard ranking metric for imbalanced churn, since a fixed
probability cutoff is meaningless when churn is a single-digit-percent event.)

## Install

```bash
git clone https://github.com/ahmeddoghri/churnfm
cd churnfm && pip install -e .
python examples/quickstart.py
```

## Monitor a stream

```

Or with Docker:

```bash
docker build -t churnfm .
docker run --rm churnfm
```python
from churnfm import ChurnMonitor, generate

rows = generate(n=1000, seed=0, drift_at=0.5)
monitor = ChurnMonitor.fit(rows[:400])

recent = list(rows[:400])
for i in range(400, len(rows), 200):
    batch = rows[i:i + 200]
    result = monitor.process_batch(i, batch, recent[-400:])
    recent.extend(batch)
    print(result.drift.psi, result.retrained, result.precision)
```

## How it works

```
ChurnMonitor.fit(reference_window)
  └─ trains ChurnModel, stores reference prediction distribution

process_batch(batch, recent_history)
  ├─ score the batch
  ├─ PSI(reference_scores, batch_scores)   -- distribution shift, not just accuracy
  └─ if PSI >= threshold: retrain on recent_history, reset reference distribution
```

Retraining on a **recent sliding window** rather than all accumulated history
matters: mixing in stale pre-drift examples would keep re-triggering drift
indefinitely and the model would never stabilize after the relationship
actually settles into its new state.

## Bring your own model

```python
class MyTabularModel:
    def fit(self, rows): ...
    def predict_proba(self, row): ...
    def predict_proba_batch(self, rows): ...

ChurnMonitor(model=MyTabularModel().fit(reference_rows), reference_scores=[...])
```

## Tests

```bash
pip install pytest && pytest -q      # 6 passing
```

## License

MIT © Ahmed Doghri
